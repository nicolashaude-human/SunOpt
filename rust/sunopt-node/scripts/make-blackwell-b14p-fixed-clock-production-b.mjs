// rust/sunopt-node/scripts/make-blackwell-b14p-fixed-clock-production-b.mjs
//
// B14P:
//   1. Production Approach-B causal path:
//        - persistent FP32 annual arrays/tank state in VRAM
//        - per-step GPU energy-cut calculation from requestedElectricalEnergy
//        - B14M/B14K depth-2 speculative causal search
//        - mapped host output + event polling
//        - automatic FP64 fallback for unsupported B jobs or CUDA errors
//        - existing runLockstepDynamic/Into API remains unchanged
//        - default path B; set SUNOPT_CUDA_CAUSAL_PATH=A to force legacy FP64
//   2. Fixed-clock benchmark workflow:
//        - starts existing NVAPI guard at 1050 mV / native 2865 MHz +625 MHz
//        - fan 100% show effect
//        - samples clock under active GPU load
//        - rejects run unless active SM clock remains 3460..3510 MHz
//   3. Re-runs pageable vs pinned B14L:
//        - warmup per variant
//        - 5 measured runs each, ABBA alternating
//        - median authoritative
//   4. Replays the new production B path against the persistent strict FP64
//      reference, warmup + 5 measured runs.
//
// One user command after installing this generator watcher:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14p-fixed-clock-production-b.mjs
//
// It creates the real watcher package, waits for the already-running watcher,
// builds strict CUDA, and starts the benchmark automatically.
//
// Failure files:
//   updates/outbox/failed-prompt.txt
//   updates/outbox/failed-build.txt
//   updates/outbox/failed-run.txt
//
// Persistent reference:
//   rust/sunopt-node/benchmarks/references/blackwell-b14m-strict-reference.bin
//
// Reports:
//   updates/outbox/blackwell-b14p-fixed-clock-production-b.json
//   updates/outbox/llm.txt

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");
const nativeRoot = path.join(root, "rust", "sunopt-node");
const inbox = path.join(root, "updates", "inbox");
const rejected = path.join(root, "updates", "rejected");
const outbox = path.join(root, "updates", "outbox");
const hashFile = path.join(root, "updates", "history", "hash.txt");

const H_REL = "rust/sunopt-node/cuda/cuda_context.h";
const CU_REL = "rust/sunopt-node/cuda/cuda_context.cu";
const RS_REL = "rust/sunopt-node/src/cuda_context.rs";
const CHILD_REL = "rust/sunopt-node/scripts/blackwell-b14p-child.mjs";
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14p-fixed-clock-production-b.mjs";

const hPath = path.join(root, H_REL);
const cuPath = path.join(root, CU_REL);
const rsPath = path.join(root, RS_REL);

fs.mkdirSync(outbox, { recursive: true });

const failedPrompt = path.join(outbox, "failed-prompt.txt");
const failedBuild = path.join(outbox, "failed-build.txt");
const failedRun = path.join(outbox, "failed-run.txt");

for (const file of [failedPrompt, failedBuild, failedRun]) {
  try { fs.unlinkSync(file); } catch {}
}

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function writeFailure(file, stage, error, extra = "") {
  fs.writeFileSync(
    file,
    [
      `stage=${stage}`,
      `time=${new Date().toISOString()}`,
      `error=${error?.stack ?? error}`,
      extra,
      "",
    ].join("\n"),
    "utf8",
  );
}

function insertBefore(text, anchor, block, label) {
  const marker = block.trimStart().split(/\r?\n/, 1)[0];
  if (text.includes(marker)) {
    throw new Error(`${label}: block already present: ${marker}`);
  }
  const pos = text.indexOf(anchor);
  if (pos < 0) {
    throw new Error(`${label}: anchor not found: ${anchor}`);
  }
  return text.slice(0, pos) + block.trimEnd() + "\n\n" + text.slice(pos);
}

function replaceExactly(text, oldText, newText, label) {
  const count = text.split(oldText).length - 1;
  if (count !== 1) {
    throw new Error(`${label}: expected exactly 1 occurrence, found ${count}`);
  }
  return text.replace(oldText, newText);
}

function sleepMs(ms) {
  const wait = new Int32Array(new SharedArrayBuffer(4));
  Atomics.wait(wait, 0, 0, ms);
}

function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; ++k) {
      c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
    }
  }
  return (c ^ 0xffffffff) >>> 0;
}
function u16(n) { const b = Buffer.alloc(2); b.writeUInt16LE(n >>> 0); return b; }
function u32(n) { const b = Buffer.alloc(4); b.writeUInt32LE(n >>> 0); return b; }

try {
  if (!fs.existsSync(hashFile)) {
    throw new Error(`Missing current release hash: ${hashFile}`);
  }

  const parent = fs.readFileSync(hashFile, "utf8").trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(parent)) {
    throw new Error(`Invalid current release hash: ${parent}`);
  }

  for (const required of [hPath, cuPath, rsPath]) {
    if (!fs.existsSync(required)) {
      throw new Error(`Missing current source file: ${required}`);
    }
  }

  let h = fs.readFileSync(hPath, "utf8");
  let cu = fs.readFileSync(cuPath, "utf8");
  let rs = fs.readFileSync(rsPath, "utf8");

  const H_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14P_PRODUCTION_B_BEGIN
typedef struct SunoptB14PContext SunoptB14PContext;

SUNOPT_CUDA_EXPORT int sunopt_cuda_b14p_create(
    int device_id,
    const float* host_export_power,
    const float* host_demand_energy,
    size_t data_length,
    SunoptB14PContext** out_context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_b14p_destroy(
    SunoptB14PContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_b14p_configure(
    SunoptB14PContext* context,
    const float* host_tank_parameters,
    size_t tank_count
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_b14p_run_step(
    SunoptB14PContext* context,
    size_t current_index,
    size_t horizon_steps,
    const float* host_dynamic_values,
    size_t tank_count,
    float* host_cut_results,
    uint32_t* host_status
);
// SUNOPT_BLACKWELL_B14P_PRODUCTION_B_END
`.trim();

  const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14P_PRODUCTION_B_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14P_WARPS = 3u;
constexpr unsigned int SUNOPT_B14P_THREADS = 96u;
constexpr unsigned int SUNOPT_B14P_MASK = 0xffffffffu;

__device__ __forceinline__ float
sunopt_b14p_warp_sum(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(
            SUNOPT_B14P_MASK,
            value,
            delta
        );
    }
    return value;
}

__device__ __forceinline__ float
sunopt_b14p_warp_max(float value) {
    for (int delta = 16; delta > 0; delta >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(
                SUNOPT_B14P_MASK,
                value,
                delta
            )
        );
    }
    return value;
}

__device__ __forceinline__ size_t
sunopt_b14p_wrap_index(
    size_t current_index,
    size_t offset,
    size_t data_length
) {
    size_t index =
        current_index + offset;

    if (index >= data_length) {
        index -= data_length;
    }

    return index;
}

__device__ __forceinline__ float
sunopt_b14p_energy_at_cut_warp(
    const float* export_power,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    float cut,
    float max_hp,
    float dt_hours,
    int lane
) {
    float local = 0.0f;

    for (
        size_t offset =
            static_cast<size_t>(lane);
        offset < horizon;
        offset += 32u
    ) {
        const size_t index =
            sunopt_b14p_wrap_index(
                current_index,
                offset,
                data_length
            );

        const float raw =
            export_power[index] -
            cut;

        const float hp =
            fminf(
                max_hp,
                fmaxf(
                    0.0f,
                    raw
                )
            );

        local += hp * dt_hours;
    }

    const float total =
        sunopt_b14p_warp_sum(
            local
        );

    return __shfl_sync(
        SUNOPT_B14P_MASK,
        total,
        0
    );
}

__device__ __forceinline__ float
sunopt_b14p_energy_cut_warp(
    const float* export_power,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    float requested_electrical_energy,
    float max_hp,
    float dt_hours,
    int lane
) {
    float local_max = 0.0f;

    for (
        size_t offset =
            static_cast<size_t>(lane);
        offset < horizon;
        offset += 32u
    ) {
        const size_t index =
            sunopt_b14p_wrap_index(
                current_index,
                offset,
                data_length
            );

        local_max =
            fmaxf(
                local_max,
                export_power[index]
            );
    }

    const float maximum_export =
        __shfl_sync(
            SUNOPT_B14P_MASK,
            sunopt_b14p_warp_max(
                local_max
            ),
            0
        );

    const float maximum_schedulable =
        sunopt_b14p_energy_at_cut_warp(
            export_power,
            data_length,
            current_index,
            horizon,
            0.0f,
            max_hp,
            dt_hours,
            lane
        );

    const float target =
        fminf(
            maximum_schedulable,
            fmaxf(
                0.0f,
                requested_electrical_energy
            )
        );

    if (
        target <= 0.0f ||
        maximum_export <= 0.0f
    ) {
        return maximum_export;
    }

    if (
        target >=
        maximum_schedulable
    ) {
        return 0.0f;
    }

    float lower = 0.0f;
    float upper = maximum_export;

    for (
        int iteration = 0;
        iteration < 32;
        ++iteration
    ) {
        const float middle =
            (lower + upper) *
            0.5f;

        const float energy =
            sunopt_b14p_energy_at_cut_warp(
                export_power,
                data_length,
                current_index,
                horizon,
                middle,
                max_hp,
                dt_hours,
                lane
            );

        if (lane == 0) {
            if (energy > target) {
                lower = middle;
            } else {
                upper = middle;
            }
        }

        lower =
            __shfl_sync(
                SUNOPT_B14P_MASK,
                lower,
                0
            );

        upper =
            __shfl_sync(
                SUNOPT_B14P_MASK,
                upper,
                0
            );
    }

    return (lower + upper) * 0.5f;
}

/*
 * Production B kernel.
 *
 * tank_parameters stride = 7 floats:
 *   0 target_storage_energy
 *   1 maximum_heat_pump_electrical_power
 *   2 hp_thermal_energy_per_electrical_power (= COP * dt)
 *   3 heat_loss_retention
 *   4 heat_loss_intercept
 *   5 affine_supported
 *   6 time_step_hours
 *
 * dynamic_values stride = 3 floats:
 *   0 current_storage_energy
 *   1 minimum_reserve_energy
 *   2 requested_electrical_energy
 *
 * output stride = two arrays in one mapped allocation:
 *   [0 .. tank_count)               cut power
 *   [tank_count .. 2*tank_count)    status encoded as float
 */
__global__ void
sunopt_b14p_production_kernel(
    const float* __restrict__ export_power,
    const float* __restrict__ demand_energy,
    size_t data_length,
    const float* __restrict__ tank_parameters,
    size_t tank_count,
    size_t current_index,
    size_t horizon,
    const float* __restrict__ dynamic_values,
    float* __restrict__ mapped_output
) {
    const size_t tank =
        static_cast<size_t>(
            blockIdx.x
        );

    if (tank >= tank_count) {
        return;
    }

    const unsigned int tid =
        static_cast<unsigned int>(
            threadIdx.x
        );

    const unsigned int warp =
        tid >> 5u;

    const int lane =
        static_cast<int>(
            tid & 31u
        );

    const size_t tb =
        tank * 7u;

    const size_t db =
        tank * 3u;

    const float target =
        tank_parameters[tb + 0u];

    const float max_hp =
        tank_parameters[tb + 1u];

    const float hp_factor =
        tank_parameters[tb + 2u];

    const float retention =
        tank_parameters[tb + 3u];

    const float loss_intercept =
        tank_parameters[tb + 4u];

    const bool affine_supported =
        tank_parameters[tb + 5u] >
        0.5f;

    const float dt_hours =
        tank_parameters[tb + 6u];

    const float current =
        dynamic_values[db + 0u];

    const float reserve =
        fminf(
            target,
            fmaxf(
                0.0f,
                dynamic_values[db + 1u]
            )
        );

    const float requested =
        dynamic_values[db + 2u];

    __shared__ float shared_energy_cut;
    __shared__ int shared_precheck;
    __shared__ int
        shared_survives[
            SUNOPT_B14P_WARPS
        ];

    __shared__ float shared_lower;
    __shared__ float shared_upper;

    if (tid == 0u) {
        mapped_output[tank] = 0.0f;
        mapped_output[
            tank_count + tank
        ] = 3.0f;
    }

    __syncthreads();

    const bool supported =
        affine_supported &&
        retention > 0.0f &&
        reserve >
            static_cast<float>(
                SUNOPT_B14C_EPSILON
            ) &&
        data_length > 0u &&
        horizon > 0u &&
        horizon <= data_length &&
        current_index < data_length;

    if (!supported) {
        return;
    }

    if (warp == 0u) {
        const float energy_cut =
            sunopt_b14p_energy_cut_warp(
                export_power,
                data_length,
                current_index,
                horizon,
                requested,
                max_hp,
                dt_hours,
                lane
            );

        if (lane == 0) {
            shared_energy_cut =
                energy_cut;
        }
    }

    __syncthreads();

    const float energy_cut =
        shared_energy_cut;

    if (warp == 0u) {
        const bool survives =
            sunopt_b14m_fixed_survives_warp(
                export_power,
                demand_energy,
                data_length,
                current_index,
                horizon,
                energy_cut,
                current,
                reserve,
                target,
                max_hp,
                hp_factor,
                retention,
                loss_intercept,
                lane
            );

        if (lane == 0) {
            shared_precheck =
                survives ? 1 : 0;
        }
    }

    __syncthreads();

    if (shared_precheck != 0) {
        if (tid == 0u) {
            mapped_output[tank] =
                energy_cut;

            mapped_output[
                tank_count + tank
            ] =
                0.0f;
        }
        return;
    }

    if (warp == 0u) {
        const bool survives =
            sunopt_b14m_fixed_survives_warp(
                export_power,
                demand_energy,
                data_length,
                current_index,
                horizon,
                0.0f,
                current,
                reserve,
                target,
                max_hp,
                hp_factor,
                retention,
                loss_intercept,
                lane
            );

        if (lane == 0) {
            shared_precheck =
                survives ? 1 : 0;
        }
    }

    __syncthreads();

    if (shared_precheck == 0) {
        if (tid == 0u) {
            mapped_output[tank] =
                0.0f;

            mapped_output[
                tank_count + tank
            ] =
                1.0f;
        }
        return;
    }

    if (tid == 0u) {
        shared_lower = 0.0f;
        shared_upper = energy_cut;
    }

    __syncthreads();

    /*
     * Depth-2 speculative binary tree:
     * 3 warps evaluate the root and its two children.
     * Two exact binary decisions are resolved per round,
     * therefore 10 rounds reproduce the 20-decision search.
     */
    for (
        unsigned int round = 0u;
        round < 10u;
        ++round
    ) {
        if (warp < SUNOPT_B14P_WARPS) {
            const unsigned int node_number =
                warp + 1u;

            unsigned int level = 0u;
            unsigned int first_at_level = 1u;

            while (
                (
                    first_at_level <<
                    1u
                ) <= node_number
            ) {
                first_at_level <<= 1u;
                ++level;
            }

            const unsigned int position =
                node_number -
                first_at_level;

            float lo = shared_lower;
            float hi = shared_upper;

            if (level > 0u) {
                for (
                    unsigned int path_level = 0u;
                    path_level < level;
                    ++path_level
                ) {
                    const float middle =
                        (lo + hi) *
                        0.5f;

                    const unsigned int shift =
                        level -
                        1u -
                        path_level;

                    const unsigned int direction =
                        (
                            position >>
                            shift
                        ) &
                        1u;

                    if (direction != 0u) {
                        lo = middle;
                    } else {
                        hi = middle;
                    }
                }
            }

            const float candidate =
                (lo + hi) * 0.5f;

            const bool survives =
                sunopt_b14m_fixed_survives_warp(
                    export_power,
                    demand_energy,
                    data_length,
                    current_index,
                    horizon,
                    candidate,
                    current,
                    reserve,
                    target,
                    max_hp,
                    hp_factor,
                    retention,
                    loss_intercept,
                    lane
                );

            if (lane == 0) {
                shared_survives[warp] =
                    survives ? 1 : 0;
            }
        }

        __syncthreads();

        if (tid == 0u) {
            float lo = shared_lower;
            float hi = shared_upper;
            unsigned int node = 0u;

            for (
                unsigned int level = 0u;
                level < 2u;
                ++level
            ) {
                const float middle =
                    (lo + hi) *
                    0.5f;

                if (
                    shared_survives[node] !=
                    0
                ) {
                    lo = middle;
                    node =
                        node * 2u + 2u;
                } else {
                    hi = middle;
                    node =
                        node * 2u + 1u;
                }
            }

            shared_lower = lo;
            shared_upper = hi;
        }

        __syncthreads();
    }

    if (tid == 0u) {
        mapped_output[tank] =
            shared_lower;

        mapped_output[
            tank_count + tank
        ] =
            2.0f;
    }
}

} // namespace

struct SunoptB14PContext {
    int device_id;
    size_t data_length;
    size_t tank_count;

    cudaStream_t stream;
    cudaEvent_t done_event;

    float* d_export;
    float* d_demand;
    float* d_tanks;
    float* d_dynamic;

    float* host_mapped_output;
    float* d_mapped_output;
};

extern "C" int
sunopt_cuda_b14p_create(
    int device_id,
    const float* host_export_power,
    const float* host_demand_energy,
    size_t data_length,
    SunoptB14PContext** out_context
) {
    if (
        !host_export_power ||
        !host_demand_energy ||
        data_length == 0u ||
        !out_context
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    *out_context = nullptr;

    cudaError_t err =
        cudaSetDevice(device_id);

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    SunoptB14PContext* context =
        new SunoptB14PContext{};

    context->device_id = device_id;
    context->data_length = data_length;
    context->tank_count = 0u;
    context->stream = nullptr;
    context->done_event = nullptr;
    context->d_export = nullptr;
    context->d_demand = nullptr;
    context->d_tanks = nullptr;
    context->d_dynamic = nullptr;
    context->host_mapped_output = nullptr;
    context->d_mapped_output = nullptr;

    auto cleanup = [&]() {
        if (context->host_mapped_output) {
            cudaFreeHost(
                context->host_mapped_output
            );
        }

        if (context->d_dynamic) {
            cudaFree(context->d_dynamic);
        }

        if (context->d_tanks) {
            cudaFree(context->d_tanks);
        }

        if (context->d_demand) {
            cudaFree(context->d_demand);
        }

        if (context->d_export) {
            cudaFree(context->d_export);
        }

        if (context->done_event) {
            cudaEventDestroy(
                context->done_event
            );
        }

        if (context->stream) {
            cudaStreamDestroy(
                context->stream
            );
        }

        delete context;
    };

    err =
        cudaStreamCreateWithFlags(
            &context->stream,
            cudaStreamNonBlocking
        );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err =
        cudaEventCreateWithFlags(
            &context->done_event,
            cudaEventDisableTiming
        );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    const size_t annual_bytes =
        data_length * sizeof(float);

    err = cudaMalloc(
        reinterpret_cast<void**>(
            &context->d_export
        ),
        annual_bytes
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMalloc(
        reinterpret_cast<void**>(
            &context->d_demand
        ),
        annual_bytes
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        context->d_export,
        host_export_power,
        annual_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        context->d_demand,
        host_demand_energy,
        annual_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err =
        cudaStreamSynchronize(
            context->stream
        );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    *out_context = context;

    return static_cast<int>(
        cudaSuccess
    );
}

extern "C" int
sunopt_cuda_b14p_destroy(
    SunoptB14PContext* context
) {
    if (!context) {
        return static_cast<int>(
            cudaSuccess
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaError_t first_error =
        cudaSuccess;

#define SUNOPT_B14P_CLEAN(call) do { \
    const cudaError_t cleanup_err = (call); \
    if (first_error == cudaSuccess && cleanup_err != cudaSuccess) { \
        first_error = cleanup_err; \
    } \
} while(false)

    if (context->host_mapped_output) {
        SUNOPT_B14P_CLEAN(
            cudaFreeHost(
                context->host_mapped_output
            )
        );
    }

    if (context->d_dynamic) {
        SUNOPT_B14P_CLEAN(
            cudaFree(context->d_dynamic)
        );
    }

    if (context->d_tanks) {
        SUNOPT_B14P_CLEAN(
            cudaFree(context->d_tanks)
        );
    }

    if (context->d_demand) {
        SUNOPT_B14P_CLEAN(
            cudaFree(context->d_demand)
        );
    }

    if (context->d_export) {
        SUNOPT_B14P_CLEAN(
            cudaFree(context->d_export)
        );
    }

    if (context->done_event) {
        SUNOPT_B14P_CLEAN(
            cudaEventDestroy(
                context->done_event
            )
        );
    }

    if (context->stream) {
        SUNOPT_B14P_CLEAN(
            cudaStreamDestroy(
                context->stream
            )
        );
    }

#undef SUNOPT_B14P_CLEAN

    delete context;

    return static_cast<int>(
        first_error
    );
}

extern "C" int
sunopt_cuda_b14p_configure(
    SunoptB14PContext* context,
    const float* host_tank_parameters,
    size_t tank_count
) {
    if (
        !context ||
        !host_tank_parameters ||
        tank_count == 0u ||
        tank_count > 32u
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err =
        cudaStreamSynchronize(
            context->stream
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    if (context->host_mapped_output) {
        cudaFreeHost(
            context->host_mapped_output
        );

        context->host_mapped_output =
            nullptr;

        context->d_mapped_output =
            nullptr;
    }

    if (context->d_dynamic) {
        cudaFree(context->d_dynamic);
        context->d_dynamic = nullptr;
    }

    if (context->d_tanks) {
        cudaFree(context->d_tanks);
        context->d_tanks = nullptr;
    }

    const size_t tank_bytes =
        tank_count *
        7u *
        sizeof(float);

    const size_t dynamic_bytes =
        tank_count *
        3u *
        sizeof(float);

    const size_t output_bytes =
        tank_count *
        2u *
        sizeof(float);

    err = cudaMalloc(
        reinterpret_cast<void**>(
            &context->d_tanks
        ),
        tank_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMalloc(
        reinterpret_cast<void**>(
            &context->d_dynamic
        ),
        dynamic_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaHostAlloc(
        reinterpret_cast<void**>(
            &context->host_mapped_output
        ),
        output_bytes,
        cudaHostAllocMapped
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaHostGetDevicePointer(
        reinterpret_cast<void**>(
            &context->d_mapped_output
        ),
        context->host_mapped_output,
        0
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        context->d_tanks,
        host_tank_parameters,
        tank_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err =
        cudaStreamSynchronize(
            context->stream
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    context->tank_count =
        tank_count;

    return static_cast<int>(
        cudaSuccess
    );
}

extern "C" int
sunopt_cuda_b14p_run_step(
    SunoptB14PContext* context,
    size_t current_index,
    size_t horizon_steps,
    const float* host_dynamic_values,
    size_t tank_count,
    float* host_cut_results,
    uint32_t* host_status
) {
    if (
        !context ||
        !host_dynamic_values ||
        !host_cut_results ||
        !host_status ||
        tank_count == 0u ||
        tank_count != context->tank_count ||
        current_index >=
            context->data_length ||
        horizon_steps == 0u ||
        horizon_steps >
            context->data_length
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t dynamic_bytes =
        tank_count *
        3u *
        sizeof(float);

    err = cudaMemcpyAsync(
        context->d_dynamic,
        host_dynamic_values,
        dynamic_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    sunopt_b14p_production_kernel
        <<<static_cast<unsigned int>(
               tank_count
           ),
           SUNOPT_B14P_THREADS,
           0,
           context->stream>>>(
            context->d_export,
            context->d_demand,
            context->data_length,
            context->d_tanks,
            tank_count,
            current_index,
            horizon_steps,
            context->d_dynamic,
            context->d_mapped_output
        );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaEventRecord(
        context->done_event,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    for (;;) {
        err =
            cudaEventQuery(
                context->done_event
            );

        if (err == cudaSuccess) {
            break;
        }

        if (
            err !=
            cudaErrorNotReady
        ) {
            return static_cast<int>(err);
        }
    }

    for (
        size_t tank = 0u;
        tank < tank_count;
        ++tank
    ) {
        host_cut_results[tank] =
            context
                ->host_mapped_output[
                    tank
                ];

        host_status[tank] =
            static_cast<uint32_t>(
                context
                    ->host_mapped_output[
                        tank_count +
                        tank
                    ]
            );
    }

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_B14P_PRODUCTION_B_END
`.trim();

  const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14P_PRODUCTION_B_FFI_BEGIN
    fn sunopt_cuda_b14p_create(
        device_id: c_int,
        host_export_power: *const f32,
        host_demand_energy: *const f32,
        data_length: usize,
        out_context:
            *mut *mut std::ffi::c_void,
    ) -> c_int;

    fn sunopt_cuda_b14p_destroy(
        context: *mut std::ffi::c_void,
    ) -> c_int;

    fn sunopt_cuda_b14p_configure(
        context: *mut std::ffi::c_void,
        host_tank_parameters:
            *const f32,
        tank_count: usize,
    ) -> c_int;

    fn sunopt_cuda_b14p_run_step(
        context: *mut std::ffi::c_void,
        current_index: usize,
        horizon_steps: usize,
        host_dynamic_values:
            *const f32,
        tank_count: usize,
        host_cut_results: *mut f32,
        host_status: *mut u32,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14P_PRODUCTION_B_FFI_END
`.trim();

  const RS_CONTEXT_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14P_PRODUCTION_B_CONTEXT_BEGIN
#[derive(Debug)]
struct CudaB14PContext {
    raw:
        NonNull<
            std::ffi::c_void
        >,
}

unsafe impl Send
    for CudaB14PContext
{
}

impl CudaB14PContext {
    fn new(
        device_id: i32,
        export_power: &[f64],
        demand_energy: &[f64],
    ) -> Result<Self> {
        let export32 =
            export_power
                .iter()
                .map(
                    |value| {
                        *value as f32
                    },
                )
                .collect::<Vec<_>>();

        let demand32 =
            demand_energy
                .iter()
                .map(
                    |value| {
                        *value as f32
                    },
                )
                .collect::<Vec<_>>();

        let mut raw =
            std::ptr::null_mut();

        let code = unsafe {
            sunopt_cuda_b14p_create(
                device_id as c_int,
                export32.as_ptr(),
                demand32.as_ptr(),
                export32.len(),
                &mut raw,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_b14p_create",
                    code,
                ),
            );
        }

        let raw =
            NonNull::new(raw)
                .ok_or_else(|| {
                    Error::from_reason(
                        "sunopt_cuda_b14p_create returned success with a null context.",
                    )
                })?;

        Ok(Self { raw })
    }

    fn configure_from_packed(
        &mut self,
        packed: &[f64],
    ) -> Result<usize> {
        const STRIDE: usize = 12;

        if (
            packed.is_empty() ||
            packed.len() % STRIDE != 0
        ) {
            return Err(
                Error::from_reason(
                    "B14P packed configuration must be a non-zero multiple of 12.",
                ),
            );
        }

        let tank_count =
            packed.len() /
            STRIDE;

        let mut tanks =
            Vec::<f32>::with_capacity(
                tank_count * 7,
            );

        for tank_index
            in 0..tank_count
        {
            let tank =
                research_tank_from_packed(
                    packed,
                    tank_index,
                );

            tanks.extend_from_slice(
                &[
                    tank
                        .target_storage_energy
                        as f32,

                    tank
                        .maximum_heat_pump_electrical_power
                        as f32,

                    (
                        tank.cop *
                        tank.time_step_hours
                    ) as f32,

                    tank
                        .heat_loss_retention
                        as f32,

                    tank
                        .heat_loss_intercept
                        as f32,

                    if tank.affine_supported {
                        1.0
                    } else {
                        0.0
                    },

                    tank
                        .time_step_hours
                        as f32,
                ],
            );
        }

        let code = unsafe {
            sunopt_cuda_b14p_configure(
                self.raw.as_ptr(),
                tanks.as_ptr(),
                tank_count,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_b14p_configure",
                    code,
                ),
            );
        }

        Ok(tank_count)
    }

    fn run_step(
        &mut self,
        current_index: usize,
        horizon_steps: usize,
        dynamic: &[f32],
        cuts: &mut [f32],
        status: &mut [u32],
    ) -> Result<()> {
        let tank_count =
            cuts.len();

        if (
            tank_count == 0 ||
            status.len() != tank_count ||
            dynamic.len() !=
                tank_count * 3
        ) {
            return Err(
                Error::from_reason(
                    "B14P run_step buffer shape mismatch.",
                ),
            );
        }

        let code = unsafe {
            sunopt_cuda_b14p_run_step(
                self.raw.as_ptr(),
                current_index,
                horizon_steps,
                dynamic.as_ptr(),
                tank_count,
                cuts.as_mut_ptr(),
                status.as_mut_ptr(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_b14p_run_step",
                    code,
                ),
            );
        }

        Ok(())
    }
}

impl Drop
    for CudaB14PContext
{
    fn drop(&mut self) {
        let _ = unsafe {
            sunopt_cuda_b14p_destroy(
                self.raw.as_ptr(),
            )
        };
    }
}

#[napi(object)]
pub struct BlackwellB14PProductionReplayResult {
    pub snapshot_path: String,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub wall_ms: f64,
    pub checksum_hex: String,
    pub status_mismatches_vs_strict: u32,
    pub cut_mismatches_1e4_vs_strict: u32,
    pub max_abs_cut_error_vs_strict: f64,
    pub unsupported_jobs: u32,
}

#[napi]
pub fn run_blackwell_b14p_production_replay(
    snapshot_path: Option<String>,
    strict_reference_path: String,
    device_id: Option<i32>,
) -> Result<BlackwellB14PProductionReplayResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path
            );

    let snapshot =
        read_blackwell_snapshot(
            &path
        ).map_err(
            |error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell snapshot {}: {error}",
                        path.display(),
                    ),
                )
            },
        )?;

    if (
        snapshot.steps.is_empty() ||
        snapshot.tank_count == 0
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty.",
            ),
        );
    }

    let mut context =
        CudaB14PContext::new(
            device_id.unwrap_or(0),
            &snapshot.export_power,
            &snapshot.demand_energy,
        )?;

    let tank_count =
        context
            .configure_from_packed(
                &snapshot.packed_config,
            )?;

    if tank_count !=
        snapshot.tank_count
    {
        return Err(
            Error::from_reason(
                "B14P tank-count mismatch.",
            ),
        );
    }

    let jobs =
        snapshot
            .steps
            .len()
            .checked_mul(
                tank_count
            )
            .ok_or_else(|| {
                Error::from_reason(
                    "B14P job-count overflow.",
                )
            })?;

    let reference_path =
        PathBuf::from(
            strict_reference_path
        );

    let reference =
        std::fs::read(
            &reference_path
        ).map_err(
            |error| {
                Error::from_reason(
                    format!(
                        "Unable to read strict reference {}: {error}",
                        reference_path.display(),
                    ),
                )
            },
        )?;

    const HEADER: usize = 16;

    if (
        reference.len() < HEADER ||
        &reference[0..8] !=
            b"B14MREF1"
    ) {
        return Err(
            Error::from_reason(
                "Invalid B14M strict-reference header.",
            ),
        );
    }

    let mut count_bytes =
        [0_u8; 8];

    count_bytes
        .copy_from_slice(
            &reference[8..16],
        );

    let count =
        u64::from_le_bytes(
            count_bytes
        ) as usize;

    if count != jobs {
        return Err(
            Error::from_reason(
                format!(
                    "B14P strict-reference job count mismatch: expected {jobs}, got {count}.",
                ),
            ),
        );
    }

    let expected_bytes =
        HEADER +
        jobs * 8 +
        jobs * 4;

    if reference.len() !=
        expected_bytes
    {
        return Err(
            Error::from_reason(
                "B14P strict-reference byte size mismatch.",
            ),
        );
    }

    let cut_base =
        HEADER;

    let status_base =
        HEADER +
        jobs * 8;

    let mut dynamic32 =
        vec![
            0.0_f32;
            tank_count * 3
        ];

    let mut cuts =
        vec![
            0.0_f32;
            tank_count
        ];

    let mut status =
        vec![
            0_u32;
            tank_count
        ];

    let mut checksum =
        1469598103934665603_u64;

    let mut status_mismatches =
        0_u32;

    let mut cut_mismatches_1e4 =
        0_u32;

    let mut max_abs_cut_error =
        0.0_f64;

    let mut unsupported_jobs =
        0_u32;

    let wall_start =
        Instant::now();

    for (
        step_index,
        step
    ) in snapshot
        .steps
        .iter()
        .enumerate()
    {
        if step.dynamic.len() !=
            tank_count * 3
        {
            return Err(
                Error::from_reason(
                    format!(
                        "B14P snapshot dynamic shape mismatch at step {step_index}.",
                    ),
                ),
            );
        }

        for (
            target,
            source
        ) in dynamic32
            .iter_mut()
            .zip(
                step.dynamic
                    .iter()
            )
        {
            *target =
                *source as f32;
        }

        context.run_step(
            step.current_index as usize,
            step.horizon_steps as usize,
            &dynamic32,
            &mut cuts,
            &mut status,
        )?;

        for tank in 0..tank_count {
            let job =
                step_index *
                tank_count +
                tank;

            let value =
                cuts[tank] as f64;

            blackwell_fnv1a_update(
                &mut checksum,
                value,
            );

            let cut_offset =
                cut_base +
                job * 8;

            let mut strict_cut_bytes =
                [0_u8; 8];

            strict_cut_bytes
                .copy_from_slice(
                    &reference[
                        cut_offset
                            ..
                        cut_offset + 8
                    ],
                );

            let strict_cut =
                f64::from_le_bytes(
                    strict_cut_bytes
                );

            let status_offset =
                status_base +
                job * 4;

            let mut strict_status_bytes =
                [0_u8; 4];

            strict_status_bytes
                .copy_from_slice(
                    &reference[
                        status_offset
                            ..
                        status_offset + 4
                    ],
                );

            let strict_status =
                u32::from_le_bytes(
                    strict_status_bytes
                );

            if status[tank] == 3 {
                unsupported_jobs += 1;
            }

            if status[tank] !=
                strict_status
            {
                status_mismatches += 1;
            }

            let error =
                (
                    value -
                    strict_cut
                ).abs();

            max_abs_cut_error =
                max_abs_cut_error
                    .max(error);

            if error > 1.0e-4 {
                cut_mismatches_1e4 += 1;
            }
        }
    }

    let wall_ms =
        wall_start
            .elapsed()
            .as_secs_f64() *
        1000.0;

    Ok(
        BlackwellB14PProductionReplayResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            annual_step_count:
                snapshot
                    .steps
                    .len()
                    as u32,

            tank_count:
                tank_count as u32,

            causal_jobs:
                jobs as u32,

            wall_ms,

            checksum_hex:
                format!(
                    "{checksum:016x}"
                ),

            status_mismatches_vs_strict:
                status_mismatches,

            cut_mismatches_1e4_vs_strict:
                cut_mismatches_1e4,

            max_abs_cut_error_vs_strict:
                max_abs_cut_error,

            unsupported_jobs,
        },
    )
}
// SUNOPT_BLACKWELL_B14P_PRODUCTION_B_CONTEXT_END
`.trim();

  // Header / CUDA blocks.
  h = insertBefore(
    h,
    "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN",
    H_BLOCK,
    "cuda_context.h",
  );

  cu = insertBefore(
    cu,
    "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN",
    CU_BLOCK,
    "cuda_context.cu",
  );

  // Rust FFI.
  rs = insertBefore(
    rs,
    "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_BEGIN",
    RS_FFI_BLOCK,
    "cuda_context.rs FFI",
  );

  // Rust B context + benchmark before CudaCausalWorker.
  rs = insertBefore(
    rs,
    "#[napi]\npub struct CudaCausalWorker {",
    RS_CONTEXT_BLOCK,
    "cuda_context.rs B14P context",
  );

  // Add worker fields.
  rs = replaceExactly(
    rs,
    `    blackwell_capture:
        Option<BlackwellCapture>,
}`,
    `    blackwell_capture:
        Option<BlackwellCapture>,

    /*
     * Production Approach B.
     *
     * B14P is enabled by default.  Set
     * SUNOPT_CUDA_CAUSAL_PATH=A before creating the worker to force
     * the legacy FP64 production path.
     */
    b14p_context:
        Option<CudaB14PContext>,

    b14p_dynamic:
        Vec<f32>,

    b14p_cuts:
        Vec<f32>,

    b14p_status:
        Vec<u32>,
}`,
    "CudaCausalWorker fields",
  );

  // Add internal B-or-A dispatcher.
  rs = replaceExactly(
    rs,
    `impl CudaCausalWorker {
    fn validate_job(`,
    `impl CudaCausalWorker {
    fn run_lockstep_b14p_or_a(
        &mut self,
        current_index: usize,
        horizon_steps: usize,
        output: &mut [f64],
    ) -> Result<()> {
        let tank_count =
            self.lockstep_dynamic.len();

        if output.len() !=
            tank_count
        {
            return Err(
                Error::from_reason(
                    "B14P output buffer length mismatch.",
                ),
            );
        }

        if self
            .b14p_context
            .is_some()
        {
            self.b14p_dynamic.resize(
                tank_count * 3,
                0.0,
            );

            self.b14p_cuts.resize(
                tank_count,
                0.0,
            );

            self.b14p_status.resize(
                tank_count,
                3,
            );

            for (
                tank,
                dynamic
            ) in self
                .lockstep_dynamic
                .iter()
                .enumerate()
            {
                let base =
                    tank * 3;

                self.b14p_dynamic[
                    base + 0
                ] =
                    dynamic
                        .current_storage_energy
                        as f32;

                self.b14p_dynamic[
                    base + 1
                ] =
                    dynamic
                        .minimum_reserve_energy
                        as f32;

                self.b14p_dynamic[
                    base + 2
                ] =
                    dynamic
                        .requested_electrical_energy
                        as f32;
            }

            let result =
                self
                    .b14p_context
                    .as_mut()
                    .expect(
                        "checked above",
                    )
                    .run_step(
                        current_index,
                        horizon_steps,
                        &self.b14p_dynamic,
                        &mut self.b14p_cuts,
                        &mut self.b14p_status,
                    );

            match result {
                Ok(()) => {
                    let supported =
                        self
                            .b14p_status
                            .iter()
                            .all(
                                |status| {
                                    *status != 3
                                },
                            );

                    if supported {
                        for (
                            target,
                            source
                        ) in output
                            .iter_mut()
                            .zip(
                                self
                                    .b14p_cuts
                                    .iter()
                            )
                        {
                            *target =
                                *source as f64;
                        }

                        return Ok(());
                    }
                }

                Err(error) => {
                    eprintln!(
                        "[SunOpt] B14P production path failed; using FP64 fallback for this worker: {error}",
                    );

                    self.b14p_context =
                        None;
                }
            }
        }

        self.context
            .run_causal_dynamic_batch_into(
                current_index,
                horizon_steps,
                &self.lockstep_dynamic,
                output,
            )
    }

    fn validate_job(`,
    "CudaCausalWorker B dispatcher",
  );

  // Create B context by default in constructor.
  const CONSTRUCTOR_OLD = `        Ok(Self {
            context,
            data_length,

            lockstep_parameters:
                Vec::new(),

            lockstep_dynamic:
                Vec::new(),

            blackwell_capture,
        })`;

  const CONSTRUCTOR_NEW = `        let use_b14p =
            std::env::var(
                "SUNOPT_CUDA_CAUSAL_PATH",
            )
            .map(
                |value| {
                    !value
                        .trim()
                        .eq_ignore_ascii_case(
                            "A",
                        )
                },
            )
            .unwrap_or(true);

        let b14p_context =
            if use_b14p {
                match CudaB14PContext::new(
                    device_id.unwrap_or(0),
                    &export_power_nonnegative,
                    &demand_energy_nonnegative,
                ) {
                    Ok(context) =>
                        Some(context),

                    Err(error) => {
                        eprintln!(
                            "[SunOpt] B14P initialization failed; continuing with FP64 path: {error}",
                        );

                        None
                    }
                }
            } else {
                None
            };

        Ok(Self {
            context,
            data_length,

            lockstep_parameters:
                Vec::new(),

            lockstep_dynamic:
                Vec::new(),

            blackwell_capture,

            b14p_context,

            b14p_dynamic:
                Vec::new(),

            b14p_cuts:
                Vec::new(),

            b14p_status:
                Vec::new(),
        })`;

  rs = replaceExactly(
    rs,
    CONSTRUCTOR_OLD,
    CONSTRUCTOR_NEW,
    "CudaCausalWorker constructor",
  );

  // Configure B static tank state after the existing A configuration.
  rs = replaceExactly(
    rs,
    `        self.context
            .configure_causal_static_parameters(
                &self.lockstep_parameters,
            )?;


        Ok(())
    }`,
    `        self.context
            .configure_causal_static_parameters(
                &self.lockstep_parameters,
            )?;

        if let Some(
            b14p
        ) = self
            .b14p_context
            .as_mut()
        {
            let packed_for_b14p =
                packed
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();

            match b14p
                .configure_from_packed(
                    &packed_for_b14p,
                )
            {
                Ok(tank_count) => {
                    self.b14p_dynamic =
                        vec![
                            0.0_f32;
                            tank_count * 3
                        ];

                    self.b14p_cuts =
                        vec![
                            0.0_f32;
                            tank_count
                        ];

                    self.b14p_status =
                        vec![
                            3_u32;
                            tank_count
                        ];
                }

                Err(error) => {
                    eprintln!(
                        "[SunOpt] B14P configuration failed; continuing with FP64 path: {error}",
                    );

                    self.b14p_context =
                        None;
                }
            }
        }


        Ok(())
    }`,
    "configureLockstepPacked B14P",
  );

  // Replace Vec-return production call with dispatcher.
  rs = replaceExactly(
    rs,
    `        self.context
            .run_causal_dynamic_batch(
                current_index as usize,
                horizon_steps as usize,
                &self.lockstep_dynamic,
            )
    }`,
    `        let mut output =
            vec![
                0.0_f64;
                self
                    .lockstep_dynamic
                    .len()
            ];

        self
            .run_lockstep_b14p_or_a(
                current_index as usize,
                horizon_steps as usize,
                &mut output,
            )?;

        Ok(output)
    }`,
    "runLockstepDynamic B14P",
  );

  // Replace borrowed-output production call with dispatcher.
  rs = replaceExactly(
    rs,
    `        self.context
            .run_causal_dynamic_batch_into(
                current_index as usize,
                horizon_steps as usize,
                &self.lockstep_dynamic,
                output_slice,
            )?;`,
    `        self
            .run_lockstep_b14p_or_a(
                current_index as usize,
                horizon_steps as usize,
                output_slice,
            )?;`,
    "runLockstepDynamicInto B14P",
  );

  const CHILD = "// rust/sunopt-node/scripts/blackwell-b14p-child.mjs\nimport { createRequire } from \"node:module\";\nimport fs from \"node:fs\";\nimport path from \"node:path\";\nimport { execFileSync } from \"node:child_process\";\nimport { fileURLToPath } from \"node:url\";\n\nconst require = createRequire(import.meta.url);\nconst here = path.dirname(fileURLToPath(import.meta.url));\nconst nativeRoot = path.resolve(here, \"..\");\nconst projectRoot = path.resolve(nativeRoot, \"..\", \"..\");\nconst addon = require(\n  path.join(\n    nativeRoot,\n    \"sunopt_node.win32-x64-msvc.node\"\n  )\n);\n\nconst args = new Map();\nfor (\n  let i = 2;\n  i + 1 < process.argv.length;\n  i += 2\n) {\n  args.set(\n    process.argv[i],\n    process.argv[i + 1]\n  );\n}\n\nconst kind =\n  args.get(\"--kind\") ??\n  \"production-b\";\n\nconst snapshot =\n  args.get(\"--snapshot\") ??\n  path.join(\n    nativeRoot,\n    \"blackwell-b12-sync-blackwell.bin\"\n  );\n\nconst reference =\n  args.get(\"--reference\") ??\n  path.join(\n    nativeRoot,\n    \"benchmarks\",\n    \"references\",\n    \"blackwell-b14m-strict-reference.bin\"\n  );\n\nfunction telemetry() {\n  try {\n    const raw = execFileSync(\n      \"nvidia-smi\",\n      [\n        \"--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw,utilization.gpu\",\n        \"--format=csv,noheader,nounits\",\n        \"-i\",\n        \"0\",\n      ],\n      {\n        encoding: \"utf8\",\n        windowsHide: true,\n      }\n    ).trim();\n\n    const [\n      clock,\n      temp,\n      pstate,\n      power,\n      util,\n    ] =\n      raw\n        .split(\",\")\n        .map(\n          value =>\n            value.trim()\n        );\n\n    return {\n      clockMHz:\n        Number(clock),\n\n      temperatureC:\n        Number(temp),\n\n      pstate,\n\n      powerW:\n        Number(power),\n\n      utilizationPercent:\n        Number(util),\n    };\n  } catch {\n    return null;\n  }\n}\n\nconst before =\n  telemetry();\n\nlet result;\n\nif (kind === \"reference\") {\n  fs.mkdirSync(\n    path.dirname(reference),\n    { recursive: true }\n  );\n\n  result =\n    addon\n      .runBlackwellB14MPrecision(\n        snapshot,\n        \"strict-reference\",\n        reference,\n        true,\n        0\n      );\n} else if (kind === \"b14l\") {\n  const pinned =\n    (\n      args.get(\"--pinned\") ??\n      \"false\"\n    ) === \"true\";\n\n  result =\n    addon\n      .runBlackwellB14LPackedTransport(\n        snapshot,\n        pinned,\n        0\n      );\n} else {\n  result =\n    addon\n      .runBlackwellB14PProductionReplay(\n        snapshot,\n        reference,\n        0\n      );\n}\n\nconst after =\n  telemetry();\n\nprocess.stdout.write(\n  JSON.stringify({\n    kind,\n    before,\n    after,\n    result,\n  }) + \"\\n\"\n);\n";

  const RUNNER = "// rust/sunopt-node/scripts/blackwell-b14p-fixed-clock-production-b.mjs\nimport fs from \"node:fs\";\nimport path from \"node:path\";\nimport {\n  spawn,\n  spawnSync,\n} from \"node:child_process\";\nimport { fileURLToPath } from \"node:url\";\n\nconst here =\n  path.dirname(\n    fileURLToPath(\n      import.meta.url\n    )\n  );\n\nconst nativeRoot =\n  path.resolve(\n    here,\n    \"..\"\n  );\n\nconst projectRoot =\n  path.resolve(\n    nativeRoot,\n    \"..\",\n    \"..\"\n  );\n\nconst outbox =\n  path.join(\n    projectRoot,\n    \"updates\",\n    \"outbox\"\n  );\n\nconst child =\n  path.join(\n    nativeRoot,\n    \"scripts\",\n    \"blackwell-b14p-child.mjs\"\n  );\n\nconst guard =\n  path.join(\n    nativeRoot,\n    \"scripts\",\n    \"nvapi-blackwell-benchmark-guard.py\"\n  );\n\nconst snapshot =\n  path.join(\n    nativeRoot,\n    \"blackwell-b12-sync-blackwell.bin\"\n  );\n\nconst reference =\n  path.join(\n    nativeRoot,\n    \"benchmarks\",\n    \"references\",\n    \"blackwell-b14m-strict-reference.bin\"\n  );\n\nconst hashFile =\n  path.join(\n    projectRoot,\n    \"updates\",\n    \"history\",\n    \"hash.txt\"\n  );\n\nconst failedRun =\n  path.join(\n    outbox,\n    \"failed-run.txt\"\n  );\n\nfs.mkdirSync(\n  outbox,\n  {\n    recursive: true,\n  }\n);\n\nfunction sleepMs(ms) {\n  const wait =\n    new Int32Array(\n      new SharedArrayBuffer(4)\n    );\n\n  Atomics.wait(\n    wait,\n    0,\n    0,\n    ms\n  );\n}\n\nfunction fail(message, extra = \"\") {\n  fs.writeFileSync(\n    failedRun,\n    [\n      \"stage=run\",\n      `time=${new Date().toISOString()}`,\n      `error=${message}`,\n      extra,\n      \"\",\n    ].join(\"\\n\"),\n    \"utf8\"\n  );\n\n  throw new Error(message);\n}\n\nfunction runChild(extra) {\n  const p =\n    spawnSync(\n      process.execPath,\n      [\n        child,\n        \"--snapshot\",\n        snapshot,\n        \"--reference\",\n        reference,\n        ...extra,\n      ],\n      {\n        cwd:\n          projectRoot,\n\n        encoding:\n          \"utf8\",\n\n        windowsHide:\n          true,\n\n        env: {\n          ...process.env,\n          SUNOPT_CUDA_CAUSAL_PATH:\n            \"B\",\n        },\n      }\n    );\n\n  if (\n    p.error ||\n    p.status !== 0\n  ) {\n    fail(\n      `B14P child failed: ${extra.join(\" \")}`,\n      [\n        `status=${p.status}`,\n        `spawnError=${p.error?.stack ?? \"\"}`,\n        \"=== STDOUT ===\",\n        p.stdout ?? \"\",\n        \"=== STDERR ===\",\n        p.stderr ?? \"\",\n      ].join(\"\\n\")\n    );\n  }\n\n  const lines =\n    (p.stdout ?? \"\")\n      .trim()\n      .split(/\\r?\\n/);\n\n  return JSON.parse(\n    lines[\n      lines.length - 1\n    ]\n  );\n}\n\nfunction median(values) {\n  const a =\n    [...values]\n      .sort(\n        (x, y) =>\n          x - y\n      );\n\n  const m =\n    Math.floor(\n      a.length / 2\n    );\n\n  return a.length % 2\n    ? a[m]\n    : (\n        a[m - 1] +\n        a[m]\n      ) / 2;\n}\n\nfunction summarize(label, rows, wallKey) {\n  const wall =\n    rows.map(\n      row =>\n        row.result[wallKey]\n    );\n\n  const kernel =\n    rows\n      .map(\n        row =>\n          row.result.kernelMs\n      )\n      .filter(\n        value =>\n          Number.isFinite(\n            value\n          )\n      );\n\n  return {\n    label,\n    count:\n      rows.length,\n    wallMs:\n      wall,\n    medianWallMs:\n      median(wall),\n    minWallMs:\n      Math.min(...wall),\n    maxWallMs:\n      Math.max(...wall),\n    medianKernelMs:\n      kernel.length\n        ? median(kernel)\n        : null,\n    telemetry:\n      rows.map(\n        row => ({\n          before:\n            row.before,\n          after:\n            row.after,\n        })\n      ),\n    firstResult:\n      rows[0]?.result ??\n      null,\n  };\n}\n\nfunction parseTelemetryFile(file) {\n  if (!fs.existsSync(file)) {\n    return [];\n  }\n\n  const rows = [];\n\n  for (\n    const rawLine\n    of fs\n      .readFileSync(\n        file,\n        \"utf8\"\n      )\n      .split(/\\r?\\n/)\n  ) {\n    const line =\n      rawLine.trim();\n\n    if (\n      !line ||\n      line.startsWith(\n        \"clocks\"\n      )\n    ) {\n      continue;\n    }\n\n    const parts =\n      line\n        .split(\",\")\n        .map(\n          value =>\n            value\n              .trim()\n              .replace(\n                /[^0-9A-Za-z.+-]/g,\n                \"\"\n              )\n        );\n\n    if (\n      parts.length < 5\n    ) {\n      continue;\n    }\n\n    const [\n      clock,\n      temp,\n      pstate,\n      power,\n      util,\n    ] = parts;\n\n    rows.push({\n      clockMHz:\n        Number(clock),\n      temperatureC:\n        Number(temp),\n      pstate,\n      powerW:\n        Number(power),\n      utilizationPercent:\n        Number(util),\n    });\n  }\n\n  return rows.filter(\n    row =>\n      Number.isFinite(\n        row.clockMHz\n      ) &&\n      Number.isFinite(\n        row.utilizationPercent\n      )\n  );\n}\n\nfunction runWithClockTelemetry(\n  label,\n  extra\n) {\n  const telemetryFile =\n    path.join(\n      outbox,\n      `b14p-clock-${label}-${Date.now()}.csv`\n    );\n\n  const sampler =\n    spawn(\n      \"nvidia-smi\",\n      [\n        \"--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw,utilization.gpu\",\n        \"--format=csv,noheader,nounits\",\n        \"-i\",\n        \"0\",\n        \"-lms\",\n        \"200\",\n        \"-f\",\n        telemetryFile,\n      ],\n      {\n        cwd:\n          projectRoot,\n\n        stdio:\n          \"ignore\",\n\n        windowsHide:\n          true,\n      }\n    );\n\n  sleepMs(300);\n\n  let row;\n\n  try {\n    row =\n      runChild(extra);\n  } finally {\n    try {\n      sampler.kill();\n    } catch {}\n\n    sleepMs(250);\n  }\n\n  const telemetry =\n    parseTelemetryFile(\n      telemetryFile\n    );\n\n  const active =\n    telemetry.filter(\n      sample =>\n        sample\n          .utilizationPercent >=\n        20\n    );\n\n  if (active.length === 0) {\n    fail(\n      `CLOCK_LOCK_FAILED: no active telemetry samples for ${label}`,\n      JSON.stringify(\n        telemetry,\n        null,\n        2\n      )\n    );\n  }\n\n  const clocks =\n    active.map(\n      sample =>\n        sample.clockMHz\n    );\n\n  const minimum =\n    Math.min(...clocks);\n\n  const maximum =\n    Math.max(...clocks);\n\n  const clockMedian =\n    median(clocks);\n\n  const clockSpan =\n    maximum -\n    minimum;\n\n  const inBand =\n    clocks.filter(\n      value =>\n        value >= 3450 &&\n        value <= 3510\n    ).length;\n\n  const inBandRatio =\n    inBand /\n    clocks.length;\n\n  /*\n   * +600 previously measured about 3457 MHz under active load.\n   * +625 should land near 3.48 GHz. Gate on median/stability so one\n   * short V/F transition sample does not invalidate a valid fixed run.\n   */\n  if (\n    clockMedian < 3450 ||\n    clockMedian > 3510 ||\n    clockSpan > 35 ||\n    inBandRatio < 0.8\n  ) {\n    fail(\n      `CLOCK_LOCK_FAILED: active clock is not stably locked near the +625 MHz point for ${label}`,\n      JSON.stringify(\n        {\n          minimum,\n          maximum,\n          clockMedian,\n          clockSpan,\n          inBandRatio,\n          active,\n        },\n        null,\n        2\n      )\n    );\n  }\n\n  return {\n    ...row,\n    fixedClock: {\n      activeSamples:\n        active.length,\n      minMHz:\n        minimum,\n      maxMHz:\n        maximum,\n      medianMHz:\n        clockMedian,\n      spanMHz:\n        clockSpan,\n      inBandRatio,\n      temperatureMaxC:\n        Math.max(\n          ...active.map(\n            sample =>\n              sample\n                .temperatureC\n          )\n        ),\n    },\n  };\n}\n\nconst release =\n  fs.existsSync(hashFile)\n    ? fs\n        .readFileSync(\n          hashFile,\n          \"utf8\"\n        )\n        .trim()\n    : \"unknown\";\n\nif (!fs.existsSync(guard)) {\n  fail(\n    `NVAPI guard missing: ${guard}`\n  );\n}\n\nconst stateDir =\n  path.join(\n    nativeRoot,\n    \"blackwell-runtime-control\",\n    `b14p-${process.pid}`\n  );\n\nfs.mkdirSync(\n  stateDir,\n  {\n    recursive: true,\n  }\n);\n\nconst readyFile =\n  path.join(\n    stateDir,\n    \"ready.json\"\n  );\n\nconst stopFile =\n  path.join(\n    stateDir,\n    \"stop.signal\"\n  );\n\nconst doneFile =\n  path.join(\n    stateDir,\n    \"done.json\"\n  );\n\nconst guardLog =\n  path.join(\n    outbox,\n    \"b14p-nvapi-guard.txt\"\n  );\n\nfor (\n  const file\n  of [\n    readyFile,\n    stopFile,\n    doneFile,\n    guardLog,\n  ]\n) {\n  try {\n    fs.unlinkSync(file);\n  } catch {}\n}\n\nconst python =\n  process.env\n    .SUNOPT_PYTHON ??\n  \"python\";\n\nconst guardArgs = [\n  guard,\n  \"--gpu\",\n  \"0\",\n  \"--target-voltage-mv\",\n  \"1050\",\n  \"--native-vf-frequency-mhz\",\n  \"2865\",\n  \"--curve-offset-mhz\",\n  \"625\",\n  \"--cap-band-points\",\n  \"8\",\n  \"--parent-pid\",\n  String(\n    process.pid\n  ),\n  \"--ready-file\",\n  readyFile,\n  \"--stop-file\",\n  stopFile,\n  \"--done-file\",\n  doneFile,\n  \"--log-file\",\n  guardLog,\n  \"--fan-percent\",\n  \"100\",\n];\n\nconst guardProcess =\n  spawn(\n    python,\n    guardArgs,\n    {\n      cwd:\n        projectRoot,\n\n      stdio:\n        \"ignore\",\n\n      windowsHide:\n        true,\n    }\n  );\n\nlet guardStopped =\n  false;\n\nfunction stopGuard() {\n  if (guardStopped) {\n    return;\n  }\n\n  guardStopped =\n    true;\n\n  try {\n    fs.writeFileSync(\n      stopFile,\n      \"stop\\n\",\n      \"utf8\"\n    );\n  } catch {}\n\n  const deadline =\n    Date.now() +\n    10_000;\n\n  while (\n    Date.now() <\n    deadline\n  ) {\n    if (\n      fs.existsSync(\n        doneFile\n      )\n    ) {\n      return;\n    }\n\n    sleepMs(200);\n  }\n\n  try {\n    guardProcess.kill();\n  } catch {}\n}\n\nprocess.on(\n  \"exit\",\n  stopGuard\n);\n\ntry {\n  const readyDeadline =\n    Date.now() +\n    30_000;\n\n  while (\n    Date.now() <\n    readyDeadline &&\n    !fs.existsSync(\n      readyFile\n    )\n  ) {\n    sleepMs(200);\n  }\n\n  if (\n    !fs.existsSync(\n      readyFile\n    )\n  ) {\n    fail(\n      \"CLOCK_LOCK_FAILED: NVAPI guard did not reach READY\",\n      fs.existsSync(guardLog)\n        ? fs.readFileSync(\n            guardLog,\n            \"utf8\"\n          )\n        : \"\"\n    );\n  }\n\n  const ready =\n    JSON.parse(\n      fs.readFileSync(\n        readyFile,\n        \"utf8\"\n      )\n    );\n\n  if (\n    ready.status !==\n    \"READY\"\n  ) {\n    fail(\n      `CLOCK_LOCK_FAILED: unexpected NVAPI guard state ${JSON.stringify(ready)}`,\n      fs.existsSync(guardLog)\n        ? fs.readFileSync(\n            guardLog,\n            \"utf8\"\n          )\n        : \"\"\n    );\n  }\n\n  // Reference is persistent, not outbox.\n  if (\n    !fs.existsSync(\n      reference\n    )\n  ) {\n    runChild(\n      [\n        \"--kind\",\n        \"reference\",\n      ]\n    );\n  }\n\n  /*\n   * Clock gate warmup.\n   * This is not counted.  If +625 is not actually active,\n   * the workflow ends before any A/B result is accepted.\n   */\n  const clockWarmup =\n    runWithClockTelemetry(\n      \"clock-warmup\",\n      [\n        \"--kind\",\n        \"b14l\",\n        \"--pinned\",\n        \"false\",\n      ]\n    );\n\n  // One additional warmup per pinned variant.\n  runWithClockTelemetry(\n    \"pageable-warmup\",\n    [\n      \"--kind\",\n      \"b14l\",\n      \"--pinned\",\n      \"false\",\n    ]\n  );\n\n  runWithClockTelemetry(\n    \"pinned-warmup\",\n    [\n      \"--kind\",\n      \"b14l\",\n      \"--pinned\",\n      \"true\",\n    ]\n  );\n\n  const pinnedRows = {\n    pageable: [],\n    pinned: [],\n  };\n\n  const abba = [\n    \"pageable\",\n    \"pinned\",\n    \"pinned\",\n    \"pageable\",\n    \"pageable\",\n    \"pinned\",\n    \"pinned\",\n    \"pageable\",\n    \"pageable\",\n    \"pinned\",\n  ];\n\n  let measuredIndex = 0;\n\n  for (const name of abba) {\n    const row =\n      runWithClockTelemetry(\n        `pinned-${measuredIndex}-${name}`,\n        [\n          \"--kind\",\n          \"b14l\",\n          \"--pinned\",\n          String(\n            name ===\n            \"pinned\"\n          ),\n        ]\n      );\n\n    pinnedRows[name]\n      .push(row);\n\n    measuredIndex += 1;\n  }\n\n  const pageable =\n    summarize(\n      \"pageable\",\n      pinnedRows.pageable,\n      \"wallMs\"\n    );\n\n  const pinned =\n    summarize(\n      \"pinned\",\n      pinnedRows.pinned,\n      \"wallMs\"\n    );\n\n  const pinnedDeltaPct =\n    100 *\n    (\n      pinned\n        .medianWallMs -\n      pageable\n        .medianWallMs\n    ) /\n    pageable\n      .medianWallMs;\n\n  const pinnedConclusion =\n    Math.abs(\n      pinnedDeltaPct\n    ) < 10\n      ? \"INCONCLUSIVE_LT_10_PERCENT\"\n      : (\n          pinnedDeltaPct < 0\n            ? \"PINNED_FASTER\"\n            : \"PAGEABLE_FASTER\"\n        );\n\n  // Production-B warmup.\n  runWithClockTelemetry(\n    \"production-b-warmup\",\n    [\n      \"--kind\",\n      \"production-b\",\n    ]\n  );\n\n  const productionRows = [];\n\n  for (\n    let i = 0;\n    i < 5;\n    ++i\n  ) {\n    productionRows.push(\n      runWithClockTelemetry(\n        `production-b-${i}`,\n        [\n          \"--kind\",\n          \"production-b\",\n        ]\n      )\n    );\n  }\n\n  const productionB =\n    summarize(\n      \"production-b\",\n      productionRows,\n      \"wallMs\"\n    );\n\n  productionB\n    .statusMismatchesVsStrict =\n      productionRows\n        .map(\n          row =>\n            row\n              .result\n              .statusMismatchesVsStrict\n        );\n\n  productionB\n    .cutMismatches1E4VsStrict =\n      productionRows\n        .map(\n          row =>\n            row\n              .result\n              .cutMismatches1E4VsStrict\n        );\n\n  productionB\n    .maxAbsCutErrorVsStrict =\n      Math.max(\n        ...productionRows\n          .map(\n            row =>\n              row\n                .result\n                .maxAbsCutErrorVsStrict\n          )\n      );\n\n  productionB\n    .unsupportedJobs =\n      Math.max(\n        ...productionRows\n          .map(\n            row =>\n              row\n                .result\n                .unsupportedJobs\n          )\n      );\n\n  const report = {\n    stage:\n      \"B14P_FIXED_CLOCK_PRODUCTION_B\",\n\n    release,\n\n    productionPath: {\n      default:\n        \"B14P_FP32\",\n\n      forceLegacy:\n        \"SUNOPT_CUDA_CAUSAL_PATH=A\",\n\n      fallback:\n        \"FP64 for unsupported/error\",\n    },\n\n    fixedClock: {\n      method:\n        \"NVAPI V/F guard\",\n\n      voltageMv:\n        1050,\n\n      nativeFrequencyMHz:\n        2865,\n\n      coreOffsetMHz:\n        625,\n\n      expectedActiveClockMedianMHz:\n        [\n          3450,\n          3510,\n        ],\n\n      maximumActiveClockSpanMHz:\n        35,\n\n      minimumInBandRatio:\n        0.8,\n\n      fanPercent:\n        100,\n\n      warmup:\n        clockWarmup\n          .fixedClock,\n    },\n\n    measurementPolicy: {\n      warmupPerVariant:\n        1,\n\n      measuredRunsPerVariant:\n        5,\n\n      medianAuthoritative:\n        true,\n\n      noiseThresholdPercent:\n        10,\n\n      pinnedOrder:\n        \"ABBA alternating\",\n    },\n\n    pinnedAB: {\n      pageable,\n      pinned,\n      deltaPct:\n        pinnedDeltaPct,\n      conclusion:\n        pinnedConclusion,\n    },\n\n    productionB,\n\n    persistentStrictReference:\n      reference,\n  };\n\n  const reportPath =\n    path.join(\n      outbox,\n      \"blackwell-b14p-fixed-clock-production-b.json\"\n    );\n\n  fs.writeFileSync(\n    reportPath,\n    JSON.stringify(\n      report,\n      null,\n      2\n    ) + \"\\n\",\n    \"utf8\"\n  );\n\n  const llmPath =\n    path.join(\n      outbox,\n      \"llm.txt\"\n    );\n\n  fs.appendFileSync(\n    llmPath,\n    [\n      \"\",\n      `=== ${new Date().toISOString()} B14P_FIXED_CLOCK_PRODUCTION_B ===`,\n      `release=${release}`,\n      `clock target=1050mV native=2865 offset=+625 medianRange=3450..3510 maxSpan=35 inBand>=0.8 warmupMedian=${clockWarmup.fixedClock.medianMHz}`,\n      `pinned pageableMedianWallMs=${pageable.medianWallMs.toFixed(6)} pinnedMedianWallMs=${pinned.medianWallMs.toFixed(6)} deltaPct=${pinnedDeltaPct.toFixed(3)} conclusion=${pinnedConclusion}`,\n      `productionB medianWallMs=${productionB.medianWallMs.toFixed(6)} min=${productionB.minWallMs.toFixed(6)} max=${productionB.maxWallMs.toFixed(6)}`,\n      `productionB statusMismatchRuns=${productionB.statusMismatchesVsStrict.join(\",\")} cutMismatch1e4Runs=${productionB.cutMismatches1E4VsStrict.join(\",\")} maxAbsCutError=${productionB.maxAbsCutErrorVsStrict} unsupportedJobs=${productionB.unsupportedJobs}`,\n      `persistentStrictReference=${reference}`,\n      `report=${reportPath}`,\n      \"\",\n    ].join(\"\\n\"),\n    \"utf8\"\n  );\n\n  console.log(\n    JSON.stringify(\n      report,\n      null,\n      2\n    )\n  );\n\n  console.log(\n    `report=${reportPath}`\n  );\n\n  console.log(\n    `llm=${llmPath}`\n  );\n\n  console.log(\n    \"B14P_FIXED_CLOCK_PRODUCTION_B_RESULT=OK\"\n  );\n} finally {\n  stopGuard();\n}\n";

  const files = new Map([
    [H_REL, Buffer.from(h, "utf8")],
    [CU_REL, Buffer.from(cu, "utf8")],
    [RS_REL, Buffer.from(rs, "utf8")],
    [CHILD_REL, Buffer.from(CHILD, "utf8")],
    [RUNNER_REL, Buffer.from(RUNNER, "utf8")],
  ]);

  const manifest = {
    format: 2,
    parent_sha256: parent,
    files: Object.fromEntries(
      [...files.entries()].map(
        ([name, data]) => [
          name,
          sha256(data),
        ]
      )
    ),
    delete: [],
  };

  const manifestBytes =
    Buffer.from(
      JSON.stringify(
        manifest,
        null,
        2
      ) + "\n",
      "utf8"
    );

  const release =
    sha256(
      manifestBytes
    );

  const entries = [
    [
      "manifest.json",
      manifestBytes,
    ],
    [
      "manifest.sha256",
      Buffer.from(
        release + "\n",
        "ascii"
      ),
    ],
    ...files.entries(),
  ];

  const locals = [];
  const centrals = [];
  let offset = 0;

  for (
    const [name, data]
    of entries
  ) {
    const nameBuf =
      Buffer.from(
        name,
        "utf8"
      );

    const crc =
      crc32(data);

    const local =
      Buffer.concat([
        u32(0x04034b50),
        u16(20),
        u16(0),
        u16(0),
        u16(0),
        u16(0),
        u32(crc),
        u32(data.length),
        u32(data.length),
        u16(nameBuf.length),
        u16(0),
        nameBuf,
        data,
      ]);

    locals.push(local);

    const central =
      Buffer.concat([
        u32(0x02014b50),
        u16(20),
        u16(20),
        u16(0),
        u16(0),
        u16(0),
        u16(0),
        u32(crc),
        u32(data.length),
        u32(data.length),
        u16(nameBuf.length),
        u16(0),
        u16(0),
        u16(0),
        u16(0),
        u32(0),
        u32(offset),
        nameBuf,
      ]);

    centrals.push(
      central
    );

    offset +=
      local.length;
  }

  const centralOffset =
    offset;

  const central =
    Buffer.concat(
      centrals
    );

  const end =
    Buffer.concat([
      u32(0x06054b50),
      u16(0),
      u16(0),
      u16(entries.length),
      u16(entries.length),
      u32(central.length),
      u32(centralOffset),
      u16(0),
    ]);

  const zip =
    Buffer.concat([
      ...locals,
      central,
      end,
    ]);

  fs.mkdirSync(
    inbox,
    {
      recursive: true,
    }
  );

  const packageName =
    "20261112-blackwell-b14p-fixed-clock-production-b.zip";

  const packagePath =
    path.join(
      inbox,
      packageName
    );

  fs.writeFileSync(
    packagePath,
    zip
  );

  console.log(
    `parent=${parent}`
  );

  console.log(
    `release=${release}`
  );

  console.log(
    `package=${packagePath}`
  );

  console.log(
    "B14P_PACKAGE_CREATED=OK"
  );

  // Wait for the already-running watcher.
  const deadline =
    Date.now() +
    120_000;

  for (;;) {
    if (
      fs.existsSync(
        hashFile
      )
    ) {
      const current =
        fs
          .readFileSync(
            hashFile,
            "utf8"
          )
          .trim();

      if (
        current ===
        release
      ) {
        break;
      }
    }

    const rejectedPath =
      path.join(
        rejected,
        packageName
      );

    if (
      fs.existsSync(
        rejectedPath
      )
    ) {
      throw new Error(
        `Watcher rejected ${packageName}`
      );
    }

    if (
      Date.now() >
      deadline
    ) {
      throw new Error(
        `Timed out waiting for watcher install: ${packageName}`
      );
    }

    sleepMs(250);
  }

  console.log(
    "B14P_WATCHER_INSTALL=OK"
  );

  // Build strict CUDA automatically.
  const comspec =
    process.env.ComSpec ||
    process.env.COMSPEC ||
    "C:\\Windows\\System32\\cmd.exe";

  const build =
    spawnSync(
      comspec,
      [
        "/d",
        "/s",
        "/c",
        "npm run build",
      ],
      {
        cwd:
          nativeRoot,

        encoding:
          "utf8",

        windowsHide:
          true,

        env: {
          ...process.env,
          SUNOPT_CUDA_FAST_MATH:
            "0",
        },
      }
    );

  fs.writeFileSync(
    path.join(
      outbox,
      "b14p-build.txt"
    ),
    [
      `status=${build.status}`,
      `spawnError=${build.error?.stack ?? "none"}`,
      "",
      "=== STDOUT ===",
      build.stdout ?? "",
      "",
      "=== STDERR ===",
      build.stderr ?? "",
      "",
    ].join("\n"),
    "utf8"
  );

  if (
    build.error ||
    build.status !== 0
  ) {
    writeFailure(
      failedBuild,
      "build",
      build.error ??
        `build exited with status ${build.status}`,
      [
        build.stdout ?? "",
        build.stderr ?? "",
      ].join("\n")
    );

    process.exit(2);
  }

  console.log(
    "B14P_BUILD=OK"
  );

  const run =
    spawnSync(
      process.execPath,
      [
        path.join(
          nativeRoot,
          "scripts",
          "blackwell-b14p-fixed-clock-production-b.mjs"
        ),
      ],
      {
        cwd:
          root,

        encoding:
          "utf8",

        windowsHide:
          true,

        env: {
          ...process.env,
          SUNOPT_CUDA_FAST_MATH:
            "0",
          SUNOPT_CUDA_CAUSAL_PATH:
            "B",
        },
      }
    );

  fs.writeFileSync(
    path.join(
      outbox,
      "b14p-run.txt"
    ),
    [
      `status=${run.status}`,
      `spawnError=${run.error?.stack ?? "none"}`,
      "",
      "=== STDOUT ===",
      run.stdout ?? "",
      "",
      "=== STDERR ===",
      run.stderr ?? "",
      "",
    ].join("\n"),
    "utf8"
  );

  if (
    run.error ||
    run.status !== 0
  ) {
    if (
      !fs.existsSync(
        failedRun
      )
    ) {
      writeFailure(
        failedRun,
        "run",
        run.error ??
          `run exited with status ${run.status}`,
        [
          run.stdout ?? "",
          run.stderr ?? "",
        ].join("\n")
      );
    }

    process.stdout.write(
      run.stdout ?? ""
    );

    process.stderr.write(
      run.stderr ?? ""
    );

    process.exit(3);
  }

  process.stdout.write(
    run.stdout ?? ""
  );

  process.stderr.write(
    run.stderr ?? ""
  );

  console.log(
    "B14P_SELF_RUNNING_WORKFLOW=OK"
  );
} catch (error) {
  writeFailure(
    failedPrompt,
    "prompt/package/install",
    error
  );

  console.error(error);
  process.exit(1);
}
