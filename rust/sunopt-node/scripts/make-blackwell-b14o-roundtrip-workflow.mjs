// rust/sunopt-node/scripts/make-blackwell-b14o-roundtrip-workflow.mjs
//
// B14O self-running workflow.
//
// One user command:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14o-roundtrip-workflow.mjs
//
// The generator:
//   1. creates the format-2 B14O package in updates/inbox
//   2. waits for the already-running update watcher to install it
//   3. builds strict CUDA automatically
//   4. runs the benchmark automatically
//
// Failure markers:
//   updates/outbox/failed-prompt.txt
//   updates/outbox/failed-build.txt
//   updates/outbox/failed-run.txt
//
// Benchmark:
//   A) B14L pageable vs pinned:
//      - one warmup per variant
//      - five measured runs per variant
//      - alternating ABBA... order
//      - median is authoritative
//      - <10% wall difference is reported as inconclusive/noise
//
//   B) FP32 packed roundtrip:
//      - one H2D input packet per timestep
//      - one output path per timestep
//      - pageable + stream sync / default scheduling
//      - pageable + stream sync / cudaDeviceScheduleSpin
//      - pageable + cudaEventQuery polling / ScheduleSpin
//      - mapped host output + cudaEventQuery polling / ScheduleSpin
//      - warmup then five measured runs each
//      - alternating order
//
// Persistent strict reference:
//   rust/sunopt-node/benchmarks/references/blackwell-b14m-strict-reference.bin
//
// Reports:
//   updates/outbox/blackwell-b14o-roundtrip-workflow.json
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
const CHILD_REL = "rust/sunopt-node/scripts/blackwell-b14o-child.mjs";
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14o-roundtrip-workflow.mjs";

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

function writeFailure(file, stage, error, extra = "") {
  const body = [
    `stage=${stage}`,
    `time=${new Date().toISOString()}`,
    `error=${error?.stack ?? error}`,
    extra,
    "",
  ].join("\n");
  fs.writeFileSync(file, body, "utf8");
}

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function insertBefore(text, anchor, block, label) {
  const marker = block.split(/\r?\n/, 1)[0];
  if (text.includes(marker)) {
    throw new Error(`${label}: B14O block already exists`);
  }
  const pos = text.indexOf(anchor);
  if (pos < 0) {
    throw new Error(`${label}: anchor not found: ${anchor}`);
  }
  return text.slice(0, pos) + block.trimEnd() + "\n\n" + text.slice(pos);
}

function sleepMs(ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) {}
}

try {
  if (!fs.existsSync(hashFile)) {
    throw new Error(`Current watcher hash not found: ${hashFile}`);
  }

  const parent = fs.readFileSync(hashFile, "utf8").trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(parent)) {
    throw new Error(`Invalid current watcher hash: ${parent}`);
  }

  const H_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14O_ROUNDTRIP_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14o_fp32_packed_benchmark(
    int device_id,
    const float* host_export_power,
    const float* host_demand_energy,
    size_t data_length,
    const float* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const float* host_dynamic_values,
    const float* host_energy_cut_levels,
    size_t step_count,
    uint32_t pinned_input,
    uint32_t wait_mode,
    uint32_t mapped_output,
    uint32_t schedule_spin,
    float* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_host_pack_ms,
    double* out_input_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14O_ROUNDTRIP_END
`.trim();

  const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14O_ROUNDTRIP_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14O_WARPS = 3u;
constexpr unsigned int SUNOPT_B14O_THREADS = 96u;

__global__ void sunopt_b14o_fp32_packed_kernel(
    const float* __restrict__ export_power,
    const float* __restrict__ demand_energy,
    size_t data_length,
    const float* __restrict__ tank_parameters,
    size_t tank_count,
    const float* __restrict__ packed_input,
    float* __restrict__ packed_output
) {
    const size_t tank =
        static_cast<size_t>(blockIdx.x);

    if (tank >= tank_count) {
        return;
    }

    const unsigned int tid =
        static_cast<unsigned int>(threadIdx.x);

    const unsigned int warp =
        tid >> 5u;

    const int lane =
        static_cast<int>(tid & 31u);

    __shared__ int
        shared_survives[SUNOPT_B14O_WARPS];
    __shared__ int shared_precheck;
    __shared__ float shared_lower;
    __shared__ float shared_upper;

    const size_t tb = tank * 6u;
    const size_t dynamic_base =
        2u + tank * 2u;
    const size_t cut_base =
        2u + tank_count * 2u;

    const size_t current_index =
        static_cast<size_t>(
            packed_input[0]
        );

    const size_t horizon =
        static_cast<size_t>(
            packed_input[1]
        );

    const float current =
        packed_input[
            dynamic_base + 0u
        ];

    const float target =
        tank_parameters[tb + 0u];

    const float reserve =
        fminf(
            target,
            fmaxf(
                0.0f,
                packed_input[
                    dynamic_base + 1u
                ]
            )
        );

    const float energy_cut =
        packed_input[
            cut_base + tank
        ];

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

    if (tid == 0u) {
        packed_output[tank] =
            energy_cut;

        packed_output[
            tank_count + tank
        ] =
            3.0f;
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
        horizon <= data_length;

    if (!supported) {
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
            packed_output[
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
            packed_output[tank] =
                0.0f;

            packed_output[
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

    for (
        unsigned int round = 0u;
        round < 10u;
        ++round
    ) {
        if (warp < SUNOPT_B14O_WARPS) {
            const unsigned int node_number =
                warp + 1u;

            unsigned int level = 0u;
            unsigned int first_at_level = 1u;

            while (
                (first_at_level << 1u) <=
                node_number
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
                    const float mid =
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
                        lo = mid;
                    } else {
                        hi = mid;
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
                const float mid =
                    (lo + hi) *
                    0.5f;

                if (
                    shared_survives[node] !=
                    0
                ) {
                    lo = mid;
                    node =
                        node * 2u + 2u;
                } else {
                    hi = mid;
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
        packed_output[tank] =
            shared_lower;

        packed_output[
            tank_count + tank
        ] =
            2.0f;
    }
}

inline cudaError_t sunopt_b14o_wait(
    cudaStream_t stream,
    cudaEvent_t done_event,
    uint32_t wait_mode
) {
    if (wait_mode == 0u) {
        return cudaStreamSynchronize(stream);
    }

    for (;;) {
        const cudaError_t query =
            cudaEventQuery(done_event);

        if (query == cudaSuccess) {
            return cudaSuccess;
        }

        if (query != cudaErrorNotReady) {
            return query;
        }
    }
}

} // namespace

extern "C" int
sunopt_cuda_run_b14o_fp32_packed_benchmark(
    int device_id,
    const float* host_export_power,
    const float* host_demand_energy,
    size_t data_length,
    const float* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const float* host_dynamic_values,
    const float* host_energy_cut_levels,
    size_t step_count,
    uint32_t pinned_input,
    uint32_t wait_mode,
    uint32_t mapped_output,
    uint32_t schedule_spin,
    float* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_host_pack_ms,
    double* out_input_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        !host_export_power ||
        !host_demand_energy ||
        !host_tank_parameters ||
        !host_current_indices ||
        !host_horizon_steps ||
        !host_dynamic_values ||
        !host_energy_cut_levels ||
        !host_cut_results ||
        !host_status ||
        !out_static_h2d_ms ||
        !out_host_pack_ms ||
        !out_input_enqueue_ms ||
        !out_kernel_ms ||
        !out_result_wait_ms ||
        !out_wall_ms ||
        data_length == 0u ||
        tank_count == 0u ||
        step_count == 0u ||
        wait_mode > 1u ||
        pinned_input > 1u ||
        mapped_output > 1u ||
        schedule_spin > 1u
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    unsigned int device_flags =
        schedule_spin != 0u
            ? cudaDeviceScheduleSpin
            : cudaDeviceScheduleAuto;

    if (mapped_output != 0u) {
        device_flags |=
            cudaDeviceMapHost;
    }

    cudaError_t err =
        cudaSetDeviceFlags(
            device_flags
        );

    if (
        err != cudaSuccess &&
        err != cudaErrorSetOnActiveProcess
    ) {
        return static_cast<int>(err);
    }

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    cudaEvent_t done_event = nullptr;

    float* d_export = nullptr;
    float* d_demand = nullptr;
    float* d_tanks = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;

    float* host_input = nullptr;
    float* host_output = nullptr;

    bool input_pinned = false;
    bool output_pinned = false;
    bool output_mapped = false;

    const size_t input_count =
        2u + tank_count * 3u;

    const size_t output_count =
        tank_count * 2u;

    const size_t input_bytes =
        input_count * sizeof(float);

    const size_t output_bytes =
        output_count * sizeof(float);

    const size_t annual_bytes =
        data_length * sizeof(float);

    const size_t tank_bytes =
        tank_count *
        6u *
        sizeof(float);

    auto cleanup = [&]() {
        if (
            d_output &&
            !output_mapped
        ) {
            cudaFree(d_output);
        }

        if (d_input) cudaFree(d_input);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);

        if (done_event) cudaEventDestroy(done_event);
        if (kernel_stop) cudaEventDestroy(kernel_stop);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (stream) cudaStreamDestroy(stream);

        if (output_pinned) {
            cudaFreeHost(host_output);
        } else if (host_output) {
            free(host_output);
        }

        if (input_pinned) {
            cudaFreeHost(host_input);
        } else if (host_input) {
            free(host_input);
        }
    };

    if (pinned_input != 0u) {
        err = cudaHostAlloc(
            reinterpret_cast<void**>(
                &host_input
            ),
            input_bytes,
            cudaHostAllocDefault
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        input_pinned = true;
    } else {
        host_input =
            static_cast<float*>(
                malloc(input_bytes)
            );

        if (!host_input) {
            cleanup();
            return static_cast<int>(
                cudaErrorMemoryAllocation
            );
        }
    }

    if (mapped_output != 0u) {
        err = cudaHostAlloc(
            reinterpret_cast<void**>(
                &host_output
            ),
            output_bytes,
            cudaHostAllocMapped
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        output_pinned = true;
        output_mapped = true;

        err = cudaHostGetDevicePointer(
            reinterpret_cast<void**>(
                &d_output
            ),
            host_output,
            0
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }
    } else {
        host_output =
            static_cast<float*>(
                malloc(output_bytes)
            );

        if (!host_output) {
            cleanup();
            return static_cast<int>(
                cudaErrorMemoryAllocation
            );
        }
    }

    err =
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaEventCreate(&kernel_start);
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaEventCreate(&kernel_stop);
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaEventCreateWithFlags(
        &done_event,
        cudaEventDisableTiming
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

#define B14O_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14O_MALLOC(
        d_export,
        annual_bytes
    );

    B14O_MALLOC(
        d_demand,
        annual_bytes
    );

    B14O_MALLOC(
        d_tanks,
        tank_bytes
    );

    B14O_MALLOC(
        d_input,
        input_bytes
    );

    if (!output_mapped) {
        B14O_MALLOC(
            d_output,
            output_bytes
        );
    }

#undef B14O_MALLOC

    const auto wall_start =
        Clock::now();

    const auto static_start =
        Clock::now();

    err = cudaMemcpyAsync(
        d_export,
        host_export_power,
        annual_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        d_demand,
        host_demand_energy,
        annual_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        d_tanks,
        host_tank_parameters,
        tank_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    *out_static_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            Clock::now() -
            static_start
        ).count();

    double host_pack_ms = 0.0;
    double input_enqueue_ms = 0.0;
    double kernel_ms = 0.0;
    double result_wait_ms = 0.0;

    const size_t dynamic_offset = 2u;
    const size_t cut_offset =
        2u + tank_count * 2u;

    for (
        size_t step = 0u;
        step < step_count;
        ++step
    ) {
        const auto pack_start =
            Clock::now();

        host_input[0] =
            static_cast<float>(
                host_current_indices[step]
            );

        host_input[1] =
            static_cast<float>(
                host_horizon_steps[step]
            );

        const size_t dynamic_source =
            step *
            tank_count *
            2u;

        const size_t cut_source =
            step *
            tank_count;

        for (
            size_t tank = 0u;
            tank < tank_count;
            ++tank
        ) {
            host_input[
                dynamic_offset +
                tank * 2u +
                0u
            ] =
                host_dynamic_values[
                    dynamic_source +
                    tank * 2u +
                    0u
                ];

            host_input[
                dynamic_offset +
                tank * 2u +
                1u
            ] =
                host_dynamic_values[
                    dynamic_source +
                    tank * 2u +
                    1u
                ];

            host_input[
                cut_offset +
                tank
            ] =
                host_energy_cut_levels[
                    cut_source +
                    tank
                ];
        }

        host_pack_ms +=
            std::chrono::duration<
                double,
                std::milli
            >(
                Clock::now() -
                pack_start
            ).count();

        const auto input_start =
            Clock::now();

        err = cudaMemcpyAsync(
            d_input,
            host_input,
            input_bytes,
            cudaMemcpyHostToDevice,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        input_enqueue_ms +=
            std::chrono::duration<
                double,
                std::milli
            >(
                Clock::now() -
                input_start
            ).count();

        err = cudaEventRecord(
            kernel_start,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        sunopt_b14o_fp32_packed_kernel
            <<<static_cast<unsigned int>(
                   tank_count
               ),
               SUNOPT_B14O_THREADS,
               0,
               stream>>>(
                d_export,
                d_demand,
                data_length,
                d_tanks,
                tank_count,
                d_input,
                d_output
            );

        err = cudaGetLastError();

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = cudaEventRecord(
            kernel_stop,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        const auto wait_start =
            Clock::now();

        if (!output_mapped) {
            err = cudaMemcpyAsync(
                host_output,
                d_output,
                output_bytes,
                cudaMemcpyDeviceToHost,
                stream
            );

            if (err != cudaSuccess) {
                cleanup();
                return static_cast<int>(err);
            }
        }

        err = cudaEventRecord(
            done_event,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = sunopt_b14o_wait(
            stream,
            done_event,
            wait_mode
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        result_wait_ms +=
            std::chrono::duration<
                double,
                std::milli
            >(
                Clock::now() -
                wait_start
            ).count();

        float step_kernel_ms = 0.0f;

        err = cudaEventElapsedTime(
            &step_kernel_ms,
            kernel_start,
            kernel_stop
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        kernel_ms +=
            static_cast<double>(
                step_kernel_ms
            );

        const size_t output_base =
            step *
            tank_count;

        for (
            size_t tank = 0u;
            tank < tank_count;
            ++tank
        ) {
            host_cut_results[
                output_base + tank
            ] =
                host_output[tank];

            host_status[
                output_base + tank
            ] =
                static_cast<uint32_t>(
                    host_output[
                        tank_count + tank
                    ]
                );
        }
    }

    *out_host_pack_ms =
        host_pack_ms;

    *out_input_enqueue_ms =
        input_enqueue_ms;

    *out_kernel_ms =
        kernel_ms;

    *out_result_wait_ms =
        result_wait_ms;

    *out_wall_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            Clock::now() -
            wall_start
        ).count();

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_B14O_ROUNDTRIP_END
`.trim();

  const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14O_ROUNDTRIP_FFI_BEGIN
    fn sunopt_cuda_run_b14o_fp32_packed_benchmark(
        device_id: c_int,
        host_export_power: *const f32,
        host_demand_energy: *const f32,
        data_length: usize,
        host_tank_parameters: *const f32,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f32,
        host_energy_cut_levels: *const f32,
        step_count: usize,
        pinned_input: u32,
        wait_mode: u32,
        mapped_output: u32,
        schedule_spin: u32,
        host_cut_results: *mut f32,
        host_status: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_host_pack_ms: *mut f64,
        out_input_enqueue_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14O_ROUNDTRIP_FFI_END
`.trim();

  const RS_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14O_ROUNDTRIP_FN_BEGIN
#[napi(object)]
pub struct BlackwellB14oRoundtripResult {
    pub pinned_input: bool,
    pub wait_mode: u32,
    pub mapped_output: bool,
    pub schedule_spin: bool,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub static_h2d_ms: f64,
    pub host_pack_ms: f64,
    pub input_enqueue_ms: f64,
    pub kernel_ms: f64,
    pub result_wait_ms: f64,
    pub wall_ms: f64,
    pub checksum_hex: String,
    pub status_mismatches_vs_strict: u32,
    pub max_abs_cut_error_vs_strict: f64,
}

#[napi]
pub fn run_blackwell_b14o_fp32_packed(
    snapshot_path: Option<String>,
    strict_reference_path: String,
    pinned_input: bool,
    wait_mode: u32,
    mapped_output: bool,
    schedule_spin: bool,
    device_id: Option<i32>,
) -> Result<BlackwellB14oRoundtripResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path
            );

    let snapshot =
        read_blackwell_snapshot(&path)
            .map_err(|error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell snapshot {}: {error}",
                        path.display(),
                    ),
                )
            })?;

    if snapshot.steps.is_empty() ||
        snapshot.tank_count == 0
    {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty.",
            ),
        );
    }

    let tank_parameters =
        (0..snapshot.tank_count)
            .map(|tank| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank,
                )
            })
            .collect::<Vec<_>>();

    let mut packed_tanks64 =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 6,
        );

    for tank in &tank_parameters {
        packed_tanks64.extend_from_slice(&[
            tank.target_storage_energy,
            tank.maximum_heat_pump_electrical_power,
            tank.cop * tank.time_step_hours,
            tank.heat_loss_retention,
            tank.heat_loss_intercept,
            if tank.affine_supported {
                1.0
            } else {
                0.0
            },
        ]);
    }

    let step_count =
        snapshot.steps.len();

    let jobs =
        step_count
            .checked_mul(
                snapshot.tank_count
            )
            .ok_or_else(|| {
                Error::from_reason(
                    "B14O job count overflow.",
                )
            })?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count
        );

    let mut horizons =
        Vec::<u32>::with_capacity(
            step_count
        );

    let mut dynamic64 =
        Vec::<f64>::with_capacity(
            jobs * 2
        );

    let mut energy_cut64 =
        Vec::<f64>::with_capacity(
            jobs
        );

    for step in &snapshot.steps {
        current_indices.push(
            step.current_index
        );

        horizons.push(
            step.horizon_steps
        );

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        for tank_index in
            0..snapshot.tank_count
        {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let tank =
                tank_parameters[
                    tank_index
                ];

            dynamic64.push(
                step.dynamic[base + 0]
            );

            dynamic64.push(
                step.dynamic[base + 1]
            );

            energy_cut64.push(
                research_energy_cut_level(
                    &export_window,
                    step.dynamic[base + 2],
                    tank,
                ),
            );
        }
    }

    let export32 =
        snapshot.export_power
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

    let demand32 =
        snapshot.demand_energy
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

    let tanks32 =
        packed_tanks64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

    let dynamic32 =
        dynamic64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

    let cuts_in32 =
        energy_cut64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

    let mut cuts_out =
        vec![0.0_f32; jobs];

    let mut status_out =
        vec![0_u32; jobs];

    let mut static_h2d_ms = 0.0;
    let mut host_pack_ms = 0.0;
    let mut input_enqueue_ms = 0.0;
    let mut kernel_ms = 0.0;
    let mut result_wait_ms = 0.0;
    let mut wall_ms = 0.0;

    let code = unsafe {
        sunopt_cuda_run_b14o_fp32_packed_benchmark(
            device_id.unwrap_or(0) as c_int,
            export32.as_ptr(),
            demand32.as_ptr(),
            snapshot.data_length,
            tanks32.as_ptr(),
            snapshot.tank_count,
            current_indices.as_ptr(),
            horizons.as_ptr(),
            dynamic32.as_ptr(),
            cuts_in32.as_ptr(),
            step_count,
            if pinned_input { 1 } else { 0 },
            wait_mode,
            if mapped_output { 1 } else { 0 },
            if schedule_spin { 1 } else { 0 },
            cuts_out.as_mut_ptr(),
            status_out.as_mut_ptr(),
            &mut static_h2d_ms,
            &mut host_pack_ms,
            &mut input_enqueue_ms,
            &mut kernel_ms,
            &mut result_wait_ms,
            &mut wall_ms,
        )
    };

    if code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_b14o_fp32_packed_benchmark",
                code,
            ),
        );
    }

    let reference_path =
        PathBuf::from(
            strict_reference_path
        );

    let bytes =
        std::fs::read(
            &reference_path
        ).map_err(|error| {
            Error::from_reason(
                format!(
                    "Unable to read strict reference {}: {error}",
                    reference_path.display(),
                ),
            )
        })?;

    const HEADER: usize = 16;

    if bytes.len() < HEADER ||
        &bytes[0..8] != b"B14MREF1"
    {
        return Err(
            Error::from_reason(
                "Invalid B14M strict reference header.",
            ),
        );
    }

    let mut count_bytes =
        [0_u8; 8];

    count_bytes.copy_from_slice(
        &bytes[8..16]
    );

    let count =
        u64::from_le_bytes(
            count_bytes
        ) as usize;

    let expected =
        HEADER +
        count * 8 +
        count * 4;

    if count != jobs ||
        bytes.len() != expected
    {
        return Err(
            Error::from_reason(
                format!(
                    "Strict reference shape mismatch: jobs={jobs} reference={count} bytes={} expected={expected}",
                    bytes.len(),
                ),
            ),
        );
    }

    let cut_base = HEADER;
    let status_base =
        HEADER + jobs * 8;

    let mut checksum =
        1469598103934665603_u64;

    let mut status_mismatches =
        0_u32;

    let mut max_abs_cut_error =
        0.0_f64;

    for job in 0..jobs {
        let value64 =
            cuts_out[job] as f64;

        blackwell_fnv1a_update(
            &mut checksum,
            value64,
        );

        let cut_offset =
            cut_base +
            job * 8;

        let mut cut_bytes =
            [0_u8; 8];

        cut_bytes.copy_from_slice(
            &bytes[
                cut_offset
                    ..
                cut_offset + 8
            ]
        );

        let strict_cut =
            f64::from_le_bytes(
                cut_bytes
            );

        let status_offset =
            status_base +
            job * 4;

        let mut status_bytes =
            [0_u8; 4];

        status_bytes.copy_from_slice(
            &bytes[
                status_offset
                    ..
                status_offset + 4
            ]
        );

        let strict_status =
            u32::from_le_bytes(
                status_bytes
            );

        if status_out[job] !=
            strict_status
        {
            status_mismatches += 1;
        }

        max_abs_cut_error =
            max_abs_cut_error.max(
                (
                    value64 -
                    strict_cut
                ).abs()
            );
    }

    Ok(
        BlackwellB14oRoundtripResult {
            pinned_input,
            wait_mode,
            mapped_output,
            schedule_spin,
            annual_step_count:
                step_count as u32,
            tank_count:
                snapshot.tank_count as u32,
            causal_jobs:
                jobs as u32,
            static_h2d_ms,
            host_pack_ms,
            input_enqueue_ms,
            kernel_ms,
            result_wait_ms,
            wall_ms,
            checksum_hex:
                format!(
                    "{checksum:016x}"
                ),
            status_mismatches_vs_strict:
                status_mismatches,
            max_abs_cut_error_vs_strict:
                max_abs_cut_error,
        },
    )
}
// SUNOPT_BLACKWELL_B14O_ROUNDTRIP_FN_END
`.trim();

  const CHILD = [
    '// rust/sunopt-node/scripts/blackwell-b14o-child.mjs',
    'import { createRequire } from "node:module";',
    'import fs from "node:fs";',
    'import path from "node:path";',
    'import { execFileSync } from "node:child_process";',
    'import { fileURLToPath } from "node:url";',
    '',
    'const require = createRequire(import.meta.url);',
    'const here = path.dirname(fileURLToPath(import.meta.url));',
    'const nativeRoot = path.resolve(here, "..");',
    'const projectRoot = path.resolve(nativeRoot, "..", "..");',
    'const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));',
    '',
    'const args = new Map();',
    'for (let i = 2; i + 1 < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);',
    '',
    'const kind = args.get("--kind") ?? "b14o";',
    'const snapshot = args.get("--snapshot") ?? path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
    'const reference = args.get("--reference") ?? path.join(nativeRoot, "benchmarks", "references", "blackwell-b14m-strict-reference.bin");',
    '',
    'function boolArg(name) { return (args.get(name) ?? "false") === "true"; }',
    'function telemetry() {',
    '  try {',
    '    const text = execFileSync("nvidia-smi", [',
    '      "--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw",',
    '      "--format=csv,noheader,nounits"',
    '    ], { encoding: "utf8", windowsHide: true }).trim();',
    '    const [clock, temp, pstate, power] = text.split(",").map(x => x.trim());',
    '    return { clockMHz: Number(clock), temperatureC: Number(temp), pstate, powerW: Number(power) };',
    '  } catch { return null; }',
    '}',
    '',
    'const before = telemetry();',
    'let result;',
    '',
    'if (kind === "b14l") {',
    '  result = addon.runBlackwellB14LPackedTransport(snapshot, boolArg("--pinned"), 0);',
    '} else if (kind === "reference") {',
    '  fs.mkdirSync(path.dirname(reference), { recursive: true });',
    '  result = addon.runBlackwellB14MPrecision(snapshot, "strict-reference", reference, true, 0);',
    '} else {',
    '  result = addon.runBlackwellB14OFp32Packed(',
    '    snapshot,',
    '    reference,',
    '    boolArg("--pinned-input"),',
    '    Number(args.get("--wait-mode") ?? "0"),',
    '    boolArg("--mapped-output"),',
    '    boolArg("--schedule-spin"),',
    '    0',
    '  );',
    '}',
    '',
    'const after = telemetry();',
    'process.stdout.write(JSON.stringify({ kind, before, after, result }) + "\\n");',
    ''
  ].join("\n");

  const RUNNER = [
    '// rust/sunopt-node/scripts/blackwell-b14o-roundtrip-workflow.mjs',
    'import fs from "node:fs";',
    'import path from "node:path";',
    'import { spawnSync } from "node:child_process";',
    'import { fileURLToPath } from "node:url";',
    '',
    'const here = path.dirname(fileURLToPath(import.meta.url));',
    'const nativeRoot = path.resolve(here, "..");',
    'const projectRoot = path.resolve(nativeRoot, "..", "..");',
    'const outbox = path.join(projectRoot, "updates", "outbox");',
    'const hashFile = path.join(projectRoot, "updates", "history", "hash.txt");',
    'const child = path.join(nativeRoot, "scripts", "blackwell-b14o-child.mjs");',
    'const snapshot = path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
    'const reference = path.join(nativeRoot, "benchmarks", "references", "blackwell-b14m-strict-reference.bin");',
    'fs.mkdirSync(outbox, { recursive: true });',
    '',
    'function runChild(extra) {',
    '  const args = [child, "--snapshot", snapshot, "--reference", reference, ...extra];',
    '  const p = spawnSync(process.execPath, args, { cwd: projectRoot, encoding: "utf8", windowsHide: true });',
    '  if (p.status !== 0) throw new Error(`child failed ${extra.join(" ")}\\nstdout:\\n${p.stdout}\\nstderr:\\n${p.stderr}`);',
    '  const lines = p.stdout.trim().split(/\\r?\\n/);',
    '  return JSON.parse(lines[lines.length - 1]);',
    '}',
    '',
    'function median(values) {',
    '  const a = [...values].sort((x, y) => x - y);',
    '  const m = Math.floor(a.length / 2);',
    '  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;',
    '}',
    '',
    'function summarize(label, rows) {',
    '  const wall = rows.map(x => x.result.wallMs);',
    '  const kernel = rows.map(x => x.result.kernelMs);',
    '  return {',
    '    label,',
    '    count: rows.length,',
    '    wallMs: wall,',
    '    kernelMs: kernel,',
    '    medianWallMs: median(wall),',
    '    medianKernelMs: median(kernel),',
    '    minWallMs: Math.min(...wall),',
    '    maxWallMs: Math.max(...wall),',
    '    telemetry: rows.map(x => ({ before: x.before, after: x.after })),',
    '    firstResult: rows[0]?.result ?? null,',
    '  };',
    '}',
    '',
    'const release = fs.existsSync(hashFile) ? fs.readFileSync(hashFile, "utf8").trim() : "unknown";',
    '',
    '// Persistent strict reference, generated outside the outbox.',
    'runChild(["--kind", "reference"]);',
    '',
    '// A) B14L pageable/pinned robust A/B.',
    'runChild(["--kind", "b14l", "--pinned", "false"]);',
    'runChild(["--kind", "b14l", "--pinned", "true"]);',
    '',
    'const pinnedRows = { pageable: [], pinned: [] };',
    'const abba = ["pageable","pinned","pinned","pageable","pageable","pinned","pinned","pageable","pageable","pinned"];',
    'for (const name of abba) {',
    '  const pinned = name === "pinned";',
    '  pinnedRows[name].push(runChild(["--kind","b14l","--pinned",String(pinned)]));',
    '}',
    '',
    'const pageable = summarize("pageable", pinnedRows.pageable);',
    'const pinned = summarize("pinned", pinnedRows.pinned);',
    'const pinnedDeltaPct = 100 * (pinned.medianWallMs - pageable.medianWallMs) / pageable.medianWallMs;',
    'const pinnedConclusion = Math.abs(pinnedDeltaPct) < 10 ? "INCONCLUSIVE_LT_10_PERCENT" : (pinnedDeltaPct < 0 ? "PINNED_FASTER" : "PAGEABLE_FASTER");',
    '',
    '// B) Packed FP32 roundtrip variants.',
    'const variants = {',
    '  A: { label: "fp32-pageable-stream-auto", args: ["--kind","b14o","--pinned-input","false","--wait-mode","0","--mapped-output","false","--schedule-spin","false"] },',
    '  B: { label: "fp32-pageable-stream-spin", args: ["--kind","b14o","--pinned-input","false","--wait-mode","0","--mapped-output","false","--schedule-spin","true"] },',
    '  C: { label: "fp32-pageable-event-spin", args: ["--kind","b14o","--pinned-input","false","--wait-mode","1","--mapped-output","false","--schedule-spin","true"] },',
    '  D: { label: "fp32-mapped-event-spin", args: ["--kind","b14o","--pinned-input","false","--wait-mode","1","--mapped-output","true","--schedule-spin","true"] },',
    '};',
    '',
    'for (const v of Object.values(variants)) runChild(v.args);',
    '',
    'const rows = { A: [], B: [], C: [], D: [] };',
    'const sequence = [',
    '  ["A","B","C","D"],',
    '  ["D","C","B","A"],',
    '  ["B","A","D","C"],',
    '  ["C","D","A","B"],',
    '  ["A","C","B","D"],',
    '];',
    'for (const cycle of sequence) {',
    '  for (const key of cycle) rows[key].push(runChild(variants[key].args));',
    '}',
    '',
    'const roundtrip = Object.fromEntries(Object.entries(rows).map(([key, value]) => [key, summarize(variants[key].label, value)]));',
    'const baselineWall = roundtrip.A.medianWallMs;',
    'for (const key of Object.keys(roundtrip)) {',
    '  const item = roundtrip[key];',
    '  item.speedupVsA = baselineWall / item.medianWallMs;',
    '  item.deltaPctVsA = 100 * (item.medianWallMs - baselineWall) / baselineWall;',
    '  item.meaningfulVsA = Math.abs(item.deltaPctVsA) >= 10;',
    '}',
    '',
    'const bestKey = Object.keys(roundtrip).reduce((a,b) => roundtrip[a].medianWallMs <= roundtrip[b].medianWallMs ? a : b);',
    '',
    'const report = {',
    '  stage: "B14O_ROUNDTRIP_WORKFLOW",',
    '  release,',
    '  measurementPolicy: { measuredRunsPerVariant: 5, warmupRunsPerVariant: 1, medianAuthoritative: true, noiseThresholdPercent: 10 },',
    '  pinnedAB: { pageable, pinned, deltaPct: pinnedDeltaPct, conclusion: pinnedConclusion },',
    '  roundtrip,',
    '  bestRoundtrip: { key: bestKey, ...roundtrip[bestKey] },',
    '  persistentStrictReference: reference,',
    '};',
    '',
    'const reportPath = path.join(outbox, "blackwell-b14o-roundtrip-workflow.json");',
    'fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\\n");',
    '',
    'const llmPath = path.join(outbox, "llm.txt");',
    'fs.appendFileSync(llmPath, [',
    '  "",',
    '  `=== ${new Date().toISOString()} B14O_ROUNDTRIP_WORKFLOW ===`,',
    '  `release=${release}`,',
    '  `pinned pageableMedianWallMs=${pageable.medianWallMs.toFixed(6)} pinnedMedianWallMs=${pinned.medianWallMs.toFixed(6)} deltaPct=${pinnedDeltaPct.toFixed(3)} conclusion=${pinnedConclusion}`,',
    '  ...Object.entries(roundtrip).map(([k,v]) => `${k} ${v.label} medianKernelMs=${v.medianKernelMs.toFixed(6)} medianWallMs=${v.medianWallMs.toFixed(6)} deltaPctVsA=${v.deltaPctVsA.toFixed(3)} meaningfulVsA=${v.meaningfulVsA}`),',
    '  `bestRoundtrip=${bestKey} ${roundtrip[bestKey].label} wallMs=${roundtrip[bestKey].medianWallMs.toFixed(6)}`,',
    '  `persistentStrictReference=${reference}`,',
    '  `report=${reportPath}`,',
    '  "",',
    '].join("\\n"), "utf8");',
    '',
    'console.log(JSON.stringify(report, null, 2));',
    'console.log(`report=${reportPath}`);',
    'console.log(`llm=${llmPath}`);',
    ''
  ].join("\n");

  let h = fs.readFileSync(hPath, "utf8");
  let cu = fs.readFileSync(cuPath, "utf8");
  let rs = fs.readFileSync(rsPath, "utf8");

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

  rs = insertBefore(
    rs,
    "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_BEGIN",
    RS_FFI_BLOCK,
    "cuda_context.rs FFI",
  );

  rs = insertBefore(
    rs,
    "// SUNOPT_BLACKWELL_B14G_WARP_BINARY_FN_BEGIN",
    RS_BLOCK,
    "cuda_context.rs NAPI",
  );

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
      [...files.entries()].map(([name, data]) => [name, sha256(data)]),
    ),
    delete: [],
  };

  const manifestBytes = Buffer.from(
    JSON.stringify(manifest, null, 2) + "\n",
    "utf8",
  );

  const release = sha256(manifestBytes);

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

  function u16(n) {
    const b = Buffer.alloc(2);
    b.writeUInt16LE(n >>> 0);
    return b;
  }

  function u32(n) {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(n >>> 0);
    return b;
  }

  const entries = [
    ["manifest.json", manifestBytes],
    ["manifest.sha256", Buffer.from(release + "\n", "ascii")],
    ...files.entries(),
  ];

  const locals = [];
  const centrals = [];
  let offset = 0;

  for (const [name, data] of entries) {
    const nameBuf = Buffer.from(name, "utf8");
    const crc = crc32(data);

    const local = Buffer.concat([
      u32(0x04034b50),
      u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc),
      u32(data.length),
      u32(data.length),
      u16(nameBuf.length),
      u16(0),
      nameBuf,
      data,
    ]);

    locals.push(local);

    const central = Buffer.concat([
      u32(0x02014b50),
      u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc),
      u32(data.length),
      u32(data.length),
      u16(nameBuf.length),
      u16(0), u16(0), u16(0), u16(0),
      u32(0),
      u32(offset),
      nameBuf,
    ]);

    centrals.push(central);
    offset += local.length;
  }

  const centralOffset = offset;
  const central = Buffer.concat(centrals);

  const end = Buffer.concat([
    u32(0x06054b50),
    u16(0), u16(0),
    u16(entries.length),
    u16(entries.length),
    u32(central.length),
    u32(centralOffset),
    u16(0),
  ]);

  const zip = Buffer.concat([
    ...locals,
    central,
    end,
  ]);

  fs.mkdirSync(inbox, { recursive: true });

  const packageName =
    "20261112-blackwell-b14o-roundtrip-workflow.zip";

  const packagePath =
    path.join(inbox, packageName);

  fs.writeFileSync(
    packagePath,
    zip
  );

  console.log(`parent=${parent}`);
  console.log(`release=${release}`);
  console.log(`package=${packagePath}`);
  console.log("B14O_PACKAGE_CREATED=OK");

  // Wait for the already-running watcher.
  const waitStart = Date.now();
  const maxWaitMs = 120000;

  for (;;) {
    if (fs.existsSync(hashFile)) {
      const current =
        fs.readFileSync(
          hashFile,
          "utf8"
        ).trim();

      if (current === release) {
        break;
      }
    }

    const rejectedPath =
      path.join(
        rejected,
        packageName
      );

    if (fs.existsSync(rejectedPath)) {
      throw new Error(
        `Watcher rejected ${packageName}`
      );
    }

    if (
      Date.now() -
      waitStart >
      maxWaitMs
    ) {
      throw new Error(
        `Timed out waiting for watcher to install ${packageName}`
      );
    }

    sleepMs(250);
  }

  console.log("B14O_WATCHER_INSTALL=OK");

  // Build strict automatically.
  const build =
    spawnSync(
      process.platform === "win32"
        ? "npm.cmd"
        : "npm",
      ["run", "build"],
      {
        cwd: nativeRoot,
        encoding: "utf8",
        windowsHide: true,
        env: {
          ...process.env,
          SUNOPT_CUDA_FAST_MATH: "0",
        },
      }
    );

  fs.writeFileSync(
    path.join(
      outbox,
      "b14o-build.txt"
    ),
    [
      build.stdout ?? "",
      build.stderr ?? "",
    ].join("\n"),
    "utf8"
  );

  if (build.status !== 0) {
    writeFailure(
      failedBuild,
      "build",
      `exit=${build.status}`,
      [
        build.stdout ?? "",
        build.stderr ?? "",
      ].join("\n")
    );
    process.exit(2);
  }

  console.log("B14O_BUILD=OK");

  // Run automatically.
  const run =
    spawnSync(
      process.execPath,
      [
        path.join(
          nativeRoot,
          "scripts",
          "blackwell-b14o-roundtrip-workflow.mjs"
        ),
      ],
      {
        cwd: root,
        encoding: "utf8",
        windowsHide: true,
      }
    );

  fs.writeFileSync(
    path.join(
      outbox,
      "b14o-run.txt"
    ),
    [
      run.stdout ?? "",
      run.stderr ?? "",
    ].join("\n"),
    "utf8"
  );

  if (run.status !== 0) {
    writeFailure(
      failedRun,
      "run",
      `exit=${run.status}`,
      [
        run.stdout ?? "",
        run.stderr ?? "",
      ].join("\n")
    );
    process.exit(3);
  }

  process.stdout.write(
    run.stdout ?? ""
  );

  console.log(
    "B14O_SELF_RUNNING_WORKFLOW=OK"
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
