// rust/sunopt-node/scripts/make-blackwell-b14m-fp32-fastmath.mjs
//
// B14M: serial precision/compiler sweep around the B14L transport shape.
//
// Installs:
//   - SUNOPT_CUDA_FAST_MATH=1 build toggle -> nvcc --use_fast_math
//   - FP32 depth-2 speculative kernel (96 threads/tank)
//   - strict FP64 reference writer/comparator
//   - serial PowerShell runner:
//       strict build -> strict FP64 + strict FP32
//       fast-math build -> fast FP64 + fast FP32
//       strict build restored at the end
//
// Reports go to updates/outbox and llm.txt is appended.
//
// Generator usage from SunOpt root:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14m-fp32-fastmath.mjs
//
// The normal always-running watcher installs the generated inbox package.
// Then run:
//   powershell -ExecutionPolicy Bypass -File .\rust\sunopt-node\scripts\run-blackwell-b14m-fp32-fastmath.ps1

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");
const inbox = path.join(root, "updates", "inbox");
const hashFile = path.join(root, "updates", "history", "hash.txt");

const BUILD_REL = "rust/sunopt-node/build.rs";
const H_REL = "rust/sunopt-node/cuda/cuda_context.h";
const CU_REL = "rust/sunopt-node/cuda/cuda_context.cu";
const RS_REL = "rust/sunopt-node/src/cuda_context.rs";
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14m-precision.mjs";
const SWEEP_REL = "rust/sunopt-node/scripts/run-blackwell-b14m-fp32-fastmath.ps1";

const buildPath = path.join(root, BUILD_REL);
const hPath = path.join(root, H_REL);
const cuPath = path.join(root, CU_REL);
const rsPath = path.join(root, RS_REL);

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function insertBefore(text, anchor, block, label) {
  const marker = block.split(/\r?\n/, 1)[0];
  if (text.includes(marker)) {
    throw new Error(`${label}: B14M block already exists`);
  }
  const pos = text.indexOf(anchor);
  if (pos < 0) {
    throw new Error(`${label}: anchor not found: ${anchor}`);
  }
  return text.slice(0, pos) + block.trimEnd() + "\n\n" + text.slice(pos);
}

function patchBuildRs(text) {
  if (text.includes("SUNOPT_CUDA_FAST_MATH")) {
    return text;
  }

  const rerunAnchor =
    '    println!("cargo:rerun-if-env-changed=SUNOPT_CUDA_ARCH");';

  if (!text.includes(rerunAnchor)) {
    throw new Error("build.rs: SUNOPT_CUDA_ARCH rerun anchor not found");
  }

  text = text.replace(
    rerunAnchor,
    rerunAnchor +
      '\n    println!("cargo:rerun-if-env-changed=SUNOPT_CUDA_FAST_MATH");'
  );

  const startNeedle =
    "    let status = Command::new(&nvcc)\n";
  const start = text.indexOf(startNeedle);

  if (start < 0) {
    throw new Error("build.rs: nvcc command start not found");
  }

  const endNeedle =
    '        .expect("failed to execute nvcc");';
  const endStart = text.indexOf(endNeedle, start);

  if (endStart < 0) {
    throw new Error("build.rs: nvcc command end not found");
  }

  const end = endStart + endNeedle.length;

  const replacement = String.raw`    let fast_math = env::var("SUNOPT_CUDA_FAST_MATH")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            value == "1" || value == "true" || value == "yes" || value == "on"
        })
        .unwrap_or(false);

    let mut nvcc_command = Command::new(&nvcc);
    nvcc_command
        .arg("-lib")
        .arg("-O3")
        // SUNOPT_CUDA_CPP17
        .arg("-std=c++17")
        // SUNOPT_MSVC_STANDARD_PREPROCESSOR
        .arg("-Xcompiler")
        .arg("/Zc:preprocessor")
        .arg(format!("-arch={arch}"));

    if fast_math {
        nvcc_command.arg("--use_fast_math");
        println!("cargo:warning=SunOpt CUDA fast math ENABLED");
    } else {
        println!("cargo:warning=SunOpt CUDA fast math disabled");
    }

    let status = nvcc_command
        .arg("cuda/cuda_context.cu")
        .arg("-o")
        .arg(&output_lib)
        .status()
        .expect("failed to execute nvcc");`;

  return text.slice(0, start) + replacement + text.slice(end);
}

if (!fs.existsSync(hashFile)) {
  throw new Error(`Current watcher hash not found: ${hashFile}`);
}

const parent = fs.readFileSync(hashFile, "utf8").trim().toLowerCase();

if (!/^[0-9a-f]{64}$/.test(parent)) {
  throw new Error(`Invalid current watcher hash: ${parent}`);
}

const H_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14M_FP32_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14m_fp32_benchmark(
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
    float* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_input_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14M_FP32_END
`.trim();

const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14M_FP32_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14M_WARPS = 3u;
constexpr unsigned int SUNOPT_B14M_THREADS = 96u;
constexpr unsigned int SUNOPT_B14M_MASK = 0xffffffffu;

struct SunoptB14MTransform {
    float a;
    float b;
    float c;
    float max_input;
};

__device__ __forceinline__
SunoptB14MTransform sunopt_b14m_identity() {
    SunoptB14MTransform t{};
    t.a = 1.0f;
    t.b = 0.0f;
    t.c = -CUDART_INF_F;
    t.max_input = CUDART_INF_F;
    return t;
}

__device__ __forceinline__
SunoptB14MTransform sunopt_b14m_compose(
    const SunoptB14MTransform left,
    const SunoptB14MTransform right
) {
    SunoptB14MTransform out{};

    float max_input = right.max_input;

    if (right.c > left.max_input + 1.0e-12f) {
        max_input = -CUDART_INF_F;
    } else if (right.a > 0.0f) {
        const float bound =
            (left.max_input - right.b) /
            right.a;
        max_input = fminf(max_input, bound);
    } else if (right.b > left.max_input + 1.0e-12f) {
        max_input = -CUDART_INF_F;
    }

    out.a = left.a * right.a;
    out.b = left.a * right.b + left.b;
    out.c =
        fmaxf(
            left.c,
            left.a * right.c + left.b
        );
    out.max_input = max_input;

    return out;
}

__device__ __forceinline__
SunoptB14MTransform sunopt_b14m_shuffle_down(
    const SunoptB14MTransform value,
    unsigned int delta
) {
    SunoptB14MTransform out{};
    out.a = __shfl_down_sync(
        SUNOPT_B14M_MASK,
        value.a,
        delta
    );
    out.b = __shfl_down_sync(
        SUNOPT_B14M_MASK,
        value.b,
        delta
    );
    out.c = __shfl_down_sync(
        SUNOPT_B14M_MASK,
        value.c,
        delta
    );
    out.max_input = __shfl_down_sync(
        SUNOPT_B14M_MASK,
        value.max_input,
        delta
    );
    return out;
}

__device__ __forceinline__
bool sunopt_b14m_fixed_survives_warp(
    const float* export_power,
    const float* demand_energy,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    float cut,
    float current_storage_energy,
    float reserve,
    float target,
    float max_hp,
    float hp_factor,
    float retention,
    float loss_intercept,
    int lane
) {
    const size_t future_count =
        horizon > 1u ? horizon - 1u : 0u;

    const size_t chunk_size =
        (future_count + 31u) / 32u;

    const size_t chunk_begin =
        1u +
        static_cast<size_t>(lane) *
        chunk_size;

    const size_t chunk_end =
        chunk_size == 0u
            ? 1u
            : (
                (chunk_begin + chunk_size) <
                horizon
                    ? chunk_begin + chunk_size
                    : horizon
            );

    SunoptB14MTransform local =
        sunopt_b14m_identity();

    const float inverse_retention =
        1.0f / retention;

    for (
        size_t offset = chunk_begin;
        offset < chunk_end &&
        offset < horizon;
        ++offset
    ) {
        const size_t index =
            (current_index + offset) %
            data_length;

        const float demand =
            fmaxf(
                demand_energy[index],
                0.0f
            );

        const float raw =
            export_power[index] - cut;

        const float hp =
            fminf(
                max_hp,
                fmaxf(0.0f, raw)
            );

        const float charge =
            hp * hp_factor;

        SunoptB14MTransform step{};
        step.a = inverse_retention;
        step.b =
            (
                loss_intercept +
                demand -
                charge
            ) *
            inverse_retention;

        step.c =
            (
                loss_intercept +
                demand +
                reserve -
                static_cast<float>(
                    SUNOPT_B14C_EPSILON
                )
            ) *
            inverse_retention;

        step.max_input = target;

        local =
            sunopt_b14m_compose(
                local,
                step
            );
    }

    for (
        unsigned int stride = 1u;
        stride < 32u;
        stride <<= 1u
    ) {
        const SunoptB14MTransform right =
            sunopt_b14m_shuffle_down(
                local,
                stride
            );

        const unsigned int group =
            stride << 1u;

        if (
            (
                static_cast<unsigned int>(lane) &
                (group - 1u)
            ) == 0u &&
            static_cast<unsigned int>(lane) +
                stride <
                32u
        ) {
            local =
                sunopt_b14m_compose(
                    local,
                    right
                );
        }
    }

    int survives = 0;

    if (lane == 0) {
        const SunoptB14MTransform summary =
            local;

        bool feasible =
            0.0f <=
            summary.max_input +
                1.0e-12f;

        const float required =
            fmaxf(
                summary.c,
                summary.b
            );

        if (
            required >
            target + 1.0e-9f
        ) {
            feasible = false;
        }

        const float raw0 =
            export_power[
                current_index %
                data_length
            ] -
            cut;

        const float hp0 =
            fminf(
                max_hp,
                fmaxf(0.0f, raw0)
            );

        const float charge0 =
            hp0 * hp_factor;

        const float current =
            fminf(
                target,
                fmaxf(
                    0.0f,
                    current_storage_energy
                )
            );

        const float available =
            fminf(
                target,
                current + charge0
            );

        survives =
            feasible &&
            available + 1.0e-12f >=
                required
                ? 1
                : 0;
    }

    return
        __shfl_sync(
            SUNOPT_B14M_MASK,
            survives,
            0
        ) != 0;
}

__global__ void sunopt_b14m_fp32_kernel(
    const float* __restrict__ export_power,
    const float* __restrict__ demand_energy,
    size_t data_length,
    const float* __restrict__ tank_parameters,
    size_t tank_count,
    uint32_t current_index_u32,
    uint32_t horizon_u32,
    const float* __restrict__ dynamic_values,
    const float* __restrict__ energy_cut_levels,
    float* __restrict__ cut_results,
    uint32_t* __restrict__ status
) {
    const size_t tank =
        static_cast<size_t>(blockIdx.x);

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

    __shared__ int
        shared_survives[
            SUNOPT_B14M_WARPS
        ];

    __shared__ int shared_precheck;
    __shared__ float shared_lower;
    __shared__ float shared_upper;

    const size_t tb = tank * 6u;
    const size_t db = tank * 2u;

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

    const size_t current_index =
        static_cast<size_t>(
            current_index_u32
        );

    const size_t horizon =
        static_cast<size_t>(
            horizon_u32
        );

    const float energy_cut =
        energy_cut_levels[tank];

    if (tid == 0u) {
        cut_results[tank] =
            energy_cut;
        status[tank] = 3u;
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
            status[tank] = 0u;
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
            cut_results[tank] = 0.0f;
            status[tank] = 1u;
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
        if (warp < SUNOPT_B14M_WARPS) {
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
                (lo + hi) *
                0.5f;

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
        cut_results[tank] =
            shared_lower;
        status[tank] = 2u;
    }
}

} // namespace

extern "C" int
sunopt_cuda_run_b14m_fp32_benchmark(
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
    float* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
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
        !out_input_enqueue_ms ||
        !out_kernel_ms ||
        !out_result_wait_ms ||
        !out_wall_ms ||
        data_length == 0u ||
        tank_count == 0u ||
        step_count == 0u
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(device_id);

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    float* d_export = nullptr;
    float* d_demand = nullptr;
    float* d_tanks = nullptr;
    float* d_dynamic = nullptr;
    float* d_energy_cut = nullptr;
    float* d_cut_results = nullptr;
    uint32_t* d_status = nullptr;

    const size_t annual_bytes =
        data_length * sizeof(float);

    const size_t tank_bytes =
        tank_count *
        6u *
        sizeof(float);

    const size_t dynamic_step_bytes =
        tank_count *
        2u *
        sizeof(float);

    const size_t cut_step_bytes =
        tank_count *
        sizeof(float);

    const size_t status_step_bytes =
        tank_count *
        sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_status) cudaFree(d_status);
        if (d_cut_results) cudaFree(d_cut_results);
        if (d_energy_cut) cudaFree(d_energy_cut);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (kernel_stop) cudaEventDestroy(kernel_stop);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (stream) cudaStreamDestroy(stream);
    };

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

#define B14M_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14M_MALLOC(d_export, annual_bytes);
    B14M_MALLOC(d_demand, annual_bytes);
    B14M_MALLOC(d_tanks, tank_bytes);
    B14M_MALLOC(d_dynamic, dynamic_step_bytes);
    B14M_MALLOC(d_energy_cut, cut_step_bytes);
    B14M_MALLOC(d_cut_results, cut_step_bytes);
    B14M_MALLOC(d_status, status_step_bytes);

#undef B14M_MALLOC

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
        std::chrono::duration<double,std::milli>(
            Clock::now() - static_start
        ).count();

    double input_enqueue_ms = 0.0;
    double kernel_ms = 0.0;
    double result_wait_ms = 0.0;

    for (
        size_t step = 0u;
        step < step_count;
        ++step
    ) {
        const auto input_start =
            Clock::now();

        err = cudaMemcpyAsync(
            d_dynamic,
            host_dynamic_values +
                step *
                tank_count *
                2u,
            dynamic_step_bytes,
            cudaMemcpyHostToDevice,
            stream
        );
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = cudaMemcpyAsync(
            d_energy_cut,
            host_energy_cut_levels +
                step *
                tank_count,
            cut_step_bytes,
            cudaMemcpyHostToDevice,
            stream
        );
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        input_enqueue_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - input_start
            ).count();

        err = cudaEventRecord(
            kernel_start,
            stream
        );
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        sunopt_b14m_fp32_kernel
            <<<static_cast<unsigned int>(
                   tank_count
               ),
               SUNOPT_B14M_THREADS,
               0,
               stream>>>(
                d_export,
                d_demand,
                data_length,
                d_tanks,
                tank_count,
                host_current_indices[step],
                host_horizon_steps[step],
                d_dynamic,
                d_energy_cut,
                d_cut_results,
                d_status
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

        err = cudaMemcpyAsync(
            host_cut_results +
                step *
                tank_count,
            d_cut_results,
            cut_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = cudaMemcpyAsync(
            host_status +
                step *
                tank_count,
            d_status,
            status_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = cudaStreamSynchronize(
            stream
        );
        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        result_wait_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - wait_start
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
    }

    *out_input_enqueue_ms =
        input_enqueue_ms;

    *out_kernel_ms =
        kernel_ms;

    *out_result_wait_ms =
        result_wait_ms;

    *out_wall_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - wall_start
        ).count();

    cleanup();
    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_B14M_FP32_END
`.trim();

const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14M_FP32_FFI_BEGIN
    fn sunopt_cuda_run_b14m_fp32_benchmark(
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
        host_cut_results: *mut f32,
        host_status: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_input_enqueue_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14M_FP32_FFI_END
`.trim();

const RS_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14M_PRECISION_FN_BEGIN
#[napi(object)]
pub struct BlackwellB14mPrecisionResult {
    pub build_mode: String,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,

    pub fp64_kernel_ms: f64,
    pub fp64_wall_ms: f64,
    pub fp64_checksum_hex: String,

    pub fp32_static_h2d_ms: f64,
    pub fp32_input_enqueue_ms: f64,
    pub fp32_kernel_ms: f64,
    pub fp32_result_wait_ms: f64,
    pub fp32_wall_ms: f64,
    pub fp32_checksum_hex: String,

    pub fp32_status_mismatches_vs_fp64: u32,
    pub fp32_max_abs_cut_error_vs_fp64: f64,
    pub fp32_mean_abs_cut_error_vs_fp64: f64,
    pub fp32_cut_mismatches_1e6_vs_fp64: u32,
    pub fp32_cut_mismatches_1e4_vs_fp64: u32,

    pub strict_reference_loaded: bool,
    pub fp64_status_mismatches_vs_strict: u32,
    pub fp64_max_abs_cut_error_vs_strict: f64,
    pub fp32_status_mismatches_vs_strict: u32,
    pub fp32_max_abs_cut_error_vs_strict: f64,
}

#[napi]
pub fn run_blackwell_b14m_precision(
    snapshot_path: Option<String>,
    build_mode: String,
    strict_reference_path: Option<String>,
    write_strict_reference: Option<bool>,
    device_id: Option<i32>,
) -> Result<BlackwellB14mPrecisionResult> {
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

    if snapshot.steps.is_empty() || snapshot.tank_count == 0 {
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

    let step_count = snapshot.steps.len();

    let jobs =
        step_count
            .checked_mul(snapshot.tank_count)
            .ok_or_else(|| {
                Error::from_reason(
                    "B14M job count overflow.",
                )
            })?;

    let mut current_indices =
        Vec::<u32>::with_capacity(step_count);

    let mut horizons =
        Vec::<u32>::with_capacity(step_count);

    let mut dynamic64 =
        Vec::<f64>::with_capacity(jobs * 2);

    let mut energy_cut64 =
        Vec::<f64>::with_capacity(jobs);

    for step in &snapshot.steps {
        current_indices.push(step.current_index);
        horizons.push(step.horizon_steps);

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        for tank_index in 0..snapshot.tank_count {
            let base =
                tank_index * BLACKWELL_DYNAMIC_STRIDE;

            let tank =
                tank_parameters[tank_index];

            let current =
                step.dynamic[base + 0];

            let reserve =
                step.dynamic[base + 1];

            let requested =
                step.dynamic[base + 2];

            dynamic64.push(current);
            dynamic64.push(reserve);

            energy_cut64.push(
                research_energy_cut_level(
                    &export_window,
                    requested,
                    tank,
                ),
            );
        }
    }

    let selected_device =
        device_id.unwrap_or(0);

    // Current-build FP64 B14L reference.
    let mut cuts64 =
        vec![0.0_f64; jobs];

    let mut status64 =
        vec![0_u32; jobs];

    let mut fp64_static = 0.0_f64;
    let mut fp64_pack = 0.0_f64;
    let mut fp64_enqueue = 0.0_f64;
    let mut fp64_kernel = 0.0_f64;
    let mut fp64_wait = 0.0_f64;
    let mut fp64_wall = 0.0_f64;

    let fp64_code = unsafe {
        sunopt_cuda_run_b14l_packed_transport_benchmark(
            selected_device as c_int,
            snapshot.export_power.as_ptr(),
            snapshot.demand_energy.as_ptr(),
            snapshot.data_length,
            packed_tanks64.as_ptr(),
            snapshot.tank_count,
            current_indices.as_ptr(),
            horizons.as_ptr(),
            dynamic64.as_ptr(),
            energy_cut64.as_ptr(),
            step_count,
            0,
            cuts64.as_mut_ptr(),
            status64.as_mut_ptr(),
            &mut fp64_static,
            &mut fp64_pack,
            &mut fp64_enqueue,
            &mut fp64_kernel,
            &mut fp64_wait,
            &mut fp64_wall,
        )
    };

    if fp64_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_b14l_packed_transport_benchmark",
                fp64_code,
            ),
        );
    }

    // FP32 inputs prepared outside the timed native region.
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

    let energy_cut32 =
        energy_cut64
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();

    let mut cuts32 =
        vec![0.0_f32; jobs];

    let mut status32 =
        vec![0_u32; jobs];

    let mut fp32_static = 0.0_f64;
    let mut fp32_enqueue = 0.0_f64;
    let mut fp32_kernel = 0.0_f64;
    let mut fp32_wait = 0.0_f64;
    let mut fp32_wall = 0.0_f64;

    let fp32_code = unsafe {
        sunopt_cuda_run_b14m_fp32_benchmark(
            selected_device as c_int,
            export32.as_ptr(),
            demand32.as_ptr(),
            snapshot.data_length,
            tanks32.as_ptr(),
            snapshot.tank_count,
            current_indices.as_ptr(),
            horizons.as_ptr(),
            dynamic32.as_ptr(),
            energy_cut32.as_ptr(),
            step_count,
            cuts32.as_mut_ptr(),
            status32.as_mut_ptr(),
            &mut fp32_static,
            &mut fp32_enqueue,
            &mut fp32_kernel,
            &mut fp32_wait,
            &mut fp32_wall,
        )
    };

    if fp32_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_b14m_fp32_benchmark",
                fp32_code,
            ),
        );
    }

    let mut fp64_checksum =
        1469598103934665603_u64;

    for value in &cuts64 {
        blackwell_fnv1a_update(
            &mut fp64_checksum,
            *value,
        );
    }

    let mut fp32_checksum =
        1469598103934665603_u64;

    let mut status_mismatch_current =
        0_u32;

    let mut max_error_current =
        0.0_f64;

    let mut sum_error_current =
        0.0_f64;

    let mut mismatch_1e6 =
        0_u32;

    let mut mismatch_1e4 =
        0_u32;

    for job in 0..jobs {
        let fp32_value =
            cuts32[job] as f64;

        blackwell_fnv1a_update(
            &mut fp32_checksum,
            fp32_value,
        );

        if status32[job] != status64[job] {
            status_mismatch_current += 1;
        }

        let error =
            (fp32_value - cuts64[job]).abs();

        max_error_current =
            max_error_current.max(error);

        sum_error_current += error;

        if error > 1.0e-6 {
            mismatch_1e6 += 1;
        }

        if error > 1.0e-4 {
            mismatch_1e4 += 1;
        }
    }

    let reference_path =
        strict_reference_path
            .map(PathBuf::from);

    if write_strict_reference.unwrap_or(false) {
        if let Some(reference_path) =
            reference_path.as_ref()
        {
            let mut bytes =
                Vec::<u8>::with_capacity(
                    16 +
                    jobs * 12
                );

            bytes.extend_from_slice(
                b"B14MREF1"
            );

            bytes.extend_from_slice(
                &(jobs as u64)
                    .to_le_bytes()
            );

            for value in &cuts64 {
                bytes.extend_from_slice(
                    &value.to_le_bytes()
                );
            }

            for value in &status64 {
                bytes.extend_from_slice(
                    &value.to_le_bytes()
                );
            }

            std::fs::write(
                reference_path,
                bytes,
            ).map_err(|error| {
                Error::from_reason(
                    format!(
                        "Unable to write B14M strict reference {}: {error}",
                        reference_path.display(),
                    ),
                )
            })?;
        }
    }

    let mut reference_loaded = false;
    let mut fp64_status_vs_strict = 0_u32;
    let mut fp64_max_vs_strict = 0.0_f64;
    let mut fp32_status_vs_strict = 0_u32;
    let mut fp32_max_vs_strict = 0.0_f64;

    if let Some(reference_path) =
        reference_path.as_ref()
    {
        if reference_path.is_file() {
            let bytes =
                std::fs::read(
                    reference_path
                ).map_err(|error| {
                    Error::from_reason(
                        format!(
                            "Unable to read B14M strict reference {}: {error}",
                            reference_path.display(),
                        ),
                    )
                })?;

            const HEADER: usize = 15;

            if (
                bytes.len() >= HEADER &&
                &bytes[0..7] == b"B14MREF1"
            ) {
                let mut count_bytes =
                    [0_u8; 8];

                count_bytes.copy_from_slice(
                    &bytes[7..15]
                );

                let count =
                    u64::from_le_bytes(
                        count_bytes
                    ) as usize;

                let expected =
                    HEADER +
                    count * 8 +
                    count * 4;

                if count == jobs &&
                    bytes.len() == expected
                {
                    reference_loaded = true;

                    let cut_base = HEADER;
                    let status_base =
                        HEADER +
                        jobs * 8;

                    for job in 0..jobs {
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

                        if status64[job] != strict_status {
                            fp64_status_vs_strict += 1;
                        }

                        fp64_max_vs_strict =
                            fp64_max_vs_strict.max(
                                (
                                    cuts64[job] -
                                    strict_cut
                                ).abs()
                            );

                        if status32[job] != strict_status {
                            fp32_status_vs_strict += 1;
                        }

                        fp32_max_vs_strict =
                            fp32_max_vs_strict.max(
                                (
                                    cuts32[job] as f64 -
                                    strict_cut
                                ).abs()
                            );
                    }
                }
            }
        }
    }

    Ok(
        BlackwellB14mPrecisionResult {
            build_mode,
            annual_step_count:
                step_count as u32,
            tank_count:
                snapshot.tank_count as u32,
            causal_jobs:
                jobs as u32,

            fp64_kernel_ms:
                fp64_kernel,
            fp64_wall_ms:
                fp64_wall,
            fp64_checksum_hex:
                format!(
                    "{fp64_checksum:016x}"
                ),

            fp32_static_h2d_ms:
                fp32_static,
            fp32_input_enqueue_ms:
                fp32_enqueue,
            fp32_kernel_ms:
                fp32_kernel,
            fp32_result_wait_ms:
                fp32_wait,
            fp32_wall_ms:
                fp32_wall,
            fp32_checksum_hex:
                format!(
                    "{fp32_checksum:016x}"
                ),

            fp32_status_mismatches_vs_fp64:
                status_mismatch_current,
            fp32_max_abs_cut_error_vs_fp64:
                max_error_current,
            fp32_mean_abs_cut_error_vs_fp64:
                if jobs > 0 {
                    sum_error_current /
                    jobs as f64
                } else {
                    0.0
                },
            fp32_cut_mismatches_1e6_vs_fp64:
                mismatch_1e6,
            fp32_cut_mismatches_1e4_vs_fp64:
                mismatch_1e4,

            strict_reference_loaded:
                reference_loaded,
            fp64_status_mismatches_vs_strict:
                fp64_status_vs_strict,
            fp64_max_abs_cut_error_vs_strict:
                fp64_max_vs_strict,
            fp32_status_mismatches_vs_strict:
                fp32_status_vs_strict,
            fp32_max_abs_cut_error_vs_strict:
                fp32_max_vs_strict,
        },
    )
}
// SUNOPT_BLACKWELL_B14M_PRECISION_FN_END
`.trim();

const RUNNER = [
  '// rust/sunopt-node/scripts/blackwell-b14m-precision.mjs',
  'import { createRequire } from "node:module";',
  'import fs from "node:fs";',
  'import path from "node:path";',
  'import { fileURLToPath } from "node:url";',
  '',
  'const require = createRequire(import.meta.url);',
  'const here = path.dirname(fileURLToPath(import.meta.url));',
  'const nativeRoot = path.resolve(here, "..");',
  'const projectRoot = path.resolve(nativeRoot, "..", "..");',
  'const outbox = path.join(projectRoot, "updates", "outbox");',
  'const historyHash = path.join(projectRoot, "updates", "history", "hash.txt");',
  'fs.mkdirSync(outbox, { recursive: true });',
  '',
  'const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));',
  'const run = addon.runBlackwellB14MPrecision;',
  'if (typeof run !== "function") throw new Error("runBlackwellB14MPrecision native export missing");',
  '',
  'const args = new Map();',
  'for (let i = 2; i < process.argv.length; i += 2) {',
  '  args.set(process.argv[i], process.argv[i + 1]);',
  '}',
  '',
  'const mode = args.get("--mode") ?? "strict";',
  'const snapshot = args.get("--snapshot")',
  '  ? path.resolve(args.get("--snapshot"))',
  '  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
  'const reference = args.get("--reference")',
  '  ? path.resolve(args.get("--reference"))',
  '  : path.join(outbox, "blackwell-b14m-strict-reference.bin");',
  'const writeReference = (args.get("--write-reference") ?? "false") === "true";',
  'const currentRelease = fs.existsSync(historyHash) ? fs.readFileSync(historyHash, "utf8").trim() : "unknown";',
  '',
  'console.log(`B14M mode=${mode} release=${currentRelease}`);',
  'const result = run(snapshot, mode, reference, writeReference, 0);',
  '',
  'const report = {',
  '  stage: "B14M_FP32_FASTMATH",',
  '  release: currentRelease,',
  '  snapshotPath: snapshot,',
  '  strictReferencePath: reference,',
  '  ...result,',
  '  fp32KernelSpeedupVsFp64: result.fp64KernelMs / result.fp32KernelMs,',
  '  fp32WallSpeedupVsFp64: result.fp64WallMs / result.fp32WallMs,',
  '};',
  '',
  'const reportPath = path.join(outbox, `blackwell-b14m-${mode}.json`);',
  'fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\\n");',
  '',
  'const llmPath = path.join(outbox, "llm.txt");',
  'const line = [',
  '  "",',
  '  `=== ${new Date().toISOString()} B14M_${mode.toUpperCase()} ===`,',
  '  `release=${currentRelease}`,',
  '  `fp64 kernelMs=${result.fp64KernelMs.toFixed(6)} wallMs=${result.fp64WallMs.toFixed(6)} checksum=${result.fp64ChecksumHex}`,',
  '  `fp32 kernelMs=${result.fp32KernelMs.toFixed(6)} wallMs=${result.fp32WallMs.toFixed(6)} checksum=${result.fp32ChecksumHex}`,',
  '  `fp32 speedupKernel=${report.fp32KernelSpeedupVsFp64.toFixed(6)}x speedupWall=${report.fp32WallSpeedupVsFp64.toFixed(6)}x`,',
  '  `fp32 statusMismatchVsFp64=${result.fp32StatusMismatchesVsFp64} maxAbsCutErrorVsFp64=${result.fp32MaxAbsCutErrorVsFp64}`,',
  '  `strictReferenceLoaded=${result.strictReferenceLoaded} fp64StatusMismatchVsStrict=${result.fp64StatusMismatchesVsStrict} fp64MaxAbsCutErrorVsStrict=${result.fp64MaxAbsCutErrorVsStrict}`,',
  '  `fp32StatusMismatchVsStrict=${result.fp32StatusMismatchesVsStrict} fp32MaxAbsCutErrorVsStrict=${result.fp32MaxAbsCutErrorVsStrict}`,',
  '  `report=${reportPath}`,',
  '  "",',
  '].join("\\n");',
  'fs.appendFileSync(llmPath, line, "utf8");',
  '',
  'console.log(JSON.stringify(report, null, 2));',
  'console.log(`report=${reportPath}`);',
  'console.log(`llm=${llmPath}`);',
  ''
].join("\n");

const SWEEP = String.raw`# rust/sunopt-node/scripts/run-blackwell-b14m-fp32-fastmath.ps1
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$NativeRoot = Join-Path $ProjectRoot "rust\sunopt-node"
$Runner = Join-Path $NativeRoot "scripts\blackwell-b14m-precision.mjs"
$Outbox = Join-Path $ProjectRoot "updates\outbox"
$Reference = Join-Path $Outbox "blackwell-b14m-strict-reference.bin"
$SweepReport = Join-Path $Outbox "blackwell-b14m-serial-sweep.txt"

New-Item -ItemType Directory -Force -Path $Outbox | Out-Null

function Invoke-Build([string]$FastMath, [string]$Label) {
    $env:SUNOPT_CUDA_FAST_MATH = $FastMath
    Push-Location $NativeRoot
    try {
        & npm run build 2>&1 |
            Tee-Object -FilePath (Join-Path $Outbox "b14m-build-$Label.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "B14M $Label build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-Run([string]$Mode, [bool]$WriteReference) {
    Push-Location $ProjectRoot
    try {
        $write = if ($WriteReference) { "true" } else { "false" }
        & node $Runner `
            --mode $Mode `
            --reference $Reference `
            --write-reference $write 2>&1 |
            Tee-Object -FilePath (Join-Path $Outbox "b14m-run-$Mode.txt")
        if ($LASTEXITCODE -ne 0) {
            throw "B14M $Mode run failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$oldFastMath = $env:SUNOPT_CUDA_FAST_MATH

try {
    "=== B14M SERIAL FP32 / FASTMATH SWEEP ===" |
        Set-Content -Encoding UTF8 $SweepReport

    "STRICT BUILD" |
        Add-Content -Encoding UTF8 $SweepReport

    Invoke-Build "0" "strict"
    Invoke-Run "strict" $true

    "FASTMATH BUILD" |
        Add-Content -Encoding UTF8 $SweepReport

    Invoke-Build "1" "fastmath"
    Invoke-Run "fastmath" $false

    "RESTORE STRICT BUILD" |
        Add-Content -Encoding UTF8 $SweepReport

    Invoke-Build "0" "restore-strict"

    "B14M_SERIAL_SWEEP_RESULT=OK" |
        Add-Content -Encoding UTF8 $SweepReport

    Write-Host ""
    Write-Host "B14M_SERIAL_SWEEP_RESULT=OK"
    Write-Host "strict report:  updates\outbox\blackwell-b14m-strict.json"
    Write-Host "fastmath report: updates\outbox\blackwell-b14m-fastmath.json"
    Write-Host "llm:            updates\outbox\llm.txt"
}
finally {
    if ($null -eq $oldFastMath) {
        Remove-Item Env:SUNOPT_CUDA_FAST_MATH -ErrorAction SilentlyContinue
    }
    else {
        $env:SUNOPT_CUDA_FAST_MATH = $oldFastMath
    }
}
`;

let build = fs.readFileSync(buildPath, "utf8");
let h = fs.readFileSync(hPath, "utf8");
let cu = fs.readFileSync(cuPath, "utf8");
let rs = fs.readFileSync(rsPath, "utf8");

build = patchBuildRs(build);

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
  [BUILD_REL, Buffer.from(build, "utf8")],
  [H_REL, Buffer.from(h, "utf8")],
  [CU_REL, Buffer.from(cu, "utf8")],
  [RS_REL, Buffer.from(rs, "utf8")],
  [RUNNER_REL, Buffer.from(RUNNER, "utf8")],
  [SWEEP_REL, Buffer.from(SWEEP, "utf8")],
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

const packagePath = path.join(
  inbox,
  "20261112-blackwell-b14m-fp32-fastmath.zip",
);

fs.writeFileSync(packagePath, zip);

console.log(`parent=${parent}`);
console.log(`release=${release}`);
console.log(`package=${packagePath}`);
console.log("B14M_PACKAGE_RESULT=OK");
