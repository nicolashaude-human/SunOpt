// rust/sunopt-node/scripts/make-blackwell-b14j-small-batch.mjs
//
// B14J small-batch occupancy sweep.
// Run from SunOpt root:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14j-small-batch.mjs
//
// This generator reads the CURRENT source tree and CURRENT watcher hash,
// then creates a normal format-2 watcher package in updates/inbox.
// It does not modify the sources directly.
//
// After generation:
//   python .\tools\update_watcher.py --once
//   cd .\rust\sunopt-node
//   npm run build
//   cd ..\..
//   node .\rust\sunopt-node\scripts\blackwell-b14j-small-batch.mjs
//
// Benchmark report:
//   updates/outbox/blackwell-b14j-small-batch.json

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");
const inbox = path.join(root, "updates", "inbox");
const hashFile = path.join(root, "updates", "history", "hash.txt");

const hPath = path.join(root, "rust", "sunopt-node", "cuda", "cuda_context.h");
const cuPath = path.join(root, "rust", "sunopt-node", "cuda", "cuda_context.cu");
const rsPath = path.join(root, "rust", "sunopt-node", "src", "cuda_context.rs");

const H_REL = "rust/sunopt-node/cuda/cuda_context.h";
const CU_REL = "rust/sunopt-node/cuda/cuda_context.cu";
const RS_REL = "rust/sunopt-node/src/cuda_context.rs";
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14j-small-batch.mjs";

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function insertBefore(text, anchor, block, label) {
  const marker = block.split(/\r?\n/, 1)[0];
  if (text.includes(marker)) {
    throw new Error(`${label}: B14J block already exists`);
  }
  const pos = text.indexOf(anchor);
  if (pos < 0) {
    throw new Error(`${label}: anchor not found: ${anchor}`);
  }
  return text.slice(0, pos) + block.trimEnd() + "\n\n" + text.slice(pos);
}

const parent = fs.existsSync(hashFile)
  ? fs.readFileSync(hashFile, "utf8").trim().toLowerCase()
  : "0".repeat(64);

if (!/^[0-9a-f]{64}$/.test(parent)) {
  throw new Error(`Invalid current watcher hash: ${parent}`);
}

const H_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14j_small_batch_benchmark(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    const double* host_energy_cut_levels,
    size_t step_count,
    uint32_t block_threads,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_dynamic_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_END
`.trim();

const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_BEGIN
namespace {

template <unsigned int BLOCK_THREADS>
__device__ __forceinline__ bool
sunopt_b14j_fixed_survives_block(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    double cut,
    double current_storage_energy,
    double reserve,
    double target,
    double max_hp,
    double hp_factor,
    double retention,
    double loss_intercept,
    SunoptB14FTransform* shared_transform,
    int* shared_survives
) {
    const unsigned int tid = static_cast<unsigned int>(threadIdx.x);
    const size_t future_count = horizon > 1 ? horizon - 1 : 0;
    const size_t chunk_size =
        (future_count + static_cast<size_t>(BLOCK_THREADS) - 1u) /
        static_cast<size_t>(BLOCK_THREADS);

    const size_t chunk_begin =
        1u + static_cast<size_t>(tid) * chunk_size;

    const size_t chunk_end =
        chunk_size == 0
            ? 1u
            : ((chunk_begin + chunk_size) < horizon
                ? (chunk_begin + chunk_size)
                : horizon);

    SunoptB14FTransform local =
        sunopt_b14f_identity();

    const double inverse_retention =
        1.0 / retention;

    for (
        size_t offset = chunk_begin;
        offset < chunk_end && offset < horizon;
        ++offset
    ) {
        const size_t index =
            (current_index + offset) % data_length;

        const double demand =
            fmax(demand_energy[index], 0.0);

        const double raw =
            export_power[index] - cut;

        const double hp =
            fmin(max_hp, fmax(0.0, raw));

        const double charge =
            hp * hp_factor;

        SunoptB14FTransform step{};
        step.a = inverse_retention;
        step.b =
            (loss_intercept + demand - charge) *
            inverse_retention;
        step.c =
            (
                loss_intercept +
                demand +
                reserve -
                SUNOPT_B14C_EPSILON
            ) *
            inverse_retention;
        step.max_input = target;

        local =
            sunopt_b14f_compose(
                local,
                step
            );
    }

    shared_transform[tid] = local;
    __syncthreads();

    for (
        unsigned int stride = 1u;
        stride < BLOCK_THREADS;
        stride <<= 1u
    ) {
        const unsigned int group =
            stride << 1u;

        if (
            (tid & (group - 1u)) == 0u &&
            tid + stride < BLOCK_THREADS
        ) {
            shared_transform[tid] =
                sunopt_b14f_compose(
                    shared_transform[tid],
                    shared_transform[tid + stride]
                );
        }

        __syncthreads();
    }

    if (tid == 0u) {
        const SunoptB14FTransform summary =
            shared_transform[0];

        bool feasible =
            0.0 <= summary.max_input + 1.0e-12;

        const double required =
            fmax(summary.c, summary.b);

        if (required > target + 1.0e-9) {
            feasible = false;
        }

        const double raw0 =
            export_power[current_index % data_length] -
            cut;

        const double hp0 =
            fmin(max_hp, fmax(0.0, raw0));

        const double charge0 =
            hp0 * hp_factor;

        const double current =
            sunopt_b14c_clamp(
                current_storage_energy,
                0.0,
                target
            );

        const double available =
            fmin(
                target,
                current + charge0
            );

        *shared_survives =
            feasible &&
            available + 1.0e-12 >= required
                ? 1
                : 0;
    }

    __syncthreads();
    return *shared_survives != 0;
}

template <unsigned int BLOCK_THREADS>
__global__ void
sunopt_b14j_small_batch_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    uint32_t current_index_u32,
    uint32_t horizon_u32,
    const double* __restrict__ dynamic_values,
    const double* __restrict__ energy_cut_levels,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status
) {
    const size_t tank =
        static_cast<size_t>(blockIdx.x);

    if (tank >= tank_count) {
        return;
    }

    const unsigned int tid =
        static_cast<unsigned int>(threadIdx.x);

    __shared__ SunoptB14FTransform
        shared_transform[BLOCK_THREADS];

    __shared__ int shared_survives;
    __shared__ double shared_lower;
    __shared__ double shared_upper;

    const size_t tb = tank * 6u;
    const size_t db = tank * 2u;

    const double target =
        tank_parameters[tb + 0];

    const double max_hp =
        tank_parameters[tb + 1];

    const double hp_factor =
        tank_parameters[tb + 2];

    const double retention =
        tank_parameters[tb + 3];

    const double loss_intercept =
        tank_parameters[tb + 4];

    const bool affine_supported =
        tank_parameters[tb + 5] > 0.5;

    const double current =
        dynamic_values[db + 0];

    const double reserve =
        sunopt_b14c_clamp(
            dynamic_values[db + 1],
            0.0,
            target
        );

    const size_t current_index =
        static_cast<size_t>(current_index_u32);

    const size_t horizon =
        static_cast<size_t>(horizon_u32);

    const double energy_cut =
        energy_cut_levels[tank];

    if (tid == 0u) {
        cut_results[tank] = energy_cut;
        status[tank] = 3u;
    }

    __syncthreads();

    const bool supported =
        affine_supported &&
        retention > 0.0 &&
        reserve > SUNOPT_B14C_EPSILON &&
        data_length > 0 &&
        horizon > 0 &&
        horizon <= data_length;

    if (!supported) {
        return;
    }

    const bool normal_survives =
        sunopt_b14j_fixed_survives_block<BLOCK_THREADS>(
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
            shared_transform,
            &shared_survives
        );

    if (normal_survives) {
        if (tid == 0u) {
            status[tank] = 0u;
        }
        return;
    }

    const bool maximum_pv_survives =
        sunopt_b14j_fixed_survives_block<BLOCK_THREADS>(
            export_power,
            demand_energy,
            data_length,
            current_index,
            horizon,
            0.0,
            current,
            reserve,
            target,
            max_hp,
            hp_factor,
            retention,
            loss_intercept,
            shared_transform,
            &shared_survives
        );

    if (!maximum_pv_survives) {
        if (tid == 0u) {
            cut_results[tank] = 0.0;
            status[tank] = 1u;
        }
        return;
    }

    if (tid == 0u) {
        shared_lower = 0.0;
        shared_upper = energy_cut;
    }

    __syncthreads();

    for (
        int iteration = 0;
        iteration < 20;
        ++iteration
    ) {
        const double middle =
            (shared_lower + shared_upper) * 0.5;

        const bool survives =
            sunopt_b14j_fixed_survives_block<BLOCK_THREADS>(
                export_power,
                demand_energy,
                data_length,
                current_index,
                horizon,
                middle,
                current,
                reserve,
                target,
                max_hp,
                hp_factor,
                retention,
                loss_intercept,
                shared_transform,
                &shared_survives
            );

        if (tid == 0u) {
            if (survives) {
                shared_lower = middle;
            } else {
                shared_upper = middle;
            }
        }

        __syncthreads();
    }

    if (tid == 0u) {
        cut_results[tank] = shared_lower;
        status[tank] = 2u;
    }
}

template <unsigned int BLOCK_THREADS>
cudaError_t sunopt_b14j_launch(
    cudaStream_t stream,
    const double* d_export,
    const double* d_demand,
    size_t data_length,
    const double* d_tanks,
    size_t tank_count,
    uint32_t current_index,
    uint32_t horizon,
    const double* d_dynamic,
    const double* d_energy_cut,
    double* d_cut_results,
    uint32_t* d_status
) {
    sunopt_b14j_small_batch_kernel<BLOCK_THREADS>
        <<<static_cast<unsigned int>(tank_count),
           BLOCK_THREADS,
           0,
           stream>>>(
            d_export,
            d_demand,
            data_length,
            d_tanks,
            tank_count,
            current_index,
            horizon,
            d_dynamic,
            d_energy_cut,
            d_cut_results,
            d_status
        );

    return cudaGetLastError();
}

} // namespace

extern "C" int
sunopt_cuda_run_b14j_small_batch_benchmark(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    const double* host_energy_cut_levels,
    size_t step_count,
    uint32_t block_threads,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_dynamic_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

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
        !out_dynamic_enqueue_ms ||
        !out_kernel_ms ||
        !out_result_wait_ms ||
        !out_wall_ms ||
        data_length == 0 ||
        tank_count == 0 ||
        step_count == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    if (
        block_threads != 32u &&
        block_threads != 64u &&
        block_threads != 128u &&
        block_threads != 256u &&
        block_threads != 512u
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err =
        cudaSetDevice(device_id);

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_dynamic = nullptr;
    double* d_energy_cut = nullptr;
    double* d_cut_results = nullptr;
    uint32_t* d_status = nullptr;

    const size_t annual_bytes =
        data_length * sizeof(double);

    const size_t tank_bytes =
        tank_count * 6u * sizeof(double);

    const size_t dynamic_step_bytes =
        tank_count * 2u * sizeof(double);

    const size_t cut_step_bytes =
        tank_count * sizeof(double);

    const size_t status_step_bytes =
        tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_status) cudaFree(d_status);
        if (d_cut_results) cudaFree(d_cut_results);
        if (d_energy_cut) cudaFree(d_energy_cut);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);

        if (kernel_stop) {
            cudaEventDestroy(kernel_stop);
        }

        if (kernel_start) {
            cudaEventDestroy(kernel_start);
        }

        if (stream) {
            cudaStreamDestroy(stream);
        }
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

#define B14J_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14J_MALLOC(d_export, annual_bytes);
    B14J_MALLOC(d_demand, annual_bytes);
    B14J_MALLOC(d_tanks, tank_bytes);
    B14J_MALLOC(d_dynamic, dynamic_step_bytes);
    B14J_MALLOC(d_energy_cut, cut_step_bytes);
    B14J_MALLOC(d_cut_results, cut_step_bytes);
    B14J_MALLOC(d_status, status_step_bytes);

#undef B14J_MALLOC

    const auto wall_start = Clock::now();
    const auto static_start = Clock::now();

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

    double dynamic_enqueue_ms = 0.0;
    double kernel_ms_total = 0.0;
    double result_wait_ms = 0.0;

    for (
        size_t step = 0;
        step < step_count;
        ++step
    ) {
        const auto enqueue_start =
            Clock::now();

        err = cudaMemcpyAsync(
            d_dynamic,
            host_dynamic_values +
                step * tank_count * 2u,
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
                step * tank_count,
            cut_step_bytes,
            cudaMemcpyHostToDevice,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        dynamic_enqueue_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - enqueue_start
            ).count();

        err =
            cudaEventRecord(
                kernel_start,
                stream
            );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        switch (block_threads) {
            case 32u:
                err = sunopt_b14j_launch<32u>(
                    stream,
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
                break;

            case 64u:
                err = sunopt_b14j_launch<64u>(
                    stream,
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
                break;

            case 128u:
                err = sunopt_b14j_launch<128u>(
                    stream,
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
                break;

            case 256u:
                err = sunopt_b14j_launch<256u>(
                    stream,
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
                break;

            case 512u:
                err = sunopt_b14j_launch<512u>(
                    stream,
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
                break;

            default:
                err = cudaErrorInvalidValue;
                break;
        }

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err =
            cudaEventRecord(
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
                step * tank_count,
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
                step * tank_count,
            d_status,
            status_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err =
            cudaStreamSynchronize(stream);

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        result_wait_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - wait_start
            ).count();

        float step_kernel_ms = 0.0f;

        err =
            cudaEventElapsedTime(
                &step_kernel_ms,
                kernel_start,
                kernel_stop
            );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        kernel_ms_total +=
            static_cast<double>(
                step_kernel_ms
            );
    }

    *out_dynamic_enqueue_ms =
        dynamic_enqueue_ms;

    *out_kernel_ms =
        kernel_ms_total;

    *out_result_wait_ms =
        result_wait_ms;

    *out_wall_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - wall_start
        ).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_END
`.trim();

const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14J_SMALL_BATCH_FFI_BEGIN
    fn sunopt_cuda_run_b14j_small_batch_benchmark(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        host_energy_cut_levels: *const f64,
        step_count: usize,
        block_threads: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_dynamic_enqueue_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14J_SMALL_BATCH_FFI_END
`.trim();

const RS_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_FN_BEGIN
#[napi(object)]
pub struct BlackwellB14jVariantResult {
    pub block_threads: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub static_h2d_ms: f64,
    pub dynamic_enqueue_ms: f64,
    pub kernel_ms: f64,
    pub result_wait_ms: f64,
    pub wall_ms: f64,
    pub checksum_hex: String,
    pub status0: u32,
    pub status1: u32,
    pub status2: u32,
    pub status3: u32,
}

#[napi]
pub fn run_blackwell_b14j_small_batch(
    snapshot_path: Option<String>,
    block_threads: u32,
    device_id: Option<i32>,
) -> Result<BlackwellB14jVariantResult> {
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

    if !matches!(
        block_threads,
        32 | 64 | 128 | 256 | 512
    ) {
        return Err(
            Error::from_reason(
                "B14J block_threads must be 32, 64, 128, 256, or 512.",
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

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 6,
        );

    for tank in &tank_parameters {
        packed_tanks.extend_from_slice(&[
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
                    "B14J job count overflow.",
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

    let mut dynamic_values =
        Vec::<f64>::with_capacity(
            jobs * 2
        );

    let mut energy_cut_levels =
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

        for tank_index in 0..snapshot.tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let tank =
                tank_parameters[tank_index];

            let current =
                step.dynamic[base + 0];

            let reserve =
                step.dynamic[base + 1];

            let requested =
                step.dynamic[base + 2];

            dynamic_values.push(
                current
            );

            dynamic_values.push(
                reserve
            );

            energy_cut_levels.push(
                research_energy_cut_level(
                    &export_window,
                    requested,
                    tank,
                ),
            );
        }
    }

    let mut cuts =
        vec![
            0.0_f64;
            jobs
        ];

    let mut status =
        vec![
            0_u32;
            jobs
        ];

    let mut static_h2d_ms =
        0.0_f64;

    let mut dynamic_enqueue_ms =
        0.0_f64;

    let mut kernel_ms =
        0.0_f64;

    let mut result_wait_ms =
        0.0_f64;

    let mut wall_ms =
        0.0_f64;

    let selected_device =
        device_id.unwrap_or(0);

    let code = unsafe {
        sunopt_cuda_run_b14j_small_batch_benchmark(
            selected_device as c_int,
            snapshot.export_power.as_ptr(),
            snapshot.demand_energy.as_ptr(),
            snapshot.data_length,
            packed_tanks.as_ptr(),
            snapshot.tank_count,
            current_indices.as_ptr(),
            horizons.as_ptr(),
            dynamic_values.as_ptr(),
            energy_cut_levels.as_ptr(),
            step_count,
            block_threads,
            cuts.as_mut_ptr(),
            status.as_mut_ptr(),
            &mut static_h2d_ms,
            &mut dynamic_enqueue_ms,
            &mut kernel_ms,
            &mut result_wait_ms,
            &mut wall_ms,
        )
    };

    if code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_b14j_small_batch_benchmark",
                code,
            ),
        );
    }

    let mut checksum =
        1469598103934665603_u64;

    for value in &cuts {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    let mut status0 = 0_u32;
    let mut status1 = 0_u32;
    let mut status2 = 0_u32;
    let mut status3 = 0_u32;

    for value in status {
        match value {
            0 => status0 += 1,
            1 => status1 += 1,
            2 => status2 += 1,
            _ => status3 += 1,
        }
    }

    Ok(
        BlackwellB14jVariantResult {
            block_threads,
            annual_step_count:
                step_count as u32,
            tank_count:
                snapshot.tank_count as u32,
            causal_jobs:
                jobs as u32,
            static_h2d_ms,
            dynamic_enqueue_ms,
            kernel_ms,
            result_wait_ms,
            wall_ms,
            checksum_hex:
                format!(
                    "{checksum:016x}"
                ),
            status0,
            status1,
            status2,
            status3,
        },
    )
}
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_FN_END
`.trim();

const RUNNER = [
  '// rust/sunopt-node/scripts/blackwell-b14j-small-batch.mjs',
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
  'fs.mkdirSync(outbox, { recursive: true });',
  '',
  'const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));',
  'const run = addon.runBlackwellB14JSmallBatch;',
  '',
  'if (typeof run !== "function") {',
  '  throw new Error("runBlackwellB14JSmallBatch native export missing");',
  '}',
  '',
  'const snapshot = process.argv[2]',
  '  ? path.resolve(process.argv[2])',
  '  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
  '',
  'const threadCounts = [32, 64, 128, 256, 512];',
  'const expectedChecksum = "bf0e8c9e7b5c879e";',
  'const variants = [];',
  '',
  'console.log("--- B14J SMALL-BATCH OCCUPANCY SWEEP ---");',
  'console.log(`snapshot=${snapshot}`);',
  '',
  'for (const blockThreads of threadCounts) {',
  '  console.log(`running blockThreads=${blockThreads} ...`);',
  '  const r = run(snapshot, blockThreads, 0);',
  '  const exact = r.checksumHex === expectedChecksum;',
  '  const row = {',
  '    ...r,',
  '    exactVsB14H: exact,',
  '    speedupVsB14I1Wall: 5256.6251 / r.wallMs,',
  '    speedupVsB14I1Kernel: 3987.9596444517374 / r.kernelMs,',
  '  };',
  '  variants.push(row);',
  '  console.log(',
  '    `threads=${blockThreads} kernel=${r.kernelMs.toFixed(3)} ms wall=${r.wallMs.toFixed(3)} ms checksum=${r.checksumHex} exact=${exact}`',
  '  );',
  '}',
  '',
  'const exactVariants = variants.filter(v => v.exactVsB14H);',
  'const bestExact = exactVariants.length',
  '  ? exactVariants.reduce((a, b) => a.wallMs <= b.wallMs ? a : b)',
  '  : null;',
  '',
  'const report = {',
  '  stage: "B14J_SMALL_BATCH_OCCUPANCY",',
  '  snapshotPath: snapshot,',
  '  baselineB14I1: {',
  '    kernelMs: 3987.9596444517374,',
  '    wallMs: 5256.6251,',
  '    checksumHex: expectedChecksum,',
  '  },',
  '  variants,',
  '  bestExact,',
  '};',
  '',
  'const reportPath = path.join(outbox, "blackwell-b14j-small-batch.json");',
  'fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\\n");',
  '',
  'console.log("");',
  'console.log(`report=${reportPath}`);',
  'if (bestExact) {',
  '  console.log(`bestExactThreads=${bestExact.blockThreads}`);',
  '  console.log(`bestExactWallMs=${bestExact.wallMs.toFixed(3)}`);',
  '  console.log(`bestExactSpeedup=${bestExact.speedupVsB14I1Wall.toFixed(4)}x`);',
  '  console.log("B14J_RESULT=EXACT_VARIANT_FOUND");',
  '} else {',
  '  console.log("B14J_RESULT=NO_BITWISE_EXACT_VARIANT");',
  '}',
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
  [RUNNER_REL, Buffer.from(RUNNER, "utf8")],
]);

const manifest = {
  format: 2,
  parent_sha256: parent,
  files: Object.fromEntries(
    [...files.entries()].map(
      ([name, data]) => [name, sha256(data)],
    ),
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
      c =
        (c >>> 1) ^
        (0xedb88320 & -(c & 1));
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

  const central = Buffer.concat([
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

  centrals.push(central);
  offset += local.length;
}

const centralOffset = offset;
const central = Buffer.concat(centrals);

const end = Buffer.concat([
  u32(0x06054b50),
  u16(0),
  u16(0),
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

const packagePath =
  path.join(
    inbox,
    "20261112-blackwell-b14j-small-batch.zip",
  );

fs.writeFileSync(packagePath, zip);

console.log(`parent=${parent}`);
console.log(`release=${release}`);
console.log(`package=${packagePath}`);
console.log("B14J_PACKAGE_RESULT=OK");
