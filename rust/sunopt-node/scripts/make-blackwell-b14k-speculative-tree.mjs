// rust/sunopt-node/scripts/make-blackwell-b14k-speculative-tree.mjs
//
// Generates a watcher-compatible B14K package from the CURRENT SunOpt tree.
// B14K evaluates complete speculative binary-search subtrees in parallel.
// Depths 1..5 use 1,3,7,15,31 warps per tank respectively while preserving
// exactly the same 20 sequential dyadic decisions as B14G.
//
// Run from SunOpt root:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14k-speculative-tree.mjs
//
// Then:
//   python .\tools\update_watcher.py --once
//   cd .\rust\sunopt-node
//   npm run build
//   cd ..\..
//   node .\rust\sunopt-node\scripts\blackwell-b14k-speculative-tree.mjs
//
// Output:
//   updates/outbox/blackwell-b14k-speculative-tree.json

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");
const inbox = path.join(root, "updates", "inbox");
const hashFile = path.join(root, "updates", "history", "hash.txt");

const H_REL = "rust/sunopt-node/cuda/cuda_context.h";
const CU_REL = "rust/sunopt-node/cuda/cuda_context.cu";
const RS_REL = "rust/sunopt-node/src/cuda_context.rs";
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14k-speculative-tree.mjs";

const hPath = path.join(root, H_REL);
const cuPath = path.join(root, CU_REL);
const rsPath = path.join(root, RS_REL);

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function insertBefore(text, anchor, block, label) {
  const marker = block.split(/\r?\n/, 1)[0];
  if (text.includes(marker)) {
    throw new Error(`${label}: B14K block already exists`);
  }
  const pos = text.indexOf(anchor);
  if (pos < 0) {
    throw new Error(`${label}: anchor not found: ${anchor}`);
  }
  return text.slice(0, pos) + block.trimEnd() + "\n\n" + text.slice(pos);
}

if (!fs.existsSync(hashFile)) {
  throw new Error(`Current watcher hash not found: ${hashFile}`);
}

const parent = fs.readFileSync(hashFile, "utf8").trim().toLowerCase();
if (!/^[0-9a-f]{64}$/.test(parent)) {
  throw new Error(`Invalid current watcher hash: ${parent}`);
}

const H_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14k_speculative_tree_benchmark(
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
    uint32_t tree_depth,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_dynamic_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_END
`.trim();

const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_BEGIN
namespace {

template <unsigned int TREE_DEPTH>
__global__ void sunopt_b14k_speculative_tree_kernel(
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
    constexpr unsigned int MAX_WARPS =
        (1u << TREE_DEPTH) - 1u;

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

    __shared__ int shared_survives[MAX_WARPS];
    __shared__ int shared_precheck;
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

    if (warp == 0u) {
        const bool normal_survives =
            sunopt_b14f_fixed_survives_warp(
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
                normal_survives ? 1 : 0;
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
        const bool maximum_pv_survives =
            sunopt_b14f_fixed_survives_warp(
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
                lane
            );

        if (lane == 0) {
            shared_precheck =
                maximum_pv_survives ? 1 : 0;
        }
    }

    __syncthreads();

    if (shared_precheck == 0) {
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

    unsigned int completed_bits = 0u;

    while (completed_bits < 20u) {
        const unsigned int remaining =
            20u - completed_bits;

        const unsigned int round_depth =
            remaining < TREE_DEPTH
                ? remaining
                : TREE_DEPTH;

        const unsigned int node_count =
            (1u << round_depth) - 1u;

        if (warp < node_count) {
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

            double lo = shared_lower;
            double hi = shared_upper;

            if (level > 0u) {
                for (
                    unsigned int path_level = 0u;
                    path_level < level;
                    ++path_level
                ) {
                    const double mid =
                        (lo + hi) * 0.5;

                    const unsigned int shift =
                        level - 1u - path_level;

                    const unsigned int direction =
                        (position >> shift) & 1u;

                    if (direction != 0u) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
            }

            const double candidate =
                (lo + hi) * 0.5;

            const bool survives =
                sunopt_b14f_fixed_survives_warp(
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
            double lo = shared_lower;
            double hi = shared_upper;
            unsigned int node = 0u;

            for (
                unsigned int level = 0u;
                level < round_depth;
                ++level
            ) {
                const double mid =
                    (lo + hi) * 0.5;

                if (shared_survives[node] != 0) {
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

        completed_bits += round_depth;
        __syncthreads();
    }

    if (tid == 0u) {
        cut_results[tank] = shared_lower;
        status[tank] = 2u;
    }
}

template <unsigned int TREE_DEPTH>
cudaError_t sunopt_b14k_launch(
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
    constexpr unsigned int WARPS =
        (1u << TREE_DEPTH) - 1u;

    constexpr unsigned int THREADS =
        WARPS * 32u;

    sunopt_b14k_speculative_tree_kernel<TREE_DEPTH>
        <<<static_cast<unsigned int>(tank_count),
           THREADS,
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
sunopt_cuda_run_b14k_speculative_tree_benchmark(
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
    uint32_t tree_depth,
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
        step_count == 0 ||
        tree_depth == 0 ||
        tree_depth > 5
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

#define B14K_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14K_MALLOC(d_export, annual_bytes);
    B14K_MALLOC(d_demand, annual_bytes);
    B14K_MALLOC(d_tanks, tank_bytes);
    B14K_MALLOC(d_dynamic, dynamic_step_bytes);
    B14K_MALLOC(d_energy_cut, cut_step_bytes);
    B14K_MALLOC(d_cut_results, cut_step_bytes);
    B14K_MALLOC(d_status, status_step_bytes);

#undef B14K_MALLOC

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

        switch (tree_depth) {
            case 1u:
                err = sunopt_b14k_launch<1u>(
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

            case 2u:
                err = sunopt_b14k_launch<2u>(
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

            case 3u:
                err = sunopt_b14k_launch<3u>(
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

            case 4u:
                err = sunopt_b14k_launch<4u>(
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

            case 5u:
                err = sunopt_b14k_launch<5u>(
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

        err = cudaStreamSynchronize(stream);
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
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_END
`.trim();

const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_FFI_BEGIN
    fn sunopt_cuda_run_b14k_speculative_tree_benchmark(
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
        tree_depth: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_dynamic_enqueue_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_FFI_END
`.trim();

const RS_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_FN_BEGIN
#[napi(object)]
pub struct BlackwellB14kVariantResult {
    pub tree_depth: u32,
    pub warps_per_tank: u32,
    pub threads_per_tank: u32,
    pub serial_rounds: u32,
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
pub fn run_blackwell_b14k_speculative_tree(
    snapshot_path: Option<String>,
    tree_depth: u32,
    device_id: Option<i32>,
) -> Result<BlackwellB14kVariantResult> {
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

    if !(1..=5).contains(&tree_depth) {
        return Err(
            Error::from_reason(
                "B14K tree_depth must be 1..=5.",
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
                    "B14K job count overflow.",
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

            dynamic_values.push(current);
            dynamic_values.push(reserve);

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
        vec![0.0_f64; jobs];

    let mut status =
        vec![0_u32; jobs];

    let mut static_h2d_ms = 0.0_f64;
    let mut dynamic_enqueue_ms = 0.0_f64;
    let mut kernel_ms = 0.0_f64;
    let mut result_wait_ms = 0.0_f64;
    let mut wall_ms = 0.0_f64;

    let selected_device =
        device_id.unwrap_or(0);

    let code = unsafe {
        sunopt_cuda_run_b14k_speculative_tree_benchmark(
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
            tree_depth,
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
                "sunopt_cuda_run_b14k_speculative_tree_benchmark",
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

    let warps_per_tank =
        (1_u32 << tree_depth) - 1;

    let serial_rounds =
        (20 + tree_depth - 1) /
        tree_depth;

    Ok(
        BlackwellB14kVariantResult {
            tree_depth,
            warps_per_tank,
            threads_per_tank:
                warps_per_tank * 32,
            serial_rounds,
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
                format!("{checksum:016x}"),
            status0,
            status1,
            status2,
            status3,
        },
    )
}
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_FN_END
`.trim();

const RUNNER = [
  '// rust/sunopt-node/scripts/blackwell-b14k-speculative-tree.mjs',
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
  'const run = addon.runBlackwellB14KSpeculativeTree;',
  '',
  'if (typeof run !== "function") {',
  '  throw new Error("runBlackwellB14KSpeculativeTree native export missing");',
  '}',
  '',
  'const snapshot = process.argv[2]',
  '  ? path.resolve(process.argv[2])',
  '  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
  '',
  'const expectedChecksum = "bf0e8c9e7b5c879e";',
  'const variants = [];',
  '',
  'console.log("--- B14K SPECULATIVE BINARY-TREE SWEEP ---");',
  'console.log(`snapshot=${snapshot}`);',
  '',
  'for (const treeDepth of [1, 2, 3, 4, 5]) {',
  '  console.log(`running treeDepth=${treeDepth} ...`);',
  '  const r = run(snapshot, treeDepth, 0);',
  '  const exact = r.checksumHex === expectedChecksum;',
  '  const row = {',
  '    ...r,',
  '    exactVsB14H: exact,',
  '    speedupVsB14I1Wall: 5256.6251 / r.wallMs,',
  '    speedupVsB14I1Kernel: 3987.9596444517374 / r.kernelMs,',
  '    speedupVsB14J64Wall: 5182.2679 / r.wallMs,',
  '    speedupVsB14J64Kernel: 3376.2135968068615 / r.kernelMs,',
  '  };',
  '  variants.push(row);',
  '  console.log(',
  '    `depth=${treeDepth} warps=${r.warpsPerTank} threads=${r.threadsPerTank} rounds=${r.serialRounds} kernel=${r.kernelMs.toFixed(3)} ms wall=${r.wallMs.toFixed(3)} ms checksum=${r.checksumHex} exact=${exact}`',
  '  );',
  '}',
  '',
  'const exactVariants = variants.filter(v => v.exactVsB14H);',
  'const bestExact = exactVariants.length',
  '  ? exactVariants.reduce((a, b) => a.wallMs <= b.wallMs ? a : b)',
  '  : null;',
  '',
  'const report = {',
  '  stage: "B14K_SPECULATIVE_BINARY_TREE",',
  '  snapshotPath: snapshot,',
  '  expectedChecksum,',
  '  baselineB14I1: { kernelMs: 3987.9596444517374, wallMs: 5256.6251 },',
  '  baselineB14J64: { kernelMs: 3376.2135968068615, wallMs: 5182.2679 },',
  '  variants,',
  '  bestExact,',
  '};',
  '',
  'const reportPath = path.join(outbox, "blackwell-b14k-speculative-tree.json");',
  'fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\\n");',
  '',
  'console.log("");',
  'console.log(`report=${reportPath}`);',
  'if (bestExact) {',
  '  console.log(`bestExactDepth=${bestExact.treeDepth}`);',
  '  console.log(`bestExactWallMs=${bestExact.wallMs.toFixed(3)}`);',
  '  console.log(`bestExactSpeedupVsB14I1=${bestExact.speedupVsB14I1Wall.toFixed(4)}x`);',
  '  console.log("B14K_RESULT=EXACT_VARIANT_FOUND");',
  '} else {',
  '  console.log("B14K_RESULT=NO_BITWISE_EXACT_VARIANT");',
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

const packagePath = path.join(
  inbox,
  "20261112-blackwell-b14k-speculative-tree.zip",
);

fs.writeFileSync(packagePath, zip);

console.log(`parent=${parent}`);
console.log(`release=${release}`);
console.log(`package=${packagePath}`);
console.log("B14K_PACKAGE_RESULT=OK");
