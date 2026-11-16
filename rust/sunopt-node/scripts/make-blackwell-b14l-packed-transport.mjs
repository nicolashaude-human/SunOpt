// rust/sunopt-node/scripts/make-blackwell-b14l-packed-transport.mjs
//
// B14L: keep exact B14K depth-2 speculative search, but reduce host/device
// transaction overhead by packing each lockstep input/output into one transfer.
// Compares pageable vs pinned host staging.
//
// Run from SunOpt root:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14l-packed-transport.mjs
//   python .\tools\update_watcher.py --once
//   cd .\rust\sunopt-node
//   npm run build
//   cd ..\..
//   node .\rust\sunopt-node\scripts\blackwell-b14l-packed-transport.mjs
//
// Reports:
//   updates/outbox/blackwell-b14l-packed-transport.json
//   updates/outbox/llm.txt

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
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14l-packed-transport.mjs";

const hPath = path.join(root, H_REL);
const cuPath = path.join(root, CU_REL);
const rsPath = path.join(root, RS_REL);

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function insertBefore(text, anchor, block, label) {
  const marker = block.split(/\r?\n/, 1)[0];
  if (text.includes(marker)) {
    throw new Error(`${label}: B14L block already exists`);
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
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14l_packed_transport_benchmark(
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
    uint32_t pinned_host_memory,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_host_pack_ms,
    double* out_input_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_END
`.trim();

const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14L_TREE_DEPTH = 2u;
constexpr unsigned int SUNOPT_B14L_WARPS = 3u;
constexpr unsigned int SUNOPT_B14L_THREADS = 96u;

__global__ void sunopt_b14l_packed_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const double* __restrict__ packed_input,
    double* __restrict__ packed_output
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

    __shared__ int shared_survives[SUNOPT_B14L_WARPS];
    __shared__ int shared_precheck;
    __shared__ double shared_lower;
    __shared__ double shared_upper;

    const size_t tb = tank * 6u;
    const size_t dynamic_base = 2u + tank * 2u;
    const size_t cut_base = 2u + tank_count * 2u;

    const size_t current_index =
        static_cast<size_t>(packed_input[0]);

    const size_t horizon =
        static_cast<size_t>(packed_input[1]);

    const double current =
        packed_input[dynamic_base + 0u];

    const double target =
        tank_parameters[tb + 0u];

    const double reserve =
        sunopt_b14c_clamp(
            packed_input[dynamic_base + 1u],
            0.0,
            target
        );

    const double energy_cut =
        packed_input[cut_base + tank];

    const double max_hp =
        tank_parameters[tb + 1u];

    const double hp_factor =
        tank_parameters[tb + 2u];

    const double retention =
        tank_parameters[tb + 3u];

    const double loss_intercept =
        tank_parameters[tb + 4u];

    const bool affine_supported =
        tank_parameters[tb + 5u] > 0.5;

    if (tid == 0u) {
        packed_output[tank] = energy_cut;
        packed_output[tank_count + tank] = 3.0;
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
        const bool survives =
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
            shared_precheck = survives ? 1 : 0;
        }
    }

    __syncthreads();

    if (shared_precheck != 0) {
        if (tid == 0u) {
            packed_output[tank_count + tank] = 0.0;
        }
        return;
    }

    if (warp == 0u) {
        const bool survives =
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
            shared_precheck = survives ? 1 : 0;
        }
    }

    __syncthreads();

    if (shared_precheck == 0) {
        if (tid == 0u) {
            packed_output[tank] = 0.0;
            packed_output[tank_count + tank] = 1.0;
        }
        return;
    }

    if (tid == 0u) {
        shared_lower = 0.0;
        shared_upper = energy_cut;
    }

    __syncthreads();

    for (unsigned int round = 0u; round < 10u; ++round) {
        if (warp < SUNOPT_B14L_WARPS) {
            const unsigned int node_number = warp + 1u;
            unsigned int level = 0u;
            unsigned int first_at_level = 1u;

            while ((first_at_level << 1u) <= node_number) {
                first_at_level <<= 1u;
                ++level;
            }

            const unsigned int position =
                node_number - first_at_level;

            double lo = shared_lower;
            double hi = shared_upper;

            if (level > 0u) {
                for (
                    unsigned int path_level = 0u;
                    path_level < level;
                    ++path_level
                ) {
                    const double mid = (lo + hi) * 0.5;
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

            const double candidate = (lo + hi) * 0.5;

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

            for (unsigned int level = 0u; level < 2u; ++level) {
                const double mid = (lo + hi) * 0.5;

                if (shared_survives[node] != 0) {
                    lo = mid;
                    node = node * 2u + 2u;
                } else {
                    hi = mid;
                    node = node * 2u + 1u;
                }
            }

            shared_lower = lo;
            shared_upper = hi;
        }

        __syncthreads();
    }

    if (tid == 0u) {
        packed_output[tank] = shared_lower;
        packed_output[tank_count + tank] = 2.0;
    }
}

} // namespace

extern "C" int sunopt_cuda_run_b14l_packed_transport_benchmark(
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
    uint32_t pinned_host_memory,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_host_pack_ms,
    double* out_input_enqueue_ms,
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
        !out_host_pack_ms ||
        !out_input_enqueue_ms ||
        !out_kernel_ms ||
        !out_result_wait_ms ||
        !out_wall_ms ||
        data_length == 0 ||
        tank_count == 0 ||
        step_count == 0 ||
        pinned_host_memory > 1u
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t input_count =
        2u + tank_count * 3u;
    const size_t output_count =
        tank_count * 2u;

    const size_t input_bytes =
        input_count * sizeof(double);
    const size_t output_bytes =
        output_count * sizeof(double);

    double* host_input = nullptr;
    double* host_output = nullptr;
    bool host_input_pinned = false;
    bool host_output_pinned = false;

    if (pinned_host_memory != 0u) {
        err = cudaHostAlloc(
            reinterpret_cast<void**>(&host_input),
            input_bytes,
            cudaHostAllocDefault
        );
        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
        host_input_pinned = true;

        err = cudaHostAlloc(
            reinterpret_cast<void**>(&host_output),
            output_bytes,
            cudaHostAllocDefault
        );
        if (err != cudaSuccess) {
            cudaFreeHost(host_input);
            return static_cast<int>(err);
        }
        host_output_pinned = true;
    } else {
        host_input =
            static_cast<double*>(malloc(input_bytes));
        host_output =
            static_cast<double*>(malloc(output_bytes));

        if (!host_input || !host_output) {
            if (host_input) free(host_input);
            if (host_output) free(host_output);
            return static_cast<int>(cudaErrorMemoryAllocation);
        }
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_input = nullptr;
    double* d_output = nullptr;

    auto cleanup = [&]() {
        if (d_output) cudaFree(d_output);
        if (d_input) cudaFree(d_input);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);

        if (kernel_stop) cudaEventDestroy(kernel_stop);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (stream) cudaStreamDestroy(stream);

        if (host_output_pinned) {
            cudaFreeHost(host_output);
        } else if (host_output) {
            free(host_output);
        }

        if (host_input_pinned) {
            cudaFreeHost(host_input);
        } else if (host_input) {
            free(host_input);
        }
    };

    err = cudaStreamCreateWithFlags(
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

#define B14L_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    const size_t annual_bytes =
        data_length * sizeof(double);
    const size_t tank_bytes =
        tank_count * 6u * sizeof(double);

    B14L_MALLOC(d_export, annual_bytes);
    B14L_MALLOC(d_demand, annual_bytes);
    B14L_MALLOC(d_tanks, tank_bytes);
    B14L_MALLOC(d_input, input_bytes);
    B14L_MALLOC(d_output, output_bytes);

#undef B14L_MALLOC

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

    double host_pack_ms = 0.0;
    double input_enqueue_ms = 0.0;
    double kernel_ms = 0.0;
    double result_wait_ms = 0.0;

    const size_t dynamic_offset = 2u;
    const size_t cut_offset =
        2u + tank_count * 2u;

    for (size_t step = 0; step < step_count; ++step) {
        const auto pack_start = Clock::now();

        host_input[0] =
            static_cast<double>(
                host_current_indices[step]
            );
        host_input[1] =
            static_cast<double>(
                host_horizon_steps[step]
            );

        const size_t dynamic_source =
            step * tank_count * 2u;
        const size_t cut_source =
            step * tank_count;

        for (size_t tank = 0; tank < tank_count; ++tank) {
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
                cut_offset + tank
            ] =
                host_energy_cut_levels[
                    cut_source + tank
                ];
        }

        host_pack_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - pack_start
            ).count();

        const auto input_start = Clock::now();

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

        sunopt_b14l_packed_kernel
            <<<static_cast<unsigned int>(tank_count),
               SUNOPT_B14L_THREADS,
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

        const auto wait_start = Clock::now();

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

        kernel_ms +=
            static_cast<double>(
                step_kernel_ms
            );

        const size_t output_base =
            step * tank_count;

        for (size_t tank = 0; tank < tank_count; ++tank) {
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

    *out_host_pack_ms = host_pack_ms;
    *out_input_enqueue_ms = input_enqueue_ms;
    *out_kernel_ms = kernel_ms;
    *out_result_wait_ms = result_wait_ms;
    *out_wall_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - wall_start
        ).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_END
`.trim();

const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_FFI_BEGIN
    fn sunopt_cuda_run_b14l_packed_transport_benchmark(
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
        pinned_host_memory: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_host_pack_ms: *mut f64,
        out_input_enqueue_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_FFI_END
`.trim();

const RS_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_FN_BEGIN
#[napi(object)]
pub struct BlackwellB14lVariantResult {
    pub pinned_host_memory: bool,
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
    pub status0: u32,
    pub status1: u32,
    pub status2: u32,
    pub status3: u32,
}

#[napi]
pub fn run_blackwell_b14l_packed_transport(
    snapshot_path: Option<String>,
    pinned_host_memory: bool,
    device_id: Option<i32>,
) -> Result<BlackwellB14lVariantResult> {
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
            if tank.affine_supported { 1.0 } else { 0.0 },
        ]);
    }

    let step_count = snapshot.steps.len();
    let jobs =
        step_count
            .checked_mul(snapshot.tank_count)
            .ok_or_else(|| {
                Error::from_reason(
                    "B14L job count overflow.",
                )
            })?;

    let mut current_indices =
        Vec::<u32>::with_capacity(step_count);
    let mut horizons =
        Vec::<u32>::with_capacity(step_count);
    let mut dynamic_values =
        Vec::<f64>::with_capacity(jobs * 2);
    let mut energy_cut_levels =
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

    let mut cuts = vec![0.0_f64; jobs];
    let mut status = vec![0_u32; jobs];

    let mut static_h2d_ms = 0.0_f64;
    let mut host_pack_ms = 0.0_f64;
    let mut input_enqueue_ms = 0.0_f64;
    let mut kernel_ms = 0.0_f64;
    let mut result_wait_ms = 0.0_f64;
    let mut wall_ms = 0.0_f64;

    let selected_device =
        device_id.unwrap_or(0);

    let code = unsafe {
        sunopt_cuda_run_b14l_packed_transport_benchmark(
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
            if pinned_host_memory { 1 } else { 0 },
            cuts.as_mut_ptr(),
            status.as_mut_ptr(),
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
                "sunopt_cuda_run_b14l_packed_transport_benchmark",
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

    Ok(BlackwellB14lVariantResult {
        pinned_host_memory,
        annual_step_count: step_count as u32,
        tank_count: snapshot.tank_count as u32,
        causal_jobs: jobs as u32,
        static_h2d_ms,
        host_pack_ms,
        input_enqueue_ms,
        kernel_ms,
        result_wait_ms,
        wall_ms,
        checksum_hex: format!("{checksum:016x}"),
        status0,
        status1,
        status2,
        status3,
    })
}
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_FN_END
`.trim();

const RUNNER = [
  '// rust/sunopt-node/scripts/blackwell-b14l-packed-transport.mjs',
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
  'const run = addon.runBlackwellB14LPackedTransport;',
  '',
  'if (typeof run !== "function") {',
  '  throw new Error("runBlackwellB14LPackedTransport native export missing");',
  '}',
  '',
  'const snapshot = process.argv[2]',
  '  ? path.resolve(process.argv[2])',
  '  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
  '',
  'const expectedChecksum = "bf0e8c9e7b5c879e";',
  'const currentRelease = fs.existsSync(historyHash)',
  '  ? fs.readFileSync(historyHash, "utf8").trim()',
  '  : "unknown";',
  '',
  'const variants = [];',
  'console.log("--- B14L PACKED/PINNED TRANSPORT ---");',
  'console.log(`release=${currentRelease}`);',
  'console.log(`snapshot=${snapshot}`);',
  '',
  'for (const pinned of [false, true]) {',
  '  console.log(`running pinned=${pinned} ...`);',
  '  const r = run(snapshot, pinned, 0);',
  '  const exact = r.checksumHex === expectedChecksum;',
  '  const row = {',
  '    ...r,',
  '    exactVsB14H: exact,',
  '    speedupVsB14K2Wall: 4547.4845 / r.wallMs,',
  '    speedupVsB14K2Kernel: 2702.7629009000957 / r.kernelMs,',
  '    nonKernelWallMs: r.wallMs - r.kernelMs,',
  '  };',
  '  variants.push(row);',
  '  console.log(`pinned=${pinned} kernel=${r.kernelMs.toFixed(3)} ms wall=${r.wallMs.toFixed(3)} ms exact=${exact}`);',
  '}',
  '',
  'const exactVariants = variants.filter(v => v.exactVsB14H);',
  'const bestExact = exactVariants.length',
  '  ? exactVariants.reduce((a, b) => a.wallMs <= b.wallMs ? a : b)',
  '  : null;',
  '',
  'const report = {',
  '  stage: "B14L_PACKED_PINNED_TRANSPORT",',
  '  release: currentRelease,',
  '  snapshotPath: snapshot,',
  '  expectedChecksum,',
  '  baselineB14KDepth2: { kernelMs: 2702.7629009000957, wallMs: 4547.4845 },',
  '  variants,',
  '  bestExact,',
  '};',
  '',
  'const reportPath = path.join(outbox, "blackwell-b14l-packed-transport.json");',
  'fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\\n");',
  '',
  'const llmPath = path.join(outbox, "llm.txt");',
  'const timestamp = new Date().toISOString();',
  'const llmLines = [',
  '  "",',
  '  `=== ${timestamp} B14L_PACKED_PINNED_TRANSPORT ===`,',
  '  `release=${currentRelease}`,',
  '  `snapshot=${snapshot}`,',
  '  `expectedChecksum=${expectedChecksum}`,',
  '  ...variants.map(v => `variant pinned=${v.pinnedHostMemory} kernelMs=${v.kernelMs.toFixed(6)} wallMs=${v.wallMs.toFixed(6)} checksum=${v.checksumHex} exact=${v.exactVsB14H}`),',
  '  bestExact ? `best pinned=${bestExact.pinnedHostMemory} kernelMs=${bestExact.kernelMs.toFixed(6)} wallMs=${bestExact.wallMs.toFixed(6)} speedupVsB14K2Wall=${bestExact.speedupVsB14K2Wall.toFixed(6)}x` : "best=none",',
  '  `report=${reportPath}`,',
  '  "",',
  '].join("\\n");',
  'fs.appendFileSync(llmPath, llmLines, "utf8");',
  '',
  'console.log("");',
  'console.log(`report=${reportPath}`);',
  'console.log(`llm=${llmPath}`);',
  'if (bestExact) {',
  '  console.log(`bestPinned=${bestExact.pinnedHostMemory}`);',
  '  console.log(`bestExactWallMs=${bestExact.wallMs.toFixed(3)}`);',
  '  console.log(`bestSpeedupVsB14K2=${bestExact.speedupVsB14K2Wall.toFixed(4)}x`);',
  '  console.log("B14L_RESULT=EXACT_VARIANT_FOUND");',
  '} else {',
  '  console.log("B14L_RESULT=NO_BITWISE_EXACT_VARIANT");',
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
    u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
    u32(crc), u32(data.length), u32(data.length),
    u16(nameBuf.length), u16(0), nameBuf, data,
  ]);
  locals.push(local);

  const central = Buffer.concat([
    u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
    u32(crc), u32(data.length), u32(data.length),
    u16(nameBuf.length), u16(0), u16(0), u16(0), u16(0), u32(0),
    u32(offset), nameBuf,
  ]);
  centrals.push(central);
  offset += local.length;
}

const centralOffset = offset;
const central = Buffer.concat(centrals);
const end = Buffer.concat([
  u32(0x06054b50), u16(0), u16(0),
  u16(entries.length), u16(entries.length),
  u32(central.length), u32(centralOffset), u16(0),
]);

const zip = Buffer.concat([...locals, central, end]);

fs.mkdirSync(inbox, { recursive: true });
const packagePath = path.join(
  inbox,
  "20261112-blackwell-b14l-packed-transport.zip",
);
fs.writeFileSync(packagePath, zip);

console.log(`parent=${parent}`);
console.log(`release=${release}`);
console.log(`package=${packagePath}`);
console.log("B14L_PACKAGE_RESULT=OK");
