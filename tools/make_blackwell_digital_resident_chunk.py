#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path.cwd()
INBOX = ROOT / "updates" / "inbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"

H_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.h"
CU_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.cu"
RS_PATH = ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs"

H_REL = "rust/sunopt-node/cuda/cuda_context.h"
CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
RS_REL = "rust/sunopt-node/src/cuda_context.rs"
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-resident-chunk.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FN_END"

H_BLOCK = r"""
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_resident_chunk_replay(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t chunk_steps,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_END
""".strip()

CHUNK_KERNEL = r"""
__global__ void
sunopt_digital_resident_chunk_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_windows,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    size_t chunk_start,
    size_t chunk_count,
    uint32_t search_iterations,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    const size_t tank_index =
        static_cast<size_t>(blockIdx.x);

    if (tank_index >= tank_count) {
        return;
    }

    const size_t chunk_end =
        min(
            step_count,
            chunk_start + chunk_count
        );

    for (
        size_t step_index = chunk_start;
        step_index < chunk_end;
        ++step_index
    ) {
        const size_t job_index =
            step_index * tank_count + tank_index;

        sunopt_digital_hp_presorted_warp_job(
            export_power,
            demand_energy,
            presorted_export_windows +
                step_index * SUNOPT_DIGITAL_THREADS,
            data_length,
            tank_parameters,
            tank_count,
            current_indices,
            horizon_steps,
            dynamic_values,
            step_count,
            job_index,
            search_iterations,
            cut_results,
            status,
            survival_evaluations
        );

        __syncthreads();
    }
}
""".strip()

C_RUNNER = r"""
extern "C" int sunopt_cuda_run_digital_resident_chunk_replay(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t chunk_steps,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_input_h2d_ms == nullptr ||
        out_kernel_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        chunk_steps == 0 ||
        repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_presorted = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    const size_t job_count = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 7 * sizeof(double);
    const size_t presorted_bytes =
        step_count * SUNOPT_DIGITAL_THREADS * sizeof(double);
    const size_t step_u32_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = job_count * 3 * sizeof(double);
    const size_t cut_bytes = job_count * sizeof(double);
    const size_t u32_job_bytes = job_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_presorted) cudaFree(d_presorted);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (stop_event) cudaEventDestroy(stop_event);
        if (start_event) cudaEventDestroy(start_event);
        if (stream) cudaStreamDestroy(stream);
    };

#define SUNOPT_CHUNK_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_CHUNK_MALLOC(ptr, bytes) \
    SUNOPT_CHUNK_CHECK( \
        cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)) \
    )

    SUNOPT_CHUNK_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_CHUNK_CHECK(cudaEventCreate(&start_event));
    SUNOPT_CHUNK_CHECK(cudaEventCreate(&stop_event));

    SUNOPT_CHUNK_MALLOC(d_export, annual_bytes);
    SUNOPT_CHUNK_MALLOC(d_demand, annual_bytes);
    SUNOPT_CHUNK_MALLOC(d_tanks, tank_bytes);
    SUNOPT_CHUNK_MALLOC(d_presorted, presorted_bytes);
    SUNOPT_CHUNK_MALLOC(d_current, step_u32_bytes);
    SUNOPT_CHUNK_MALLOC(d_horizon, step_u32_bytes);
    SUNOPT_CHUNK_MALLOC(d_dynamic, dynamic_bytes);
    SUNOPT_CHUNK_MALLOC(d_cuts, cut_bytes);
    SUNOPT_CHUNK_MALLOC(d_status, u32_job_bytes);
    SUNOPT_CHUNK_MALLOC(d_evaluations, u32_job_bytes);

#undef SUNOPT_CHUNK_MALLOC

    const auto h2d_start = Clock::now();

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_export,
        host_export_power,
        annual_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_demand,
        host_demand_energy,
        annual_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_tanks,
        host_tank_parameters,
        tank_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_presorted,
        host_presorted_export_windows,
        presorted_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_current,
        host_current_indices,
        step_u32_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_horizon,
        host_horizon_steps,
        step_u32_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_dynamic,
        host_dynamic_values,
        dynamic_bytes,
        cudaMemcpyHostToDevice
    ));

    const auto h2d_stop = Clock::now();

    *out_input_h2d_ms =
        std::chrono::duration<double, std::milli>(
            h2d_stop - h2d_start
        ).count();

    const auto wall_start = Clock::now();

    SUNOPT_CHUNK_CHECK(cudaEventRecord(start_event, stream));

    const size_t chunk = static_cast<size_t>(chunk_steps);

    for (
        uint32_t repetition = 0;
        repetition < repetitions;
        ++repetition
    ) {
        for (
            size_t start = 0;
            start < step_count;
            start += chunk
        ) {
            const size_t count =
                min(
                    chunk,
                    step_count - start
                );

            sunopt_digital_resident_chunk_kernel<<<
                static_cast<unsigned int>(tank_count),
                SUNOPT_DIGITAL_THREADS,
                0,
                stream
            >>>(
                d_export,
                d_demand,
                d_presorted,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                step_count,
                start,
                count,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_CHUNK_CHECK(cudaGetLastError());
        }
    }

    SUNOPT_CHUNK_CHECK(cudaEventRecord(stop_event, stream));
    SUNOPT_CHUNK_CHECK(cudaEventSynchronize(stop_event));

    float kernel_total_ms = 0.0f;

    SUNOPT_CHUNK_CHECK(cudaEventElapsedTime(
        &kernel_total_ms,
        start_event,
        stop_event
    ));

    *out_kernel_ms =
        static_cast<double>(kernel_total_ms) /
        static_cast<double>(repetitions);

    const auto d2h_start = Clock::now();

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        host_cut_results,
        d_cuts,
        cut_bytes,
        cudaMemcpyDeviceToHost
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        host_status,
        d_status,
        u32_job_bytes,
        cudaMemcpyDeviceToHost
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        host_survival_evaluations,
        d_evaluations,
        u32_job_bytes,
        cudaMemcpyDeviceToHost
    ));

    const auto d2h_stop = Clock::now();

    *out_result_d2h_ms =
        std::chrono::duration<double, std::milli>(
            d2h_stop - d2h_start
        ).count();

    const auto wall_stop = Clock::now();

    *out_wall_ms =
        std::chrono::duration<double, std::milli>(
            wall_stop - wall_start
        ).count() /
        static_cast<double>(repetitions);

#undef SUNOPT_CHUNK_CHECK

    cleanup();

    return static_cast<int>(cudaSuccess);
}
""".strip()

RS_FFI_BLOCK = r"""
    // SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FFI_BEGIN
    fn sunopt_cuda_run_digital_resident_chunk_replay(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        chunk_steps: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_input_h2d_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_d2h_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FFI_END
""".strip()

RS_RESULT_BLOCK = r"""
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalResidentChunkVariant {
    pub chunk_steps: u32,
    pub launches_per_year: u32,
    pub input_h2d_ms: f64,
    pub kernel_ms: f64,
    pub result_d2h_ms: f64,
    pub wall_ms: f64,
    pub speedup_vs_lockstep: f64,
    pub reduction_vs_lockstep_percent: f64,
    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,
    pub evaluation_mismatches: u32,
    pub checksum_hex: String,
}

#[napi(object)]
pub struct BlackwellDigitalResidentChunkResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub presorted_mib: f64,
    pub lockstep_gpu_total_ms: f64,
    pub lockstep_wall_ms: f64,
    pub lockstep_checksum_hex: String,
    pub variants: Vec<BlackwellDigitalResidentChunkVariant>,
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_RESULT_END
""".strip()

RS_FN_BLOCK = r"""
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_resident_chunk(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalResidentChunkResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(&path)
            .map_err(
                |error| {
                    Error::from_reason(
                        format!(
                            "Unable to read Blackwell benchmark snapshot {}: {error}",
                            path.display(),
                        ),
                    )
                },
            )?;

    if (
        snapshot.steps.is_empty() ||
        snapshot.tank_count == 0 ||
        snapshot.export_power.is_empty() ||
        snapshot.demand_energy.len() != snapshot.export_power.len() ||
        snapshot.steps.len() != snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device = device_id.unwrap_or(0);
    let repetition_count = repetitions.unwrap_or(1).clamp(1, 3);
    let search_iterations = 8_u32;

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(tank.target_storage_energy);
        packed_tanks.push(pmax);
        packed_tanks.push(
            pmax * tank.cop * tank.time_step_hours,
        );
        packed_tanks.push(tank.heat_loss_retention);
        packed_tanks.push(tank.heat_loss_intercept);
        packed_tanks.push(
            if tank.affine_supported { 1.0 } else { 0.0 },
        );
        packed_tanks.push(
            pmax * tank.time_step_hours,
        );
    }

    let step_count = snapshot.steps.len();
    let tank_count = snapshot.tank_count;
    let job_count =
        step_count
            .checked_mul(tank_count)
            .ok_or_else(
                || Error::from_reason(
                    "Resident-chunk job count overflow.",
                ),
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(step_count);
    let mut horizon_steps =
        Vec::<u32>::with_capacity(step_count);
    let mut dynamic_flat =
        Vec::<f64>::with_capacity(job_count * 3);

    for step in &snapshot.steps {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Resident-chunk benchmark supports horizon 1..=256; got {}.",
                        step.horizon_steps,
                    ),
                ),
            );
        }

        current_indices.push(step.current_index);
        horizon_steps.push(step.horizon_steps);

        for tank_index in 0..tank_count {
            let base =
                tank_index * BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(step.dynamic[base + 0]);
            dynamic_flat.push(step.dynamic[base + 1]);
            dynamic_flat.push(step.dynamic[base + 2]);
        }
    }

    let presorted_count =
        step_count
            .checked_mul(256)
            .ok_or_else(
                || Error::from_reason(
                    "Presorted export matrix size overflow.",
                ),
            )?;

    let mut presorted =
        vec![f64::NEG_INFINITY; presorted_count];

    for (step_index, step) in snapshot.steps.iter().enumerate() {
        let horizon = step.horizon_steps as usize;
        let base = step_index * 256;
        let first = base + (256 - horizon);

        for offset in 0..horizon {
            let index =
                (step.current_index as usize + offset) %
                snapshot.export_power.len();

            presorted[first + offset] =
                snapshot.export_power[index];
        }

        presorted[first..base + 256]
            .sort_by(
                |a, b| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                },
            );
    }

    let presorted_mib =
        (
            presorted.len() * std::mem::size_of::<f64>()
        ) as f64 /
        1048576.0;

    let mut reference_cuts = vec![0.0_f64; job_count];
    let mut reference_status = vec![0_u32; job_count];
    let mut reference_evaluations = vec![0_u32; job_count];

    let mut ref_static_h2d_ms = 0.0_f64;
    let mut ref_presorted_h2d_ms = 0.0_f64;
    let mut ref_gpu_total_raw_ms = 0.0_f64;
    let mut ref_host_wait_raw_ms = 0.0_f64;
    let mut ref_wall_raw_ms = 0.0_f64;

    let reference_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                reference_cuts.as_mut_ptr(),
                reference_status.as_mut_ptr(),
                reference_evaluations.as_mut_ptr(),
                &mut ref_static_h2d_ms,
                &mut ref_presorted_h2d_ms,
                &mut ref_gpu_total_raw_ms,
                &mut ref_host_wait_raw_ms,
                &mut ref_wall_raw_ms,
            )
        };

    if reference_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",
                reference_code,
            ),
        );
    }

    let divisor = repetition_count as f64;
    let lockstep_gpu_total_ms = ref_gpu_total_raw_ms / divisor;
    let lockstep_wall_ms = ref_wall_raw_ms / divisor;

    let mut reference_checksum = 1469598103934665603u64;
    for value in &reference_cuts {
        blackwell_fnv1a_update(&mut reference_checksum, *value);
    }

    let chunk_sizes: [u32; 7] =
        [1, 4, 16, 64, 256, 1024, 4096];

    let mut variants =
        Vec::<BlackwellDigitalResidentChunkVariant>::with_capacity(
            chunk_sizes.len(),
        );

    for chunk_steps in chunk_sizes {
        let mut cuts = vec![0.0_f64; job_count];
        let mut status = vec![0_u32; job_count];
        let mut evaluations = vec![0_u32; job_count];

        let mut input_h2d_ms = 0.0_f64;
        let mut kernel_ms = 0.0_f64;
        let mut result_d2h_ms = 0.0_f64;
        let mut wall_ms = 0.0_f64;

        let code =
            unsafe {
                sunopt_cuda_run_digital_resident_chunk_replay(
                    selected_device,
                    snapshot.export_power.as_ptr(),
                    snapshot.demand_energy.as_ptr(),
                    snapshot.export_power.len(),
                    packed_tanks.as_ptr(),
                    tank_count,
                    presorted.as_ptr(),
                    current_indices.as_ptr(),
                    horizon_steps.as_ptr(),
                    dynamic_flat.as_ptr(),
                    step_count,
                    search_iterations,
                    chunk_steps,
                    repetition_count,
                    cuts.as_mut_ptr(),
                    status.as_mut_ptr(),
                    evaluations.as_mut_ptr(),
                    &mut input_h2d_ms,
                    &mut kernel_ms,
                    &mut result_d2h_ms,
                    &mut wall_ms,
                )
            };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_digital_resident_chunk_replay",
                    code,
                ),
            );
        }

        let mut status_mismatches = 0_u32;
        let mut cut_mismatches = 0_u32;
        let mut maximum_cut_error = 0.0_f64;
        let mut evaluation_mismatches = 0_u32;

        for job in 0..job_count {
            if status[job] != reference_status[job] {
                status_mismatches += 1;
            }

            if evaluations[job] != reference_evaluations[job] {
                evaluation_mismatches += 1;
            }

            let error =
                (cuts[job] - reference_cuts[job]).abs();

            maximum_cut_error =
                maximum_cut_error.max(error);

            if error > 1.0e-9 {
                cut_mismatches += 1;
            }
        }

        let mut checksum = 1469598103934665603u64;
        for value in &cuts {
            blackwell_fnv1a_update(&mut checksum, *value);
        }

        let speedup_vs_lockstep =
            if wall_ms > 0.0 {
                lockstep_wall_ms / wall_ms
            } else {
                f64::INFINITY
            };

        let reduction_vs_lockstep_percent =
            if lockstep_wall_ms > 0.0 {
                (lockstep_wall_ms - wall_ms) /
                    lockstep_wall_ms *
                    100.0
            } else {
                0.0
            };

        let launches_per_year =
            (
                (step_count as u64 + chunk_steps as u64 - 1) /
                chunk_steps as u64
            ) as u32;

        variants.push(
            BlackwellDigitalResidentChunkVariant {
                chunk_steps,
                launches_per_year,
                input_h2d_ms,
                kernel_ms,
                result_d2h_ms,
                wall_ms,
                speedup_vs_lockstep,
                reduction_vs_lockstep_percent,
                status_mismatches,
                cut_mismatches,
                maximum_cut_error,
                evaluation_mismatches,
                checksum_hex: format!("{checksum:016x}"),
            },
        );
    }

    Ok(
        BlackwellDigitalResidentChunkResult {
            snapshot_path: path.display().to_string(),
            device_id: selected_device,
            repetitions: repetition_count,
            search_iterations,
            annual_step_count: step_count as u32,
            tank_count: tank_count as u32,
            causal_jobs: job_count as u32,
            presorted_mib,
            lockstep_gpu_total_ms,
            lockstep_wall_ms,
            lockstep_checksum_hex:
                format!("{reference_checksum:016x}"),
            variants,
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FN_END
""".strip()

MJS = r"""
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const addon = require(
  path.join(root, "sunopt_node.win32-x64-msvc.node"),
);

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");

const repetitions =
  Number.parseInt(process.argv[3] ?? "1", 10);

const deviceId =
  Number.parseInt(process.argv[4] ?? "0", 10);

const result =
  addon.runBlackwellDigitalResidentChunk(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_RESIDENT_CHUNK_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log("--- DIGITAL RESIDENT CHUNK REPLAY ---");
console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);

console.log();
console.log("LOCKSTEP REFERENCE");
console.log(`gpuTotalMs=${result.lockstepGpuTotalMs.toFixed(3)}`);
console.log(`wallMs=${result.lockstepWallMs.toFixed(3)}`);
console.log(`checksum=${result.lockstepChecksumHex}`);

console.log();
console.log(
  "chunk launches H2Dms kernelMs D2Hms wallMs speedup reduction statusMis cutMis evalMis maxCutErr",
);

for (const v of result.variants) {
  console.log(
    [
      String(v.chunkSteps),
      String(v.launchesPerYear),
      v.inputH2DMs.toFixed(3),
      v.kernelMs.toFixed(3),
      v.resultD2HMs.toFixed(3),
      v.wallMs.toFixed(3),
      `${v.speedupVsLockstep.toFixed(3)}x`,
      `${v.reductionVsLockstepPercent.toFixed(2)}%`,
      String(v.statusMismatches),
      String(v.cutMismatches),
      String(v.evaluationMismatches),
      v.maximumCutError.toExponential(3),
    ].join("\t"),
  );
}

const report =
  path.join(root, "blackwell-digital-resident-chunk.json");

writeFileSync(
  report,
  JSON.stringify(result, null, 2) + "\n",
  "utf8",
);

console.log();
console.log(`report=${report}`);
""".strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="utf-8").strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid SHA-256 in {HASH_FILE}")

    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid sha256 in {RELEASE_FILE}")

    return GENESIS


def replace_between(source: str, begin: str, end: str, block: str) -> str:
    has_begin = begin in source
    has_end = end in source

    if has_begin != has_end:
        raise RuntimeError(f"Only one marker present: {begin!r} / {end!r}")
    if not has_begin:
        raise RuntimeError(f"Marker pair missing: {begin!r} / {end!r}")

    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[:start] + block + source[finish:]


def insert_or_replace(
    source: str,
    begin: str,
    end: str,
    block: str,
    anchors: list[str],
) -> str:
    if begin in source or end in source:
        return replace_between(source, begin, end, block)

    for anchor in anchors:
        position = source.find(anchor)
        if position >= 0:
            return source[:position] + block + "\n\n\n" + source[position:]

    raise RuntimeError(
        "Could not find insertion anchor. Tried: " +
        ", ".join(repr(x) for x in anchors)
    )


def extract_braced_function(source: str, token: str) -> str:
    start = source.find(token)
    if start < 0:
        raise RuntimeError(f"Could not find function token {token!r}")

    brace = source.find("{", start)
    if brace < 0:
        raise RuntimeError(f"Could not find opening brace for {token!r}")

    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]

    raise RuntimeError(f"Unterminated function {token!r}")


def build_job_helper(source: str) -> str:
    token = (
        "__global__ void\n"
        "sunopt_digital_hp_presorted_warp_kernel("
    )

    helper = extract_braced_function(source, token)

    helper = helper.replace(
        "__global__ void\n"
        "sunopt_digital_hp_presorted_warp_kernel(",
        "__device__ __forceinline__ void\n"
        "sunopt_digital_hp_presorted_warp_job(",
        1,
    )

    signature_anchor = (
        "    size_t step_count,\n"
        "    uint32_t search_iterations,"
    )

    if signature_anchor not in helper:
        raise RuntimeError(
            "Could not locate step_count in warp-kernel signature."
        )

    helper = helper.replace(
        signature_anchor,
        (
            "    size_t step_count,\n"
            "    size_t forced_job_index,\n"
            "    uint32_t search_iterations,"
        ),
        1,
    )

    old_job_prefix = """    const size_t job_index =
        static_cast<size_t>(
            blockIdx.x
        );

    const size_t job_count =
        step_count *
        tank_count;

    if (
        job_index >=
        job_count
    ) {
        return;
    }
"""

    if old_job_prefix not in helper:
        raise RuntimeError(
            "Could not locate the warp kernel blockIdx job mapping."
        )

    helper = helper.replace(
        old_job_prefix,
        """    const size_t job_index =
        forced_job_index;
""",
        1,
    )

    return helper


def patch_header(source: str) -> str:
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_BEGIN",
        "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(",
    ):
        if token not in source:
            raise RuntimeError(
                f"Expected installed warp-tree header missing {token!r}"
            )

    return insert_or_replace(
        source,
        H_BEGIN,
        H_END,
        H_BLOCK,
        [
            "SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile(",
            "#ifdef __cplusplus\n}",
        ],
    )


def patch_cuda(source: str) -> str:
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_BEGIN",
        "sunopt_digital_hp_presorted_warp_kernel(",
        "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(",
        "SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN",
    ):
        if token not in source:
            raise RuntimeError(
                f"Expected current warp CUDA generation missing {token!r}"
            )

    helper = build_job_helper(source)

    block = (
        CU_BEGIN + "\n" +
        helper + "\n\n\n" +
        CHUNK_KERNEL + "\n\n\n" +
        C_RUNNER + "\n" +
        CU_END
    )

    return insert_or_replace(
        source,
        CU_BEGIN,
        CU_END,
        block,
        [
            'extern "C" int sunopt_cuda_get_profile(',
            'extern "C" const char* sunopt_cuda_error_string(',
        ],
    )


def patch_rust(source: str) -> str:
    for token in (
        "fn sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(",
        "fn research_tank_from_packed(",
        "fn blackwell_fnv1a_update(",
        "BLACKWELL_DYNAMIC_STRIDE",
    ):
        if token not in source:
            raise RuntimeError(
                f"Expected current Rust generation missing {token!r}"
            )

    source = insert_or_replace(
        source,
        RS_FFI_BEGIN,
        RS_FFI_END,
        RS_FFI_BLOCK,
        [
            "    fn sunopt_cuda_get_profile(",
            "    fn sunopt_cuda_error_string(",
        ],
    )

    source = insert_or_replace(
        source,
        RS_RESULT_BEGIN,
        RS_RESULT_END,
        RS_RESULT_BLOCK,
        [
            "#[napi]\npub struct CudaCausalWorker {",
        ],
    )

    source = insert_or_replace(
        source,
        RS_FN_BEGIN,
        RS_FN_END,
        RS_FN_BLOCK,
        [
            "#[napi]\npub fn run_blackwell_replay(",
            "pub fn run_blackwell_replay(",
        ],
    )

    return source


def build_package() -> Path:
    for path in (H_PATH, CU_PATH, RS_PATH):
        if not path.is_file():
            raise RuntimeError(f"Missing project file: {path}")

    h = patch_header(H_PATH.read_text(encoding="utf-8"))
    cu = patch_cuda(CU_PATH.read_text(encoding="utf-8"))
    rs = patch_rust(RS_PATH.read_text(encoding="utf-8"))

    payloads = {
        H_REL: h.encode("utf-8"),
        CU_REL: cu.encode("utf-8"),
        RS_REL: rs.encode("utf-8"),
        MJS_REL: (MJS + "\n").encode("utf-8"),
    }

    parent = read_parent_hash()

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {
            name: sha256_bytes(data)
            for name, data in payloads.items()
        },
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    release = sha256_bytes(manifest_bytes)

    INBOX.mkdir(parents=True, exist_ok=True)
    output = INBOX / "20261109-blackwell-digital-resident-chunk.zip"

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr(
            "manifest.sha256",
            (release + "\n").encode("ascii"),
        )
        for name, data in payloads.items():
            archive.writestr(name, data)

    print("SunOpt digital resident-chunk benchmark package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print("  production worker: UNCHANGED")
    print("  variants: 1,4,16,64,256,1024,4096 steps per launch")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(
        r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-resident-chunk.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node'
    )

    return output


if __name__ == "__main__":
    build_package()
