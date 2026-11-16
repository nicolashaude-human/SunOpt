#!/usr/bin/env python3
# tools/make_blackwell_digital_lockstep_candidate.py
#
# Run from:
#   C:\NextJS\SunOpt\sunopt
#
# Creates a verified format-2 update package.
#
# This is a production-shaped A/B benchmark, NOT yet a production switch:
#   - 35,040 causal steps are executed sequentially
#   - 28 tanks are launched per step
#   - per step: dynamic H2D -> digital breakpoint kernel -> result D2H -> sync
#   - search depth fixed to 6 (proven exact in the sampled benchmark)
#   - unsupported exact max-affine jobs are completed by exact scalar digital
#     fallback on CPU and timed separately
#
# Requires the previously installed digital-HP breakpoint research package.
# The existing continuous production worker remains untouched.

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
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-lockstep-candidate.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_END"


H_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN
/*
 * Production-shaped direct lockstep benchmark for the ideal digital HP.
 *
 * Static annual arrays and tank parameters are uploaded once.
 * Every causal step then performs:
 *   dynamic H2D -> 28-block digital breakpoint kernel -> result D2H -> sync.
 *
 * This deliberately models the host/GPU boundary of the real annual
 * simulation rather than batching thousands of independent recorded jobs.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_candidate(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_static_h2d_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_END
'''.strip()


CU_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN
extern "C" int sunopt_cuda_run_digital_lockstep_candidate(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_static_h2d_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
    double* out_wall_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_static_h2d_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        repetitions == 0 ||
        search_iterations == 0
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            device_id
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    cudaStream_t stream =
        nullptr;

    cudaEvent_t gpu_start =
        nullptr;

    cudaEvent_t gpu_stop =
        nullptr;

    double* d_export =
        nullptr;

    double* d_demand =
        nullptr;

    double* d_tanks =
        nullptr;

    uint32_t* d_current =
        nullptr;

    uint32_t* d_horizon =
        nullptr;

    double* d_dynamic =
        nullptr;

    double* d_cuts =
        nullptr;

    uint32_t* d_status =
        nullptr;

    uint32_t* d_evaluations =
        nullptr;

    const size_t annual_bytes =
        data_length *
        sizeof(double);

    const size_t tank_bytes =
        tank_count *
        7 *
        sizeof(double);

    const size_t dynamic_step_bytes =
        tank_count *
        3 *
        sizeof(double);

    const size_t cut_step_bytes =
        tank_count *
        sizeof(double);

    const size_t u32_step_bytes =
        tank_count *
        sizeof(uint32_t);

    auto cleanup =
        [&]() {
            if (d_evaluations) cudaFree(d_evaluations);
            if (d_status) cudaFree(d_status);
            if (d_cuts) cudaFree(d_cuts);
            if (d_dynamic) cudaFree(d_dynamic);
            if (d_horizon) cudaFree(d_horizon);
            if (d_current) cudaFree(d_current);
            if (d_tanks) cudaFree(d_tanks);
            if (d_demand) cudaFree(d_demand);
            if (d_export) cudaFree(d_export);
            if (gpu_stop) cudaEventDestroy(gpu_stop);
            if (gpu_start) cudaEventDestroy(gpu_start);
            if (stream) cudaStreamDestroy(stream);
        };

#define SUNOPT_LOCKSTEP_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_LOCKSTEP_MALLOC(ptr, bytes) \
    SUNOPT_LOCKSTEP_CHECK( \
        cudaMalloc( \
            reinterpret_cast<void**>(&(ptr)), \
            (bytes) \
        ) \
    )

    SUNOPT_LOCKSTEP_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventCreate(
            &gpu_start
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventCreate(
            &gpu_stop
        )
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_export,
        annual_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_demand,
        annual_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_tanks,
        tank_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_current,
        sizeof(uint32_t)
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_horizon,
        sizeof(uint32_t)
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_dynamic,
        dynamic_step_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_cuts,
        cut_step_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_status,
        u32_step_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_evaluations,
        u32_step_bytes
    );

#undef SUNOPT_LOCKSTEP_MALLOC

    const auto static_h2d_start =
        Clock::now();

    SUNOPT_LOCKSTEP_CHECK(
        cudaMemcpy(
            d_export,
            host_export_power,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaMemcpy(
            d_demand,
            host_demand_energy,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaMemcpy(
            d_tanks,
            host_tank_parameters,
            tank_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const auto static_h2d_stop =
        Clock::now();

    *out_static_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            static_h2d_stop -
            static_h2d_start
        ).count();

    double host_wait_ms =
        0.0;

    const auto wall_start =
        Clock::now();

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventRecord(
            gpu_start,
            stream
        )
    );

    const unsigned int blocks =
        static_cast<unsigned int>(
            tank_count
        );

    for (
        uint32_t repetition = 0;
        repetition <
            repetitions;
        ++repetition
    ) {
        for (
            size_t step = 0;
            step <
                step_count;
            ++step
        ) {
            const size_t dynamic_offset =
                step *
                tank_count *
                3;

            const size_t result_offset =
                step *
                tank_count;

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    d_current,
                    host_current_indices +
                        step,
                    sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    d_horizon,
                    host_horizon_steps +
                        step,
                    sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    d_dynamic,
                    host_dynamic_values +
                        dynamic_offset,
                    dynamic_step_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            sunopt_digital_hp_breakpoint_kernel<<<
                blocks,
                SUNOPT_DIGITAL_THREADS,
                0,
                stream
            >>>(
                d_export,
                d_demand,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                1,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaGetLastError()
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    host_cut_results +
                        result_offset,
                    d_cuts,
                    cut_step_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    host_status +
                        result_offset,
                    d_status,
                    u32_step_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    host_survival_evaluations +
                        result_offset,
                    d_evaluations,
                    u32_step_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                )
            );

            const auto wait_start =
                Clock::now();

            SUNOPT_LOCKSTEP_CHECK(
                cudaStreamSynchronize(
                    stream
                )
            );

            const auto wait_stop =
                Clock::now();

            host_wait_ms +=
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    wait_stop -
                    wait_start
                ).count();
        }
    }

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventRecord(
            gpu_stop,
            stream
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventSynchronize(
            gpu_stop
        )
    );

    float gpu_total_ms =
        0.0f;

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventElapsedTime(
            &gpu_total_ms,
            gpu_start,
            gpu_stop
        )
    );

    const auto wall_stop =
        Clock::now();

    *out_gpu_total_ms =
        static_cast<double>(
            gpu_total_ms
        );

    *out_host_wait_ms =
        host_wait_ms;

    *out_wall_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            wall_stop -
            wall_start
        ).count();

#undef SUNOPT_LOCKSTEP_CHECK

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_END
'''.strip()


RS_FFI_BLOCK = r'''
    // SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FFI_BEGIN
    fn sunopt_cuda_run_digital_lockstep_candidate(
        device_id:
            c_int,

        host_export_power:
            *const f64,

        host_demand_energy:
            *const f64,

        data_length:
            usize,

        host_tank_parameters:
            *const f64,

        tank_count:
            usize,

        host_current_indices:
            *const u32,

        host_horizon_steps:
            *const u32,

        host_dynamic_values:
            *const f64,

        step_count:
            usize,

        search_iterations:
            u32,

        repetitions:
            u32,

        host_cut_results:
            *mut f64,

        host_status:
            *mut u32,

        host_survival_evaluations:
            *mut u32,

        out_static_h2d_ms:
            *mut f64,

        out_gpu_total_ms:
            *mut f64,

        out_host_wait_ms:
            *mut f64,

        out_wall_ms:
            *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FFI_END
'''.strip()


RS_RESULT_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalLockstepCandidateResult {
    pub snapshot_path:
        String,

    pub device_id:
        i32,

    pub repetitions:
        u32,

    pub search_iterations:
        u32,

    pub annual_step_count:
        u32,

    pub tank_count:
        u32,

    pub causal_jobs:
        u32,

    pub gpu_supported_jobs:
        u32,

    pub fallback_jobs:
        u32,

    pub static_h2d_ms:
        f64,

    pub direct_gpu_total_ms:
        f64,

    pub direct_host_wait_ms:
        f64,

    pub direct_wall_ms:
        f64,

    pub fallback_cpu_ms:
        f64,

    pub candidate_causal_ms:
        f64,

    pub candidate_average_us_per_job:
        f64,

    pub parity_sample_steps:
        u32,

    pub parity_status_mismatches:
        u32,

    pub parity_cut_mismatches:
        u32,

    pub parity_max_cut_error:
        f64,

    pub average_survival_evaluations:
        f64,

    pub maximum_survival_evaluations:
        u32,

    pub checksum_hex:
        String,
}
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_END
'''.strip()


RS_FN_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_lockstep_candidate(
    snapshot_path:
        Option<String>,

    repetitions:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<BlackwellDigitalLockstepCandidateResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(
            &path,
        )
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
        snapshot.demand_energy.len() !=
            snapshot.export_power.len()
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or inconsistent.",
            ),
        );
    }

    if (
        snapshot.steps.len() !=
        snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "Expected full annual lockstep snapshot: {} steps vs data length {}.",
                    snapshot.steps.len(),
                    snapshot.data_length,
                ),
            ),
        );
    }

    let selected_device =
        device_id
            .unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(
                1,
                5,
            );

    let search_iterations =
        6_u32;

    let tanks =
        (
            0..
            snapshot.tank_count
        )
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
            snapshot.tank_count *
            7,
        );

    for tank in
        &tanks
    {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            pmax,
        );

        packed_tanks.push(
            pmax *
            tank.cop *
            tank.time_step_hours,
        );

        packed_tanks.push(
            tank.heat_loss_retention,
        );

        packed_tanks.push(
            tank.heat_loss_intercept,
        );

        packed_tanks.push(
            if tank.affine_supported {
                1.0
            } else {
                0.0
            },
        );

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Digital lockstep candidate job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count *
            3,
        );

    for step in
        &snapshot.steps
    {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital lockstep candidate supports horizon 1..=256; got {}.",
                        step.horizon_steps,
                    ),
                ),
            );
        }

        current_indices.push(
            step.current_index,
        );

        horizon_steps.push(
            step.horizon_steps,
        );

        for tank in
            0..
            tank_count
        {
            let base =
                tank *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(
                step.dynamic[
                    base + 0
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 1
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 2
                ],
            );
        }
    }

    let mut cuts =
        vec![
            0.0_f64;
            job_count
        ];

    let mut statuses =
        vec![
            0_u32;
            job_count
        ];

    let mut evaluations =
        vec![
            0_u32;
            job_count
        ];

    let mut static_h2d_ms =
        0.0_f64;

    let mut gpu_total_ms =
        0.0_f64;

    let mut host_wait_ms =
        0.0_f64;

    let mut wall_ms =
        0.0_f64;

    let code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_candidate(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                tank_count,
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_flat
                    .as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                cuts
                    .as_mut_ptr(),
                statuses
                    .as_mut_ptr(),
                evaluations
                    .as_mut_ptr(),
                &mut static_h2d_ms,
                &mut gpu_total_ms,
                &mut host_wait_ms,
                &mut wall_ms,
            )
        };

    if (
        code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_candidate",
                code,
            ),
        );
    }

    let direct_gpu_total_ms =
        gpu_total_ms /
        repetition_count
            as f64;

    let direct_host_wait_ms =
        host_wait_ms /
        repetition_count
            as f64;

    let direct_wall_ms =
        wall_ms /
        repetition_count
            as f64;

    /*
     * Preserve the raw GPU status for parity statistics before replacing
     * unsupported jobs by the exact scalar digital fallback.
     */
    let gpu_statuses =
        statuses.clone();

    let gpu_cuts =
        cuts.clone();

    let fallback_started =
        Instant::now();

    let mut fallback_jobs =
        0_u32;

    for (
        step_index,
        step,
    ) in
        snapshot.steps
            .iter()
            .enumerate()
    {
        let mut export_window:
            Option<Vec<f64>> =
            None;

        let mut demand_window:
            Option<Vec<f64>> =
            None;

        for tank_index in
            0..
            tank_count
        {
            let job =
                step_index *
                tank_count +
                tank_index;

            if (
                statuses[
                    job
                ] !=
                3
            ) {
                continue;
            }

            fallback_jobs +=
                1;

            if (
                export_window
                    .is_none()
            ) {
                export_window =
                    Some(
                        research_window(
                            &snapshot.export_power,
                            step.current_index
                                as usize,
                            step.horizon_steps
                                as usize,
                        ),
                    );

                demand_window =
                    Some(
                        research_window(
                            &snapshot.demand_energy,
                            step.current_index
                                as usize,
                            step.horizon_steps
                                as usize,
                        ),
                    );
            }

            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let current_storage =
                step.dynamic[
                    base + 0
                ];

            let reserve =
                step.dynamic[
                    base + 1
                ];

            let requested =
                step.dynamic[
                    base + 2
                ];

            let reference =
                digital_run_scalar(
                    export_window
                        .as_ref()
                        .unwrap(),
                    demand_window
                        .as_ref()
                        .unwrap(),
                    current_storage,
                    reserve,
                    requested,
                    tanks[
                        tank_index
                    ],
                );

            cuts[
                job
            ] =
                reference
                    .cut_level_power;

            statuses[
                job
            ] =
                if reference
                    .normal_plan_survives
                {
                    0
                } else if !reference
                    .maximum_pv_plan_survives
                {
                    1
                } else {
                    2
                };
        }
    }

    let fallback_cpu_ms =
        fallback_started
            .elapsed()
            .as_secs_f64() *
        1000.0;

    let gpu_supported_jobs =
        job_count
            as u32 -
        fallback_jobs;

    /*
     * Production-shaped parity sample: verify the direct 28-block lockstep
     * outputs against the scalar ideal-digital reference at 192 evenly spaced
     * annual steps. Unsupported GPU jobs are excluded from GPU parity because
     * they intentionally use the scalar fallback.
     */
    let parity_sample_steps =
        192_usize.min(
            step_count,
        );

    let mut parity_status_mismatches =
        0_u32;

    let mut parity_cut_mismatches =
        0_u32;

    let mut parity_max_cut_error =
        0.0_f64;

    for sample in
        0..
        parity_sample_steps
    {
        let step_index =
            if (
                parity_sample_steps <=
                1
            ) {
                0
            } else {
                sample *
                (
                    step_count -
                    1
                ) /
                (
                    parity_sample_steps -
                    1
                )
            };

        let step =
            &snapshot.steps[
                step_index
            ];

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index
                    as usize,
                step.horizon_steps
                    as usize,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index
                    as usize,
                step.horizon_steps
                    as usize,
            );

        for tank_index in
            0..
            tank_count
        {
            let job =
                step_index *
                tank_count +
                tank_index;

            if (
                gpu_statuses[
                    job
                ] ==
                3
            ) {
                continue;
            }

            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reference =
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    step.dynamic[
                        base + 0
                    ],
                    step.dynamic[
                        base + 1
                    ],
                    step.dynamic[
                        base + 2
                    ],
                    tanks[
                        tank_index
                    ],
                );

            let expected_status =
                if reference
                    .normal_plan_survives
                {
                    0_u32
                } else if !reference
                    .maximum_pv_plan_survives
                {
                    1_u32
                } else {
                    2_u32
                };

            if (
                gpu_statuses[
                    job
                ] !=
                expected_status
            ) {
                parity_status_mismatches +=
                    1;
            }

            let cut_error =
                (
                    gpu_cuts[
                        job
                    ] -
                    reference
                        .cut_level_power
                )
                .abs();

            parity_max_cut_error =
                parity_max_cut_error.max(
                    cut_error,
                );

            if (
                cut_error >
                1.0e-9
            ) {
                parity_cut_mismatches +=
                    1;
            }
        }
    }

    let mut evaluation_sum =
        0_u64;

    let mut maximum_survival_evaluations =
        0_u32;

    for (
        job,
        &value,
    ) in
        evaluations
            .iter()
            .enumerate()
    {
        if (
            gpu_statuses[
                job
            ] ==
            3
        ) {
            continue;
        }

        evaluation_sum +=
            value
                as u64;

        maximum_survival_evaluations =
            maximum_survival_evaluations.max(
                value,
            );
    }

    let average_survival_evaluations =
        if (
            gpu_supported_jobs >
            0
        ) {
            evaluation_sum
                as f64 /
            gpu_supported_jobs
                as f64
        } else {
            0.0
        };

    let candidate_causal_ms =
        direct_wall_ms +
        fallback_cpu_ms;

    let candidate_average_us_per_job =
        candidate_causal_ms *
        1000.0 /
        job_count
            as f64;

    let mut checksum =
        1469598103934665603u64;

    for value in
        &cuts
    {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    Ok(
        BlackwellDigitalLockstepCandidateResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            search_iterations,

            annual_step_count:
                step_count
                    as u32,

            tank_count:
                tank_count
                    as u32,

            causal_jobs:
                job_count
                    as u32,

            gpu_supported_jobs,

            fallback_jobs,

            static_h2d_ms,

            direct_gpu_total_ms,

            direct_host_wait_ms,

            direct_wall_ms,

            fallback_cpu_ms,

            candidate_causal_ms,

            candidate_average_us_per_job,

            parity_sample_steps:
                parity_sample_steps
                    as u32,

            parity_status_mismatches,

            parity_cut_mismatches,

            parity_max_cut_error,

            average_survival_evaluations,

            maximum_survival_evaluations,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_END
'''.strip()


MJS = r'''import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const addon = require(
  path.join(
    root,
    "sunopt_node.win32-x64-msvc.node",
  ),
);

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(
      root,
      "blackwell-benchmark.bin",
    );

const repetitions =
  Number.parseInt(
    process.argv[3] ?? "1",
    10,
  );

const deviceId =
  Number.parseInt(
    process.argv[4] ?? "0",
    10,
  );

const result =
  addon.runBlackwellDigitalLockstepCandidate(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_LOCKSTEP_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- BLACKWELL DIGITAL-HP PRODUCTION-SHAPED LOCKSTEP ---",
);

console.log(
  `snapshot=${result.snapshotPath}`,
);

console.log(
  `steps=${result.annualStepCount}`,
);

console.log(
  `tanks=${result.tankCount}`,
);

console.log(
  `jobs=${result.causalJobs}`,
);

console.log(
  `searchIterations=${result.searchIterations}`,
);

console.log(
  `gpuSupportedJobs=${result.gpuSupportedJobs}`,
);

console.log(
  `fallbackJobs=${result.fallbackJobs}`,
);

console.log();
console.log(
  "TIMING",
);

console.log(
  `staticH2DMs=${result.staticH2DMs.toFixed(3)}`,
);

console.log(
  `directGpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`,
);

console.log(
  `directHostWaitMs=${result.directHostWaitMs.toFixed(3)}`,
);

console.log(
  `directWallMs=${result.directWallMs.toFixed(3)}`,
);

console.log(
  `fallbackCpuMs=${result.fallbackCpuMs.toFixed(3)}`,
);

console.log(
  `candidateCausalMs=${result.candidateCausalMs.toFixed(3)}`,
);

console.log(
  `candidateAverageUsPerJob=${result.candidateAverageUsPerJob.toFixed(3)}`,
);

console.log();
console.log(
  "PARITY SAMPLE",
);

console.log(
  `sampleSteps=${result.paritySampleSteps}`,
);

console.log(
  `statusMismatches=${result.parityStatusMismatches}`,
);

console.log(
  `cutMismatches=${result.parityCutMismatches}`,
);

console.log(
  `maxCutError=${result.parityMaxCutError.toExponential(6)}`,
);

console.log();
console.log(
  "SEARCH WORK",
);

console.log(
  `averageSurvivalEvaluations=${result.averageSurvivalEvaluations.toFixed(3)}`,
);

console.log(
  `maximumSurvivalEvaluations=${result.maximumSurvivalEvaluations}`,
);

console.log(
  `checksum=${result.checksumHex}`,
);

const report =
  path.join(
    root,
    "blackwell-digital-lockstep-candidate.json",
  );

writeFileSync(
  report,
  JSON.stringify(
    result,
    null,
    2,
  ) + "\n",
  "utf8",
);

console.log();
console.log(
  `report=${report}`,
);
'''.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(
            encoding="utf-8",
        ).strip().lower()

        if HASH_RE.fullmatch(value):
            return value

        raise RuntimeError(
            f"Invalid SHA-256 in {HASH_FILE}"
        )

    if RELEASE_FILE.is_file():
        state = json.loads(
            RELEASE_FILE.read_text(
                encoding="utf-8",
            )
        )

        value = str(
            state.get(
                "sha256",
                "",
            )
        ).strip().lower()

        if HASH_RE.fullmatch(value):
            return value

        raise RuntimeError(
            f"Invalid sha256 in {RELEASE_FILE}"
        )

    return GENESIS


def replace_between(
    source: str,
    begin: str,
    end: str,
    block: str,
) -> str:
    has_begin = begin in source
    has_end = end in source

    if has_begin != has_end:
        raise RuntimeError(
            f"Only one marker present: {begin!r} / {end!r}"
        )

    if not has_begin:
        raise RuntimeError(
            f"Marker pair missing: {begin!r} / {end!r}"
        )

    start = source.index(begin)
    finish = source.index(
        end,
        start,
    ) + len(end)

    return (
        source[:start] +
        block +
        source[finish:]
    )


def insert_or_replace(
    source: str,
    begin: str,
    end: str,
    block: str,
    anchors: list[str],
) -> str:
    if begin in source or end in source:
        return replace_between(
            source,
            begin,
            end,
            block,
        )

    for anchor in anchors:
        position = source.find(
            anchor
        )

        if position >= 0:
            return (
                source[:position] +
                block +
                "\n\n\n" +
                source[position:]
            )

    raise RuntimeError(
        "Could not find insertion anchor. Tried: " +
        ", ".join(
            repr(x)
            for x in anchors
        )
    )


def patch_header(source: str) -> str:
    if (
        "SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN"
        not in source
    ):
        raise RuntimeError(
            "Digital-HP breakpoint package is not installed in cuda_context.h."
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
    required = [
        "SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN",
        "sunopt_digital_hp_breakpoint_kernel(",
        "SUNOPT_DIGITAL_THREADS",
    ]

    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Required digital breakpoint CUDA code missing: {token!r}"
            )

    return insert_or_replace(
        source,
        CU_BEGIN,
        CU_END,
        CU_BLOCK,
        [
            'extern "C" int sunopt_cuda_get_profile(',
            'extern "C" const char* sunopt_cuda_error_string(',
        ],
    )


def patch_rust(source: str) -> str:
    required = [
        "SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_BEGIN",
        "fn digital_run_scalar(",
        "struct DigitalResearchResult",
        "fn research_tank_from_packed(",
        "fn research_window(",
        "fn read_blackwell_snapshot(",
        "BLACKWELL_DYNAMIC_STRIDE",
        "fn blackwell_fnv1a_update(",
    ]

    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Required installed research helper missing: {token!r}"
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
    for path in (
        H_PATH,
        CU_PATH,
        RS_PATH,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing project file: {path}"
            )

    h = patch_header(
        H_PATH.read_text(
            encoding="utf-8",
        )
    )

    cu = patch_cuda(
        CU_PATH.read_text(
            encoding="utf-8",
        )
    )

    rs = patch_rust(
        RS_PATH.read_text(
            encoding="utf-8",
        )
    )

    payloads = {
        H_REL:
            h.encode("utf-8"),

        CU_REL:
            cu.encode("utf-8"),

        RS_REL:
            rs.encode("utf-8"),

        MJS_REL:
            (MJS + "\n").encode(
                "utf-8",
            ),
    }

    parent = read_parent_hash()

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {
            name:
                sha256_bytes(data)
            for name, data in
            payloads.items()
        },
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) +
        "\n"
    ).encode("utf-8")

    release = sha256_bytes(
        manifest_bytes
    )

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        INBOX /
        "20261109-blackwell-digital-lockstep-candidate.zip"
    )

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "manifest.json",
            manifest_bytes,
        )

        archive.writestr(
            "manifest.sha256",
            (release + "\n").encode(
                "ascii",
            ),
        )

        for name, data in payloads.items():
            archive.writestr(
                name,
                data,
            )

    print(
        "SunOpt digital lockstep candidate package created"
    )

    print(
        f"  package: {output}"
    )

    print(
        f"  parent : {parent}"
    )

    print(
        f"  release: {release}"
    )

    print(
        "  production continuous worker: UNCHANGED"
    )

    print(
        "  digital search iterations: 6"
    )

    print()
    print("Next:")
    print(
        r"  python .\tools\update_watcher.py --once"
    )
    print(
        r"  cd .\rust\sunopt-node"
    )
    print(
        r"  npm run build"
    )
    print(
        r"  node .\scripts\blackwell-digital-lockstep-candidate.mjs .\blackwell-benchmark.bin 1 0"
    )

    return output


if __name__ == "__main__":
    build_package()
