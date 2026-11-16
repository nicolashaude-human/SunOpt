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
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-resident-job-batch.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FN_END"

H_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_BEGIN\nSUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_resident_job_batch(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const double* host_presorted_export_windows,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t search_iterations,\n    uint32_t jobs_per_launch,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_input_h2d_ms,\n    double* out_kernel_ms,\n    double* out_result_d2h_ms,\n    double* out_wall_ms\n);\n// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_END'
CUDA_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_BEGIN\n__global__ void\nsunopt_digital_resident_job_batch_kernel(\n    const double* __restrict__ export_power,\n    const double* __restrict__ demand_energy,\n    const double* __restrict__ presorted_export_windows,\n    size_t data_length,\n    const double* __restrict__ tank_parameters,\n    size_t tank_count,\n    const uint32_t* __restrict__ current_indices,\n    const uint32_t* __restrict__ horizon_steps,\n    const double* __restrict__ dynamic_values,\n    size_t step_count,\n    size_t first_job,\n    size_t batch_job_count,\n    uint32_t search_iterations,\n    double* __restrict__ cut_results,\n    uint32_t* __restrict__ status,\n    uint32_t* __restrict__ survival_evaluations\n) {\n    const size_t local_job =\n        static_cast<size_t>(\n            blockIdx.x\n        );\n\n    if (\n        local_job >=\n        batch_job_count\n    ) {\n        return;\n    }\n\n    const size_t job_index =\n        first_job +\n        local_job;\n\n    const size_t total_jobs =\n        step_count *\n        tank_count;\n\n    if (\n        job_index >=\n        total_jobs\n    ) {\n        return;\n    }\n\n    const size_t step_index =\n        job_index /\n        tank_count;\n\n    sunopt_digital_hp_presorted_warp_job(\n        export_power,\n        demand_energy,\n        presorted_export_windows +\n            step_index *\n            SUNOPT_DIGITAL_THREADS,\n        data_length,\n        tank_parameters,\n        tank_count,\n        current_indices,\n        horizon_steps,\n        dynamic_values,\n        step_count,\n        job_index,\n        search_iterations,\n        cut_results,\n        status,\n        survival_evaluations\n    );\n}\n\n\nextern "C" int sunopt_cuda_run_digital_resident_job_batch(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const double* host_presorted_export_windows,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t search_iterations,\n    uint32_t jobs_per_launch,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_input_h2d_ms,\n    double* out_kernel_ms,\n    double* out_result_d2h_ms,\n    double* out_wall_ms\n) {\n    using Clock =\n        std::chrono::steady_clock;\n\n    if (\n        host_export_power == nullptr ||\n        host_demand_energy == nullptr ||\n        host_tank_parameters == nullptr ||\n        host_presorted_export_windows == nullptr ||\n        host_current_indices == nullptr ||\n        host_horizon_steps == nullptr ||\n        host_dynamic_values == nullptr ||\n        host_cut_results == nullptr ||\n        host_status == nullptr ||\n        host_survival_evaluations == nullptr ||\n        out_input_h2d_ms == nullptr ||\n        out_kernel_ms == nullptr ||\n        out_result_d2h_ms == nullptr ||\n        out_wall_ms == nullptr ||\n        data_length == 0 ||\n        tank_count == 0 ||\n        tank_count > 32 ||\n        step_count == 0 ||\n        jobs_per_launch == 0 ||\n        repetitions == 0\n    ) {\n        return static_cast<int>(\n            cudaErrorInvalidValue\n        );\n    }\n\n    if (\n        step_count >\n        SIZE_MAX /\n        tank_count\n    ) {\n        return static_cast<int>(\n            cudaErrorInvalidValue\n        );\n    }\n\n    const size_t total_jobs =\n        step_count *\n        tank_count;\n\n    cudaError_t err =\n        cudaSetDevice(\n            device_id\n        );\n\n    if (\n        err !=\n        cudaSuccess\n    ) {\n        return static_cast<int>(\n            err\n        );\n    }\n\n    cudaStream_t stream =\n        nullptr;\n\n    cudaEvent_t start_event =\n        nullptr;\n\n    cudaEvent_t stop_event =\n        nullptr;\n\n    double* d_export =\n        nullptr;\n\n    double* d_demand =\n        nullptr;\n\n    double* d_tanks =\n        nullptr;\n\n    double* d_presorted =\n        nullptr;\n\n    uint32_t* d_current =\n        nullptr;\n\n    uint32_t* d_horizon =\n        nullptr;\n\n    double* d_dynamic =\n        nullptr;\n\n    double* d_cuts =\n        nullptr;\n\n    uint32_t* d_status =\n        nullptr;\n\n    uint32_t* d_evaluations =\n        nullptr;\n\n    const size_t annual_bytes =\n        data_length *\n        sizeof(double);\n\n    const size_t tank_bytes =\n        tank_count *\n        7 *\n        sizeof(double);\n\n    const size_t presorted_bytes =\n        step_count *\n        SUNOPT_DIGITAL_THREADS *\n        sizeof(double);\n\n    const size_t step_u32_bytes =\n        step_count *\n        sizeof(uint32_t);\n\n    const size_t dynamic_bytes =\n        total_jobs *\n        3 *\n        sizeof(double);\n\n    const size_t cut_bytes =\n        total_jobs *\n        sizeof(double);\n\n    const size_t u32_job_bytes =\n        total_jobs *\n        sizeof(uint32_t);\n\n    auto cleanup =\n        [&]() {\n            if (d_evaluations) cudaFree(d_evaluations);\n            if (d_status) cudaFree(d_status);\n            if (d_cuts) cudaFree(d_cuts);\n            if (d_dynamic) cudaFree(d_dynamic);\n            if (d_horizon) cudaFree(d_horizon);\n            if (d_current) cudaFree(d_current);\n            if (d_presorted) cudaFree(d_presorted);\n            if (d_tanks) cudaFree(d_tanks);\n            if (d_demand) cudaFree(d_demand);\n            if (d_export) cudaFree(d_export);\n\n            if (stop_event) cudaEventDestroy(stop_event);\n            if (start_event) cudaEventDestroy(start_event);\n            if (stream) cudaStreamDestroy(stream);\n        };\n\n#define SUNOPT_JOB_BATCH_CHECK(call) \\\n    do { \\\n        err = (call); \\\n        if (err != cudaSuccess) { \\\n            cleanup(); \\\n            return static_cast<int>(err); \\\n        } \\\n    } while (false)\n\n#define SUNOPT_JOB_BATCH_MALLOC(ptr, bytes) \\\n    SUNOPT_JOB_BATCH_CHECK( \\\n        cudaMalloc( \\\n            reinterpret_cast<void**>(&(ptr)), \\\n            (bytes) \\\n        ) \\\n    )\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaStreamCreateWithFlags(\n            &stream,\n            cudaStreamNonBlocking\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaEventCreate(\n            &start_event\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaEventCreate(\n            &stop_event\n        )\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_export,\n        annual_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_demand,\n        annual_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_tanks,\n        tank_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_presorted,\n        presorted_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_current,\n        step_u32_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_horizon,\n        step_u32_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_dynamic,\n        dynamic_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_cuts,\n        cut_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_status,\n        u32_job_bytes\n    );\n\n    SUNOPT_JOB_BATCH_MALLOC(\n        d_evaluations,\n        u32_job_bytes\n    );\n\n#undef SUNOPT_JOB_BATCH_MALLOC\n\n    const auto h2d_start =\n        Clock::now();\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_export,\n            host_export_power,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_demand,\n            host_demand_energy,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_tanks,\n            host_tank_parameters,\n            tank_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_presorted,\n            host_presorted_export_windows,\n            presorted_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_current,\n            host_current_indices,\n            step_u32_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_horizon,\n            host_horizon_steps,\n            step_u32_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            d_dynamic,\n            host_dynamic_values,\n            dynamic_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    const auto h2d_stop =\n        Clock::now();\n\n    *out_input_h2d_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            h2d_stop -\n            h2d_start\n        ).count();\n\n    const auto wall_start =\n        Clock::now();\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaEventRecord(\n            start_event,\n            stream\n        )\n    );\n\n    const size_t launch_jobs =\n        static_cast<size_t>(\n            jobs_per_launch\n        );\n\n    for (\n        uint32_t repetition = 0;\n        repetition <\n            repetitions;\n        ++repetition\n    ) {\n        for (\n            size_t first_job = 0;\n            first_job <\n                total_jobs;\n            first_job +=\n                launch_jobs\n        ) {\n            const size_t current_count =\n                min(\n                    launch_jobs,\n                    total_jobs -\n                        first_job\n                );\n\n            sunopt_digital_resident_job_batch_kernel<<<\n                static_cast<unsigned int>(\n                    current_count\n                ),\n                SUNOPT_DIGITAL_THREADS,\n                0,\n                stream\n            >>>(\n                d_export,\n                d_demand,\n                d_presorted,\n                data_length,\n                d_tanks,\n                tank_count,\n                d_current,\n                d_horizon,\n                d_dynamic,\n                step_count,\n                first_job,\n                current_count,\n                search_iterations,\n                d_cuts,\n                d_status,\n                d_evaluations\n            );\n\n            SUNOPT_JOB_BATCH_CHECK(\n                cudaGetLastError()\n            );\n        }\n    }\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaEventRecord(\n            stop_event,\n            stream\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaEventSynchronize(\n            stop_event\n        )\n    );\n\n    float kernel_total_ms =\n        0.0f;\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaEventElapsedTime(\n            &kernel_total_ms,\n            start_event,\n            stop_event\n        )\n    );\n\n    *out_kernel_ms =\n        static_cast<double>(\n            kernel_total_ms\n        ) /\n        static_cast<double>(\n            repetitions\n        );\n\n    const auto d2h_start =\n        Clock::now();\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            host_cut_results,\n            d_cuts,\n            cut_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            host_status,\n            d_status,\n            u32_job_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    SUNOPT_JOB_BATCH_CHECK(\n        cudaMemcpy(\n            host_survival_evaluations,\n            d_evaluations,\n            u32_job_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    const auto d2h_stop =\n        Clock::now();\n\n    *out_result_d2h_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            d2h_stop -\n            d2h_start\n        ).count();\n\n    const auto wall_stop =\n        Clock::now();\n\n    *out_wall_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            wall_stop -\n            wall_start\n        ).count() /\n        static_cast<double>(\n            repetitions\n        );\n\n#undef SUNOPT_JOB_BATCH_CHECK\n\n    cleanup();\n\n    return static_cast<int>(\n        cudaSuccess\n    );\n}\n// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_END'
RS_FFI_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FFI_BEGIN\n    fn sunopt_cuda_run_digital_resident_job_batch(\n        device_id: c_int,\n        host_export_power: *const f64,\n        host_demand_energy: *const f64,\n        data_length: usize,\n        host_tank_parameters: *const f64,\n        tank_count: usize,\n        host_presorted_export_windows: *const f64,\n        host_current_indices: *const u32,\n        host_horizon_steps: *const u32,\n        host_dynamic_values: *const f64,\n        step_count: usize,\n        search_iterations: u32,\n        jobs_per_launch: u32,\n        repetitions: u32,\n        host_cut_results: *mut f64,\n        host_status: *mut u32,\n        host_survival_evaluations: *mut u32,\n        out_input_h2d_ms: *mut f64,\n        out_kernel_ms: *mut f64,\n        out_result_d2h_ms: *mut f64,\n        out_wall_ms: *mut f64,\n    ) -> c_int;\n    // SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FFI_END'
RS_RESULT_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_RESULT_BEGIN\n#[napi(object)]\npub struct BlackwellDigitalResidentJobBatchVariant {\n    pub jobs_per_launch: u32,\n    pub launches_per_year: u32,\n    pub input_h2d_ms: f64,\n    pub kernel_ms: f64,\n    pub result_d2h_ms: f64,\n    pub wall_ms: f64,\n    pub speedup_vs_chunk_28: f64,\n    pub speedup_vs_lockstep: f64,\n    pub reduction_vs_lockstep_percent: f64,\n    pub status_mismatches: u32,\n    pub cut_mismatches: u32,\n    pub maximum_cut_error: f64,\n    pub evaluation_mismatches: u32,\n    pub checksum_hex: String,\n}\n\n\n#[napi(object)]\npub struct BlackwellDigitalResidentJobBatchResult {\n    pub snapshot_path: String,\n    pub device_id: i32,\n    pub repetitions: u32,\n    pub search_iterations: u32,\n    pub annual_step_count: u32,\n    pub tank_count: u32,\n    pub causal_jobs: u32,\n    pub presorted_mib: f64,\n    pub lockstep_wall_ms: f64,\n    pub chunk_28_wall_ms: f64,\n    pub reference_checksum_hex: String,\n    pub variants: Vec<BlackwellDigitalResidentJobBatchVariant>,\n}\n// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_RESULT_END'
RS_FN_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FN_BEGIN\n#[napi]\npub fn run_blackwell_digital_resident_job_batch(\n    snapshot_path: Option<String>,\n    repetitions: Option<u32>,\n    device_id: Option<i32>,\n) -> Result<BlackwellDigitalResidentJobBatchResult> {\n    let path =\n        snapshot_path\n            .map(PathBuf::from)\n            .unwrap_or_else(\n                blackwell_default_snapshot_path,\n            );\n\n    let snapshot =\n        read_blackwell_snapshot(\n            &path,\n        )\n        .map_err(\n            |error| {\n                Error::from_reason(\n                    format!(\n                        "Unable to read Blackwell benchmark snapshot {}: {error}",\n                        path.display(),\n                    ),\n                )\n            },\n        )?;\n\n    if (\n        snapshot.steps.is_empty() ||\n        snapshot.tank_count == 0 ||\n        snapshot.export_power.is_empty() ||\n        snapshot.demand_energy.len() !=\n            snapshot.export_power.len() ||\n        snapshot.steps.len() !=\n            snapshot.data_length\n    ) {\n        return Err(\n            Error::from_reason(\n                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",\n            ),\n        );\n    }\n\n    let selected_device =\n        device_id.unwrap_or(0);\n\n    let repetition_count =\n        repetitions\n            .unwrap_or(1)\n            .clamp(\n                1,\n                3,\n            );\n\n    let search_iterations =\n        8_u32;\n\n    let tanks =\n        (\n            0..\n            snapshot.tank_count\n        )\n        .map(\n            |tank| {\n                research_tank_from_packed(\n                    &snapshot.packed_config,\n                    tank,\n                )\n            },\n        )\n        .collect::<Vec<_>>();\n\n    let mut packed_tanks =\n        Vec::<f64>::with_capacity(\n            snapshot.tank_count *\n            7,\n        );\n\n    for tank in\n        &tanks\n    {\n        let pmax =\n            tank.maximum_heat_pump_electrical_power;\n\n        packed_tanks.push(\n            tank.target_storage_energy,\n        );\n\n        packed_tanks.push(\n            pmax,\n        );\n\n        packed_tanks.push(\n            pmax *\n            tank.cop *\n            tank.time_step_hours,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_retention,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_intercept,\n        );\n\n        packed_tanks.push(\n            if tank.affine_supported {\n                1.0\n            } else {\n                0.0\n            },\n        );\n\n        packed_tanks.push(\n            pmax *\n            tank.time_step_hours,\n        );\n    }\n\n    let step_count =\n        snapshot.steps.len();\n\n    let tank_count =\n        snapshot.tank_count;\n\n    let job_count =\n        step_count\n            .checked_mul(\n                tank_count,\n            )\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Resident-job-batch job count overflow.",\n                    )\n                },\n            )?;\n\n    let mut current_indices =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut horizon_steps =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut dynamic_flat =\n        Vec::<f64>::with_capacity(\n            job_count *\n            3,\n        );\n\n    for step in\n        &snapshot.steps\n    {\n        if (\n            step.horizon_steps ==\n                0 ||\n            step.horizon_steps >\n                256\n        ) {\n            return Err(\n                Error::from_reason(\n                    format!(\n                        "Resident-job-batch supports horizon 1..=256; got {}.",\n                        step.horizon_steps,\n                    ),\n                ),\n            );\n        }\n\n        current_indices.push(\n            step.current_index,\n        );\n\n        horizon_steps.push(\n            step.horizon_steps,\n        );\n\n        for tank_index in\n            0..\n            tank_count\n        {\n            let base =\n                tank_index *\n                BLACKWELL_DYNAMIC_STRIDE;\n\n            dynamic_flat.push(\n                step.dynamic[\n                    base + 0\n                ],\n            );\n\n            dynamic_flat.push(\n                step.dynamic[\n                    base + 1\n                ],\n            );\n\n            dynamic_flat.push(\n                step.dynamic[\n                    base + 2\n                ],\n            );\n        }\n    }\n\n    let presorted_count =\n        step_count\n            .checked_mul(\n                256\n            )\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Presorted export matrix size overflow.",\n                    )\n                },\n            )?;\n\n    let mut presorted =\n        vec![\n            f64::NEG_INFINITY;\n            presorted_count\n        ];\n\n    for (\n        step_index,\n        step,\n    ) in\n        snapshot.steps\n            .iter()\n            .enumerate()\n    {\n        let horizon =\n            step.horizon_steps\n                as usize;\n\n        let base =\n            step_index *\n            256;\n\n        let first =\n            base +\n            (\n                256 -\n                horizon\n            );\n\n        for offset in\n            0..\n            horizon\n        {\n            let index =\n                (\n                    step.current_index\n                        as usize +\n                    offset\n                ) %\n                snapshot.export_power\n                    .len();\n\n            presorted[\n                first +\n                offset\n            ] =\n                snapshot.export_power[\n                    index\n                ];\n        }\n\n        presorted[\n            first..\n            base + 256\n        ]\n        .sort_by(\n            |a, b| {\n                a.partial_cmp(\n                    b\n                )\n                .unwrap_or(\n                    std::cmp::Ordering::Equal,\n                )\n            },\n        );\n    }\n\n    let presorted_mib =\n        (\n            presorted.len() *\n            std::mem::size_of::<f64>()\n        ) as f64 /\n        1048576.0;\n\n    /*\n     * Reference = current B8 resident chunk=1 behaviour.\n     * This has the same 28 jobs per launch as the true lockstep but with\n     * all annual inputs/results resident.\n     */\n    let mut reference_cuts =\n        vec![\n            0.0_f64;\n            job_count\n        ];\n\n    let mut reference_status =\n        vec![\n            0_u32;\n            job_count\n        ];\n\n    let mut reference_evaluations =\n        vec![\n            0_u32;\n            job_count\n        ];\n\n    let mut ref_input_h2d_ms =\n        0.0_f64;\n\n    let mut ref_kernel_ms =\n        0.0_f64;\n\n    let mut ref_result_d2h_ms =\n        0.0_f64;\n\n    let mut ref_wall_ms =\n        0.0_f64;\n\n    let reference_code =\n        unsafe {\n            sunopt_cuda_run_digital_resident_chunk_replay(\n                selected_device,\n                snapshot.export_power\n                    .as_ptr(),\n                snapshot.demand_energy\n                    .as_ptr(),\n                snapshot.export_power\n                    .len(),\n                packed_tanks\n                    .as_ptr(),\n                tank_count,\n                presorted\n                    .as_ptr(),\n                current_indices\n                    .as_ptr(),\n                horizon_steps\n                    .as_ptr(),\n                dynamic_flat\n                    .as_ptr(),\n                step_count,\n                search_iterations,\n                1,\n                repetition_count,\n                reference_cuts\n                    .as_mut_ptr(),\n                reference_status\n                    .as_mut_ptr(),\n                reference_evaluations\n                    .as_mut_ptr(),\n                &mut ref_input_h2d_ms,\n                &mut ref_kernel_ms,\n                &mut ref_result_d2h_ms,\n                &mut ref_wall_ms,\n            )\n        };\n\n    if (\n        reference_code !=\n        0\n    ) {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_resident_chunk_replay",\n                reference_code,\n            ),\n        );\n    }\n\n    /*\n     * For context, retain the most recently measured lockstep shape by\n     * running the actual pre-sorted warp lockstep once in this same process.\n     */\n    let mut lock_cuts =\n        vec![\n            0.0_f64;\n            job_count\n        ];\n\n    let mut lock_status =\n        vec![\n            0_u32;\n            job_count\n        ];\n\n    let mut lock_eval =\n        vec![\n            0_u32;\n            job_count\n        ];\n\n    let mut lock_static_h2d =\n        0.0_f64;\n\n    let mut lock_presorted_h2d =\n        0.0_f64;\n\n    let mut lock_gpu_total =\n        0.0_f64;\n\n    let mut lock_host_wait =\n        0.0_f64;\n\n    let mut lock_wall_raw =\n        0.0_f64;\n\n    let lock_code =\n        unsafe {\n            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(\n                selected_device,\n                snapshot.export_power\n                    .as_ptr(),\n                snapshot.demand_energy\n                    .as_ptr(),\n                snapshot.export_power\n                    .len(),\n                packed_tanks\n                    .as_ptr(),\n                tank_count,\n                presorted\n                    .as_ptr(),\n                current_indices\n                    .as_ptr(),\n                horizon_steps\n                    .as_ptr(),\n                dynamic_flat\n                    .as_ptr(),\n                step_count,\n                search_iterations,\n                repetition_count,\n                lock_cuts\n                    .as_mut_ptr(),\n                lock_status\n                    .as_mut_ptr(),\n                lock_eval\n                    .as_mut_ptr(),\n                &mut lock_static_h2d,\n                &mut lock_presorted_h2d,\n                &mut lock_gpu_total,\n                &mut lock_host_wait,\n                &mut lock_wall_raw,\n            )\n        };\n\n    if (\n        lock_code !=\n        0\n    ) {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",\n                lock_code,\n            ),\n        );\n    }\n\n    let lockstep_wall_ms =\n        lock_wall_raw /\n        repetition_count\n            as f64;\n\n    let mut reference_checksum =\n        1469598103934665603u64;\n\n    for value in\n        &reference_cuts\n    {\n        blackwell_fnv1a_update(\n            &mut reference_checksum,\n            *value,\n        );\n    }\n\n    /*\n     * 28 jobs == one annual time slice. Larger values deliberately expose\n     * more independent snapshot jobs to the GPU. They are a throughput lower\n     * bound, NOT a production-causal execution order.\n     */\n    let batch_sizes:\n        [u32; 7] =\n        [\n            28,\n            112,\n            448,\n            1792,\n            7168,\n            28672,\n            114688,\n        ];\n\n    let mut variants =\n        Vec::<BlackwellDigitalResidentJobBatchVariant>::with_capacity(\n            batch_sizes.len(),\n        );\n\n    for jobs_per_launch in\n        batch_sizes\n    {\n        let mut cuts =\n            vec![\n                0.0_f64;\n                job_count\n            ];\n\n        let mut status =\n            vec![\n                0_u32;\n                job_count\n            ];\n\n        let mut evaluations =\n            vec![\n                0_u32;\n                job_count\n            ];\n\n        let mut input_h2d_ms =\n            0.0_f64;\n\n        let mut kernel_ms =\n            0.0_f64;\n\n        let mut result_d2h_ms =\n            0.0_f64;\n\n        let mut wall_ms =\n            0.0_f64;\n\n        let code =\n            unsafe {\n                sunopt_cuda_run_digital_resident_job_batch(\n                    selected_device,\n                    snapshot.export_power\n                        .as_ptr(),\n                    snapshot.demand_energy\n                        .as_ptr(),\n                    snapshot.export_power\n                        .len(),\n                    packed_tanks\n                        .as_ptr(),\n                    tank_count,\n                    presorted\n                        .as_ptr(),\n                    current_indices\n                        .as_ptr(),\n                    horizon_steps\n                        .as_ptr(),\n                    dynamic_flat\n                        .as_ptr(),\n                    step_count,\n                    search_iterations,\n                    jobs_per_launch,\n                    repetition_count,\n                    cuts\n                        .as_mut_ptr(),\n                    status\n                        .as_mut_ptr(),\n                    evaluations\n                        .as_mut_ptr(),\n                    &mut input_h2d_ms,\n                    &mut kernel_ms,\n                    &mut result_d2h_ms,\n                    &mut wall_ms,\n                )\n            };\n\n        if (\n            code !=\n            0\n        ) {\n            return Err(\n                cuda_error(\n                    "sunopt_cuda_run_digital_resident_job_batch",\n                    code,\n                ),\n            );\n        }\n\n        let mut status_mismatches =\n            0_u32;\n\n        let mut cut_mismatches =\n            0_u32;\n\n        let mut maximum_cut_error =\n            0.0_f64;\n\n        let mut evaluation_mismatches =\n            0_u32;\n\n        for job in\n            0..\n            job_count\n        {\n            if (\n                status[\n                    job\n                ] !=\n                reference_status[\n                    job\n                ]\n            ) {\n                status_mismatches +=\n                    1;\n            }\n\n            if (\n                evaluations[\n                    job\n                ] !=\n                reference_evaluations[\n                    job\n                ]\n            ) {\n                evaluation_mismatches +=\n                    1;\n            }\n\n            let error =\n                (\n                    cuts[\n                        job\n                    ] -\n                    reference_cuts[\n                        job\n                    ]\n                )\n                .abs();\n\n            maximum_cut_error =\n                maximum_cut_error.max(\n                    error,\n                );\n\n            if (\n                error >\n                1.0e-9\n            ) {\n                cut_mismatches +=\n                    1;\n            }\n        }\n\n        let mut checksum =\n            1469598103934665603u64;\n\n        for value in\n            &cuts\n        {\n            blackwell_fnv1a_update(\n                &mut checksum,\n                *value,\n            );\n        }\n\n        let speedup_vs_chunk_28 =\n            if (\n                wall_ms >\n                0.0\n            ) {\n                ref_wall_ms /\n                wall_ms\n            } else {\n                f64::INFINITY\n            };\n\n        let speedup_vs_lockstep =\n            if (\n                wall_ms >\n                0.0\n            ) {\n                lockstep_wall_ms /\n                wall_ms\n            } else {\n                f64::INFINITY\n            };\n\n        let reduction_vs_lockstep_percent =\n            if (\n                lockstep_wall_ms >\n                0.0\n            ) {\n                (\n                    lockstep_wall_ms -\n                    wall_ms\n                ) /\n                lockstep_wall_ms *\n                100.0\n            } else {\n                0.0\n            };\n\n        let launches_per_year =\n            (\n                (\n                    job_count\n                        as u64 +\n                    jobs_per_launch\n                        as u64 -\n                    1\n                ) /\n                jobs_per_launch\n                    as u64\n            ) as u32;\n\n        variants.push(\n            BlackwellDigitalResidentJobBatchVariant {\n                jobs_per_launch,\n                launches_per_year,\n                input_h2d_ms,\n                kernel_ms,\n                result_d2h_ms,\n                wall_ms,\n                speedup_vs_chunk_28,\n                speedup_vs_lockstep,\n                reduction_vs_lockstep_percent,\n                status_mismatches,\n                cut_mismatches,\n                maximum_cut_error,\n                evaluation_mismatches,\n                checksum_hex:\n                    format!(\n                        "{checksum:016x}",\n                    ),\n            },\n        );\n    }\n\n    Ok(\n        BlackwellDigitalResidentJobBatchResult {\n            snapshot_path:\n                path.display()\n                    .to_string(),\n\n            device_id:\n                selected_device,\n\n            repetitions:\n                repetition_count,\n\n            search_iterations,\n\n            annual_step_count:\n                step_count\n                    as u32,\n\n            tank_count:\n                tank_count\n                    as u32,\n\n            causal_jobs:\n                job_count\n                    as u32,\n\n            presorted_mib,\n\n            lockstep_wall_ms,\n\n            chunk_28_wall_ms:\n                ref_wall_ms,\n\n            reference_checksum_hex:\n                format!(\n                    "{reference_checksum:016x}",\n                ),\n\n            variants,\n        },\n    )\n}\n// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FN_END'
MJS = 'import { createRequire } from "node:module";\nimport path from "node:path";\nimport { fileURLToPath } from "node:url";\nimport { writeFileSync } from "node:fs";\n\nconst require = createRequire(import.meta.url);\nconst here = path.dirname(fileURLToPath(import.meta.url));\nconst root = path.resolve(here, "..");\n\nconst addon = require(\n  path.join(\n    root,\n    "sunopt_node.win32-x64-msvc.node",\n  ),\n);\n\nconst snapshot = process.argv[2]\n  ? path.resolve(process.argv[2])\n  : path.join(\n      root,\n      "blackwell-benchmark.bin",\n    );\n\nconst repetitions =\n  Number.parseInt(\n    process.argv[3] ?? "1",\n    10,\n  );\n\nconst deviceId =\n  Number.parseInt(\n    process.argv[4] ?? "0",\n    10,\n  );\n\nconst result =\n  addon.runBlackwellDigitalResidentJobBatch(\n    snapshot,\n    repetitions,\n    deviceId,\n  );\n\nconsole.log(\n  `SUNOPT_DIGITAL_RESIDENT_JOB_BATCH_JSON=${JSON.stringify(result)}`,\n);\n\nconsole.log();\nconsole.log(\n  "--- DIGITAL RESIDENT JOB-PARALLEL LOWER BOUND ---",\n);\n\nconsole.log(`steps=${result.annualStepCount}`);\nconsole.log(`tanks=${result.tankCount}`);\nconsole.log(`jobs=${result.causalJobs}`);\nconsole.log(`searchIterations=${result.searchIterations}`);\nconsole.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);\n\nconsole.log();\nconsole.log("REFERENCES");\nconsole.log(`lockstepWallMs=${result.lockstepWallMs.toFixed(3)}`);\nconsole.log(`resident28WallMs=${result.chunk28WallMs.toFixed(3)}`);\nconsole.log(`checksum=${result.referenceChecksumHex}`);\n\nconsole.log();\nconsole.log(\n  "jobs/launch launches H2Dms kernelMs D2Hms wallMs vs28 vsLock reduction status cut eval maxErr",\n);\n\nfor (const v of result.variants) {\n  console.log(\n    [\n      String(v.jobsPerLaunch).padEnd(11),\n      String(v.launchesPerYear).padEnd(8),\n      v.inputH2DMs.toFixed(3).padEnd(7),\n      v.kernelMs.toFixed(3).padEnd(9),\n      v.resultD2HMs.toFixed(3).padEnd(7),\n      v.wallMs.toFixed(3).padEnd(9),\n      `${v.speedupVsChunk28.toFixed(3)}x`.padEnd(7),\n      `${v.speedupVsLockstep.toFixed(3)}x`.padEnd(8),\n      `${v.reductionVsLockstepPercent.toFixed(2)}%`.padEnd(9),\n      String(v.statusMismatches).padEnd(6),\n      String(v.cutMismatches).padEnd(4),\n      String(v.evaluationMismatches).padEnd(4),\n      v.maximumCutError.toExponential(3),\n    ].join(" "),\n  );\n}\n\nconst report =\n  path.join(\n    root,\n    "blackwell-digital-resident-job-batch.json",\n  );\n\nwriteFileSync(\n  report,\n  JSON.stringify(\n    result,\n    null,\n    2,\n  ) + "\\n",\n  "utf8",\n);\n\nconsole.log();\nconsole.log(`report=${report}`);'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(
            encoding="utf-8",
        ).strip().lower()

        if HASH_RE.fullmatch(
            value
        ):
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

        if HASH_RE.fullmatch(
            value
        ):
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

    start = source.index(
        begin
    )

    finish = source.index(
        end,
        start,
    ) + len(
        end
    )

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
    if (
        begin in source or
        end in source
    ):
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

        if (
            position >=
            0
        ):
            return (
                source[:position] +
                block +
                "\n\n\n" +
                source[position:]
            )

    raise RuntimeError(
        "Could not find insertion anchor. Tried: " +
        ", ".join(
            repr(
                value
            )
            for value in
            anchors
        )
    )


def patch_header(
    source: str,
) -> str:
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN",
        "sunopt_cuda_run_digital_resident_chunk_replay(",
    ):
        if (
            token not in
            source
        ):
            raise RuntimeError(
                f"Expected installed resident-chunk header missing {token!r}"
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


def patch_cuda(
    source: str,
) -> str:
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN",
        "sunopt_digital_hp_presorted_warp_job(",
        "sunopt_cuda_run_digital_resident_chunk_replay(",
        "SUNOPT_DIGITAL_THREADS",
    ):
        if (
            token not in
            source
        ):
            raise RuntimeError(
                f"Expected current resident-chunk CUDA generation missing {token!r}"
            )

    return insert_or_replace(
        source,
        CU_BEGIN,
        CU_END,
        CUDA_BLOCK,
        [
            'extern "C" int sunopt_cuda_get_profile(',
            'extern "C" const char* sunopt_cuda_error_string(',
        ],
    )


def patch_rust(
    source: str,
) -> str:
    for token in (
        "fn sunopt_cuda_run_digital_resident_chunk_replay(",
        "fn sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(",
        "fn research_tank_from_packed(",
        "fn blackwell_fnv1a_update(",
        "BLACKWELL_DYNAMIC_STRIDE",
    ):
        if (
            token not in
            source
        ):
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
    for path in (
        H_PATH,
        CU_PATH,
        RS_PATH,
    ):
        if (
            not path.is_file()
        ):
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
            h.encode(
                "utf-8"
            ),

        CU_REL:
            cu.encode(
                "utf-8"
            ),

        RS_REL:
            rs.encode(
                "utf-8"
            ),

        MJS_REL:
            (
                MJS +
                "\n"
            ).encode(
                "utf-8"
            ),
    }

    parent = read_parent_hash()

    manifest = {
        "format":
            2,

        "parent_sha256":
            parent,

        "files": {
            name:
                sha256_bytes(
                    data
                )
            for name, data in
            payloads.items()
        },

        "delete":
            [],
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) +
        "\n"
    ).encode(
        "utf-8"
    )

    release = sha256_bytes(
        manifest_bytes
    )

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        INBOX /
        "20261109-blackwell-digital-resident-job-batch.zip"
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
            (
                release +
                "\n"
            ).encode(
                "ascii"
            ),
        )

        for name, data in payloads.items():
            archive.writestr(
                name,
                data,
            )

    print(
        "SunOpt digital resident job-batch benchmark package created"
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
        "  production worker: UNCHANGED"
    )

    print(
        "  purpose: measure the job-parallel GPU throughput lower bound"
    )

    print()
    print(
        "Next:"
    )

    print(
        r"  python .\tools\update_watcher.py --once"
    )

    print(
        r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-resident-job-batch.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node'
    )

    return output


if __name__ == "__main__":
    build_package()
