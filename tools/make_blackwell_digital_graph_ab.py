#!/usr/bin/env python3
# tools/make_blackwell_digital_graph_ab.py
#
# A/B benchmark:
#   A = current direct all-GPU digital lockstep
#   B = same kernel in a reusable CUDA Graph with pinned staging buffers
#
# Production worker remains untouched.

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
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-graph-ab.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_END"
CU_BEGIN = H_BEGIN
CU_END = H_END
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FN_END"

H_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_BEGIN\nSUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_graph_candidate(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t search_iterations,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_static_h2d_ms,\n    double* out_graph_build_ms,\n    double* out_gpu_total_ms,\n    double* out_host_wait_ms,\n    double* out_wall_ms\n);\n// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_END'
CU_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_BEGIN\nextern "C" int sunopt_cuda_run_digital_lockstep_graph_candidate(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t search_iterations,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_static_h2d_ms,\n    double* out_graph_build_ms,\n    double* out_gpu_total_ms,\n    double* out_host_wait_ms,\n    double* out_wall_ms\n) {\n    using Clock = std::chrono::steady_clock;\n\n    if (\n        host_export_power == nullptr ||\n        host_demand_energy == nullptr ||\n        host_tank_parameters == nullptr ||\n        host_current_indices == nullptr ||\n        host_horizon_steps == nullptr ||\n        host_dynamic_values == nullptr ||\n        host_cut_results == nullptr ||\n        host_status == nullptr ||\n        host_survival_evaluations == nullptr ||\n        out_static_h2d_ms == nullptr ||\n        out_graph_build_ms == nullptr ||\n        out_gpu_total_ms == nullptr ||\n        out_host_wait_ms == nullptr ||\n        out_wall_ms == nullptr ||\n        data_length == 0 ||\n        tank_count == 0 ||\n        tank_count > 32 ||\n        step_count == 0 ||\n        search_iterations == 0 ||\n        repetitions == 0\n    ) {\n        return static_cast<int>(cudaErrorInvalidValue);\n    }\n\n    cudaError_t err = cudaSetDevice(device_id);\n    if (err != cudaSuccess) {\n        return static_cast<int>(err);\n    }\n\n    cudaStream_t stream = nullptr;\n    cudaGraph_t graph = nullptr;\n    cudaGraphExec_t graph_exec = nullptr;\n    cudaEvent_t gpu_start = nullptr;\n    cudaEvent_t gpu_stop = nullptr;\n\n    double* d_export = nullptr;\n    double* d_demand = nullptr;\n    double* d_tanks = nullptr;\n    uint32_t* d_current = nullptr;\n    uint32_t* d_horizon = nullptr;\n    double* d_dynamic = nullptr;\n    double* d_cuts = nullptr;\n    uint32_t* d_status = nullptr;\n    uint32_t* d_evaluations = nullptr;\n\n    uint32_t* h_current = nullptr;\n    uint32_t* h_horizon = nullptr;\n    double* h_dynamic = nullptr;\n    double* h_cuts = nullptr;\n    uint32_t* h_status = nullptr;\n    uint32_t* h_evaluations = nullptr;\n\n    const size_t annual_bytes =\n        data_length * sizeof(double);\n\n    const size_t tank_bytes =\n        tank_count * 7 * sizeof(double);\n\n    const size_t dynamic_step_count =\n        tank_count * 3;\n\n    const size_t dynamic_step_bytes =\n        dynamic_step_count * sizeof(double);\n\n    const size_t cut_step_bytes =\n        tank_count * sizeof(double);\n\n    const size_t u32_step_bytes =\n        tank_count * sizeof(uint32_t);\n\n    auto cleanup = [&]() {\n        if (graph_exec) cudaGraphExecDestroy(graph_exec);\n        if (graph) cudaGraphDestroy(graph);\n\n        if (h_evaluations) cudaFreeHost(h_evaluations);\n        if (h_status) cudaFreeHost(h_status);\n        if (h_cuts) cudaFreeHost(h_cuts);\n        if (h_dynamic) cudaFreeHost(h_dynamic);\n        if (h_horizon) cudaFreeHost(h_horizon);\n        if (h_current) cudaFreeHost(h_current);\n\n        if (d_evaluations) cudaFree(d_evaluations);\n        if (d_status) cudaFree(d_status);\n        if (d_cuts) cudaFree(d_cuts);\n        if (d_dynamic) cudaFree(d_dynamic);\n        if (d_horizon) cudaFree(d_horizon);\n        if (d_current) cudaFree(d_current);\n        if (d_tanks) cudaFree(d_tanks);\n        if (d_demand) cudaFree(d_demand);\n        if (d_export) cudaFree(d_export);\n\n        if (gpu_stop) cudaEventDestroy(gpu_stop);\n        if (gpu_start) cudaEventDestroy(gpu_start);\n        if (stream) cudaStreamDestroy(stream);\n    };\n\n#define SUNOPT_GRAPH_CHECK(call) \\\n    do { \\\n        err = (call); \\\n        if (err != cudaSuccess) { \\\n            cleanup(); \\\n            return static_cast<int>(err); \\\n        } \\\n    } while (false)\n\n#define SUNOPT_GRAPH_MALLOC(ptr, bytes) \\\n    SUNOPT_GRAPH_CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)))\n\n#define SUNOPT_GRAPH_HOST_MALLOC(ptr, bytes) \\\n    SUNOPT_GRAPH_CHECK(cudaMallocHost(reinterpret_cast<void**>(&(ptr)), (bytes)))\n\n    SUNOPT_GRAPH_CHECK(\n        cudaStreamCreateWithFlags(\n            &stream,\n            cudaStreamNonBlocking\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(cudaEventCreate(&gpu_start));\n    SUNOPT_GRAPH_CHECK(cudaEventCreate(&gpu_stop));\n\n    SUNOPT_GRAPH_MALLOC(d_export, annual_bytes);\n    SUNOPT_GRAPH_MALLOC(d_demand, annual_bytes);\n    SUNOPT_GRAPH_MALLOC(d_tanks, tank_bytes);\n    SUNOPT_GRAPH_MALLOC(d_current, sizeof(uint32_t));\n    SUNOPT_GRAPH_MALLOC(d_horizon, sizeof(uint32_t));\n    SUNOPT_GRAPH_MALLOC(d_dynamic, dynamic_step_bytes);\n    SUNOPT_GRAPH_MALLOC(d_cuts, cut_step_bytes);\n    SUNOPT_GRAPH_MALLOC(d_status, u32_step_bytes);\n    SUNOPT_GRAPH_MALLOC(d_evaluations, u32_step_bytes);\n\n    SUNOPT_GRAPH_HOST_MALLOC(h_current, sizeof(uint32_t));\n    SUNOPT_GRAPH_HOST_MALLOC(h_horizon, sizeof(uint32_t));\n    SUNOPT_GRAPH_HOST_MALLOC(h_dynamic, dynamic_step_bytes);\n    SUNOPT_GRAPH_HOST_MALLOC(h_cuts, cut_step_bytes);\n    SUNOPT_GRAPH_HOST_MALLOC(h_status, u32_step_bytes);\n    SUNOPT_GRAPH_HOST_MALLOC(h_evaluations, u32_step_bytes);\n\n#undef SUNOPT_GRAPH_MALLOC\n#undef SUNOPT_GRAPH_HOST_MALLOC\n\n    const auto static_h2d_start = Clock::now();\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpy(\n            d_export,\n            host_export_power,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpy(\n            d_demand,\n            host_demand_energy,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpy(\n            d_tanks,\n            host_tank_parameters,\n            tank_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    const auto static_h2d_stop = Clock::now();\n\n    *out_static_h2d_ms =\n        std::chrono::duration<double, std::milli>(\n            static_h2d_stop - static_h2d_start\n        ).count();\n\n    const auto graph_build_start = Clock::now();\n\n    SUNOPT_GRAPH_CHECK(\n        cudaStreamBeginCapture(\n            stream,\n            cudaStreamCaptureModeThreadLocal\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpyAsync(\n            d_current,\n            h_current,\n            sizeof(uint32_t),\n            cudaMemcpyHostToDevice,\n            stream\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpyAsync(\n            d_horizon,\n            h_horizon,\n            sizeof(uint32_t),\n            cudaMemcpyHostToDevice,\n            stream\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpyAsync(\n            d_dynamic,\n            h_dynamic,\n            dynamic_step_bytes,\n            cudaMemcpyHostToDevice,\n            stream\n        )\n    );\n\n    sunopt_digital_hp_breakpoint_kernel<<<\n        static_cast<unsigned int>(tank_count),\n        SUNOPT_DIGITAL_THREADS,\n        0,\n        stream\n    >>>(\n        d_export,\n        d_demand,\n        data_length,\n        d_tanks,\n        tank_count,\n        d_current,\n        d_horizon,\n        d_dynamic,\n        1,\n        search_iterations,\n        d_cuts,\n        d_status,\n        d_evaluations\n    );\n\n    SUNOPT_GRAPH_CHECK(cudaGetLastError());\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpyAsync(\n            h_cuts,\n            d_cuts,\n            cut_step_bytes,\n            cudaMemcpyDeviceToHost,\n            stream\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpyAsync(\n            h_status,\n            d_status,\n            u32_step_bytes,\n            cudaMemcpyDeviceToHost,\n            stream\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaMemcpyAsync(\n            h_evaluations,\n            d_evaluations,\n            u32_step_bytes,\n            cudaMemcpyDeviceToHost,\n            stream\n        )\n    );\n\n    err = cudaStreamEndCapture(stream, &graph);\n    if (err != cudaSuccess) {\n        cleanup();\n        return static_cast<int>(err);\n    }\n\n    err = cudaGraphInstantiate(\n        &graph_exec,\n        graph,\n        nullptr,\n        nullptr,\n        0\n    );\n\n    if (err != cudaSuccess) {\n        cleanup();\n        return static_cast<int>(err);\n    }\n\n    SUNOPT_GRAPH_CHECK(cudaGraphDestroy(graph));\n    graph = nullptr;\n\n    const auto graph_build_stop = Clock::now();\n\n    *out_graph_build_ms =\n        std::chrono::duration<double, std::milli>(\n            graph_build_stop - graph_build_start\n        ).count();\n\n    double host_wait_ms = 0.0;\n\n    const auto wall_start = Clock::now();\n\n    SUNOPT_GRAPH_CHECK(\n        cudaEventRecord(\n            gpu_start,\n            stream\n        )\n    );\n\n    for (\n        uint32_t repetition = 0;\n        repetition < repetitions;\n        ++repetition\n    ) {\n        for (\n            size_t step = 0;\n            step < step_count;\n            ++step\n        ) {\n            const size_t dynamic_offset =\n                step * dynamic_step_count;\n\n            const size_t result_offset =\n                step * tank_count;\n\n            *h_current =\n                host_current_indices[step];\n\n            *h_horizon =\n                host_horizon_steps[step];\n\n            for (\n                size_t i = 0;\n                i < dynamic_step_count;\n                ++i\n            ) {\n                h_dynamic[i] =\n                    host_dynamic_values[\n                        dynamic_offset + i\n                    ];\n            }\n\n            SUNOPT_GRAPH_CHECK(\n                cudaGraphLaunch(\n                    graph_exec,\n                    stream\n                )\n            );\n\n            const auto wait_start =\n                Clock::now();\n\n            SUNOPT_GRAPH_CHECK(\n                cudaStreamSynchronize(\n                    stream\n                )\n            );\n\n            const auto wait_stop =\n                Clock::now();\n\n            host_wait_ms +=\n                std::chrono::duration<double, std::milli>(\n                    wait_stop - wait_start\n                ).count();\n\n            for (\n                size_t tank = 0;\n                tank < tank_count;\n                ++tank\n            ) {\n                host_cut_results[\n                    result_offset + tank\n                ] = h_cuts[tank];\n\n                host_status[\n                    result_offset + tank\n                ] = h_status[tank];\n\n                host_survival_evaluations[\n                    result_offset + tank\n                ] = h_evaluations[tank];\n            }\n        }\n    }\n\n    SUNOPT_GRAPH_CHECK(\n        cudaEventRecord(\n            gpu_stop,\n            stream\n        )\n    );\n\n    SUNOPT_GRAPH_CHECK(\n        cudaEventSynchronize(\n            gpu_stop\n        )\n    );\n\n    float gpu_total_ms = 0.0f;\n\n    SUNOPT_GRAPH_CHECK(\n        cudaEventElapsedTime(\n            &gpu_total_ms,\n            gpu_start,\n            gpu_stop\n        )\n    );\n\n    const auto wall_stop = Clock::now();\n\n    *out_gpu_total_ms =\n        static_cast<double>(gpu_total_ms);\n\n    *out_host_wait_ms =\n        host_wait_ms;\n\n    *out_wall_ms =\n        std::chrono::duration<double, std::milli>(\n            wall_stop - wall_start\n        ).count();\n\n#undef SUNOPT_GRAPH_CHECK\n\n    cleanup();\n\n    return static_cast<int>(cudaSuccess);\n}\n// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_END'
RS_FFI_BLOCK = '    // SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FFI_BEGIN\n    fn sunopt_cuda_run_digital_lockstep_graph_candidate(\n        device_id: c_int,\n        host_export_power: *const f64,\n        host_demand_energy: *const f64,\n        data_length: usize,\n        host_tank_parameters: *const f64,\n        tank_count: usize,\n        host_current_indices: *const u32,\n        host_horizon_steps: *const u32,\n        host_dynamic_values: *const f64,\n        step_count: usize,\n        search_iterations: u32,\n        repetitions: u32,\n        host_cut_results: *mut f64,\n        host_status: *mut u32,\n        host_survival_evaluations: *mut u32,\n        out_static_h2d_ms: *mut f64,\n        out_graph_build_ms: *mut f64,\n        out_gpu_total_ms: *mut f64,\n        out_host_wait_ms: *mut f64,\n        out_wall_ms: *mut f64,\n    ) -> c_int;\n    // SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FFI_END'
RS_RESULT_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_RESULT_BEGIN\n#[napi(object)]\npub struct BlackwellDigitalGraphAbResult {\n    pub snapshot_path: String,\n    pub device_id: i32,\n    pub repetitions: u32,\n    pub search_iterations: u32,\n    pub annual_step_count: u32,\n    pub tank_count: u32,\n    pub causal_jobs: u32,\n    pub gpu_compressed_jobs: u32,\n    pub gpu_serial_fallback_jobs: u32,\n\n    pub direct_static_h2d_ms: f64,\n    pub direct_gpu_total_ms: f64,\n    pub direct_host_wait_ms: f64,\n    pub direct_wall_ms: f64,\n\n    pub graph_static_h2d_ms: f64,\n    pub graph_build_ms: f64,\n    pub graph_gpu_total_ms: f64,\n    pub graph_host_wait_ms: f64,\n    pub graph_wall_ms: f64,\n\n    pub wall_saving_ms: f64,\n    pub wall_speedup: f64,\n    pub wall_reduction_percent: f64,\n\n    pub direct_graph_status_mismatches: u32,\n    pub direct_graph_cut_mismatches: u32,\n    pub direct_graph_max_cut_error: f64,\n\n    pub parity_sample_steps: u32,\n    pub parity_status_mismatches: u32,\n    pub parity_cut_mismatches: u32,\n    pub parity_max_cut_error: f64,\n\n    pub checksum_hex: String,\n}\n// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_RESULT_END'
RS_FN_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FN_BEGIN\n#[napi]\npub fn run_blackwell_digital_graph_ab(\n    snapshot_path: Option<String>,\n    repetitions: Option<u32>,\n    device_id: Option<i32>,\n) -> Result<BlackwellDigitalGraphAbResult> {\n    let path =\n        snapshot_path\n            .map(PathBuf::from)\n            .unwrap_or_else(\n                blackwell_default_snapshot_path,\n            );\n\n    let snapshot =\n        read_blackwell_snapshot(\n            &path,\n        )\n        .map_err(\n            |error| {\n                Error::from_reason(\n                    format!(\n                        "Unable to read Blackwell benchmark snapshot {}: {error}",\n                        path.display(),\n                    ),\n                )\n            },\n        )?;\n\n    if (\n        snapshot.steps.is_empty() ||\n        snapshot.tank_count == 0 ||\n        snapshot.export_power.is_empty() ||\n        snapshot.demand_energy.len() !=\n            snapshot.export_power.len() ||\n        snapshot.steps.len() !=\n            snapshot.data_length\n    ) {\n        return Err(\n            Error::from_reason(\n                "Blackwell snapshot is empty or not a complete annual lockstep snapshot.",\n            ),\n        );\n    }\n\n    let selected_device =\n        device_id.unwrap_or(0);\n\n    let repetition_count =\n        repetitions\n            .unwrap_or(1)\n            .clamp(1, 3);\n\n    let search_iterations =\n        8_u32;\n\n    let tanks =\n        (0..snapshot.tank_count)\n            .map(\n                |tank| {\n                    research_tank_from_packed(\n                        &snapshot.packed_config,\n                        tank,\n                    )\n                },\n            )\n            .collect::<Vec<_>>();\n\n    let mut packed_tanks =\n        Vec::<f64>::with_capacity(\n            snapshot.tank_count * 7,\n        );\n\n    for tank in &tanks {\n        let pmax =\n            tank.maximum_heat_pump_electrical_power;\n\n        packed_tanks.push(\n            tank.target_storage_energy,\n        );\n\n        packed_tanks.push(pmax);\n\n        packed_tanks.push(\n            pmax *\n            tank.cop *\n            tank.time_step_hours,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_retention,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_intercept,\n        );\n\n        packed_tanks.push(\n            if tank.affine_supported {\n                1.0\n            } else {\n                0.0\n            },\n        );\n\n        packed_tanks.push(\n            pmax *\n            tank.time_step_hours,\n        );\n    }\n\n    let step_count =\n        snapshot.steps.len();\n\n    let tank_count =\n        snapshot.tank_count;\n\n    let job_count =\n        step_count\n            .checked_mul(tank_count)\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Digital graph A/B job count overflow.",\n                    )\n                },\n            )?;\n\n    let mut current_indices =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut horizon_steps =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut dynamic_flat =\n        Vec::<f64>::with_capacity(\n            job_count * 3,\n        );\n\n    let mut gpu_serial_fallback_jobs =\n        0_u32;\n\n    for step in &snapshot.steps {\n        if (\n            step.horizon_steps == 0 ||\n            step.horizon_steps > 256\n        ) {\n            return Err(\n                Error::from_reason(\n                    format!(\n                        "Digital graph A/B supports horizon 1..=256; got {}.",\n                        step.horizon_steps,\n                    ),\n                ),\n            );\n        }\n\n        current_indices.push(\n            step.current_index,\n        );\n\n        horizon_steps.push(\n            step.horizon_steps,\n        );\n\n        for tank_index in 0..tank_count {\n            let base =\n                tank_index *\n                BLACKWELL_DYNAMIC_STRIDE;\n\n            let current_storage =\n                step.dynamic[base + 0];\n\n            let reserve =\n                step.dynamic[base + 1];\n\n            let requested =\n                step.dynamic[base + 2];\n\n            dynamic_flat.push(\n                current_storage,\n            );\n\n            dynamic_flat.push(\n                reserve,\n            );\n\n            dynamic_flat.push(\n                requested,\n            );\n\n            let tank =\n                tanks[tank_index];\n\n            let clamped_reserve =\n                reserve\n                    .max(0.0)\n                    .min(\n                        tank.target_storage_energy,\n                    );\n\n            let compressed =\n                tank.affine_supported &&\n                tank.heat_loss_retention >\n                    0.0 &&\n                clamped_reserve >\n                    1.0e-6;\n\n            if !compressed {\n                gpu_serial_fallback_jobs += 1;\n            }\n        }\n    }\n\n    let gpu_compressed_jobs =\n        job_count as u32 -\n        gpu_serial_fallback_jobs;\n\n    let mut direct_cuts =\n        vec![0.0_f64; job_count];\n\n    let mut direct_status =\n        vec![0_u32; job_count];\n\n    let mut direct_evaluations =\n        vec![0_u32; job_count];\n\n    let mut direct_static_h2d_ms = 0.0_f64;\n    let mut direct_gpu_total_raw_ms = 0.0_f64;\n    let mut direct_host_wait_raw_ms = 0.0_f64;\n    let mut direct_wall_raw_ms = 0.0_f64;\n\n    let direct_code =\n        unsafe {\n            sunopt_cuda_run_digital_lockstep_candidate(\n                selected_device,\n                snapshot.export_power.as_ptr(),\n                snapshot.demand_energy.as_ptr(),\n                snapshot.export_power.len(),\n                packed_tanks.as_ptr(),\n                tank_count,\n                current_indices.as_ptr(),\n                horizon_steps.as_ptr(),\n                dynamic_flat.as_ptr(),\n                step_count,\n                search_iterations,\n                repetition_count,\n                direct_cuts.as_mut_ptr(),\n                direct_status.as_mut_ptr(),\n                direct_evaluations.as_mut_ptr(),\n                &mut direct_static_h2d_ms,\n                &mut direct_gpu_total_raw_ms,\n                &mut direct_host_wait_raw_ms,\n                &mut direct_wall_raw_ms,\n            )\n        };\n\n    if direct_code != 0 {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_lockstep_candidate",\n                direct_code,\n            ),\n        );\n    }\n\n    let mut graph_cuts =\n        vec![0.0_f64; job_count];\n\n    let mut graph_status =\n        vec![0_u32; job_count];\n\n    let mut graph_evaluations =\n        vec![0_u32; job_count];\n\n    let mut graph_static_h2d_ms = 0.0_f64;\n    let mut graph_build_ms = 0.0_f64;\n    let mut graph_gpu_total_raw_ms = 0.0_f64;\n    let mut graph_host_wait_raw_ms = 0.0_f64;\n    let mut graph_wall_raw_ms = 0.0_f64;\n\n    let graph_code =\n        unsafe {\n            sunopt_cuda_run_digital_lockstep_graph_candidate(\n                selected_device,\n                snapshot.export_power.as_ptr(),\n                snapshot.demand_energy.as_ptr(),\n                snapshot.export_power.len(),\n                packed_tanks.as_ptr(),\n                tank_count,\n                current_indices.as_ptr(),\n                horizon_steps.as_ptr(),\n                dynamic_flat.as_ptr(),\n                step_count,\n                search_iterations,\n                repetition_count,\n                graph_cuts.as_mut_ptr(),\n                graph_status.as_mut_ptr(),\n                graph_evaluations.as_mut_ptr(),\n                &mut graph_static_h2d_ms,\n                &mut graph_build_ms,\n                &mut graph_gpu_total_raw_ms,\n                &mut graph_host_wait_raw_ms,\n                &mut graph_wall_raw_ms,\n            )\n        };\n\n    if graph_code != 0 {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_lockstep_graph_candidate",\n                graph_code,\n            ),\n        );\n    }\n\n    let divisor =\n        repetition_count as f64;\n\n    let direct_gpu_total_ms =\n        direct_gpu_total_raw_ms / divisor;\n\n    let direct_host_wait_ms =\n        direct_host_wait_raw_ms / divisor;\n\n    let direct_wall_ms =\n        direct_wall_raw_ms / divisor;\n\n    let graph_gpu_total_ms =\n        graph_gpu_total_raw_ms / divisor;\n\n    let graph_host_wait_ms =\n        graph_host_wait_raw_ms / divisor;\n\n    let graph_wall_ms =\n        graph_wall_raw_ms / divisor;\n\n    let wall_saving_ms =\n        direct_wall_ms -\n        graph_wall_ms;\n\n    let wall_speedup =\n        if graph_wall_ms > 0.0 {\n            direct_wall_ms /\n            graph_wall_ms\n        } else {\n            f64::INFINITY\n        };\n\n    let wall_reduction_percent =\n        if direct_wall_ms > 0.0 {\n            wall_saving_ms /\n            direct_wall_ms *\n            100.0\n        } else {\n            0.0\n        };\n\n    let mut direct_graph_status_mismatches =\n        0_u32;\n\n    let mut direct_graph_cut_mismatches =\n        0_u32;\n\n    let mut direct_graph_max_cut_error =\n        0.0_f64;\n\n    for job in 0..job_count {\n        if (\n            direct_status[job] !=\n            graph_status[job]\n        ) {\n            direct_graph_status_mismatches += 1;\n        }\n\n        let error =\n            (\n                direct_cuts[job] -\n                graph_cuts[job]\n            )\n            .abs();\n\n        direct_graph_max_cut_error =\n            direct_graph_max_cut_error.max(\n                error,\n            );\n\n        if error > 1.0e-9 {\n            direct_graph_cut_mismatches += 1;\n        }\n    }\n\n    let parity_sample_steps =\n        192_usize.min(step_count);\n\n    let mut parity_status_mismatches =\n        0_u32;\n\n    let mut parity_cut_mismatches =\n        0_u32;\n\n    let mut parity_max_cut_error =\n        0.0_f64;\n\n    for sample in 0..parity_sample_steps {\n        let step_index =\n            if parity_sample_steps <= 1 {\n                0\n            } else {\n                sample *\n                (step_count - 1) /\n                (parity_sample_steps - 1)\n            };\n\n        let step =\n            &snapshot.steps[step_index];\n\n        let export_window =\n            research_window(\n                &snapshot.export_power,\n                step.current_index as usize,\n                step.horizon_steps as usize,\n            );\n\n        let demand_window =\n            research_window(\n                &snapshot.demand_energy,\n                step.current_index as usize,\n                step.horizon_steps as usize,\n            );\n\n        for tank_index in 0..tank_count {\n            let job =\n                step_index *\n                tank_count +\n                tank_index;\n\n            let base =\n                tank_index *\n                BLACKWELL_DYNAMIC_STRIDE;\n\n            let reference =\n                digital_run_scalar(\n                    &export_window,\n                    &demand_window,\n                    step.dynamic[base + 0],\n                    step.dynamic[base + 1],\n                    step.dynamic[base + 2],\n                    tanks[tank_index],\n                );\n\n            let expected_status =\n                if reference.normal_plan_survives {\n                    0_u32\n                } else if !reference.maximum_pv_plan_survives {\n                    1_u32\n                } else {\n                    2_u32\n                };\n\n            if (\n                graph_status[job] !=\n                expected_status\n            ) {\n                parity_status_mismatches += 1;\n            }\n\n            let error =\n                (\n                    graph_cuts[job] -\n                    reference.cut_level_power\n                )\n                .abs();\n\n            parity_max_cut_error =\n                parity_max_cut_error.max(\n                    error,\n                );\n\n            if error > 1.0e-9 {\n                parity_cut_mismatches += 1;\n            }\n        }\n    }\n\n    let mut checksum =\n        1469598103934665603u64;\n\n    for value in &graph_cuts {\n        blackwell_fnv1a_update(\n            &mut checksum,\n            *value,\n        );\n    }\n\n    Ok(\n        BlackwellDigitalGraphAbResult {\n            snapshot_path:\n                path.display()\n                    .to_string(),\n\n            device_id:\n                selected_device,\n\n            repetitions:\n                repetition_count,\n\n            search_iterations,\n\n            annual_step_count:\n                step_count as u32,\n\n            tank_count:\n                tank_count as u32,\n\n            causal_jobs:\n                job_count as u32,\n\n            gpu_compressed_jobs,\n\n            gpu_serial_fallback_jobs,\n\n            direct_static_h2d_ms,\n            direct_gpu_total_ms,\n            direct_host_wait_ms,\n            direct_wall_ms,\n\n            graph_static_h2d_ms,\n            graph_build_ms,\n            graph_gpu_total_ms,\n            graph_host_wait_ms,\n            graph_wall_ms,\n\n            wall_saving_ms,\n            wall_speedup,\n            wall_reduction_percent,\n\n            direct_graph_status_mismatches,\n            direct_graph_cut_mismatches,\n            direct_graph_max_cut_error,\n\n            parity_sample_steps:\n                parity_sample_steps as u32,\n\n            parity_status_mismatches,\n            parity_cut_mismatches,\n            parity_max_cut_error,\n\n            checksum_hex:\n                format!(\n                    "{checksum:016x}",\n                ),\n        },\n    )\n}\n// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FN_END'
MJS = 'import { createRequire } from "node:module";\nimport path from "node:path";\nimport { fileURLToPath } from "node:url";\nimport { writeFileSync } from "node:fs";\n\nconst require = createRequire(import.meta.url);\nconst here = path.dirname(fileURLToPath(import.meta.url));\nconst root = path.resolve(here, "..");\n\nconst addon = require(\n  path.join(\n    root,\n    "sunopt_node.win32-x64-msvc.node",\n  ),\n);\n\nconst snapshot = process.argv[2]\n  ? path.resolve(process.argv[2])\n  : path.join(\n      root,\n      "blackwell-benchmark.bin",\n    );\n\nconst repetitions =\n  Number.parseInt(\n    process.argv[3] ?? "1",\n    10,\n  );\n\nconst deviceId =\n  Number.parseInt(\n    process.argv[4] ?? "0",\n    10,\n  );\n\nconst result =\n  addon.runBlackwellDigitalGraphAb(\n    snapshot,\n    repetitions,\n    deviceId,\n  );\n\nconsole.log(\n  `SUNOPT_DIGITAL_GRAPH_AB_JSON=${JSON.stringify(result)}`,\n);\n\nconsole.log();\nconsole.log(\n  "--- DIGITAL LOCKSTEP: DIRECT vs CUDA GRAPH ---",\n);\n\nconsole.log(`steps=${result.annualStepCount}`);\nconsole.log(`tanks=${result.tankCount}`);\nconsole.log(`jobs=${result.causalJobs}`);\nconsole.log(`searchIterations=${result.searchIterations}`);\nconsole.log(`gpuCompressedJobs=${result.gpuCompressedJobs}`);\nconsole.log(`gpuSerialFallbackJobs=${result.gpuSerialFallbackJobs}`);\n\nconsole.log();\nconsole.log("DIRECT");\nconsole.log(`staticH2DMs=${result.directStaticH2DMs.toFixed(3)}`);\nconsole.log(`gpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`);\nconsole.log(`hostWaitMs=${result.directHostWaitMs.toFixed(3)}`);\nconsole.log(`wallMs=${result.directWallMs.toFixed(3)}`);\n\nconsole.log();\nconsole.log("CUDA GRAPH + PINNED STAGING");\nconsole.log(`staticH2DMs=${result.graphStaticH2DMs.toFixed(3)}`);\nconsole.log(`graphBuildMs=${result.graphBuildMs.toFixed(3)}`);\nconsole.log(`gpuTotalMs=${result.graphGpuTotalMs.toFixed(3)}`);\nconsole.log(`hostWaitMs=${result.graphHostWaitMs.toFixed(3)}`);\nconsole.log(`wallMs=${result.graphWallMs.toFixed(3)}`);\n\nconsole.log();\nconsole.log("DELTA");\nconsole.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);\nconsole.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);\nconsole.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);\n\nconsole.log();\nconsole.log("DIRECT / GRAPH PARITY");\nconsole.log(`statusMismatches=${result.directGraphStatusMismatches}`);\nconsole.log(`cutMismatches=${result.directGraphCutMismatches}`);\nconsole.log(`maxCutError=${result.directGraphMaxCutError.toExponential(6)}`);\n\nconsole.log();\nconsole.log("SCALAR PARITY SAMPLE");\nconsole.log(`sampleSteps=${result.paritySampleSteps}`);\nconsole.log(`statusMismatches=${result.parityStatusMismatches}`);\nconsole.log(`cutMismatches=${result.parityCutMismatches}`);\nconsole.log(`maxCutError=${result.parityMaxCutError.toExponential(6)}`);\nconsole.log(`checksum=${result.checksumHex}`);\n\nconst report =\n  path.join(\n    root,\n    "blackwell-digital-graph-ab.json",\n  );\n\nwriteFileSync(\n  report,\n  JSON.stringify(\n    result,\n    null,\n    2,\n  ) + "\\n",\n  "utf8",\n);\n\nconsole.log();\nconsole.log(`report=${report}`);\n'


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


def patch_header(source: str) -> str:
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN",
        "SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN",
    ):
        if token not in source:
            raise RuntimeError(f"Expected installed header generation missing {token!r}")

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
        "SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN",
        "SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN",
        "SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN",
        "sunopt_digital_hp_breakpoint_kernel(",
        "sunopt_cuda_run_digital_lockstep_candidate(",
    ):
        if token not in source:
            raise RuntimeError(f"Expected current all-GPU CUDA generation missing {token!r}")

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
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_BEGIN",
        "fn sunopt_cuda_run_digital_lockstep_candidate(",
        "fn digital_run_scalar(",
        "fn research_tank_from_packed(",
        "fn research_window(",
        "fn blackwell_fnv1a_update(",
        "BLACKWELL_DYNAMIC_STRIDE",
    ):
        if token not in source:
            raise RuntimeError(f"Expected current Rust generation missing {token!r}")

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

    output = INBOX / "20261109-blackwell-digital-graph-ab.zip"

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("manifest.sha256", (release + "\n").encode("ascii"))
        for name, data in payloads.items():
            archive.writestr(name, data)

    print("SunOpt digital CUDA Graph A/B package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print("  production worker: UNCHANGED")
    print("  direct and graph kernels: IDENTICAL semantics")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(
        r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-graph-ab.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node'
    )

    return output


if __name__ == "__main__":
    build_package()
