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
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-piecewise-zero-ab.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FN_END"

H_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_BEGIN\nSUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const double* host_presorted_export_windows,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t search_iterations,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_static_h2d_ms,\n    double* out_presorted_h2d_ms,\n    double* out_gpu_total_ms,\n    double* out_host_wait_ms,\n    double* out_wall_ms\n);\n// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_END'
PW_HELPER = 'struct SunoptDigitalPiecewiseTransform {\n    double a;\n    double b;\n    double c;\n    double threshold;\n    double max_input;\n};\n\n\n__device__ __forceinline__ SunoptDigitalPiecewiseTransform\nsunopt_digital_piecewise_identity() {\n    SunoptDigitalPiecewiseTransform t{};\n\n    t.a = 1.0;\n    t.b = 0.0;\n    t.c = -CUDART_INF;\n    t.threshold = -CUDART_INF;\n    t.max_input = CUDART_INF;\n\n    return t;\n}\n\n\n__device__ __forceinline__ double\nsunopt_digital_piecewise_eval(\n    const SunoptDigitalPiecewiseTransform t,\n    double x\n) {\n    return (\n        x <= t.threshold\n    ) ?\n        t.c :\n        (\n            t.a *\n            x +\n            t.b\n        );\n}\n\n\n/*\n * Exact composition of monotone two-piece transforms:\n *\n *   F(x) = c              for x <= threshold\n *          a*x + b        for x >  threshold\n *\n * The high branch is monotone and begins at or above the low constant.\n * This family is closed under composition.\n */\n__device__ __forceinline__ SunoptDigitalPiecewiseTransform\nsunopt_digital_piecewise_compose(\n    const SunoptDigitalPiecewiseTransform left,\n    const SunoptDigitalPiecewiseTransform right\n) {\n    SunoptDigitalPiecewiseTransform out{};\n\n    out.a =\n        left.a *\n        right.a;\n\n    out.b =\n        left.a *\n        right.b +\n        left.b;\n\n    out.c =\n        sunopt_digital_piecewise_eval(\n            left,\n            right.c\n        );\n\n    if (\n        right.c >\n        left.threshold\n    ) {\n        out.threshold =\n            right.threshold;\n    } else {\n        const double crossing =\n            (\n                left.threshold -\n                right.b\n            ) /\n            right.a;\n\n        out.threshold =\n            fmax(\n                right.threshold,\n                crossing\n            );\n    }\n\n    double max_input =\n        right.max_input;\n\n    if (\n        right.c >\n        left.max_input +\n        1.0e-12\n    ) {\n        max_input =\n            -CUDART_INF;\n    } else if (\n        right.a >\n        0.0\n    ) {\n        /*\n         * right() uses the low constant at x == threshold.\n         * If its high branch jumps above left.max_input immediately after\n         * threshold, the exact last feasible x is the threshold itself.\n         */\n        const double high_at_threshold =\n            right.a *\n            right.threshold +\n            right.b;\n\n        if (\n            high_at_threshold >\n            left.max_input +\n            1.0e-12\n        ) {\n            max_input =\n                fmin(\n                    max_input,\n                    right.threshold\n                );\n        } else {\n            const double bound =\n                (\n                    left.max_input -\n                    right.b\n                ) /\n                right.a;\n\n            max_input =\n                fmin(\n                    max_input,\n                    bound\n                );\n        }\n    }\n\n    out.max_input =\n        max_input;\n\n    return out;\n}\n\n\n__device__ __forceinline__ SunoptDigitalPiecewiseTransform\nsunopt_digital_piecewise_shuffle_down(\n    SunoptDigitalPiecewiseTransform value,\n    unsigned int delta\n) {\n    constexpr unsigned int FULL_MASK =\n        0xffffffffu;\n\n    SunoptDigitalPiecewiseTransform result{};\n\n    result.a =\n        __shfl_down_sync(\n            FULL_MASK,\n            value.a,\n            delta\n        );\n\n    result.b =\n        __shfl_down_sync(\n            FULL_MASK,\n            value.b,\n            delta\n        );\n\n    result.c =\n        __shfl_down_sync(\n            FULL_MASK,\n            value.c,\n            delta\n        );\n\n    result.threshold =\n        __shfl_down_sync(\n            FULL_MASK,\n            value.threshold,\n            delta\n        );\n\n    result.max_input =\n        __shfl_down_sync(\n            FULL_MASK,\n            value.max_input,\n            delta\n        );\n\n    return result;\n}\n\n\n/*\n * Exact digital survival tree INCLUDING reserve ~= 0.\n *\n * For one future bucket let:\n *\n *   y = max(0, retention*E - loss_intercept)\n *\n * Scalar survival requires:\n *   unmet demand <= EPSILON\n *   before_charge >= reserve - EPSILON\n *   before_charge + charge >= suffix_required\n *\n * The exact minimal input energy therefore has the closed two-piece form:\n *\n *   F(x) = c              for x <= threshold\n *          a*x + b        for x >  threshold\n *\n * with:\n *   reserve_req = max(0, reserve - EPSILON)\n *   low_y =\n *       demand + reserve_req             if reserve_req > 0\n *       max(0, demand - EPSILON)         otherwise\n *   c =\n *       0                                if low_y == 0\n *       (loss_intercept + low_y)/r       otherwise\n *   threshold = charge + reserve_req\n *   a = 1/r\n *   b = (loss_intercept + demand - charge)/r\n *\n * This is the missing exact zero-reserve branch: no serial GPU replay is\n * required merely because reserve is zero.\n */\n__device__ __forceinline__ bool\nsunopt_digital_survives_piecewise_warp(\n    const double* export_power,\n    const double* demand_energy,\n    size_t data_length,\n    const double* tank_parameters,\n    size_t tank_index,\n    size_t current_index,\n    size_t horizon,\n    double current_storage_energy,\n    double minimum_reserve_energy,\n    double cut,\n    SunoptDigitalPiecewiseTransform* shared_transform,\n    bool* shared_supported\n) {\n    const unsigned int tid =\n        threadIdx.x;\n\n    const unsigned int lane =\n        tid &\n        31u;\n\n    const unsigned int warp_id =\n        tid >>\n        5;\n\n    const size_t tank_base =\n        tank_index *\n        7;\n\n    const double target =\n        tank_parameters[\n            tank_base + 0\n        ];\n\n    const double maximum_hp_power =\n        tank_parameters[\n            tank_base + 1\n        ];\n\n    const double full_charge =\n        tank_parameters[\n            tank_base + 2\n        ];\n\n    const double retention =\n        tank_parameters[\n            tank_base + 3\n        ];\n\n    const double loss_intercept =\n        tank_parameters[\n            tank_base + 4\n        ];\n\n    const bool affine_supported =\n        tank_parameters[\n            tank_base + 5\n        ] >\n        0.5;\n\n    const double reserve =\n        fmin(\n            target,\n            fmax(\n                0.0,\n                minimum_reserve_energy\n            )\n        );\n\n    const bool supported =\n        affine_supported &&\n        retention >\n            0.0 &&\n        data_length >\n            0 &&\n        horizon >\n            0 &&\n        horizon <=\n            SUNOPT_DIGITAL_THREADS;\n\n    __shared__ bool piecewise_survives;\n\n    if (\n        !supported\n    ) {\n        if (\n            tid ==\n            0\n        ) {\n            *shared_supported =\n                false;\n\n            piecewise_survives =\n                false;\n        }\n\n        __syncthreads();\n\n        return false;\n    }\n\n    const size_t future_offset =\n        static_cast<size_t>(\n            tid\n        ) +\n        1;\n\n    SunoptDigitalPiecewiseTransform local =\n        sunopt_digital_piecewise_identity();\n\n    if (\n        future_offset <\n        horizon\n    ) {\n        const size_t index =\n            (\n                current_index +\n                future_offset\n            ) %\n            data_length;\n\n        const double demand =\n            fmax(\n                0.0,\n                demand_energy[\n                    index\n                ]\n            );\n\n        const double charge =\n            sunopt_digital_hp_on(\n                export_power[\n                    index\n                ],\n                cut,\n                maximum_hp_power\n            ) ?\n                full_charge :\n                0.0;\n\n        const double reserve_required =\n            fmax(\n                0.0,\n                reserve -\n                SUNOPT_DIGITAL_EPSILON\n            );\n\n        const double demand_required =\n            fmax(\n                0.0,\n                demand -\n                SUNOPT_DIGITAL_EPSILON\n            );\n\n        const double low_after_loss =\n            reserve_required >\n                0.0 ?\n                (\n                    demand +\n                    reserve_required\n                ) :\n                demand_required;\n\n        const double inverse_retention =\n            1.0 /\n            retention;\n\n        local.a =\n            inverse_retention;\n\n        local.b =\n            (\n                loss_intercept +\n                demand -\n                charge\n            ) *\n            inverse_retention;\n\n        local.c =\n            low_after_loss >\n                0.0 ?\n                (\n                    loss_intercept +\n                    low_after_loss\n                ) *\n                inverse_retention :\n                0.0;\n\n        local.threshold =\n            charge +\n            reserve_required;\n\n        local.max_input =\n            target;\n    }\n\n    for (\n        unsigned int stride = 1;\n        stride <\n            32u;\n        stride <<=\n            1\n    ) {\n        const SunoptDigitalPiecewiseTransform right =\n            sunopt_digital_piecewise_shuffle_down(\n                local,\n                stride\n            );\n\n        const unsigned int group =\n            stride <<\n            1;\n\n        if (\n            (\n                lane &\n                (\n                    group -\n                    1\n                )\n            ) ==\n            0\n        ) {\n            local =\n                sunopt_digital_piecewise_compose(\n                    local,\n                    right\n                );\n        }\n    }\n\n    if (\n        lane ==\n        0\n    ) {\n        shared_transform[\n            warp_id\n        ] =\n            local;\n    }\n\n    __syncthreads();\n\n    if (\n        warp_id ==\n        0\n    ) {\n        SunoptDigitalPiecewiseTransform warp_local =\n            lane <\n                8u ?\n                shared_transform[\n                    lane\n                ] :\n                sunopt_digital_piecewise_identity();\n\n        for (\n            unsigned int stride = 1;\n            stride <\n                8u;\n            stride <<=\n                1\n        ) {\n            const SunoptDigitalPiecewiseTransform right =\n                sunopt_digital_piecewise_shuffle_down(\n                    warp_local,\n                    stride\n                );\n\n            const unsigned int group =\n                stride <<\n                1;\n\n            if (\n                (\n                    lane &\n                    (\n                        group -\n                        1\n                    )\n                ) ==\n                0\n            ) {\n                warp_local =\n                    sunopt_digital_piecewise_compose(\n                        warp_local,\n                        right\n                    );\n            }\n        }\n\n        if (\n            lane ==\n            0\n        ) {\n            const SunoptDigitalPiecewiseTransform summary =\n                warp_local;\n\n            bool feasible =\n                0.0 <=\n                summary.max_input +\n                1.0e-12;\n\n            const double required =\n                sunopt_digital_piecewise_eval(\n                    summary,\n                    0.0\n                );\n\n            if (\n                required >\n                target +\n                1.0e-9\n            ) {\n                feasible =\n                    false;\n            }\n\n            const double current_charge =\n                sunopt_digital_hp_on(\n                    export_power[\n                        current_index %\n                        data_length\n                    ],\n                    cut,\n                    maximum_hp_power\n                ) ?\n                    full_charge :\n                    0.0;\n\n            const double current =\n                fmin(\n                    target,\n                    fmax(\n                        0.0,\n                        current_storage_energy\n                    )\n                );\n\n            const double available =\n                fmin(\n                    target,\n                    current +\n                    current_charge\n                );\n\n            piecewise_survives =\n                feasible &&\n                available +\n                    1.0e-12 >=\n                    required;\n\n            *shared_supported =\n                true;\n        }\n    }\n\n    __syncthreads();\n\n    return piecewise_survives;\n}'
RS_FFI_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FFI_BEGIN\n    fn sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(\n        device_id: c_int,\n        host_export_power: *const f64,\n        host_demand_energy: *const f64,\n        data_length: usize,\n        host_tank_parameters: *const f64,\n        tank_count: usize,\n        host_presorted_export_windows: *const f64,\n        host_current_indices: *const u32,\n        host_horizon_steps: *const u32,\n        host_dynamic_values: *const f64,\n        step_count: usize,\n        search_iterations: u32,\n        repetitions: u32,\n        host_cut_results: *mut f64,\n        host_status: *mut u32,\n        host_survival_evaluations: *mut u32,\n        out_static_h2d_ms: *mut f64,\n        out_presorted_h2d_ms: *mut f64,\n        out_gpu_total_ms: *mut f64,\n        out_host_wait_ms: *mut f64,\n        out_wall_ms: *mut f64,\n    ) -> c_int;\n    // SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FFI_END'
RS_RESULT_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_RESULT_BEGIN\n#[napi(object)]\npub struct BlackwellDigitalPiecewiseZeroAbResult {\n    pub snapshot_path: String,\n    pub device_id: i32,\n    pub repetitions: u32,\n    pub search_iterations: u32,\n    pub annual_step_count: u32,\n    pub tank_count: u32,\n    pub causal_jobs: u32,\n\n    pub legacy_compressed_jobs: u32,\n    pub legacy_serial_fallback_jobs: u32,\n    pub piecewise_compressed_jobs: u32,\n    pub piecewise_serial_fallback_jobs: u32,\n\n    pub legacy_gpu_total_ms: f64,\n    pub legacy_wall_ms: f64,\n\n    pub piecewise_gpu_total_ms: f64,\n    pub piecewise_wall_ms: f64,\n\n    pub wall_saving_ms: f64,\n    pub wall_speedup: f64,\n    pub wall_reduction_percent: f64,\n\n    pub legacy_piecewise_status_mismatches: u32,\n    pub legacy_piecewise_cut_mismatches: u32,\n    pub legacy_piecewise_max_cut_error: f64,\n    pub evaluation_mismatches: u32,\n\n    pub parity_sample_steps: u32,\n    pub parity_status_mismatches: u32,\n    pub parity_cut_mismatches: u32,\n    pub parity_max_cut_error: f64,\n\n    pub checksum_hex: String,\n}\n// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_RESULT_END'
RS_FN_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FN_BEGIN\n#[napi]\npub fn run_blackwell_digital_piecewise_zero_ab(\n    snapshot_path: Option<String>,\n    repetitions: Option<u32>,\n    device_id: Option<i32>,\n) -> Result<BlackwellDigitalPiecewiseZeroAbResult> {\n    let path =\n        snapshot_path\n            .map(PathBuf::from)\n            .unwrap_or_else(\n                blackwell_default_snapshot_path,\n            );\n\n    let snapshot =\n        read_blackwell_snapshot(&path)\n            .map_err(\n                |error| {\n                    Error::from_reason(\n                        format!(\n                            "Unable to read Blackwell benchmark snapshot {}: {error}",\n                            path.display(),\n                        ),\n                    )\n                },\n            )?;\n\n    if (\n        snapshot.steps.is_empty() ||\n        snapshot.tank_count == 0 ||\n        snapshot.export_power.is_empty() ||\n        snapshot.demand_energy.len() !=\n            snapshot.export_power.len() ||\n        snapshot.steps.len() !=\n            snapshot.data_length\n    ) {\n        return Err(\n            Error::from_reason(\n                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",\n            ),\n        );\n    }\n\n    let selected_device =\n        device_id.unwrap_or(0);\n\n    let repetition_count =\n        repetitions\n            .unwrap_or(1)\n            .clamp(1, 3);\n\n    let search_iterations =\n        8_u32;\n\n    let tanks =\n        (0..snapshot.tank_count)\n            .map(\n                |tank| {\n                    research_tank_from_packed(\n                        &snapshot.packed_config,\n                        tank,\n                    )\n                },\n            )\n            .collect::<Vec<_>>();\n\n    let mut packed_tanks =\n        Vec::<f64>::with_capacity(\n            snapshot.tank_count * 7,\n        );\n\n    for tank in &tanks {\n        let pmax =\n            tank.maximum_heat_pump_electrical_power;\n\n        packed_tanks.push(\n            tank.target_storage_energy,\n        );\n\n        packed_tanks.push(pmax);\n\n        packed_tanks.push(\n            pmax *\n            tank.cop *\n            tank.time_step_hours,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_retention,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_intercept,\n        );\n\n        packed_tanks.push(\n            if tank.affine_supported {\n                1.0\n            } else {\n                0.0\n            },\n        );\n\n        packed_tanks.push(\n            pmax *\n            tank.time_step_hours,\n        );\n    }\n\n    let step_count =\n        snapshot.steps.len();\n\n    let tank_count =\n        snapshot.tank_count;\n\n    let job_count =\n        step_count\n            .checked_mul(tank_count)\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Piecewise-zero A/B job count overflow.",\n                    )\n                },\n            )?;\n\n    let mut current_indices =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut horizon_steps =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut dynamic_flat =\n        Vec::<f64>::with_capacity(\n            job_count * 3,\n        );\n\n    let mut legacy_serial_fallback_jobs =\n        0_u32;\n\n    let mut piecewise_serial_fallback_jobs =\n        0_u32;\n\n    for step in &snapshot.steps {\n        if (\n            step.horizon_steps == 0 ||\n            step.horizon_steps > 256\n        ) {\n            return Err(\n                Error::from_reason(\n                    format!(\n                        "Piecewise-zero A/B supports horizon 1..=256; got {}.",\n                        step.horizon_steps,\n                    ),\n                ),\n            );\n        }\n\n        current_indices.push(\n            step.current_index,\n        );\n\n        horizon_steps.push(\n            step.horizon_steps,\n        );\n\n        for tank_index in 0..tank_count {\n            let base =\n                tank_index *\n                BLACKWELL_DYNAMIC_STRIDE;\n\n            let reserve =\n                step.dynamic[\n                    base + 1\n                ]\n                .max(0.0)\n                .min(\n                    tanks[\n                        tank_index\n                    ]\n                    .target_storage_energy,\n                );\n\n            dynamic_flat.push(\n                step.dynamic[base + 0],\n            );\n\n            dynamic_flat.push(\n                reserve,\n            );\n\n            dynamic_flat.push(\n                step.dynamic[base + 2],\n            );\n\n            let tank =\n                tanks[tank_index];\n\n            let legacy_supported =\n                tank.affine_supported &&\n                tank.heat_loss_retention >\n                    0.0 &&\n                reserve >\n                    1.0e-6;\n\n            let piecewise_supported =\n                tank.affine_supported &&\n                tank.heat_loss_retention >\n                    0.0;\n\n            if !legacy_supported {\n                legacy_serial_fallback_jobs +=\n                    1;\n            }\n\n            if !piecewise_supported {\n                piecewise_serial_fallback_jobs +=\n                    1;\n            }\n        }\n    }\n\n    let legacy_compressed_jobs =\n        job_count as u32 -\n        legacy_serial_fallback_jobs;\n\n    let piecewise_compressed_jobs =\n        job_count as u32 -\n        piecewise_serial_fallback_jobs;\n\n    let presorted_count =\n        step_count\n            .checked_mul(256)\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Presorted export matrix size overflow.",\n                    )\n                },\n            )?;\n\n    let mut presorted =\n        vec![\n            f64::NEG_INFINITY;\n            presorted_count\n        ];\n\n    for (\n        step_index,\n        step,\n    ) in\n        snapshot.steps\n            .iter()\n            .enumerate()\n    {\n        let horizon =\n            step.horizon_steps as usize;\n\n        let base =\n            step_index * 256;\n\n        let first =\n            base +\n            (256 - horizon);\n\n        for offset in 0..horizon {\n            let index =\n                (\n                    step.current_index as usize +\n                    offset\n                ) %\n                snapshot.export_power.len();\n\n            presorted[\n                first + offset\n            ] =\n                snapshot.export_power[index];\n        }\n\n        presorted[\n            first..\n            base + 256\n        ]\n        .sort_by(\n            |a, b| {\n                a.partial_cmp(b)\n                    .unwrap_or(\n                        std::cmp::Ordering::Equal,\n                    )\n            },\n        );\n    }\n\n    let mut legacy_cuts =\n        vec![0.0_f64; job_count];\n\n    let mut legacy_status =\n        vec![0_u32; job_count];\n\n    let mut legacy_evaluations =\n        vec![0_u32; job_count];\n\n    let mut legacy_static_h2d_ms =\n        0.0_f64;\n\n    let mut legacy_presorted_h2d_ms =\n        0.0_f64;\n\n    let mut legacy_gpu_total_raw_ms =\n        0.0_f64;\n\n    let mut legacy_host_wait_raw_ms =\n        0.0_f64;\n\n    let mut legacy_wall_raw_ms =\n        0.0_f64;\n\n    let legacy_code =\n        unsafe {\n            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(\n                selected_device,\n                snapshot.export_power.as_ptr(),\n                snapshot.demand_energy.as_ptr(),\n                snapshot.export_power.len(),\n                packed_tanks.as_ptr(),\n                tank_count,\n                presorted.as_ptr(),\n                current_indices.as_ptr(),\n                horizon_steps.as_ptr(),\n                dynamic_flat.as_ptr(),\n                step_count,\n                search_iterations,\n                repetition_count,\n                legacy_cuts.as_mut_ptr(),\n                legacy_status.as_mut_ptr(),\n                legacy_evaluations.as_mut_ptr(),\n                &mut legacy_static_h2d_ms,\n                &mut legacy_presorted_h2d_ms,\n                &mut legacy_gpu_total_raw_ms,\n                &mut legacy_host_wait_raw_ms,\n                &mut legacy_wall_raw_ms,\n            )\n        };\n\n    if legacy_code != 0 {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",\n                legacy_code,\n            ),\n        );\n    }\n\n    let mut piecewise_cuts =\n        vec![0.0_f64; job_count];\n\n    let mut piecewise_status =\n        vec![0_u32; job_count];\n\n    let mut piecewise_evaluations =\n        vec![0_u32; job_count];\n\n    let mut piecewise_static_h2d_ms =\n        0.0_f64;\n\n    let mut piecewise_presorted_h2d_ms =\n        0.0_f64;\n\n    let mut piecewise_gpu_total_raw_ms =\n        0.0_f64;\n\n    let mut piecewise_host_wait_raw_ms =\n        0.0_f64;\n\n    let mut piecewise_wall_raw_ms =\n        0.0_f64;\n\n    let piecewise_code =\n        unsafe {\n            sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(\n                selected_device,\n                snapshot.export_power.as_ptr(),\n                snapshot.demand_energy.as_ptr(),\n                snapshot.export_power.len(),\n                packed_tanks.as_ptr(),\n                tank_count,\n                presorted.as_ptr(),\n                current_indices.as_ptr(),\n                horizon_steps.as_ptr(),\n                dynamic_flat.as_ptr(),\n                step_count,\n                search_iterations,\n                repetition_count,\n                piecewise_cuts.as_mut_ptr(),\n                piecewise_status.as_mut_ptr(),\n                piecewise_evaluations.as_mut_ptr(),\n                &mut piecewise_static_h2d_ms,\n                &mut piecewise_presorted_h2d_ms,\n                &mut piecewise_gpu_total_raw_ms,\n                &mut piecewise_host_wait_raw_ms,\n                &mut piecewise_wall_raw_ms,\n            )\n        };\n\n    if piecewise_code != 0 {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate",\n                piecewise_code,\n            ),\n        );\n    }\n\n    let divisor =\n        repetition_count as f64;\n\n    let legacy_gpu_total_ms =\n        legacy_gpu_total_raw_ms /\n        divisor;\n\n    let legacy_wall_ms =\n        legacy_wall_raw_ms /\n        divisor;\n\n    let piecewise_gpu_total_ms =\n        piecewise_gpu_total_raw_ms /\n        divisor;\n\n    let piecewise_wall_ms =\n        piecewise_wall_raw_ms /\n        divisor;\n\n    let wall_saving_ms =\n        legacy_wall_ms -\n        piecewise_wall_ms;\n\n    let wall_speedup =\n        if piecewise_wall_ms > 0.0 {\n            legacy_wall_ms /\n            piecewise_wall_ms\n        } else {\n            f64::INFINITY\n        };\n\n    let wall_reduction_percent =\n        if legacy_wall_ms > 0.0 {\n            wall_saving_ms /\n            legacy_wall_ms *\n            100.0\n        } else {\n            0.0\n        };\n\n    let mut legacy_piecewise_status_mismatches =\n        0_u32;\n\n    let mut legacy_piecewise_cut_mismatches =\n        0_u32;\n\n    let mut legacy_piecewise_max_cut_error =\n        0.0_f64;\n\n    let mut evaluation_mismatches =\n        0_u32;\n\n    for job in 0..job_count {\n        if (\n            legacy_status[job] !=\n            piecewise_status[job]\n        ) {\n            legacy_piecewise_status_mismatches +=\n                1;\n        }\n\n        if (\n            legacy_evaluations[job] !=\n            piecewise_evaluations[job]\n        ) {\n            evaluation_mismatches +=\n                1;\n        }\n\n        let error =\n            (\n                legacy_cuts[job] -\n                piecewise_cuts[job]\n            )\n            .abs();\n\n        legacy_piecewise_max_cut_error =\n            legacy_piecewise_max_cut_error.max(\n                error,\n            );\n\n        if error > 1.0e-9 {\n            legacy_piecewise_cut_mismatches +=\n                1;\n        }\n    }\n\n    let parity_sample_steps =\n        192_usize.min(step_count);\n\n    let mut parity_status_mismatches =\n        0_u32;\n\n    let mut parity_cut_mismatches =\n        0_u32;\n\n    let mut parity_max_cut_error =\n        0.0_f64;\n\n    for sample in 0..parity_sample_steps {\n        let step_index =\n            if parity_sample_steps <= 1 {\n                0\n            } else {\n                sample *\n                (step_count - 1) /\n                (parity_sample_steps - 1)\n            };\n\n        let step =\n            &snapshot.steps[step_index];\n\n        let export_window =\n            research_window(\n                &snapshot.export_power,\n                step.current_index as usize,\n                step.horizon_steps as usize,\n            );\n\n        let demand_window =\n            research_window(\n                &snapshot.demand_energy,\n                step.current_index as usize,\n                step.horizon_steps as usize,\n            );\n\n        for tank_index in 0..tank_count {\n            let job =\n                step_index *\n                tank_count +\n                tank_index;\n\n            let base =\n                tank_index *\n                BLACKWELL_DYNAMIC_STRIDE;\n\n            let reference =\n                digital_run_scalar(\n                    &export_window,\n                    &demand_window,\n                    step.dynamic[base + 0],\n                    step.dynamic[base + 1],\n                    step.dynamic[base + 2],\n                    tanks[tank_index],\n                );\n\n            let expected_status =\n                if reference.normal_plan_survives {\n                    0_u32\n                } else if !reference.maximum_pv_plan_survives {\n                    1_u32\n                } else {\n                    2_u32\n                };\n\n            if (\n                piecewise_status[job] !=\n                expected_status\n            ) {\n                parity_status_mismatches +=\n                    1;\n            }\n\n            let error =\n                (\n                    piecewise_cuts[job] -\n                    reference.cut_level_power\n                )\n                .abs();\n\n            parity_max_cut_error =\n                parity_max_cut_error.max(\n                    error,\n                );\n\n            if error > 1.0e-9 {\n                parity_cut_mismatches +=\n                    1;\n            }\n        }\n    }\n\n    let mut checksum =\n        1469598103934665603u64;\n\n    for value in &piecewise_cuts {\n        blackwell_fnv1a_update(\n            &mut checksum,\n            *value,\n        );\n    }\n\n    Ok(\n        BlackwellDigitalPiecewiseZeroAbResult {\n            snapshot_path:\n                path.display()\n                    .to_string(),\n\n            device_id:\n                selected_device,\n\n            repetitions:\n                repetition_count,\n\n            search_iterations,\n\n            annual_step_count:\n                step_count as u32,\n\n            tank_count:\n                tank_count as u32,\n\n            causal_jobs:\n                job_count as u32,\n\n            legacy_compressed_jobs,\n            legacy_serial_fallback_jobs,\n            piecewise_compressed_jobs,\n            piecewise_serial_fallback_jobs,\n\n            legacy_gpu_total_ms,\n            legacy_wall_ms,\n\n            piecewise_gpu_total_ms,\n            piecewise_wall_ms,\n\n            wall_saving_ms,\n            wall_speedup,\n            wall_reduction_percent,\n\n            legacy_piecewise_status_mismatches,\n            legacy_piecewise_cut_mismatches,\n            legacy_piecewise_max_cut_error,\n            evaluation_mismatches,\n\n            parity_sample_steps:\n                parity_sample_steps as u32,\n\n            parity_status_mismatches,\n            parity_cut_mismatches,\n            parity_max_cut_error,\n\n            checksum_hex:\n                format!(\n                    "{checksum:016x}",\n                ),\n        },\n    )\n}\n// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FN_END'
MJS = 'import { createRequire } from "node:module";\nimport path from "node:path";\nimport { fileURLToPath } from "node:url";\nimport { writeFileSync } from "node:fs";\n\nconst require = createRequire(import.meta.url);\nconst here = path.dirname(fileURLToPath(import.meta.url));\nconst root = path.resolve(here, "..");\n\nconst addon = require(\n  path.join(\n    root,\n    "sunopt_node.win32-x64-msvc.node",\n  ),\n);\n\nconst snapshot = process.argv[2]\n  ? path.resolve(process.argv[2])\n  : path.join(\n      root,\n      "blackwell-benchmark.bin",\n    );\n\nconst repetitions =\n  Number.parseInt(\n    process.argv[3] ?? "1",\n    10,\n  );\n\nconst deviceId =\n  Number.parseInt(\n    process.argv[4] ?? "0",\n    10,\n  );\n\nconst result =\n  addon.runBlackwellDigitalPiecewiseZeroAb(\n    snapshot,\n    repetitions,\n    deviceId,\n  );\n\nconsole.log(\n  `SUNOPT_DIGITAL_PIECEWISE_ZERO_AB_JSON=${JSON.stringify(result)}`,\n);\n\nconsole.log();\nconsole.log(\n  "--- DIGITAL PRE-SORTED WARP: SERIAL ZERO FALLBACK vs PIECEWISE ZERO TREE ---",\n);\n\nconsole.log(`steps=${result.annualStepCount}`);\nconsole.log(`tanks=${result.tankCount}`);\nconsole.log(`jobs=${result.causalJobs}`);\nconsole.log(`searchIterations=${result.searchIterations}`);\n\nconsole.log();\nconsole.log("JOB ROUTING");\nconsole.log(`legacyCompressedJobs=${result.legacyCompressedJobs}`);\nconsole.log(`legacySerialFallbackJobs=${result.legacySerialFallbackJobs}`);\nconsole.log(`piecewiseCompressedJobs=${result.piecewiseCompressedJobs}`);\nconsole.log(`piecewiseSerialFallbackJobs=${result.piecewiseSerialFallbackJobs}`);\n\nconsole.log();\nconsole.log("LEGACY WARP + SERIAL ZERO FALLBACK");\nconsole.log(`gpuTotalMs=${result.legacyGpuTotalMs.toFixed(3)}`);\nconsole.log(`wallMs=${result.legacyWallMs.toFixed(3)}`);\n\nconsole.log();\nconsole.log("PIECEWISE ZERO TREE");\nconsole.log(`gpuTotalMs=${result.piecewiseGpuTotalMs.toFixed(3)}`);\nconsole.log(`wallMs=${result.piecewiseWallMs.toFixed(3)}`);\n\nconsole.log();\nconsole.log("DELTA");\nconsole.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);\nconsole.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);\nconsole.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);\n\nconsole.log();\nconsole.log("LEGACY / PIECEWISE PARITY");\nconsole.log(`statusMismatches=${result.legacyPiecewiseStatusMismatches}`);\nconsole.log(`cutMismatches=${result.legacyPiecewiseCutMismatches}`);\nconsole.log(`maxCutError=${result.legacyPiecewiseMaxCutError.toExponential(6)}`);\nconsole.log(`evaluationMismatches=${result.evaluationMismatches}`);\n\nconsole.log();\nconsole.log("SCALAR PARITY SAMPLE");\nconsole.log(`sampleSteps=${result.paritySampleSteps}`);\nconsole.log(`statusMismatches=${result.parityStatusMismatches}`);\nconsole.log(`cutMismatches=${result.parityCutMismatches}`);\nconsole.log(`maxCutError=${result.parityMaxCutError.toExponential(6)}`);\nconsole.log(`checksum=${result.checksumHex}`);\n\nconst report =\n  path.join(\n    root,\n    "blackwell-digital-piecewise-zero-ab.json",\n  );\n\nwriteFileSync(\n  report,\n  JSON.stringify(\n    result,\n    null,\n    2,\n  ) + "\\n",\n  "utf8",\n);\n\nconsole.log();\nconsole.log(`report=${report}`);\n'


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
            state.get("sha256", "")
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

    return source[:start] + block + source[finish:]


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
        position = source.find(anchor)

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


def extract_braced_function(
    source: str,
    token: str,
) -> str:
    start = source.find(token)

    if start < 0:
        raise RuntimeError(
            f"Could not find function token {token!r}"
        )

    brace = source.find(
        "{",
        start,
    )

    if brace < 0:
        raise RuntimeError(
            f"Could not find opening brace for {token!r}"
        )

    depth = 0

    for index in range(
        brace,
        len(source),
    ):
        char = source[index]

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return source[
                    start:
                    index + 1
                ]

    raise RuntimeError(
        f"Unterminated function {token!r}"
    )


def build_piecewise_kernel(
    source: str,
) -> str:
    token = (
        "__global__ void\n"
        "sunopt_digital_hp_presorted_warp_kernel("
    )

    kernel = extract_braced_function(
        source,
        token,
    )

    kernel = kernel.replace(
        "sunopt_digital_hp_presorted_warp_kernel(",
        "sunopt_digital_hp_piecewise_zero_kernel(",
        1,
    )

    old_type = (
        "SunoptDigitalTransform transforms"
    )

    if old_type not in kernel:
        raise RuntimeError(
            "Could not find shared max-affine transform array."
        )

    kernel = kernel.replace(
        old_type,
        "SunoptDigitalPiecewiseTransform transforms",
        1,
    )

    old_call = "sunopt_digital_survives_warp("

    call_count = kernel.count(old_call)

    if call_count < 2:
        raise RuntimeError(
            f"Expected multiple warp survival calls; got {call_count}."
        )

    kernel = kernel.replace(
        old_call,
        "sunopt_digital_survives_piecewise_warp(",
    )

    return kernel


def build_piecewise_runner(
    source: str,
) -> str:
    token = (
        'extern "C" int '
        'sunopt_cuda_run_digital_lockstep_presorted_warp_candidate('
    )

    runner = extract_braced_function(
        source,
        token,
    )

    runner = runner.replace(
        "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(",
        "sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(",
        1,
    )

    old_kernel = "sunopt_digital_hp_presorted_warp_kernel<<<"

    if old_kernel not in runner:
        raise RuntimeError(
            "Could not find warp kernel launch in runner."
        )

    runner = runner.replace(
        old_kernel,
        "sunopt_digital_hp_piecewise_zero_kernel<<<",
        1,
    )

    return runner


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
        "sunopt_digital_survives_warp(",
        "SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN",
    ):
        if token not in source:
            raise RuntimeError(
                f"Expected current CUDA generation missing {token!r}"
            )

    piecewise_kernel = build_piecewise_kernel(source)

    piecewise_runner = build_piecewise_runner(source)

    block = (
        CU_BEGIN +
        "\n" +
        PW_HELPER +
        "\n\n\n" +
        piecewise_kernel +
        "\n\n\n" +
        piecewise_runner +
        "\n" +
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
        "fn digital_run_scalar(",
        "fn research_tank_from_packed(",
        "fn research_window(",
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
        "20261109-blackwell-digital-piecewise-zero-ab.zip"
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
            (release + "\n").encode("ascii"),
        )

        for name, data in payloads.items():
            archive.writestr(
                name,
                data,
            )

    print(
        "SunOpt digital piecewise-zero A/B package created"
    )
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print("  production worker: UNCHANGED")
    print(
        "  candidate: exact two-piece affine zero-reserve transform"
    )
    print()
    print("Next:")
    print(
        r"  python .\tools\update_watcher.py --once"
    )
    print(
        r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-piecewise-zero-ab.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node'
    )

    return output


if __name__ == "__main__":
    build_package()
