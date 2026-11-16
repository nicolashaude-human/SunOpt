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
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-cluster4.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FN_END"

H_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_BEGIN\nSUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_cluster4_lockstep(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const double* host_presorted_export_windows,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_input_h2d_ms,\n    double* out_kernel_ms,\n    double* out_result_d2h_ms,\n    double* out_wall_ms\n);\n// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_END'
CUDA_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_BEGIN\n#include <cooperative_groups.h>\n\nnamespace sunopt_cg = cooperative_groups;\n\nstruct SunoptDigitalCluster4Control {\n    double normal_cut;\n    double candidate_cut;\n    double best_cut;\n\n    int first_valid;\n    int left_index;\n    int right_index;\n\n    uint32_t valid_count;\n    uint32_t eval_count;\n    uint32_t final_status;\n\n    bool parallel_supported;\n    bool done;\n    bool candidate_survives;\n};\n\n\n__device__ __forceinline__ bool\nsunopt_digital_cluster4_survives(\n    const double* export_power,\n    const double* demand_energy,\n    size_t data_length,\n    const double* tank_parameters,\n    size_t tank_index,\n    size_t current_index,\n    size_t horizon,\n    double current_storage_energy,\n    double minimum_reserve_energy,\n    double cut,\n    SunoptDigitalTransform* warp_partials,\n    SunoptDigitalTransform* block_summary,\n    SunoptDigitalCluster4Control* control\n) {\n    sunopt_cg::cluster_group cluster =\n        sunopt_cg::this_cluster();\n\n    const unsigned int cluster_rank =\n        cluster.block_rank();\n\n    const unsigned int tid =\n        threadIdx.x;\n\n    const unsigned int lane =\n        tid & 31u;\n\n    const unsigned int warp_id =\n        tid >> 5;\n\n    const size_t tank_base =\n        tank_index * 7;\n\n    const double target =\n        tank_parameters[tank_base + 0];\n\n    const double maximum_hp_power =\n        tank_parameters[tank_base + 1];\n\n    const double full_charge =\n        tank_parameters[tank_base + 2];\n\n    const double retention =\n        tank_parameters[tank_base + 3];\n\n    const double loss_intercept =\n        tank_parameters[tank_base + 4];\n\n    const double reserve =\n        fmin(\n            target,\n            fmax(\n                0.0,\n                minimum_reserve_energy\n            )\n        );\n\n    /*\n     * Four blocks x 64 threads cover offsets 1..256.\n     * The chronological partition is contiguous:\n     * block 0 -> 1..64, block 1 -> 65..128, ...\n     */\n    const size_t future_offset =\n        1 +\n        static_cast<size_t>(\n            cluster_rank\n        ) *\n            64 +\n        static_cast<size_t>(\n            tid\n        );\n\n    SunoptDigitalTransform local =\n        sunopt_digital_identity();\n\n    if (\n        future_offset <\n        horizon\n    ) {\n        const size_t index =\n            (\n                current_index +\n                future_offset\n            ) %\n            data_length;\n\n        const double demand =\n            fmax(\n                0.0,\n                demand_energy[index]\n            );\n\n        const double charge =\n            sunopt_digital_hp_on(\n                export_power[index],\n                cut,\n                maximum_hp_power\n            ) ?\n                full_charge :\n                0.0;\n\n        const double inverse_retention =\n            1.0 /\n            retention;\n\n        local.a =\n            inverse_retention;\n\n        local.b =\n            (\n                loss_intercept +\n                demand -\n                charge\n            ) *\n            inverse_retention;\n\n        local.c =\n            (\n                loss_intercept +\n                demand +\n                reserve -\n                SUNOPT_DIGITAL_EPSILON\n            ) *\n            inverse_retention;\n\n        local.max_input =\n            target;\n    }\n\n    /*\n     * First reduce each 32-thread warp.\n     */\n    for (\n        unsigned int stride = 1;\n        stride < 32u;\n        stride <<= 1\n    ) {\n        const SunoptDigitalTransform right =\n            sunopt_digital_shuffle_down_transform(\n                local,\n                stride\n            );\n\n        const unsigned int group =\n            stride << 1;\n\n        if (\n            (\n                lane &\n                (\n                    group -\n                    1\n                )\n            ) ==\n            0\n        ) {\n            local =\n                sunopt_digital_compose(\n                    local,\n                    right\n                );\n        }\n    }\n\n    if (\n        lane ==\n        0\n    ) {\n        warp_partials[\n            warp_id\n        ] =\n            local;\n    }\n\n    __syncthreads();\n\n    /*\n     * Warp 0 composes the two contiguous 32-bucket summaries.\n     */\n    if (\n        warp_id ==\n        0\n    ) {\n        SunoptDigitalTransform block_local =\n            lane < 2u ?\n                warp_partials[lane] :\n                sunopt_digital_identity();\n\n        const SunoptDigitalTransform right =\n            sunopt_digital_shuffle_down_transform(\n                block_local,\n                1u\n            );\n\n        if (\n            lane ==\n            0\n        ) {\n            block_local =\n                sunopt_digital_compose(\n                    block_local,\n                    right\n                );\n\n            *block_summary =\n                block_local;\n        }\n    }\n\n    /*\n     * All four block summaries are now valid and remain alive in DSM.\n     */\n    cluster.sync();\n\n    if (\n        cluster_rank ==\n            0u &&\n        tid ==\n            0u\n    ) {\n        SunoptDigitalTransform summary =\n            sunopt_digital_identity();\n\n        for (\n            unsigned int rank = 0;\n            rank < 4u;\n            ++rank\n        ) {\n            SunoptDigitalTransform* remote =\n                cluster.map_shared_rank(\n                    block_summary,\n                    rank\n                );\n\n            summary =\n                sunopt_digital_compose(\n                    summary,\n                    *remote\n                );\n        }\n\n        bool feasible =\n            0.0 <=\n            summary.max_input +\n                1.0e-12;\n\n        const double required =\n            fmax(\n                summary.c,\n                summary.b\n            );\n\n        if (\n            required >\n            target +\n                1.0e-9\n        ) {\n            feasible =\n                false;\n        }\n\n        const double current_charge =\n            sunopt_digital_hp_on(\n                export_power[\n                    current_index %\n                    data_length\n                ],\n                cut,\n                maximum_hp_power\n            ) ?\n                full_charge :\n                0.0;\n\n        const double current =\n            fmin(\n                target,\n                fmax(\n                    0.0,\n                    current_storage_energy\n                )\n            );\n\n        const double available =\n            fmin(\n                target,\n                current +\n                    current_charge\n            );\n\n        control->candidate_survives =\n            feasible &&\n            available +\n                1.0e-12 >=\n                required;\n    }\n\n    cluster.sync();\n\n    SunoptDigitalCluster4Control* root =\n        cluster.map_shared_rank(\n            control,\n            0\n        );\n\n    return root->candidate_survives;\n}\n\n\n__global__ void\n__cluster_dims__(4, 1, 1)\nsunopt_digital_cluster4_kernel(\n    const double* __restrict__ export_power,\n    const double* __restrict__ demand_energy,\n    const double* __restrict__ presorted_export_windows,\n    size_t data_length,\n    const double* __restrict__ tank_parameters,\n    size_t tank_count,\n    const uint32_t* __restrict__ current_indices,\n    const uint32_t* __restrict__ horizon_steps,\n    const double* __restrict__ dynamic_values,\n    size_t step_count,\n    size_t step_index,\n    double* __restrict__ cut_results,\n    uint32_t* __restrict__ status,\n    uint32_t* __restrict__ survival_evaluations\n) {\n    sunopt_cg::cluster_group cluster =\n        sunopt_cg::this_cluster();\n\n    const unsigned int cluster_rank =\n        cluster.block_rank();\n\n    const unsigned int tid =\n        threadIdx.x;\n\n    const size_t tank_index =\n        static_cast<size_t>(\n            blockIdx.x\n        ) /\n        4;\n\n    if (\n        tank_index >=\n        tank_count\n    ) {\n        return;\n    }\n\n    const size_t job_index =\n        step_index *\n            tank_count +\n        tank_index;\n\n    const size_t tank_base =\n        tank_index * 7;\n\n    const double target =\n        tank_parameters[tank_base + 0];\n\n    const double maximum_hp_power =\n        tank_parameters[tank_base + 1];\n\n    const double electrical_energy_per_on_bucket =\n        tank_parameters[tank_base + 6];\n\n    const size_t dynamic_base =\n        job_index * 3;\n\n    const double current_storage =\n        dynamic_values[dynamic_base + 0];\n\n    const double reserve =\n        dynamic_values[dynamic_base + 1];\n\n    const double requested_electrical_energy =\n        dynamic_values[dynamic_base + 2];\n\n    const size_t current_index =\n        static_cast<size_t>(\n            current_indices[step_index]\n        );\n\n    const size_t horizon =\n        static_cast<size_t>(\n            horizon_steps[step_index]\n        );\n\n    const double* sorted_export =\n        presorted_export_windows +\n        step_index *\n            SUNOPT_DIGITAL_THREADS;\n\n    __shared__ SunoptDigitalTransform warp_partials[2];\n    __shared__ SunoptDigitalTransform block_summary;\n    __shared__ SunoptDigitalCluster4Control control;\n\n    SunoptDigitalCluster4Control* root =\n        cluster.map_shared_rank(\n            &control,\n            0\n        );\n\n    if (\n        cluster_rank ==\n            0u &&\n        tid ==\n            0u\n    ) {\n        int first =\n            static_cast<int>(\n                SUNOPT_DIGITAL_THREADS\n            );\n\n        for (\n            int i = 0;\n            i <\n                static_cast<int>(\n                    SUNOPT_DIGITAL_THREADS\n                );\n            ++i\n        ) {\n            if (\n                sorted_export[i] >=\n                maximum_hp_power\n            ) {\n                first =\n                    i;\n\n                break;\n            }\n        }\n\n        control.first_valid =\n            first;\n\n        control.valid_count =\n            first <\n                static_cast<int>(\n                    SUNOPT_DIGITAL_THREADS\n                ) ?\n                static_cast<uint32_t>(\n                    SUNOPT_DIGITAL_THREADS -\n                    first\n                ) :\n                0u;\n\n        const double maximum_schedulable =\n            static_cast<double>(\n                control.valid_count\n            ) *\n            electrical_energy_per_on_bucket;\n\n        const double target_electrical =\n            fmin(\n                fmax(\n                    0.0,\n                    requested_electrical_energy\n                ),\n                maximum_schedulable\n            );\n\n        if (\n            target_electrical <= 0.0 ||\n            control.valid_count == 0 ||\n            electrical_energy_per_on_bucket <=\n                0.0\n        ) {\n            control.normal_cut =\n                control.valid_count > 0 ?\n                    sorted_export[\n                        SUNOPT_DIGITAL_THREADS -\n                        1\n                    ] :\n                    maximum_hp_power;\n        } else if (\n            target_electrical >=\n            maximum_schedulable -\n                1.0e-12\n        ) {\n            control.normal_cut =\n                0.0;\n        } else {\n            uint32_t required_on =\n                static_cast<uint32_t>(\n                    ceil(\n                        target_electrical /\n                        electrical_energy_per_on_bucket -\n                        1.0e-12\n                    )\n                );\n\n            if (\n                required_on < 1u\n            ) {\n                required_on =\n                    1u;\n            }\n\n            if (\n                required_on >\n                control.valid_count\n            ) {\n                required_on =\n                    control.valid_count;\n            }\n\n            control.normal_cut =\n                fmax(\n                    0.0,\n                    sorted_export[\n                        SUNOPT_DIGITAL_THREADS -\n                        required_on\n                    ] -\n                    maximum_hp_power\n                );\n        }\n\n        const bool affine_supported =\n            tank_parameters[\n                tank_base + 5\n            ] >\n            0.5;\n\n        const double retention =\n            tank_parameters[\n                tank_base + 3\n            ];\n\n        const double clamped_reserve =\n            fmin(\n                target,\n                fmax(\n                    0.0,\n                    reserve\n                )\n            );\n\n        control.parallel_supported =\n            affine_supported &&\n            retention > 0.0 &&\n            horizon > 0 &&\n            horizon <=\n                SUNOPT_DIGITAL_THREADS &&\n            clamped_reserve >\n                SUNOPT_DIGITAL_EPSILON;\n\n        control.eval_count =\n            0u;\n\n        control.final_status =\n            2u;\n\n        control.best_cut =\n            0.0;\n\n        control.done =\n            false;\n    }\n\n    cluster.sync();\n\n    /*\n     * Preserve the proven exact serial fallback for reserve ~= 0.\n     */\n    if (\n        !root->parallel_supported\n    ) {\n        if (\n            cluster_rank ==\n                0u &&\n            tid ==\n                0u\n        ) {\n            control.eval_count =\n                1u;\n\n            const bool normal_survives =\n                sunopt_digital_survives_serial_exact(\n                    export_power,\n                    demand_energy,\n                    data_length,\n                    tank_parameters,\n                    tank_index,\n                    current_index,\n                    horizon,\n                    current_storage,\n                    reserve,\n                    control.normal_cut\n                );\n\n            if (\n                normal_survives\n            ) {\n                control.final_status =\n                    0u;\n\n                control.best_cut =\n                    control.normal_cut;\n            } else {\n                control.eval_count +=\n                    1u;\n\n                const bool maxpv_survives =\n                    sunopt_digital_survives_serial_exact(\n                        export_power,\n                        demand_energy,\n                        data_length,\n                        tank_parameters,\n                        tank_index,\n                        current_index,\n                        horizon,\n                        current_storage,\n                        reserve,\n                        0.0\n                    );\n\n                if (\n                    !maxpv_survives\n                ) {\n                    control.final_status =\n                        1u;\n\n                    control.best_cut =\n                        0.0;\n                } else {\n                    int left =\n                        control.first_valid;\n\n                    int right =\n                        static_cast<int>(\n                            SUNOPT_DIGITAL_THREADS -\n                            1\n                        );\n\n                    double best =\n                        0.0;\n\n                    for (\n                        uint32_t iteration = 0;\n                        iteration < 8u &&\n                        left <= right;\n                        ++iteration\n                    ) {\n                        const int middle =\n                            left +\n                            (\n                                right -\n                                left\n                            ) /\n                            2;\n\n                        const double candidate =\n                            fmax(\n                                0.0,\n                                sorted_export[middle] -\n                                    maximum_hp_power\n                            );\n\n                        control.eval_count +=\n                            1u;\n\n                        if (\n                            sunopt_digital_survives_serial_exact(\n                                export_power,\n                                demand_energy,\n                                data_length,\n                                tank_parameters,\n                                tank_index,\n                                current_index,\n                                horizon,\n                                current_storage,\n                                reserve,\n                                candidate\n                            )\n                        ) {\n                            best =\n                                candidate;\n\n                            left =\n                                middle +\n                                1;\n                        } else {\n                            right =\n                                middle -\n                                1;\n                        }\n                    }\n\n                    control.final_status =\n                        2u;\n\n                    control.best_cut =\n                        best;\n                }\n            }\n\n            cut_results[job_index] =\n                control.best_cut;\n\n            status[job_index] =\n                control.final_status;\n\n            survival_evaluations[job_index] =\n                control.eval_count;\n        }\n\n        cluster.sync();\n\n        return;\n    }\n\n    /*\n     * Normal digital cut.\n     */\n    const bool normal_survives =\n        sunopt_digital_cluster4_survives(\n            export_power,\n            demand_energy,\n            data_length,\n            tank_parameters,\n            tank_index,\n            current_index,\n            horizon,\n            current_storage,\n            reserve,\n            root->normal_cut,\n            warp_partials,\n            &block_summary,\n            &control\n        );\n\n    if (\n        cluster_rank ==\n            0u &&\n        tid ==\n            0u\n    ) {\n        control.eval_count =\n            1u;\n\n        if (\n            normal_survives\n        ) {\n            control.final_status =\n                0u;\n\n            control.best_cut =\n                control.normal_cut;\n\n            control.done =\n                true;\n        }\n    }\n\n    cluster.sync();\n\n    if (\n        root->done\n    ) {\n        if (\n            cluster_rank ==\n                0u &&\n            tid ==\n                0u\n        ) {\n            cut_results[job_index] =\n                control.best_cut;\n\n            status[job_index] =\n                control.final_status;\n\n            survival_evaluations[job_index] =\n                control.eval_count;\n        }\n\n        cluster.sync();\n\n        return;\n    }\n\n    /*\n     * Maximum-PV cut = 0.\n     */\n    const bool maxpv_survives =\n        sunopt_digital_cluster4_survives(\n            export_power,\n            demand_energy,\n            data_length,\n            tank_parameters,\n            tank_index,\n            current_index,\n            horizon,\n            current_storage,\n            reserve,\n            0.0,\n            warp_partials,\n            &block_summary,\n            &control\n        );\n\n    if (\n        cluster_rank ==\n            0u &&\n        tid ==\n            0u\n    ) {\n        control.eval_count =\n            2u;\n\n        if (\n            !maxpv_survives\n        ) {\n            control.final_status =\n                1u;\n\n            control.best_cut =\n                0.0;\n\n            control.done =\n                true;\n        } else {\n            control.left_index =\n                control.first_valid;\n\n            control.right_index =\n                static_cast<int>(\n                    SUNOPT_DIGITAL_THREADS -\n                    1\n                );\n\n            control.best_cut =\n                0.0;\n        }\n    }\n\n    cluster.sync();\n\n    if (\n        root->done\n    ) {\n        if (\n            cluster_rank ==\n                0u &&\n            tid ==\n                0u\n        ) {\n            cut_results[job_index] =\n                control.best_cut;\n\n            status[job_index] =\n                control.final_status;\n\n            survival_evaluations[job_index] =\n                control.eval_count;\n        }\n\n        cluster.sync();\n\n        return;\n    }\n\n    /*\n     * Same exact breakpoint binary search as the B8 reference.\n     * Therefore the number and identity of survival evaluations should match.\n     */\n    for (\n        uint32_t iteration = 0;\n        iteration < 8u;\n        ++iteration\n    ) {\n        if (\n            cluster_rank ==\n                0u &&\n            tid ==\n                0u\n        ) {\n            if (\n                control.left_index >\n                control.right_index\n            ) {\n                control.done =\n                    true;\n            } else {\n                const int middle =\n                    control.left_index +\n                    (\n                        control.right_index -\n                        control.left_index\n                    ) /\n                    2;\n\n                control.candidate_cut =\n                    fmax(\n                        0.0,\n                        sorted_export[middle] -\n                            maximum_hp_power\n                    );\n\n                /*\n                 * Reuse right_index temporarily only after the evaluator:\n                 * middle is recoverable by another exact midpoint operation\n                 * from the current bounds, so no extra shared field is needed.\n                 */\n                control.done =\n                    false;\n            }\n        }\n\n        cluster.sync();\n\n        if (\n            root->done\n        ) {\n            break;\n        }\n\n        const int middle =\n            root->left_index +\n            (\n                root->right_index -\n                root->left_index\n            ) /\n                2;\n\n        const bool survives =\n            sunopt_digital_cluster4_survives(\n                export_power,\n                demand_energy,\n                data_length,\n                tank_parameters,\n                tank_index,\n                current_index,\n                horizon,\n                current_storage,\n                reserve,\n                root->candidate_cut,\n                warp_partials,\n                &block_summary,\n                &control\n            );\n\n        if (\n            cluster_rank ==\n                0u &&\n            tid ==\n                0u\n        ) {\n            control.eval_count +=\n                1u;\n\n            if (\n                survives\n            ) {\n                control.best_cut =\n                    control.candidate_cut;\n\n                control.left_index =\n                    middle +\n                    1;\n            } else {\n                control.right_index =\n                    middle -\n                    1;\n            }\n        }\n\n        cluster.sync();\n    }\n\n    if (\n        cluster_rank ==\n            0u &&\n        tid ==\n            0u\n    ) {\n        control.final_status =\n            2u;\n\n        cut_results[job_index] =\n            control.best_cut;\n\n        status[job_index] =\n            control.final_status;\n\n        survival_evaluations[job_index] =\n            control.eval_count;\n    }\n\n    cluster.sync();\n}\n\n\nextern "C" int sunopt_cuda_run_digital_cluster4_lockstep(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const double* host_presorted_export_windows,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_input_h2d_ms,\n    double* out_kernel_ms,\n    double* out_result_d2h_ms,\n    double* out_wall_ms\n) {\n    using Clock =\n        std::chrono::steady_clock;\n\n    if (\n        host_export_power == nullptr ||\n        host_demand_energy == nullptr ||\n        host_tank_parameters == nullptr ||\n        host_presorted_export_windows == nullptr ||\n        host_current_indices == nullptr ||\n        host_horizon_steps == nullptr ||\n        host_dynamic_values == nullptr ||\n        host_cut_results == nullptr ||\n        host_status == nullptr ||\n        host_survival_evaluations == nullptr ||\n        out_input_h2d_ms == nullptr ||\n        out_kernel_ms == nullptr ||\n        out_result_d2h_ms == nullptr ||\n        out_wall_ms == nullptr ||\n        data_length == 0 ||\n        tank_count == 0 ||\n        tank_count > 32 ||\n        step_count == 0 ||\n        repetitions == 0\n    ) {\n        return static_cast<int>(\n            cudaErrorInvalidValue\n        );\n    }\n\n    cudaError_t err =\n        cudaSetDevice(\n            device_id\n        );\n\n    if (\n        err !=\n        cudaSuccess\n    ) {\n        return static_cast<int>(\n            err\n        );\n    }\n\n    cudaStream_t stream =\n        nullptr;\n\n    cudaEvent_t start_event =\n        nullptr;\n\n    cudaEvent_t stop_event =\n        nullptr;\n\n    double* d_export =\n        nullptr;\n\n    double* d_demand =\n        nullptr;\n\n    double* d_tanks =\n        nullptr;\n\n    double* d_presorted =\n        nullptr;\n\n    uint32_t* d_current =\n        nullptr;\n\n    uint32_t* d_horizon =\n        nullptr;\n\n    double* d_dynamic =\n        nullptr;\n\n    double* d_cuts =\n        nullptr;\n\n    uint32_t* d_status =\n        nullptr;\n\n    uint32_t* d_evaluations =\n        nullptr;\n\n    const size_t total_jobs =\n        step_count *\n        tank_count;\n\n    const size_t annual_bytes =\n        data_length *\n        sizeof(double);\n\n    const size_t tank_bytes =\n        tank_count *\n        7 *\n        sizeof(double);\n\n    const size_t presorted_bytes =\n        step_count *\n        SUNOPT_DIGITAL_THREADS *\n        sizeof(double);\n\n    const size_t step_u32_bytes =\n        step_count *\n        sizeof(uint32_t);\n\n    const size_t dynamic_bytes =\n        total_jobs *\n        3 *\n        sizeof(double);\n\n    const size_t cut_bytes =\n        total_jobs *\n        sizeof(double);\n\n    const size_t u32_job_bytes =\n        total_jobs *\n        sizeof(uint32_t);\n\n    auto cleanup =\n        [&]() {\n            if (d_evaluations) cudaFree(d_evaluations);\n            if (d_status) cudaFree(d_status);\n            if (d_cuts) cudaFree(d_cuts);\n            if (d_dynamic) cudaFree(d_dynamic);\n            if (d_horizon) cudaFree(d_horizon);\n            if (d_current) cudaFree(d_current);\n            if (d_presorted) cudaFree(d_presorted);\n            if (d_tanks) cudaFree(d_tanks);\n            if (d_demand) cudaFree(d_demand);\n            if (d_export) cudaFree(d_export);\n\n            if (stop_event) cudaEventDestroy(stop_event);\n            if (start_event) cudaEventDestroy(start_event);\n            if (stream) cudaStreamDestroy(stream);\n        };\n\n#define SUNOPT_CLUSTER4_CHECK(call) \\\n    do { \\\n        err = (call); \\\n        if (err != cudaSuccess) { \\\n            cleanup(); \\\n            return static_cast<int>(err); \\\n        } \\\n    } while (false)\n\n#define SUNOPT_CLUSTER4_MALLOC(ptr, bytes) \\\n    SUNOPT_CLUSTER4_CHECK( \\\n        cudaMalloc( \\\n            reinterpret_cast<void**>(&(ptr)), \\\n            (bytes) \\\n        ) \\\n    )\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaStreamCreateWithFlags(\n            &stream,\n            cudaStreamNonBlocking\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaEventCreate(\n            &start_event\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaEventCreate(\n            &stop_event\n        )\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_export,\n        annual_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_demand,\n        annual_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_tanks,\n        tank_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_presorted,\n        presorted_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_current,\n        step_u32_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_horizon,\n        step_u32_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_dynamic,\n        dynamic_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_cuts,\n        cut_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_status,\n        u32_job_bytes\n    );\n\n    SUNOPT_CLUSTER4_MALLOC(\n        d_evaluations,\n        u32_job_bytes\n    );\n\n#undef SUNOPT_CLUSTER4_MALLOC\n\n    const auto h2d_start =\n        Clock::now();\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_export,\n            host_export_power,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_demand,\n            host_demand_energy,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_tanks,\n            host_tank_parameters,\n            tank_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_presorted,\n            host_presorted_export_windows,\n            presorted_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_current,\n            host_current_indices,\n            step_u32_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_horizon,\n            host_horizon_steps,\n            step_u32_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            d_dynamic,\n            host_dynamic_values,\n            dynamic_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    const auto h2d_stop =\n        Clock::now();\n\n    *out_input_h2d_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            h2d_stop -\n            h2d_start\n        ).count();\n\n    const auto wall_start =\n        Clock::now();\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaEventRecord(\n            start_event,\n            stream\n        )\n    );\n\n    for (\n        uint32_t repetition = 0;\n        repetition < repetitions;\n        ++repetition\n    ) {\n        for (\n            size_t step = 0;\n            step < step_count;\n            ++step\n        ) {\n            sunopt_digital_cluster4_kernel<<<\n                static_cast<unsigned int>(\n                    tank_count *\n                    4\n                ),\n                64,\n                0,\n                stream\n            >>>(\n                d_export,\n                d_demand,\n                d_presorted,\n                data_length,\n                d_tanks,\n                tank_count,\n                d_current,\n                d_horizon,\n                d_dynamic,\n                step_count,\n                step,\n                d_cuts,\n                d_status,\n                d_evaluations\n            );\n\n            SUNOPT_CLUSTER4_CHECK(\n                cudaGetLastError()\n            );\n        }\n    }\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaEventRecord(\n            stop_event,\n            stream\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaEventSynchronize(\n            stop_event\n        )\n    );\n\n    float kernel_total_ms =\n        0.0f;\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaEventElapsedTime(\n            &kernel_total_ms,\n            start_event,\n            stop_event\n        )\n    );\n\n    *out_kernel_ms =\n        static_cast<double>(\n            kernel_total_ms\n        ) /\n        static_cast<double>(\n            repetitions\n        );\n\n    const auto d2h_start =\n        Clock::now();\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            host_cut_results,\n            d_cuts,\n            cut_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            host_status,\n            d_status,\n            u32_job_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    SUNOPT_CLUSTER4_CHECK(\n        cudaMemcpy(\n            host_survival_evaluations,\n            d_evaluations,\n            u32_job_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    const auto d2h_stop =\n        Clock::now();\n\n    *out_result_d2h_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            d2h_stop -\n            d2h_start\n        ).count();\n\n    const auto wall_stop =\n        Clock::now();\n\n    *out_wall_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            wall_stop -\n            wall_start\n        ).count() /\n        static_cast<double>(\n            repetitions\n        );\n\n#undef SUNOPT_CLUSTER4_CHECK\n\n    cleanup();\n\n    return static_cast<int>(\n        cudaSuccess\n    );\n}\n// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_END'
RS_FFI_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FFI_BEGIN\n    fn sunopt_cuda_run_digital_cluster4_lockstep(\n        device_id: c_int,\n        host_export_power: *const f64,\n        host_demand_energy: *const f64,\n        data_length: usize,\n        host_tank_parameters: *const f64,\n        tank_count: usize,\n        host_presorted_export_windows: *const f64,\n        host_current_indices: *const u32,\n        host_horizon_steps: *const u32,\n        host_dynamic_values: *const f64,\n        step_count: usize,\n        repetitions: u32,\n        host_cut_results: *mut f64,\n        host_status: *mut u32,\n        host_survival_evaluations: *mut u32,\n        out_input_h2d_ms: *mut f64,\n        out_kernel_ms: *mut f64,\n        out_result_d2h_ms: *mut f64,\n        out_wall_ms: *mut f64,\n    ) -> c_int;\n    // SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FFI_END'
RS_RESULT_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_RESULT_BEGIN\n#[napi(object)]\npub struct BlackwellDigitalCluster4Result {\n    pub snapshot_path: String,\n    pub device_id: i32,\n    pub repetitions: u32,\n    pub annual_step_count: u32,\n    pub tank_count: u32,\n    pub causal_jobs: u32,\n\n    pub binary_kernel_ms: f64,\n    pub binary_wall_ms: f64,\n\n    pub cluster_blocks_per_tank: u32,\n    pub cluster_threads_per_block: u32,\n    pub cluster_total_blocks_per_step: u32,\n\n    pub cluster_input_h2d_ms: f64,\n    pub cluster_kernel_ms: f64,\n    pub cluster_result_d2h_ms: f64,\n    pub cluster_wall_ms: f64,\n\n    pub speedup_vs_binary: f64,\n    pub reduction_vs_binary_percent: f64,\n\n    pub status_mismatches: u32,\n    pub cut_mismatches: u32,\n    pub maximum_cut_error: f64,\n    pub evaluation_mismatches: u32,\n\n    pub binary_checksum_hex: String,\n    pub cluster_checksum_hex: String,\n}\n// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_RESULT_END'
RS_FN_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FN_BEGIN\n#[napi]\npub fn run_blackwell_digital_cluster4(\n    snapshot_path: Option<String>,\n    repetitions: Option<u32>,\n    device_id: Option<i32>,\n) -> Result<BlackwellDigitalCluster4Result> {\n    let path =\n        snapshot_path\n            .map(PathBuf::from)\n            .unwrap_or_else(\n                blackwell_default_snapshot_path,\n            );\n\n    let snapshot =\n        read_blackwell_snapshot(&path)\n            .map_err(\n                |error| {\n                    Error::from_reason(\n                        format!(\n                            "Unable to read Blackwell benchmark snapshot {}: {error}",\n                            path.display(),\n                        ),\n                    )\n                },\n            )?;\n\n    if (\n        snapshot.steps.is_empty() ||\n        snapshot.tank_count == 0 ||\n        snapshot.export_power.is_empty() ||\n        snapshot.demand_energy.len() !=\n            snapshot.export_power.len() ||\n        snapshot.steps.len() !=\n            snapshot.data_length\n    ) {\n        return Err(\n            Error::from_reason(\n                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",\n            ),\n        );\n    }\n\n    let selected_device =\n        device_id.unwrap_or(0);\n\n    let repetition_count =\n        repetitions\n            .unwrap_or(1)\n            .clamp(1, 3);\n\n    let tanks =\n        (0..snapshot.tank_count)\n            .map(\n                |tank| {\n                    research_tank_from_packed(\n                        &snapshot.packed_config,\n                        tank,\n                    )\n                },\n            )\n            .collect::<Vec<_>>();\n\n    let mut packed_tanks =\n        Vec::<f64>::with_capacity(\n            snapshot.tank_count * 7,\n        );\n\n    for tank in &tanks {\n        let pmax =\n            tank.maximum_heat_pump_electrical_power;\n\n        packed_tanks.push(\n            tank.target_storage_energy,\n        );\n\n        packed_tanks.push(\n            pmax,\n        );\n\n        packed_tanks.push(\n            pmax *\n            tank.cop *\n            tank.time_step_hours,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_retention,\n        );\n\n        packed_tanks.push(\n            tank.heat_loss_intercept,\n        );\n\n        packed_tanks.push(\n            if tank.affine_supported {\n                1.0\n            } else {\n                0.0\n            },\n        );\n\n        packed_tanks.push(\n            pmax *\n            tank.time_step_hours,\n        );\n    }\n\n    let step_count =\n        snapshot.steps.len();\n\n    let tank_count =\n        snapshot.tank_count;\n\n    let job_count =\n        step_count\n            .checked_mul(tank_count)\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Cluster4 job count overflow.",\n                    )\n                },\n            )?;\n\n    let mut current_indices =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut horizon_steps =\n        Vec::<u32>::with_capacity(\n            step_count,\n        );\n\n    let mut dynamic_flat =\n        Vec::<f64>::with_capacity(\n            job_count * 3,\n        );\n\n    for step in &snapshot.steps {\n        if (\n            step.horizon_steps == 0 ||\n            step.horizon_steps > 256\n        ) {\n            return Err(\n                Error::from_reason(\n                    format!(\n                        "Cluster4 benchmark supports horizon 1..=256; got {}.",\n                        step.horizon_steps,\n                    ),\n                ),\n            );\n        }\n\n        current_indices.push(\n            step.current_index,\n        );\n\n        horizon_steps.push(\n            step.horizon_steps,\n        );\n\n        for tank_index in 0..tank_count {\n            let base =\n                tank_index *\n                BLACKWELL_DYNAMIC_STRIDE;\n\n            dynamic_flat.push(\n                step.dynamic[base + 0],\n            );\n\n            dynamic_flat.push(\n                step.dynamic[base + 1],\n            );\n\n            dynamic_flat.push(\n                step.dynamic[base + 2],\n            );\n        }\n    }\n\n    let presorted_count =\n        step_count\n            .checked_mul(256)\n            .ok_or_else(\n                || {\n                    Error::from_reason(\n                        "Presorted export matrix size overflow.",\n                    )\n                },\n            )?;\n\n    let mut presorted =\n        vec![\n            f64::NEG_INFINITY;\n            presorted_count\n        ];\n\n    for (\n        step_index,\n        step,\n    ) in\n        snapshot.steps\n            .iter()\n            .enumerate()\n    {\n        let horizon =\n            step.horizon_steps as usize;\n\n        let base =\n            step_index * 256;\n\n        let first =\n            base +\n            (\n                256 -\n                horizon\n            );\n\n        for offset in 0..horizon {\n            let index =\n                (\n                    step.current_index as usize +\n                    offset\n                ) %\n                snapshot.export_power.len();\n\n            presorted[\n                first + offset\n            ] =\n                snapshot.export_power[index];\n        }\n\n        presorted[\n            first..\n            base + 256\n        ]\n        .sort_by(\n            |a, b| {\n                a.partial_cmp(b)\n                    .unwrap_or(\n                        std::cmp::Ordering::Equal,\n                    )\n            },\n        );\n    }\n\n    /*\n     * B8 exact resident reference: same annual data, same binary breakpoint\n     * search, one 256-thread block per tank.\n     */\n    let mut binary_cuts =\n        vec![0.0_f64; job_count];\n\n    let mut binary_status =\n        vec![0_u32; job_count];\n\n    let mut binary_evaluations =\n        vec![0_u32; job_count];\n\n    let mut binary_input_h2d_ms =\n        0.0_f64;\n\n    let mut binary_kernel_ms =\n        0.0_f64;\n\n    let mut binary_result_d2h_ms =\n        0.0_f64;\n\n    let mut binary_wall_ms =\n        0.0_f64;\n\n    let binary_code =\n        unsafe {\n            sunopt_cuda_run_digital_resident_chunk_replay(\n                selected_device,\n                snapshot.export_power.as_ptr(),\n                snapshot.demand_energy.as_ptr(),\n                snapshot.export_power.len(),\n                packed_tanks.as_ptr(),\n                tank_count,\n                presorted.as_ptr(),\n                current_indices.as_ptr(),\n                horizon_steps.as_ptr(),\n                dynamic_flat.as_ptr(),\n                step_count,\n                8,\n                1,\n                repetition_count,\n                binary_cuts.as_mut_ptr(),\n                binary_status.as_mut_ptr(),\n                binary_evaluations.as_mut_ptr(),\n                &mut binary_input_h2d_ms,\n                &mut binary_kernel_ms,\n                &mut binary_result_d2h_ms,\n                &mut binary_wall_ms,\n            )\n        };\n\n    if binary_code != 0 {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_resident_chunk_replay",\n                binary_code,\n            ),\n        );\n    }\n\n    let mut cluster_cuts =\n        vec![0.0_f64; job_count];\n\n    let mut cluster_status =\n        vec![0_u32; job_count];\n\n    let mut cluster_evaluations =\n        vec![0_u32; job_count];\n\n    let mut cluster_input_h2d_ms =\n        0.0_f64;\n\n    let mut cluster_kernel_ms =\n        0.0_f64;\n\n    let mut cluster_result_d2h_ms =\n        0.0_f64;\n\n    let mut cluster_wall_ms =\n        0.0_f64;\n\n    let cluster_code =\n        unsafe {\n            sunopt_cuda_run_digital_cluster4_lockstep(\n                selected_device,\n                snapshot.export_power.as_ptr(),\n                snapshot.demand_energy.as_ptr(),\n                snapshot.export_power.len(),\n                packed_tanks.as_ptr(),\n                tank_count,\n                presorted.as_ptr(),\n                current_indices.as_ptr(),\n                horizon_steps.as_ptr(),\n                dynamic_flat.as_ptr(),\n                step_count,\n                repetition_count,\n                cluster_cuts.as_mut_ptr(),\n                cluster_status.as_mut_ptr(),\n                cluster_evaluations.as_mut_ptr(),\n                &mut cluster_input_h2d_ms,\n                &mut cluster_kernel_ms,\n                &mut cluster_result_d2h_ms,\n                &mut cluster_wall_ms,\n            )\n        };\n\n    if cluster_code != 0 {\n        return Err(\n            cuda_error(\n                "sunopt_cuda_run_digital_cluster4_lockstep",\n                cluster_code,\n            ),\n        );\n    }\n\n    let mut status_mismatches =\n        0_u32;\n\n    let mut cut_mismatches =\n        0_u32;\n\n    let mut maximum_cut_error =\n        0.0_f64;\n\n    let mut evaluation_mismatches =\n        0_u32;\n\n    for job in 0..job_count {\n        if (\n            cluster_status[job] !=\n            binary_status[job]\n        ) {\n            status_mismatches +=\n                1;\n        }\n\n        if (\n            cluster_evaluations[job] !=\n            binary_evaluations[job]\n        ) {\n            evaluation_mismatches +=\n                1;\n        }\n\n        let error =\n            (\n                cluster_cuts[job] -\n                binary_cuts[job]\n            )\n            .abs();\n\n        maximum_cut_error =\n            maximum_cut_error.max(\n                error,\n            );\n\n        if error > 1.0e-9 {\n            cut_mismatches +=\n                1;\n        }\n    }\n\n    let speedup_vs_binary =\n        if cluster_wall_ms > 0.0 {\n            binary_wall_ms /\n            cluster_wall_ms\n        } else {\n            f64::INFINITY\n        };\n\n    let reduction_vs_binary_percent =\n        if binary_wall_ms > 0.0 {\n            (\n                binary_wall_ms -\n                cluster_wall_ms\n            ) /\n            binary_wall_ms *\n            100.0\n        } else {\n            0.0\n        };\n\n    let mut binary_checksum =\n        1469598103934665603u64;\n\n    for value in &binary_cuts {\n        blackwell_fnv1a_update(\n            &mut binary_checksum,\n            *value,\n        );\n    }\n\n    let mut cluster_checksum =\n        1469598103934665603u64;\n\n    for value in &cluster_cuts {\n        blackwell_fnv1a_update(\n            &mut cluster_checksum,\n            *value,\n        );\n    }\n\n    Ok(\n        BlackwellDigitalCluster4Result {\n            snapshot_path:\n                path.display()\n                    .to_string(),\n\n            device_id:\n                selected_device,\n\n            repetitions:\n                repetition_count,\n\n            annual_step_count:\n                step_count as u32,\n\n            tank_count:\n                tank_count as u32,\n\n            causal_jobs:\n                job_count as u32,\n\n            binary_kernel_ms,\n            binary_wall_ms,\n\n            cluster_blocks_per_tank:\n                4,\n\n            cluster_threads_per_block:\n                64,\n\n            cluster_total_blocks_per_step:\n                (tank_count * 4) as u32,\n\n            cluster_input_h2d_ms,\n            cluster_kernel_ms,\n            cluster_result_d2h_ms,\n            cluster_wall_ms,\n\n            speedup_vs_binary,\n            reduction_vs_binary_percent,\n\n            status_mismatches,\n            cut_mismatches,\n            maximum_cut_error,\n            evaluation_mismatches,\n\n            binary_checksum_hex:\n                format!(\n                    "{binary_checksum:016x}",\n                ),\n\n            cluster_checksum_hex:\n                format!(\n                    "{cluster_checksum:016x}",\n                ),\n        },\n    )\n}\n// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FN_END'
MJS = 'import { createRequire } from "node:module";\nimport path from "node:path";\nimport { fileURLToPath } from "node:url";\nimport { writeFileSync } from "node:fs";\n\nconst require = createRequire(import.meta.url);\nconst here = path.dirname(fileURLToPath(import.meta.url));\nconst root = path.resolve(here, "..");\n\nconst addon = require(\n  path.join(\n    root,\n    "sunopt_node.win32-x64-msvc.node",\n  ),\n);\n\nconst snapshot = process.argv[2]\n  ? path.resolve(process.argv[2])\n  : path.join(\n      root,\n      "blackwell-benchmark.bin",\n    );\n\nconst repetitions =\n  Number.parseInt(\n    process.argv[3] ?? "1",\n    10,\n  );\n\nconst deviceId =\n  Number.parseInt(\n    process.argv[4] ?? "0",\n    10,\n  );\n\nconst result =\n  addon.runBlackwellDigitalCluster4(\n    snapshot,\n    repetitions,\n    deviceId,\n  );\n\nconsole.log(\n  `SUNOPT_DIGITAL_CLUSTER4_JSON=${JSON.stringify(result)}`,\n);\n\nconsole.log();\nconsole.log(\n  "--- DIGITAL RESIDENT 28-TANK: 1 BLOCK vs 4-BLOCK CLUSTER ---",\n);\n\nconsole.log(`steps=${result.annualStepCount}`);\nconsole.log(`tanks=${result.tankCount}`);\nconsole.log(`jobs=${result.causalJobs}`);\n\nconsole.log();\nconsole.log("BINARY RESIDENT REFERENCE");\nconsole.log(`kernelMs=${result.binaryKernelMs.toFixed(3)}`);\nconsole.log(`wallMs=${result.binaryWallMs.toFixed(3)}`);\nconsole.log(`checksum=${result.binaryChecksumHex}`);\n\nconsole.log();\nconsole.log("CLUSTER4");\nconsole.log(`blocksPerTank=${result.clusterBlocksPerTank}`);\nconsole.log(`threadsPerBlock=${result.clusterThreadsPerBlock}`);\nconsole.log(`blocksPerStep=${result.clusterTotalBlocksPerStep}`);\nconsole.log(`inputH2DMs=${result.clusterInputH2DMs.toFixed(3)}`);\nconsole.log(`kernelMs=${result.clusterKernelMs.toFixed(3)}`);\nconsole.log(`resultD2HMs=${result.clusterResultD2HMs.toFixed(3)}`);\nconsole.log(`wallMs=${result.clusterWallMs.toFixed(3)}`);\nconsole.log(`speedup=${result.speedupVsBinary.toFixed(3)}x`);\nconsole.log(`reduction=${result.reductionVsBinaryPercent.toFixed(2)}%`);\n\nconsole.log();\nconsole.log("FULL ANNUAL PARITY");\nconsole.log(`statusMismatches=${result.statusMismatches}`);\nconsole.log(`cutMismatches=${result.cutMismatches}`);\nconsole.log(`maxCutError=${result.maximumCutError.toExponential(6)}`);\nconsole.log(`evaluationMismatches=${result.evaluationMismatches}`);\nconsole.log(`checksum=${result.clusterChecksumHex}`);\n\nconst report =\n  path.join(\n    root,\n    "blackwell-digital-cluster4.json",\n  );\n\nwriteFileSync(\n  report,\n  JSON.stringify(\n    result,\n    null,\n    2,\n  ) + "\\n",\n  "utf8",\n);\n\nconsole.log();\nconsole.log(`report=${report}`);'


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
            repr(value)
            for value in anchors
        )
    )


def patch_header(source: str) -> str:
    for token in (
        "SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN",
        "sunopt_cuda_run_digital_resident_chunk_replay(",
    ):
        if token not in source:
            raise RuntimeError(
                f"Expected installed B8 header missing {token!r}"
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
        "SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN",
        "sunopt_digital_survives_serial_exact(",
        "sunopt_digital_shuffle_down_transform(",
        "sunopt_digital_compose(",
        "sunopt_digital_identity(",
        "sunopt_digital_hp_on(",
    ):
        if token not in source:
            raise RuntimeError(
                f"Expected current CUDA generation missing {token!r}"
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


def patch_rust(source: str) -> str:
    for token in (
        "fn sunopt_cuda_run_digital_resident_chunk_replay(",
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
                sha256_bytes(data)
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
        "20261109-blackwell-digital-cluster4.zip"
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
        "SunOpt digital Cluster4 benchmark package created"
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
        "  cluster: 4 blocks/tank x 64 threads = 256 threads/tank"
    )
    print(
        "  per step: 28 clusters = 112 schedulable blocks"
    )

    print()
    print("Next:")
    print(
        r"  python .\tools\update_watcher.py --once"
    )
    print(
        r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-cluster4.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node'
    )

    return output


if __name__ == "__main__":
    build_package()
