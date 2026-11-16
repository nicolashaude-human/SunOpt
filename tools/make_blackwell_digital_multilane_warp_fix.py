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

CU_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.cu"
RS_PATH = ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs"

CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
RS_REL = "rust/sunopt-node/src/cuda_context.rs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_FN_END"

CUDA_BLOCK = '// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_BEGIN\n\n__device__ __forceinline__ bool\nsunopt_digital_survives_candidate_warp(\n    const double* export_power,\n    const double* demand_energy,\n    size_t data_length,\n    const double* tank_parameters,\n    size_t tank_index,\n    size_t current_index,\n    size_t horizon,\n    double current_storage_energy,\n    double minimum_reserve_energy,\n    double cut\n) {\n    const unsigned int lane =\n        threadIdx.x & 31u;\n\n    const size_t tank_base =\n        tank_index * 7;\n\n    const double target =\n        tank_parameters[tank_base + 0];\n\n    const double maximum_hp_power =\n        tank_parameters[tank_base + 1];\n\n    const double full_charge =\n        tank_parameters[tank_base + 2];\n\n    const double retention =\n        tank_parameters[tank_base + 3];\n\n    const double loss_intercept =\n        tank_parameters[tank_base + 4];\n\n    const double reserve =\n        fmin(\n            target,\n            fmax(\n                0.0,\n                minimum_reserve_energy\n            )\n        );\n\n    SunoptDigitalTransform local =\n        sunopt_digital_identity();\n\n#pragma unroll\n    for (\n        unsigned int local_offset = 0;\n        local_offset < 8u;\n        ++local_offset\n    ) {\n        const size_t future_offset =\n            1 +\n            static_cast<size_t>(lane) * 8 +\n            local_offset;\n\n        if (\n            future_offset >= horizon\n        ) {\n            continue;\n        }\n\n        const size_t index =\n            (\n                current_index +\n                future_offset\n            ) %\n            data_length;\n\n        const double demand =\n            fmax(\n                0.0,\n                demand_energy[index]\n            );\n\n        const double charge =\n            sunopt_digital_hp_on(\n                export_power[index],\n                cut,\n                maximum_hp_power\n            ) ?\n                full_charge :\n                0.0;\n\n        const double inverse_retention =\n            1.0 /\n            retention;\n\n        SunoptDigitalTransform step =\n            sunopt_digital_identity();\n\n        step.a =\n            inverse_retention;\n\n        step.b =\n            (\n                loss_intercept +\n                demand -\n                charge\n            ) *\n            inverse_retention;\n\n        step.c =\n            (\n                loss_intercept +\n                demand +\n                reserve -\n                SUNOPT_DIGITAL_EPSILON\n            ) *\n            inverse_retention;\n\n        step.max_input =\n            target;\n\n        local =\n            sunopt_digital_compose(\n                local,\n                step\n            );\n    }\n\n    for (\n        unsigned int stride = 1;\n        stride < 32u;\n        stride <<= 1\n    ) {\n        const SunoptDigitalTransform right =\n            sunopt_digital_shuffle_down_transform(\n                local,\n                stride\n            );\n\n        const unsigned int group =\n            stride << 1;\n\n        if (\n            (\n                lane &\n                (\n                    group -\n                    1\n                )\n            ) ==\n            0\n        ) {\n            local =\n                sunopt_digital_compose(\n                    local,\n                    right\n                );\n        }\n    }\n\n    unsigned int survives =\n        0u;\n\n    if (\n        lane ==\n        0\n    ) {\n        const SunoptDigitalTransform summary =\n            local;\n\n        bool feasible =\n            0.0 <=\n            summary.max_input +\n                1.0e-12;\n\n        const double required =\n            fmax(\n                summary.c,\n                summary.b\n            );\n\n        if (\n            required >\n            target +\n                1.0e-9\n        ) {\n            feasible =\n                false;\n        }\n\n        const double current_charge =\n            sunopt_digital_hp_on(\n                export_power[\n                    current_index %\n                    data_length\n                ],\n                cut,\n                maximum_hp_power\n            ) ?\n                full_charge :\n                0.0;\n\n        const double current =\n            fmin(\n                target,\n                fmax(\n                    0.0,\n                    current_storage_energy\n                )\n            );\n\n        const double available =\n            fmin(\n                target,\n                current +\n                    current_charge\n            );\n\n        survives =\n            (\n                feasible &&\n                available +\n                    1.0e-12 >=\n                    required\n            ) ?\n                1u :\n                0u;\n    }\n\n    survives =\n        __shfl_sync(\n            0xffffffffu,\n            survives,\n            0\n        );\n\n    return survives != 0u;\n}\n\n\n__global__ void\nsunopt_digital_multilane_kernel(\n    const double* __restrict__ export_power,\n    const double* __restrict__ demand_energy,\n    const double* __restrict__ presorted_export_windows,\n    size_t data_length,\n    const double* __restrict__ tank_parameters,\n    size_t tank_count,\n    const uint32_t* __restrict__ current_indices,\n    const uint32_t* __restrict__ horizon_steps,\n    const double* __restrict__ dynamic_values,\n    size_t step_count,\n    size_t step_index,\n    uint32_t candidate_warps,\n    double* __restrict__ cut_results,\n    uint32_t* __restrict__ status,\n    uint32_t* __restrict__ survival_evaluations\n) {\n    const size_t tank_index =\n        static_cast<size_t>(\n            blockIdx.x\n        );\n\n    if (\n        tank_index >= tank_count\n    ) {\n        return;\n    }\n\n    const unsigned int tid =\n        threadIdx.x;\n\n    const unsigned int warp_id =\n        tid >> 5;\n\n    const unsigned int lane =\n        tid & 31u;\n\n    if (\n        warp_id >= candidate_warps\n    ) {\n        return;\n    }\n\n    const size_t job_index =\n        step_index *\n            tank_count +\n        tank_index;\n\n    const size_t tank_base =\n        tank_index * 7;\n\n    const double target =\n        tank_parameters[tank_base + 0];\n\n    const double maximum_hp_power =\n        tank_parameters[tank_base + 1];\n\n    const double electrical_energy_per_on_bucket =\n        tank_parameters[tank_base + 6];\n\n    const size_t dynamic_base =\n        job_index * 3;\n\n    const double current_storage =\n        dynamic_values[dynamic_base + 0];\n\n    const double reserve =\n        dynamic_values[dynamic_base + 1];\n\n    const double requested_electrical_energy =\n        dynamic_values[dynamic_base + 2];\n\n    const size_t current_index =\n        static_cast<size_t>(\n            current_indices[step_index]\n        );\n\n    const size_t horizon =\n        static_cast<size_t>(\n            horizon_steps[step_index]\n        );\n\n    const double* sorted_export =\n        presorted_export_windows +\n        step_index *\n            SUNOPT_DIGITAL_THREADS;\n\n    __shared__ double candidate_cuts[8];\n    __shared__ int candidate_indices[8];\n    __shared__ uint32_t candidate_survives[8];\n\n    __shared__ unsigned int active_candidates;\n    __shared__ unsigned int valid_count;\n    __shared__ int first_valid;\n    __shared__ double normal_cut;\n    __shared__ bool parallel_supported;\n    __shared__ bool correction_needed;\n    __shared__ int left_index;\n    __shared__ int right_index;\n    __shared__ double best_cut;\n    __shared__ uint32_t final_status;\n    __shared__ double final_cut;\n    __shared__ uint32_t eval_count;\n\n    if (\n        tid ==\n        0\n    ) {\n        int first =\n            static_cast<int>(\n                SUNOPT_DIGITAL_THREADS\n            );\n\n        for (\n            int i = 0;\n            i <\n                static_cast<int>(\n                    SUNOPT_DIGITAL_THREADS\n                );\n            ++i\n        ) {\n            if (\n                sorted_export[i] >=\n                maximum_hp_power\n            ) {\n                first = i;\n                break;\n            }\n        }\n\n        first_valid =\n            first;\n\n        valid_count =\n            first <\n                static_cast<int>(\n                    SUNOPT_DIGITAL_THREADS\n                ) ?\n                static_cast<unsigned int>(\n                    SUNOPT_DIGITAL_THREADS -\n                    first\n                ) :\n                0u;\n\n        const double maximum_schedulable =\n            static_cast<double>(\n                valid_count\n            ) *\n            electrical_energy_per_on_bucket;\n\n        const double target_electrical =\n            fmin(\n                fmax(\n                    0.0,\n                    requested_electrical_energy\n                ),\n                maximum_schedulable\n            );\n\n        if (\n            target_electrical <= 0.0 ||\n            valid_count == 0 ||\n            electrical_energy_per_on_bucket <=\n                0.0\n        ) {\n            normal_cut =\n                valid_count > 0 ?\n                    sorted_export[\n                        SUNOPT_DIGITAL_THREADS -\n                        1\n                    ] :\n                    maximum_hp_power;\n        } else if (\n            target_electrical >=\n            maximum_schedulable -\n                1.0e-12\n        ) {\n            normal_cut =\n                0.0;\n        } else {\n            unsigned int required_on =\n                static_cast<unsigned int>(\n                    ceil(\n                        target_electrical /\n                        electrical_energy_per_on_bucket -\n                        1.0e-12\n                    )\n                );\n\n            if (\n                required_on < 1u\n            ) {\n                required_on = 1u;\n            }\n\n            if (\n                required_on > valid_count\n            ) {\n                required_on =\n                    valid_count;\n            }\n\n            normal_cut =\n                fmax(\n                    0.0,\n                    sorted_export[\n                        SUNOPT_DIGITAL_THREADS -\n                        required_on\n                    ] -\n                    maximum_hp_power\n                );\n        }\n\n        const bool affine_supported =\n            tank_parameters[\n                tank_base + 5\n            ] >\n            0.5;\n\n        const double retention =\n            tank_parameters[\n                tank_base + 3\n            ];\n\n        const double clamped_reserve =\n            fmin(\n                target,\n                fmax(\n                    0.0,\n                    reserve\n                )\n            );\n\n        parallel_supported =\n            affine_supported &&\n            retention > 0.0 &&\n            horizon > 0 &&\n            horizon <=\n                SUNOPT_DIGITAL_THREADS &&\n            clamped_reserve >\n                SUNOPT_DIGITAL_EPSILON;\n\n        correction_needed =\n            false;\n\n        final_status =\n            2u;\n\n        final_cut =\n            0.0;\n\n        eval_count =\n            0u;\n\n        candidate_cuts[0] =\n            normal_cut;\n\n        candidate_cuts[1] =\n            0.0;\n\n        for (\n            unsigned int warp = 2;\n            warp < candidate_warps;\n            ++warp\n        ) {\n            candidate_cuts[warp] =\n                0.0;\n        }\n    }\n\n    __syncthreads();\n\n    if (\n        !parallel_supported\n    ) {\n        if (\n            tid ==\n            0\n        ) {\n            eval_count =\n                1u;\n\n            const bool normal_survives =\n                sunopt_digital_survives_serial_exact(\n                    export_power,\n                    demand_energy,\n                    data_length,\n                    tank_parameters,\n                    tank_index,\n                    current_index,\n                    horizon,\n                    current_storage,\n                    reserve,\n                    normal_cut\n                );\n\n            if (\n                normal_survives\n            ) {\n                final_status =\n                    0u;\n\n                final_cut =\n                    normal_cut;\n            } else {\n                eval_count +=\n                    1u;\n\n                const bool maxpv_survives =\n                    sunopt_digital_survives_serial_exact(\n                        export_power,\n                        demand_energy,\n                        data_length,\n                        tank_parameters,\n                        tank_index,\n                        current_index,\n                        horizon,\n                        current_storage,\n                        reserve,\n                        0.0\n                    );\n\n                if (\n                    !maxpv_survives\n                ) {\n                    final_status =\n                        1u;\n\n                    final_cut =\n                        0.0;\n                } else {\n                    int left =\n                        first_valid;\n\n                    int right =\n                        static_cast<int>(\n                            SUNOPT_DIGITAL_THREADS -\n                            1\n                        );\n\n                    double best =\n                        0.0;\n\n                    for (\n                        unsigned int iteration = 0;\n                        iteration < 8u &&\n                        left <= right;\n                        ++iteration\n                    ) {\n                        const int middle =\n                            left +\n                            (\n                                right -\n                                left\n                            ) /\n                            2;\n\n                        const double candidate =\n                            fmax(\n                                0.0,\n                                sorted_export[middle] -\n                                    maximum_hp_power\n                            );\n\n                        eval_count +=\n                            1u;\n\n                        if (\n                            sunopt_digital_survives_serial_exact(\n                                export_power,\n                                demand_energy,\n                                data_length,\n                                tank_parameters,\n                                tank_index,\n                                current_index,\n                                horizon,\n                                current_storage,\n                                reserve,\n                                candidate\n                            )\n                        ) {\n                            best =\n                                candidate;\n\n                            left =\n                                middle +\n                                1;\n                        } else {\n                            right =\n                                middle -\n                                1;\n                        }\n                    }\n\n                    final_status =\n                        2u;\n\n                    final_cut =\n                        best;\n                }\n            }\n\n            cut_results[job_index] =\n                final_cut;\n\n            status[job_index] =\n                final_status;\n\n            survival_evaluations[job_index] =\n                eval_count;\n        }\n\n        return;\n    }\n\n    const bool initial_survives =\n        sunopt_digital_survives_candidate_warp(\n            export_power,\n            demand_energy,\n            data_length,\n            tank_parameters,\n            tank_index,\n            current_index,\n            horizon,\n            current_storage,\n            reserve,\n            candidate_cuts[warp_id]\n        );\n\n    if (\n        lane ==\n        0\n    ) {\n        candidate_survives[warp_id] =\n            initial_survives ?\n                1u :\n                0u;\n    }\n\n    __syncthreads();\n\n    if (\n        tid ==\n        0\n    ) {\n        eval_count =\n            2u;\n\n        const bool normal_survives =\n            candidate_survives[0] !=\n            0u;\n\n        const bool maxpv_survives =\n            candidate_survives[1] !=\n            0u;\n\n        if (\n            normal_survives\n        ) {\n            final_status =\n                0u;\n\n            final_cut =\n                normal_cut;\n\n            correction_needed =\n                false;\n        } else if (\n            !maxpv_survives\n        ) {\n            final_status =\n                1u;\n\n            final_cut =\n                0.0;\n\n            correction_needed =\n                false;\n        } else {\n            correction_needed =\n                true;\n\n            left_index =\n                first_valid;\n\n            right_index =\n                static_cast<int>(\n                    SUNOPT_DIGITAL_THREADS -\n                    1\n                );\n\n            best_cut =\n                0.0;\n        }\n    }\n\n    __syncthreads();\n\n    if (\n        !correction_needed\n    ) {\n        if (\n            tid ==\n            0\n        ) {\n            cut_results[job_index] =\n                final_cut;\n\n            status[job_index] =\n                final_status;\n\n            survival_evaluations[job_index] =\n                eval_count;\n        }\n\n        return;\n    }\n\n    for (\n        unsigned int round = 0;\n        round < 8u;\n        ++round\n    ) {\n        if (\n            tid ==\n            0\n        ) {\n            if (\n                left_index >\n                right_index\n            ) {\n                active_candidates =\n                    0u;\n            } else {\n                const int count =\n                    right_index -\n                    left_index +\n                    1;\n\n                active_candidates =\n                    static_cast<unsigned int>(\n                        count\n                    );\n\n                if (\n                    active_candidates >\n                    candidate_warps\n                ) {\n                    active_candidates =\n                        candidate_warps;\n                }\n\n                if (\n                    static_cast<unsigned int>(\n                        count\n                    ) <=\n                    candidate_warps\n                ) {\n                    for (\n                        unsigned int warp = 0;\n                        warp < candidate_warps;\n                        ++warp\n                    ) {\n                        const int index =\n                            warp <\n                                active_candidates ?\n                                (\n                                    left_index +\n                                    static_cast<int>(\n                                        warp\n                                    )\n                                ) :\n                                right_index;\n\n                        candidate_indices[warp] =\n                            index;\n\n                        candidate_cuts[warp] =\n                            fmax(\n                                0.0,\n                                sorted_export[index] -\n                                    maximum_hp_power\n                            );\n                    }\n                } else {\n                    for (\n                        unsigned int warp = 0;\n                        warp < candidate_warps;\n                        ++warp\n                    ) {\n                        const long long numerator =\n                            static_cast<long long>(\n                                warp +\n                                1\n                            ) *\n                            static_cast<long long>(\n                                count -\n                                1\n                            );\n\n                        int index =\n                            left_index +\n                            static_cast<int>(\n                                numerator /\n                                static_cast<long long>(\n                                    candidate_warps +\n                                    1\n                                )\n                            );\n\n                        if (\n                            warp > 0 &&\n                            index <=\n                                candidate_indices[\n                                    warp -\n                                    1\n                                ]\n                        ) {\n                            index =\n                                candidate_indices[\n                                    warp -\n                                    1\n                                ] +\n                                1;\n                        }\n\n                        if (\n                            index >\n                            right_index\n                        ) {\n                            index =\n                                right_index;\n                        }\n\n                        candidate_indices[warp] =\n                            index;\n\n                        candidate_cuts[warp] =\n                            fmax(\n                                0.0,\n                                sorted_export[index] -\n                                    maximum_hp_power\n                            );\n                    }\n                }\n            }\n        }\n\n        __syncthreads();\n\n        if (\n            active_candidates ==\n            0u\n        ) {\n            break;\n        }\n\n        const bool survives =\n            sunopt_digital_survives_candidate_warp(\n                export_power,\n                demand_energy,\n                data_length,\n                tank_parameters,\n                tank_index,\n                current_index,\n                horizon,\n                current_storage,\n                reserve,\n                candidate_cuts[warp_id]\n            );\n\n        if (\n            lane ==\n            0\n        ) {\n            candidate_survives[warp_id] =\n                survives ?\n                    1u :\n                    0u;\n        }\n\n        __syncthreads();\n\n        if (\n            tid ==\n            0\n        ) {\n            eval_count +=\n                active_candidates;\n\n            int highest_surviving_warp =\n                -1;\n\n            for (\n                unsigned int warp = 0;\n                warp < active_candidates;\n                ++warp\n            ) {\n                if (\n                    candidate_survives[warp] !=\n                    0u\n                ) {\n                    highest_surviving_warp =\n                        static_cast<int>(\n                            warp\n                        );\n                } else {\n                    break;\n                }\n            }\n\n            if (\n                highest_surviving_warp <\n                0\n            ) {\n                right_index =\n                    candidate_indices[0] -\n                    1;\n            } else {\n                const unsigned int winner =\n                    static_cast<unsigned int>(\n                        highest_surviving_warp\n                    );\n\n                best_cut =\n                    candidate_cuts[winner];\n\n                left_index =\n                    candidate_indices[winner] +\n                    1;\n\n                if (\n                    winner +\n                        1 <\n                    active_candidates\n                ) {\n                    right_index =\n                        candidate_indices[\n                            winner +\n                            1\n                        ] -\n                        1;\n                }\n            }\n        }\n\n        __syncthreads();\n    }\n\n    if (\n        tid ==\n        0\n    ) {\n        cut_results[job_index] =\n            best_cut;\n\n        status[job_index] =\n            2u;\n\n        survival_evaluations[job_index] =\n            eval_count;\n    }\n}\n\n\nextern "C" int sunopt_cuda_run_digital_multilane_lockstep(\n    int device_id,\n    const double* host_export_power,\n    const double* host_demand_energy,\n    size_t data_length,\n    const double* host_tank_parameters,\n    size_t tank_count,\n    const double* host_presorted_export_windows,\n    const uint32_t* host_current_indices,\n    const uint32_t* host_horizon_steps,\n    const double* host_dynamic_values,\n    size_t step_count,\n    uint32_t candidate_warps,\n    uint32_t repetitions,\n    double* host_cut_results,\n    uint32_t* host_status,\n    uint32_t* host_survival_evaluations,\n    double* out_input_h2d_ms,\n    double* out_kernel_ms,\n    double* out_result_d2h_ms,\n    double* out_wall_ms\n) {\n    using Clock =\n        std::chrono::steady_clock;\n\n    if (\n        host_export_power == nullptr ||\n        host_demand_energy == nullptr ||\n        host_tank_parameters == nullptr ||\n        host_presorted_export_windows == nullptr ||\n        host_current_indices == nullptr ||\n        host_horizon_steps == nullptr ||\n        host_dynamic_values == nullptr ||\n        host_cut_results == nullptr ||\n        host_status == nullptr ||\n        host_survival_evaluations == nullptr ||\n        out_input_h2d_ms == nullptr ||\n        out_kernel_ms == nullptr ||\n        out_result_d2h_ms == nullptr ||\n        out_wall_ms == nullptr ||\n        data_length == 0 ||\n        tank_count == 0 ||\n        tank_count > 32 ||\n        step_count == 0 ||\n        candidate_warps < 2 ||\n        candidate_warps > 8 ||\n        repetitions == 0\n    ) {\n        return static_cast<int>(\n            cudaErrorInvalidValue\n        );\n    }\n\n    const unsigned int threads =\n        candidate_warps *\n        32u;\n\n    cudaError_t err =\n        cudaSetDevice(\n            device_id\n        );\n\n    if (\n        err !=\n        cudaSuccess\n    ) {\n        return static_cast<int>(\n            err\n        );\n    }\n\n    cudaStream_t stream =\n        nullptr;\n\n    cudaEvent_t start_event =\n        nullptr;\n\n    cudaEvent_t stop_event =\n        nullptr;\n\n    double* d_export =\n        nullptr;\n\n    double* d_demand =\n        nullptr;\n\n    double* d_tanks =\n        nullptr;\n\n    double* d_presorted =\n        nullptr;\n\n    uint32_t* d_current =\n        nullptr;\n\n    uint32_t* d_horizon =\n        nullptr;\n\n    double* d_dynamic =\n        nullptr;\n\n    double* d_cuts =\n        nullptr;\n\n    uint32_t* d_status =\n        nullptr;\n\n    uint32_t* d_evaluations =\n        nullptr;\n\n    const size_t total_jobs =\n        step_count *\n        tank_count;\n\n    const size_t annual_bytes =\n        data_length *\n        sizeof(double);\n\n    const size_t tank_bytes =\n        tank_count *\n        7 *\n        sizeof(double);\n\n    const size_t presorted_bytes =\n        step_count *\n        SUNOPT_DIGITAL_THREADS *\n        sizeof(double);\n\n    const size_t step_u32_bytes =\n        step_count *\n        sizeof(uint32_t);\n\n    const size_t dynamic_bytes =\n        total_jobs *\n        3 *\n        sizeof(double);\n\n    const size_t cut_bytes =\n        total_jobs *\n        sizeof(double);\n\n    const size_t u32_job_bytes =\n        total_jobs *\n        sizeof(uint32_t);\n\n    auto cleanup =\n        [&]() {\n            if (d_evaluations) cudaFree(d_evaluations);\n            if (d_status) cudaFree(d_status);\n            if (d_cuts) cudaFree(d_cuts);\n            if (d_dynamic) cudaFree(d_dynamic);\n            if (d_horizon) cudaFree(d_horizon);\n            if (d_current) cudaFree(d_current);\n            if (d_presorted) cudaFree(d_presorted);\n            if (d_tanks) cudaFree(d_tanks);\n            if (d_demand) cudaFree(d_demand);\n            if (d_export) cudaFree(d_export);\n\n            if (stop_event) cudaEventDestroy(stop_event);\n            if (start_event) cudaEventDestroy(start_event);\n            if (stream) cudaStreamDestroy(stream);\n        };\n\n#define SUNOPT_MULTILANE_CHECK(call) \\\n    do { \\\n        err = (call); \\\n        if (err != cudaSuccess) { \\\n            cleanup(); \\\n            return static_cast<int>(err); \\\n        } \\\n    } while (false)\n\n#define SUNOPT_MULTILANE_MALLOC(ptr, bytes) \\\n    SUNOPT_MULTILANE_CHECK( \\\n        cudaMalloc( \\\n            reinterpret_cast<void**>(&(ptr)), \\\n            (bytes) \\\n        ) \\\n    )\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaStreamCreateWithFlags(\n            &stream,\n            cudaStreamNonBlocking\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaEventCreate(\n            &start_event\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaEventCreate(\n            &stop_event\n        )\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_export,\n        annual_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_demand,\n        annual_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_tanks,\n        tank_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_presorted,\n        presorted_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_current,\n        step_u32_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_horizon,\n        step_u32_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_dynamic,\n        dynamic_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_cuts,\n        cut_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_status,\n        u32_job_bytes\n    );\n\n    SUNOPT_MULTILANE_MALLOC(\n        d_evaluations,\n        u32_job_bytes\n    );\n\n#undef SUNOPT_MULTILANE_MALLOC\n\n    const auto h2d_start =\n        Clock::now();\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_export,\n            host_export_power,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_demand,\n            host_demand_energy,\n            annual_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_tanks,\n            host_tank_parameters,\n            tank_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_presorted,\n            host_presorted_export_windows,\n            presorted_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_current,\n            host_current_indices,\n            step_u32_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_horizon,\n            host_horizon_steps,\n            step_u32_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            d_dynamic,\n            host_dynamic_values,\n            dynamic_bytes,\n            cudaMemcpyHostToDevice\n        )\n    );\n\n    const auto h2d_stop =\n        Clock::now();\n\n    *out_input_h2d_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            h2d_stop -\n            h2d_start\n        ).count();\n\n    const auto wall_start =\n        Clock::now();\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaEventRecord(\n            start_event,\n            stream\n        )\n    );\n\n    for (\n        uint32_t repetition = 0;\n        repetition < repetitions;\n        ++repetition\n    ) {\n        for (\n            size_t step = 0;\n            step < step_count;\n            ++step\n        ) {\n            sunopt_digital_multilane_kernel<<<\n                static_cast<unsigned int>(\n                    tank_count\n                ),\n                threads,\n                0,\n                stream\n            >>>(\n                d_export,\n                d_demand,\n                d_presorted,\n                data_length,\n                d_tanks,\n                tank_count,\n                d_current,\n                d_horizon,\n                d_dynamic,\n                step_count,\n                step,\n                candidate_warps,\n                d_cuts,\n                d_status,\n                d_evaluations\n            );\n\n            SUNOPT_MULTILANE_CHECK(\n                cudaGetLastError()\n            );\n        }\n    }\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaEventRecord(\n            stop_event,\n            stream\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaEventSynchronize(\n            stop_event\n        )\n    );\n\n    float kernel_total_ms =\n        0.0f;\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaEventElapsedTime(\n            &kernel_total_ms,\n            start_event,\n            stop_event\n        )\n    );\n\n    *out_kernel_ms =\n        static_cast<double>(\n            kernel_total_ms\n        ) /\n        static_cast<double>(\n            repetitions\n        );\n\n    const auto d2h_start =\n        Clock::now();\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            host_cut_results,\n            d_cuts,\n            cut_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            host_status,\n            d_status,\n            u32_job_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    SUNOPT_MULTILANE_CHECK(\n        cudaMemcpy(\n            host_survival_evaluations,\n            d_evaluations,\n            u32_job_bytes,\n            cudaMemcpyDeviceToHost\n        )\n    );\n\n    const auto d2h_stop =\n        Clock::now();\n\n    *out_result_d2h_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            d2h_stop -\n            d2h_start\n        ).count();\n\n    const auto wall_stop =\n        Clock::now();\n\n    *out_wall_ms =\n        std::chrono::duration<\n            double,\n            std::milli\n        >(\n            wall_stop -\n            wall_start\n        ).count() /\n        static_cast<double>(\n            repetitions\n        );\n\n#undef SUNOPT_MULTILANE_CHECK\n\n    cleanup();\n\n    return static_cast<int>(\n        cudaSuccess\n    );\n}\n\n// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_END'


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
    if begin not in source or end not in source:
        raise RuntimeError(
            f"Expected installed B10 marker missing: {begin} / {end}"
        )

    start = source.index(begin)
    finish = source.index(end, start) + len(end)

    return (
        source[:start] +
        block +
        source[finish:]
    )


def patch_cuda(source: str) -> str:
    required = (
        CU_BEGIN,
        "sunopt_digital_survives_serial_exact(",
        "sunopt_digital_shuffle_down_transform(",
        "sunopt_digital_compose(",
        "sunopt_digital_identity(",
        "sunopt_digital_hp_on(",
    )

    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Current B10 CUDA source missing {token!r}"
            )

    return replace_between(
        source,
        CU_BEGIN,
        CU_END,
        CUDA_BLOCK,
    )


def patch_rust(source: str) -> str:
    if (
        RS_FN_BEGIN not in source or
        RS_FN_END not in source
    ):
        raise RuntimeError(
            "Current B10 Rust function markers are missing."
        )

    start = source.index(RS_FN_BEGIN)
    finish = source.index(
        RS_FN_END,
        start,
    ) + len(RS_FN_END)

    block = source[start:finish]

    old_lanes = '''    let lane_counts:
        [u32; 3] =
        [
            2,
            3,
            4,
        ];'''

    new_lanes = '''    let lane_counts:
        [u32; 3] =
        [
            2,
            4,
            8,
        ];'''

    if old_lanes not in block:
        raise RuntimeError(
            "Could not locate the installed B10 lane-count table."
        )

    block = block.replace(
        old_lanes,
        new_lanes,
        1,
    )

    old_threads = '''                threads_per_block:
                    lane_count *
                    256,''';

    new_threads = '''                threads_per_block:
                    lane_count *
                    32,''';

    if old_threads not in block:
        raise RuntimeError(
            "Could not locate the installed B10 thread-count expression."
        )

    block = block.replace(
        old_threads,
        new_threads,
        1,
    )

    return (
        source[:start] +
        block +
        source[finish:]
    )


def build_package() -> Path:
    for path in (
        CU_PATH,
        RS_PATH,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing project file: {path}"
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
        CU_REL:
            cu.encode(
                "utf-8"
            ),

        RS_REL:
            rs.encode(
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
        "20261109-blackwell-digital-multilane-warp-fix.zip"
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
        "SunOpt B10b warp-granular multilane fix created"
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
        "  candidate warps: 2 / 4 / 8"
    )
    print(
        "  threads/tank: 64 / 128 / 256"
    )

    print()
    print("Next:")
    print(
        r"  python .\tools\update_watcher.py --once"
    )
    print(
        r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-multilane-search.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node'
    )

    return output


if __name__ == "__main__":
    build_package()
