#!/usr/bin/env python3
# tools/make_blackwell_digital_hp_breakpoints.py
#
# Run from:
#   C:\NextJS\SunOpt\sunopt
#
# Creates a verified format-2 SunOpt update in updates/inbox.
#
# Research benchmark only. Production physics is NOT changed.
#
# Ideal digital HP model used by BOTH scalar reference and CUDA candidate:
#
#   HP = Pmax  if exportPower - cutLevel >= Pmax
#   HP = 0     otherwise
#
# No hysteresis, no minimum run time, no compressor dynamics.
#
# For this model every future time bucket has one relevant cut breakpoint:
#
#   breakpoint = exportPower - Pmax
#
# The CUDA candidate sorts the <=256 breakpoints once per causal job and
# binary-searches the largest breakpoint whose exact max-affine survival tree
# still survives. At most 8 breakpoint survival evaluations are required for a
# 192-step horizon.
#
# Existing continuous/modulating production code remains untouched.

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
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-hp-breakpoints.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FN_END"


H_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN
/*
 * Research-only ideal digital-HP breakpoint solver.
 *
 * tank_parameters: 7 doubles/tank
 *   0 target_storage_energy
 *   1 maximum_heat_pump_electrical_power
 *   2 hp_thermal_energy_per_on_bucket (= Pmax * COP * dt)
 *   3 heat_loss_retention
 *   4 heat_loss_intercept
 *   5 affine_supported (1/0)
 *   6 hp_electrical_energy_per_on_bucket (= Pmax * dt)
 *
 * dynamic_values: 3 doubles/job
 *   0 current_storage_energy
 *   1 minimum_reserve_energy
 *   2 requested_electrical_energy
 *
 * status:
 *   0 = normal digital energy cut survives
 *   1 = even maximum-PV digital plan fails
 *   2 = correction breakpoint selected
 *   3 = unsupported exact max-affine branch
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_hp_breakpoints(
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
    double* out_kernel_total_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_END
'''.strip()


CU_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN
#include <chrono>
#include <math_constants.h>

namespace {

constexpr unsigned int SUNOPT_DIGITAL_THREADS = 256;
constexpr double SUNOPT_DIGITAL_EPSILON = 1.0e-6;


struct SunoptDigitalTransform {
    double a;
    double b;
    double c;
    double max_input;
};


__device__ __forceinline__ SunoptDigitalTransform
sunopt_digital_identity() {
    SunoptDigitalTransform t{};
    t.a = 1.0;
    t.b = 0.0;
    t.c = -CUDART_INF;
    t.max_input = CUDART_INF;
    return t;
}


__device__ __forceinline__ SunoptDigitalTransform
sunopt_digital_compose(
    const SunoptDigitalTransform left,
    const SunoptDigitalTransform right
) {
    SunoptDigitalTransform out{};

    double max_input =
        right.max_input;

    if (
        right.c >
        left.max_input +
        1.0e-12
    ) {
        max_input =
            -CUDART_INF;
    } else if (
        right.a >
        0.0
    ) {
        const double bound =
            (
                left.max_input -
                right.b
            ) /
            right.a;

        max_input =
            fmin(
                max_input,
                bound
            );
    } else if (
        right.b >
        left.max_input +
        1.0e-12
    ) {
        max_input =
            -CUDART_INF;
    }

    out.a =
        left.a *
        right.a;

    out.b =
        left.a *
        right.b +
        left.b;

    out.c =
        fmax(
            left.c,
            left.a *
            right.c +
            left.b
        );

    out.max_input =
        max_input;

    return out;
}


__device__ __forceinline__ bool
sunopt_digital_hp_on(
    double export_power,
    double cut,
    double maximum_hp_power
) {
    return (
        maximum_hp_power >
            0.0 &&
        export_power -
            cut >=
            maximum_hp_power
    );
}


/*
 * Exact max-affine survival evaluation at one DIGITAL cut.
 *
 * Every participating thread creates one future transform. Because digital
 * charge is constant inside a breakpoint interval, no cut derivative is
 * required.
 */
__device__ __forceinline__ bool
sunopt_digital_survives(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    const double* tank_parameters,
    size_t tank_index,
    size_t current_index,
    size_t horizon,
    double current_storage_energy,
    double minimum_reserve_energy,
    double cut,
    SunoptDigitalTransform* shared_transform,
    bool* shared_supported
) {
    const unsigned int tid =
        threadIdx.x;

    const size_t tank_base =
        tank_index *
        7;

    const double target =
        tank_parameters[
            tank_base + 0
        ];

    const double maximum_hp_power =
        tank_parameters[
            tank_base + 1
        ];

    const double full_charge =
        tank_parameters[
            tank_base + 2
        ];

    const double retention =
        tank_parameters[
            tank_base + 3
        ];

    const double loss_intercept =
        tank_parameters[
            tank_base + 4
        ];

    const bool affine_supported =
        tank_parameters[
            tank_base + 5
        ] >
        0.5;

    const double reserve =
        fmin(
            target,
            fmax(
                0.0,
                minimum_reserve_energy
            )
        );

    const bool supported =
        affine_supported &&
        retention >
            0.0 &&
        data_length >
            0 &&
        horizon >
            0 &&
        horizon <=
            SUNOPT_DIGITAL_THREADS &&
        reserve >
            SUNOPT_DIGITAL_EPSILON;

    if (
        !supported
    ) {
        if (
            tid ==
            0
        ) {
            *shared_supported =
                false;
        }

        __syncthreads();

        return false;
    }

    const size_t future_offset =
        static_cast<size_t>(
            tid
        ) +
        1;

    SunoptDigitalTransform local =
        sunopt_digital_identity();

    if (
        future_offset <
        horizon
    ) {
        const size_t index =
            (
                current_index +
                future_offset
            ) %
            data_length;

        const double demand =
            fmax(
                0.0,
                demand_energy[
                    index
                ]
            );

        const double charge =
            sunopt_digital_hp_on(
                export_power[
                    index
                ],
                cut,
                maximum_hp_power
            ) ?
                full_charge :
                0.0;

        const double inverse_retention =
            1.0 /
            retention;

        local.a =
            inverse_retention;

        local.b =
            (
                loss_intercept +
                demand -
                charge
            ) *
            inverse_retention;

        local.c =
            (
                loss_intercept +
                demand +
                reserve -
                SUNOPT_DIGITAL_EPSILON
            ) *
            inverse_retention;

        local.max_input =
            target;
    }

    shared_transform[
        tid
    ] =
        local;

    __syncthreads();

    for (
        unsigned int stride = 1;
        stride <
            SUNOPT_DIGITAL_THREADS;
        stride <<=
            1
    ) {
        const unsigned int group =
            stride <<
            1;

        if (
            (
                tid &
                (
                    group -
                    1
                )
            ) ==
            0
        ) {
            const unsigned int right =
                tid +
                stride;

            if (
                right <
                SUNOPT_DIGITAL_THREADS
            ) {
                shared_transform[
                    tid
                ] =
                    sunopt_digital_compose(
                        shared_transform[
                            tid
                        ],
                        shared_transform[
                            right
                        ]
                    );
            }
        }

        __syncthreads();
    }

    __shared__ bool survives;

    if (
        tid ==
        0
    ) {
        const SunoptDigitalTransform summary =
            shared_transform[
                0
            ];

        bool feasible =
            0.0 <=
            summary.max_input +
            1.0e-12;

        const double required =
            fmax(
                summary.c,
                summary.b
            );

        if (
            required >
            target +
            1.0e-9
        ) {
            feasible =
                false;
        }

        const double current_charge =
            sunopt_digital_hp_on(
                export_power[
                    current_index %
                    data_length
                ],
                cut,
                maximum_hp_power
            ) ?
                full_charge :
                0.0;

        const double current =
            fmin(
                target,
                fmax(
                    0.0,
                    current_storage_energy
                )
            );

        const double available =
            fmin(
                target,
                current +
                current_charge
            );

        survives =
            feasible &&
            available +
                1.0e-12 >=
                required;

        *shared_supported =
            true;
    }

    __syncthreads();

    return survives;
}


__global__ void
sunopt_digital_hp_breakpoint_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    const size_t job_index =
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

    const unsigned int tid =
        threadIdx.x;

    const size_t step_index =
        job_index /
        tank_count;

    const size_t tank_index =
        job_index -
        step_index *
        tank_count;

    const size_t tank_base =
        tank_index *
        7;

    const double target =
        tank_parameters[
            tank_base + 0
        ];

    const double maximum_hp_power =
        tank_parameters[
            tank_base + 1
        ];

    const double electrical_energy_per_on_bucket =
        tank_parameters[
            tank_base + 6
        ];

    const size_t dynamic_base =
        job_index *
        3;

    const double current_storage =
        dynamic_values[
            dynamic_base + 0
        ];

    const double reserve =
        dynamic_values[
            dynamic_base + 1
        ];

    const double requested_electrical_energy =
        dynamic_values[
            dynamic_base + 2
        ];

    const size_t current_index =
        static_cast<size_t>(
            current_indices[
                step_index
            ]
        );

    const size_t horizon =
        static_cast<size_t>(
            horizon_steps[
                step_index
            ]
        );

    __shared__ double breakpoints[
        SUNOPT_DIGITAL_THREADS
    ];

    __shared__ SunoptDigitalTransform transforms[
        SUNOPT_DIGITAL_THREADS
    ];

    __shared__ bool shared_supported;
    __shared__ double normal_cut;
    __shared__ unsigned int valid_count;
    __shared__ unsigned int eval_count;
    __shared__ double final_cut;
    __shared__ uint32_t final_status;

    /*
     * A breakpoint is relevant only when the HP can run at cut=0.
     * Negative breakpoints are represented as -INF and sort to the start.
     */
    double breakpoint =
        -CUDART_INF;

    if (
        static_cast<size_t>(
            tid
        ) <
        horizon
    ) {
        const size_t index =
            (
                current_index +
                static_cast<size_t>(
                    tid
                )
            ) %
            data_length;

        const double export_value =
            export_power[
                index
            ];

        if (
            maximum_hp_power >
                0.0 &&
            export_value >=
                maximum_hp_power
        ) {
            breakpoint =
                export_value -
                maximum_hp_power;
        }
    }

    breakpoints[
        tid
    ] =
        breakpoint;

    __syncthreads();

    /*
     * In-block bitonic sort, ascending.
     * 256 is a power of two; invalid -INF entries end up first.
     */
    for (
        unsigned int k = 2;
        k <=
            SUNOPT_DIGITAL_THREADS;
        k <<=
            1
    ) {
        for (
            unsigned int j =
                k >>
                1;
            j >
                0;
            j >>=
                1
        ) {
            const unsigned int partner =
                tid ^
                j;

            if (
                partner >
                tid
            ) {
                const bool ascending =
                    (
                        tid &
                        k
                    ) ==
                    0;

                const double a =
                    breakpoints[
                        tid
                    ];

                const double b =
                    breakpoints[
                        partner
                    ];

                const bool swap =
                    ascending ?
                        (
                            a >
                            b
                        ) :
                        (
                            a <
                            b
                        );

                if (
                    swap
                ) {
                    breakpoints[
                        tid
                    ] =
                        b;

                    breakpoints[
                        partner
                    ] =
                        a;
                }
            }

            __syncthreads();
        }
    }

    if (
        tid ==
        0
    ) {
        unsigned int count =
            0;

        for (
            unsigned int i = 0;
            i <
                SUNOPT_DIGITAL_THREADS;
            ++i
        ) {
            if (
                breakpoints[
                    i
                ] !=
                -CUDART_INF
            ) {
                count +=
                    1;
            }
        }

        valid_count =
            count;

        eval_count =
            0;

        final_cut =
            0.0;

        final_status =
            2;

        /*
         * Digital normal-cut analogue:
         *
         * choose the highest cut that still schedules at least the requested
         * electrical energy. Ties can schedule more than requested by one or
         * more full buckets, which is unavoidable in an ideal ON/OFF model.
         */
        const double maximum_schedulable =
            static_cast<double>(
                count
            ) *
            electrical_energy_per_on_bucket;

        const double target_electrical =
            fmin(
                fmax(
                    0.0,
                    requested_electrical_energy
                ),
                maximum_schedulable
            );

        if (
            target_electrical <=
                0.0 ||
            count ==
                0 ||
            electrical_energy_per_on_bucket <=
                0.0
        ) {
            if (
                count >
                0
            ) {
                normal_cut =
                    breakpoints[
                        SUNOPT_DIGITAL_THREADS -
                        1
                    ] +
                    maximum_hp_power;
            } else {
                normal_cut =
                    maximum_hp_power;
            }
        } else if (
            target_electrical >=
            maximum_schedulable -
                1.0e-12
        ) {
            normal_cut =
                0.0;
        } else {
            unsigned int required_on =
                static_cast<unsigned int>(
                    ceil(
                        target_electrical /
                        electrical_energy_per_on_bucket -
                        1.0e-12
                    )
                );

            required_on =
                max(
                    1u,
                    min(
                        required_on,
                        count
                    )
                );

            normal_cut =
                breakpoints[
                    SUNOPT_DIGITAL_THREADS -
                    required_on
                ];
        }
    }

    __syncthreads();

    /*
     * Exact support check is performed by the survival evaluator.
     */
    bool normal_survives =
        sunopt_digital_survives(
            export_power,
            demand_energy,
            data_length,
            tank_parameters,
            tank_index,
            current_index,
            horizon,
            current_storage,
            reserve,
            normal_cut,
            transforms,
            &shared_supported
        );

    if (
        tid ==
        0
    ) {
        eval_count +=
            1;

        if (
            !shared_supported
        ) {
            final_status =
                3;
            final_cut =
                0.0;
        } else if (
            normal_survives
        ) {
            final_status =
                0;
            final_cut =
                normal_cut;
        }
    }

    __syncthreads();

    if (
        final_status ==
            0 ||
        final_status ==
            3
    ) {
        if (
            tid ==
            0
        ) {
            cut_results[
                job_index
            ] =
                final_cut;

            status[
                job_index
            ] =
                final_status;

            survival_evaluations[
                job_index
            ] =
                eval_count;
        }

        return;
    }

    bool maxpv_survives =
        sunopt_digital_survives(
            export_power,
            demand_energy,
            data_length,
            tank_parameters,
            tank_index,
            current_index,
            horizon,
            current_storage,
            reserve,
            0.0,
            transforms,
            &shared_supported
        );

    if (
        tid ==
        0
    ) {
        eval_count +=
            1;

        if (
            !maxpv_survives
        ) {
            final_status =
                1;
            final_cut =
                0.0;
        }
    }

    __syncthreads();

    if (
        final_status ==
        1
    ) {
        if (
            tid ==
            0
        ) {
            cut_results[
                job_index
            ] =
                0.0;

            status[
                job_index
            ] =
                1;

            survival_evaluations[
                job_index
            ] =
                eval_count;
        }

        return;
    }

    /*
     * Monotonic breakpoint search.
     *
     * All valid breakpoints are in the final valid_count positions of the
     * ascending array. cut=0 already survives. We find the largest surviving
     * breakpoint. A 192-step horizon needs <=8 iterations.
     */
    __shared__ int left_index;
    __shared__ int right_index;
    __shared__ int middle_index;
    __shared__ double candidate_cut;
    __shared__ double best_cut;

    if (
        tid ==
        0
    ) {
        left_index =
            static_cast<int>(
                SUNOPT_DIGITAL_THREADS -
                valid_count
            );

        right_index =
            static_cast<int>(
                SUNOPT_DIGITAL_THREADS -
                1
            );

        best_cut =
            0.0;
    }

    __syncthreads();

    for (
        uint32_t iteration = 0;
        iteration <
            search_iterations;
        ++iteration
    ) {
        if (
            tid ==
            0
        ) {
            if (
                left_index <=
                right_index
            ) {
                middle_index =
                    left_index +
                    (
                        right_index -
                        left_index
                    ) /
                    2;

                candidate_cut =
                    fmax(
                        0.0,
                        breakpoints[
                            middle_index
                        ]
                    );
            } else {
                middle_index =
                    -1;

                candidate_cut =
                    best_cut;
            }
        }

        __syncthreads();

        if (
            middle_index <
            0
        ) {
            break;
        }

        const bool candidate_survives =
            sunopt_digital_survives(
                export_power,
                demand_energy,
                data_length,
                tank_parameters,
                tank_index,
                current_index,
                horizon,
                current_storage,
                reserve,
                candidate_cut,
                transforms,
                &shared_supported
            );

        if (
            tid ==
            0
        ) {
            eval_count +=
                1;

            if (
                candidate_survives
            ) {
                best_cut =
                    candidate_cut;

                left_index =
                    middle_index +
                    1;
            } else {
                right_index =
                    middle_index -
                    1;
            }
        }

        __syncthreads();
    }

    if (
        tid ==
        0
    ) {
        final_status =
            2;

        final_cut =
            best_cut;

        cut_results[
            job_index
        ] =
            final_cut;

        status[
            job_index
        ] =
            final_status;

        survival_evaluations[
            job_index
        ] =
            eval_count;
    }
}


} // namespace


extern "C" int sunopt_cuda_run_digital_hp_breakpoints(
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
    double* out_kernel_total_ms,
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
        out_kernel_total_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        step_count == 0 ||
        repetitions == 0
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

    cudaEvent_t start_event =
        nullptr;

    cudaEvent_t stop_event =
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

    const size_t job_count =
        step_count *
        tank_count;

    const size_t annual_bytes =
        data_length *
        sizeof(double);

    const size_t tank_bytes =
        tank_count *
        7 *
        sizeof(double);

    const size_t step_bytes =
        step_count *
        sizeof(uint32_t);

    const size_t dynamic_bytes =
        job_count *
        3 *
        sizeof(double);

    const size_t cut_bytes =
        job_count *
        sizeof(double);

    const size_t u32_job_bytes =
        job_count *
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
            if (stop_event) cudaEventDestroy(stop_event);
            if (start_event) cudaEventDestroy(start_event);
            if (stream) cudaStreamDestroy(stream);
        };

#define SUNOPT_DIGITAL_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

    SUNOPT_DIGITAL_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_DIGITAL_CHECK(
        cudaEventCreate(
            &start_event
        )
    );

    SUNOPT_DIGITAL_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );

#define SUNOPT_DIGITAL_MALLOC(ptr, bytes) \
    SUNOPT_DIGITAL_CHECK( \
        cudaMalloc( \
            reinterpret_cast<void**>(&(ptr)), \
            (bytes) \
        ) \
    )

    SUNOPT_DIGITAL_MALLOC(d_export, annual_bytes);
    SUNOPT_DIGITAL_MALLOC(d_demand, annual_bytes);
    SUNOPT_DIGITAL_MALLOC(d_tanks, tank_bytes);
    SUNOPT_DIGITAL_MALLOC(d_current, step_bytes);
    SUNOPT_DIGITAL_MALLOC(d_horizon, step_bytes);
    SUNOPT_DIGITAL_MALLOC(d_dynamic, dynamic_bytes);
    SUNOPT_DIGITAL_MALLOC(d_cuts, cut_bytes);
    SUNOPT_DIGITAL_MALLOC(d_status, u32_job_bytes);
    SUNOPT_DIGITAL_MALLOC(d_evaluations, u32_job_bytes);

#undef SUNOPT_DIGITAL_MALLOC

#define SUNOPT_DIGITAL_H2D(dst, src, bytes) \
    SUNOPT_DIGITAL_CHECK( \
        cudaMemcpy( \
            (dst), \
            (src), \
            (bytes), \
            cudaMemcpyHostToDevice \
        ) \
    )

    SUNOPT_DIGITAL_H2D(d_export, host_export_power, annual_bytes);
    SUNOPT_DIGITAL_H2D(d_demand, host_demand_energy, annual_bytes);
    SUNOPT_DIGITAL_H2D(d_tanks, host_tank_parameters, tank_bytes);
    SUNOPT_DIGITAL_H2D(d_current, host_current_indices, step_bytes);
    SUNOPT_DIGITAL_H2D(d_horizon, host_horizon_steps, step_bytes);
    SUNOPT_DIGITAL_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);

#undef SUNOPT_DIGITAL_H2D

    const auto wall_start =
        Clock::now();

    SUNOPT_DIGITAL_CHECK(
        cudaEventRecord(
            start_event,
            stream
        )
    );

    const unsigned int blocks =
        static_cast<unsigned int>(
            job_count
        );

    for (
        uint32_t repetition = 0;
        repetition <
            repetitions;
        ++repetition
    ) {
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
            step_count,
            search_iterations,
            d_cuts,
            d_status,
            d_evaluations
        );

        SUNOPT_DIGITAL_CHECK(
            cudaGetLastError()
        );
    }

    SUNOPT_DIGITAL_CHECK(
        cudaEventRecord(
            stop_event,
            stream
        )
    );

    SUNOPT_DIGITAL_CHECK(
        cudaEventSynchronize(
            stop_event
        )
    );

    float kernel_total_ms =
        0.0f;

    SUNOPT_DIGITAL_CHECK(
        cudaEventElapsedTime(
            &kernel_total_ms,
            start_event,
            stop_event
        )
    );

    *out_kernel_total_ms =
        static_cast<double>(
            kernel_total_ms
        );

    SUNOPT_DIGITAL_CHECK(
        cudaMemcpy(
            host_cut_results,
            d_cuts,
            cut_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_DIGITAL_CHECK(
        cudaMemcpy(
            host_status,
            d_status,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_DIGITAL_CHECK(
        cudaMemcpy(
            host_survival_evaluations,
            d_evaluations,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    const auto wall_stop =
        Clock::now();

    *out_wall_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            wall_stop -
            wall_start
        ).count();

#undef SUNOPT_DIGITAL_CHECK

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_END
'''.strip()


RS_FFI_BLOCK = r'''
    // SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_BEGIN
    fn sunopt_cuda_run_digital_hp_breakpoints(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_kernel_total_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_END
'''.strip()


RS_RESULT_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_BEGIN
#[derive(Clone, Copy, Default)]
struct DigitalResearchResult {
    cut_level_power: f64,
    normal_plan_survives: bool,
    maximum_pv_plan_survives: bool,
}

fn digital_hp_charge(
    export_power: f64,
    cut_level_power: f64,
    tank: ResearchTankParameters,
) -> f64 {
    if tank.maximum_heat_pump_electrical_power > 0.0 &&
        export_power - cut_level_power >=
            tank.maximum_heat_pump_electrical_power
    {
        tank.maximum_heat_pump_electrical_power *
        tank.cop *
        tank.time_step_hours
    } else {
        0.0
    }
}

fn digital_survives_scalar(
    export_window: &[f64],
    demand_window: &[f64],
    cut_level_power: f64,
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    tank: ResearchTankParameters,
) -> bool {
    const EPSILON: f64 = 1.0e-6;

    let reserve =
        minimum_reserve_energy
            .max(0.0)
            .min(tank.target_storage_energy);

    let mut stored =
        current_storage_energy
            .max(0.0)
            .min(tank.target_storage_energy);

    for offset in 0..export_window.len() {
        let charge =
            digital_hp_charge(
                export_window[offset],
                cut_level_power,
                tank,
            );

        if offset == 0 {
            stored =
                (stored + charge)
                    .min(tank.target_storage_energy);
            continue;
        }

        let requested_loss =
            (
                (1.0 - tank.heat_loss_retention) *
                stored +
                tank.heat_loss_intercept
            )
            .max(0.0);

        let heat_loss =
            requested_loss.min(stored);

        let after_loss =
            (stored - heat_loss).max(0.0);

        let demand =
            demand_window[offset].max(0.0);

        let discharged =
            demand.min(after_loss);

        let before_charge =
            after_loss -
            discharged;

        let unmet =
            (demand - discharged).max(0.0);

        if unmet > EPSILON {
            return false;
        }

        if reserve - before_charge > EPSILON {
            return false;
        }

        stored =
            (before_charge + charge)
                .min(tank.target_storage_energy);
    }

    true
}

fn digital_normal_cut(
    export_window: &[f64],
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> f64 {
    let pmax =
        tank.maximum_heat_pump_electrical_power;

    if pmax <= 0.0 ||
        tank.time_step_hours <= 0.0
    {
        return 0.0;
    }

    let mut breakpoints =
        export_window
            .iter()
            .copied()
            .filter(|&export| export >= pmax)
            .map(|export| export - pmax)
            .collect::<Vec<_>>();

    breakpoints.sort_by(
        |a, b| {
            a.partial_cmp(b)
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
        },
    );

    let count =
        breakpoints.len();

    if count == 0 {
        return pmax;
    }

    let per_bucket =
        pmax *
        tank.time_step_hours;

    let maximum_schedulable =
        count as f64 *
        per_bucket;

    let target =
        requested_electrical_energy
            .max(0.0)
            .min(maximum_schedulable);

    if target <= 0.0 {
        return breakpoints[count - 1] + pmax;
    }

    if target >= maximum_schedulable - 1.0e-12 {
        return 0.0;
    }

    let required_on =
        ((target / per_bucket - 1.0e-12).ceil() as usize)
            .max(1)
            .min(count);

    breakpoints[
        count -
        required_on
    ]
}

fn digital_run_scalar(
    export_window: &[f64],
    demand_window: &[f64],
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> DigitalResearchResult {
    let normal_cut =
        digital_normal_cut(
            export_window,
            requested_electrical_energy,
            tank,
        );

    if digital_survives_scalar(
        export_window,
        demand_window,
        normal_cut,
        current_storage_energy,
        minimum_reserve_energy,
        tank,
    ) {
        return DigitalResearchResult {
            cut_level_power:
                normal_cut,
            normal_plan_survives:
                true,
            maximum_pv_plan_survives:
                true,
        };
    }

    if !digital_survives_scalar(
        export_window,
        demand_window,
        0.0,
        current_storage_energy,
        minimum_reserve_energy,
        tank,
    ) {
        return DigitalResearchResult {
            cut_level_power:
                0.0,
            normal_plan_survives:
                false,
            maximum_pv_plan_survives:
                false,
        };
    }

    let pmax =
        tank.maximum_heat_pump_electrical_power;

    let mut candidates =
        export_window
            .iter()
            .copied()
            .filter(|&export| export >= pmax)
            .map(|export| export - pmax)
            .collect::<Vec<_>>();

    candidates.sort_by(
        |a, b| {
            a.partial_cmp(b)
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
        },
    );

    let mut left =
        0usize;

    let mut right =
        candidates.len();

    let mut best =
        0.0_f64;

    while left < right {
        let middle =
            left +
            (right - left) /
            2;

        let cut =
            candidates[middle].max(0.0);

        if digital_survives_scalar(
            export_window,
            demand_window,
            cut,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        ) {
            best =
                cut;
            left =
                middle +
                1;
        } else {
            right =
                middle;
        }
    }

    DigitalResearchResult {
        cut_level_power:
            best,
        normal_plan_survives:
            false,
        maximum_pv_plan_survives:
            true,
    }
}

#[napi(object)]
pub struct BlackwellDigitalHpBreakpointVariant {
    pub search_iterations: u32,
    pub kernel_ms: f64,
    pub wall_ms: f64,
    pub annual_projected_kernel_ms: f64,

    pub supported_jobs: u32,
    pub unsupported_jobs: u32,

    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,

    pub average_survival_evaluations: f64,
    pub maximum_survival_evaluations: u32,
}

#[napi(object)]
pub struct BlackwellDigitalHpBreakpointResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub annual_step_count: u32,
    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub supported_jobs: u32,
    pub unsupported_jobs: u32,
    pub scalar_reference_ms: f64,
    pub variants: Vec<BlackwellDigitalHpBreakpointVariant>,
}
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_END
'''.strip()


RS_FN_BLOCK = r'''
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_hp_breakpoints(
    snapshot_path: Option<String>,
    sample_steps: Option<u32>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalHpBreakpointResult> {
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

    if snapshot.steps.is_empty() ||
        snapshot.tank_count == 0
    {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty.",
            ),
        );
    }

    let requested_samples =
        sample_steps
            .unwrap_or(192)
            .clamp(
                2,
                snapshot.steps.len() as u32,
            )
            as usize;

    let repetitions =
        repetitions
            .unwrap_or(20)
            .clamp(
                1,
                10_000,
            );

    let selected_device =
        device_id.unwrap_or(0);

    let mut sample_positions =
        Vec::<usize>::with_capacity(
            requested_samples,
        );

    for sample in 0..requested_samples {
        let position =
            sample *
            (snapshot.steps.len() - 1) /
            (requested_samples - 1);

        if sample_positions
            .last()
            .copied() !=
            Some(position)
        {
            sample_positions.push(position);
        }
    }

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
            snapshot.tank_count *
            7,
        );

    for tank in &tanks {
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
        sample_positions.len();

    let job_count =
        step_count *
        snapshot.tank_count;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_values =
        Vec::<f64>::with_capacity(
            job_count *
            3,
        );

    let mut scalar =
        Vec::<DigitalResearchResult>::with_capacity(
            job_count,
        );

    let mut exact_supported =
        Vec::<bool>::with_capacity(
            job_count,
        );

    let scalar_started =
        Instant::now();

    for &position in &sample_positions {
        let step =
            &snapshot.steps[position];

        let horizon =
            step.horizon_steps as usize;

        if horizon == 0 ||
            horizon > 256
        {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital breakpoint benchmark supports horizon 1..=256; got {horizon}.",
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

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                horizon,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index as usize,
                horizon,
            );

        for tank_index in 0..snapshot.tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let current_storage =
                step.dynamic[base + 0];

            let reserve =
                step.dynamic[base + 1];

            let requested =
                step.dynamic[base + 2];

            let tank =
                tanks[tank_index];

            dynamic_values.push(
                current_storage,
            );

            dynamic_values.push(
                reserve,
            );

            dynamic_values.push(
                requested,
            );

            scalar.push(
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    current_storage,
                    reserve,
                    requested,
                    tank,
                ),
            );

            let clamped_reserve =
                reserve
                    .max(0.0)
                    .min(
                        tank.target_storage_energy,
                    );

            exact_supported.push(
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                clamped_reserve >
                    1.0e-6,
            );
        }
    }

    let scalar_reference_ms =
        scalar_started
            .elapsed()
            .as_secs_f64() *
        1000.0;

    let supported_jobs =
        exact_supported
            .iter()
            .filter(|&&x| x)
            .count()
            as u32;

    let unsupported_jobs =
        job_count as u32 -
        supported_jobs;

    let iteration_limits =
        [
            1_u32,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ];

    let mut variants =
        Vec::<BlackwellDigitalHpBreakpointVariant>::new();

    for search_iterations in iteration_limits {
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

        let mut kernel_total_ms =
            0.0_f64;

        let mut wall_ms =
            0.0_f64;

        let error =
            unsafe {
                sunopt_cuda_run_digital_hp_breakpoints(
                    selected_device,
                    snapshot.export_power.as_ptr(),
                    snapshot.demand_energy.as_ptr(),
                    snapshot.export_power.len(),
                    packed_tanks.as_ptr(),
                    snapshot.tank_count,
                    current_indices.as_ptr(),
                    horizon_steps.as_ptr(),
                    dynamic_values.as_ptr(),
                    step_count,
                    search_iterations,
                    repetitions,
                    cuts.as_mut_ptr(),
                    statuses.as_mut_ptr(),
                    evaluations.as_mut_ptr(),
                    &mut kernel_total_ms,
                    &mut wall_ms,
                )
            };

        if error != 0 {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital HP CUDA benchmark failed with CUDA error {error} at search_iterations={search_iterations}.",
                    ),
                ),
            );
        }

        let kernel_ms =
            kernel_total_ms /
            repetitions as f64;

        let mut status_mismatches =
            0_u32;

        let mut cut_mismatches =
            0_u32;

        let mut maximum_cut_error =
            0.0_f64;

        let mut evaluation_sum =
            0_u64;

        let mut maximum_survival_evaluations =
            0_u32;

        for job in 0..job_count {
            if !exact_supported[job] {
                continue;
            }

            let reference =
                scalar[job];

            let expected_status =
                if reference.normal_plan_survives {
                    0_u32
                } else if !reference.maximum_pv_plan_survives {
                    1_u32
                } else {
                    2_u32
                };

            if statuses[job] !=
                expected_status
            {
                status_mismatches +=
                    1;
            }

            let cut_error =
                (
                    cuts[job] -
                    reference.cut_level_power
                )
                .abs();

            maximum_cut_error =
                maximum_cut_error.max(
                    cut_error,
                );

            if cut_error >
                1.0e-9
            {
                cut_mismatches +=
                    1;
            }

            evaluation_sum +=
                evaluations[job]
                    as u64;

            maximum_survival_evaluations =
                maximum_survival_evaluations.max(
                    evaluations[job],
                );
        }

        let average_survival_evaluations =
            if supported_jobs >
                0
            {
                evaluation_sum as f64 /
                supported_jobs as f64
            } else {
                0.0
            };

        let annual_projected_kernel_ms =
            kernel_ms *
            snapshot.steps.len() as f64 /
            step_count as f64;

        variants.push(
            BlackwellDigitalHpBreakpointVariant {
                search_iterations,
                kernel_ms,
                wall_ms,
                annual_projected_kernel_ms,
                supported_jobs,
                unsupported_jobs,
                status_mismatches,
                cut_mismatches,
                maximum_cut_error,
                average_survival_evaluations,
                maximum_survival_evaluations,
            },
        );
    }

    Ok(
        BlackwellDigitalHpBreakpointResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions,

            annual_step_count:
                snapshot.steps.len()
                    as u32,

            sample_step_count:
                step_count
                    as u32,

            sample_job_count:
                job_count
                    as u32,

            supported_jobs,

            unsupported_jobs,

            scalar_reference_ms,

            variants,
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FN_END
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

const sampleSteps =
  Number.parseInt(
    process.argv[3] ?? "192",
    10,
  );

const repetitions =
  Number.parseInt(
    process.argv[4] ?? "20",
    10,
  );

const deviceId =
  Number.parseInt(
    process.argv[5] ?? "0",
    10,
  );

const result =
  addon.runBlackwellDigitalHpBreakpoints(
    snapshot,
    sampleSteps,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_HP_BREAKPOINTS_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- BLACKWELL DIGITAL-HP BREAKPOINT SOLVER ---",
);

console.log(
  `snapshot=${result.snapshotPath}`,
);

console.log(
  `annualSteps=${result.annualStepCount}`,
);

console.log(
  `sampleSteps=${result.sampleStepCount}`,
);

console.log(
  `sampleJobs=${result.sampleJobCount}`,
);

console.log(
  `supportedJobs=${result.supportedJobs}/${result.sampleJobCount}`,
);

console.log(
  `scalarReferenceMs=${result.scalarReferenceMs.toFixed(4)}`,
);

console.log();

console.log(
  [
    "iters",
    "kernelMs",
    "annualProjectedMs",
    "avgSurvivalEvals",
    "maxSurvivalEvals",
    "statusMis",
    "cutMis",
    "maxCutErr",
  ].join("\t"),
);

for (
  const variant of
  result.variants
) {
  console.log(
    [
      variant.searchIterations,
      variant.kernelMs.toFixed(6),
      variant.annualProjectedKernelMs.toFixed(3),
      variant.averageSurvivalEvaluations.toFixed(3),
      variant.maximumSurvivalEvaluations,
      variant.statusMismatches,
      variant.cutMismatches,
      variant.maximumCutError.toExponential(6),
    ].join("\t"),
  );
}

const reportPath =
  path.join(
    root,
    "blackwell-digital-hp-breakpoints.json",
  );

writeFileSync(
  reportPath,
  JSON.stringify(
    result,
    null,
    2,
  ) + "\n",
  "utf8",
);

console.log();
console.log(
  `report=${reportPath}`,
);
'''.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(
            encoding="ascii",
        ).strip().lower()

        if HASH_RE.fullmatch(value):
            return value

        raise RuntimeError(
            f"Invalid hash in {HASH_FILE}"
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
            repr(x)
            for x in anchors
        )
    )


def patch_header(source: str) -> str:
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
        "cuda_runtime.h",
        "causal_batch_kernel(",
    ):
        if token not in source:
            raise RuntimeError(
                f"Unexpected cuda_context.cu; missing {token!r}"
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
        "struct BlackwellReplaySnapshot",
        "fn read_blackwell_snapshot(",
        "BLACKWELL_DYNAMIC_STRIDE",
        "struct ResearchTankParameters",
        "fn research_tank_from_packed(",
        "fn research_window(",
    ]

    for token in required:
        if token not in source:
            raise RuntimeError(
                "Current cuda_context.rs does not contain required "
                f"research helper {token!r}"
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
            (MJS + "\n").encode("utf-8"),
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

    release = sha256_bytes(manifest_bytes)

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        INBOX /
        "20261109-blackwell-digital-hp-breakpoints.zip"
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
        "SunOpt digital-HP breakpoint research package created"
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
        "  production physics: UNCHANGED"
    )
    print(
        "  digital research rule: OFF or full Pmax"
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
        r"  node .\scripts\blackwell-digital-hp-breakpoints.mjs .\blackwell-benchmark.bin 192 20 0"
    )

    return output


if __name__ == "__main__":
    build_package()
