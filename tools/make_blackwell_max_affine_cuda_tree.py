#!/usr/bin/env python3
# tools/make_blackwell_max_affine_cuda_tree.py
#
# Run from:
#   C:\NextJS\SunOpt\sunopt
#
# Creates a verified format-2 update package in updates/inbox.
#
# Research only:
#   - production causal lockstep remains unchanged
#   - adds an exact max-affine CUDA tree benchmark
#   - benchmarks two survival checks per sampled causal job:
#       1) normal energy-cut level
#       2) maximum-PV cut = 0
#
# The package parent is read from updates/history/hash.txt.

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
MJS_REL = "rust/sunopt-node/scripts/blackwell-max-affine-cuda-tree.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN"
H_END = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_END"

CU_BEGIN = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_END"

RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_END"

RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_END"

RS_FN_BEGIN = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_END"


H_BLOCK = r'''
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN
/*
 * Benchmark-only exact max-affine survival tree.
 *
 * tank_parameters: 6 doubles per tank
 *   0 target_storage_energy
 *   1 maximum_heat_pump_electrical_power
 *   2 hp_thermal_energy_per_electrical_power (= COP * dt_hours)
 *   3 heat_loss_retention
 *   4 heat_loss_intercept
 *   5 affine_supported (1.0 / 0.0)
 *
 * dynamic_values: 2 doubles per job
 *   0 current_storage_energy
 *   1 minimum_reserve_energy
 *
 * cut_levels: 2 doubles per job
 *   0 normal energy-cut level
 *   1 maximum-PV cut level (= 0)
 *
 * host_survival: 2 uint32 values per job
 *   0 = does not survive
 *   1 = survives
 *   2 = unsupported exact max-affine branch
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_max_affine_tree_benchmark(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    const double* host_cut_levels,
    size_t step_count,
    uint32_t repetitions,
    uint32_t* host_survival,
    double* out_h2d_ms,
    double* out_kernel_total_ms,
    double* out_d2h_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_END
'''.strip()


CU_BLOCK = r'''
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN
#include <chrono>
#include <math_constants.h>

namespace {

constexpr unsigned int SUNOPT_MAX_AFFINE_THREADS = 256;
constexpr double SUNOPT_MAX_AFFINE_EPSILON = 1.0e-6;


__device__ __forceinline__ double sunopt_max_affine_clamp(
    double value,
    double lower,
    double upper
) {
    return fmin(
        fmax(
            value,
            lower
        ),
        upper
    );
}


__device__ __forceinline__ void sunopt_max_affine_compose(
    double left_a,
    double left_b,
    double left_c,
    double left_max_input,
    double right_a,
    double right_b,
    double right_c,
    double right_max_input,
    double* out_a,
    double* out_b,
    double* out_c,
    double* out_max_input
) {
    /*
     * left(right(x))
     *
     * F(x)=max(c,a*x+b), valid for x <= max_input.
     */
    double max_input =
        right_max_input;

    if (
        right_c >
        left_max_input +
        1.0e-12
    ) {
        max_input =
            -CUDART_INF;
    } else if (
        right_a >
        0.0
    ) {
        const double bound =
            (
                left_max_input -
                right_b
            ) /
            right_a;

        max_input =
            fmin(
                max_input,
                bound
            );
    } else if (
        right_b >
        left_max_input +
        1.0e-12
    ) {
        max_input =
            -CUDART_INF;
    }

    *out_a =
        left_a *
        right_a;

    *out_b =
        left_a *
        right_b +
        left_b;

    *out_c =
        fmax(
            left_c,
            left_a *
            right_c +
            left_b
        );

    *out_max_input =
        max_input;
}


__global__ void sunopt_max_affine_tree_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    const double* __restrict__ cut_levels,
    size_t step_count,
    uint32_t* __restrict__ survival
) {
    /*
     * One block = one exact survival check.
     *
     * Two blocks are generated per causal job:
     *   even: normal energy cut
     *   odd : maximum-PV cut = 0
     *
     * Future offsets 1..255 are transformed in parallel and then
     * composed in 8 ordered tree levels.
     */
    const size_t check_index =
        static_cast<size_t>(
            blockIdx.x
        );

    const size_t total_jobs =
        step_count *
        tank_count;

    const size_t total_checks =
        total_jobs *
        2;

    if (
        check_index >=
        total_checks
    ) {
        return;
    }

    const size_t job_index =
        check_index >>
        1;

    const size_t step_index =
        job_index /
        tank_count;

    const size_t tank_index =
        job_index -
        step_index *
        tank_count;

    const unsigned int tid =
        threadIdx.x;

    const size_t tank_base =
        tank_index *
        6;

    const double target_storage_energy =
        tank_parameters[
            tank_base +
            0
        ];

    const double maximum_hp_power =
        tank_parameters[
            tank_base +
            1
        ];

    const double hp_thermal_factor =
        tank_parameters[
            tank_base +
            2
        ];

    const double heat_loss_retention =
        tank_parameters[
            tank_base +
            3
        ];

    const double heat_loss_intercept =
        tank_parameters[
            tank_base +
            4
        ];

    const bool affine_supported =
        tank_parameters[
            tank_base +
            5
        ] >
        0.5;

    const size_t dynamic_base =
        job_index *
        2;

    const double current_storage_energy =
        dynamic_values[
            dynamic_base +
            0
        ];

    const double minimum_reserve_energy =
        dynamic_values[
            dynamic_base +
            1
        ];

    const double cut_level_power =
        cut_levels[
            check_index
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

    __shared__ double shared_a[
        SUNOPT_MAX_AFFINE_THREADS
    ];

    __shared__ double shared_b[
        SUNOPT_MAX_AFFINE_THREADS
    ];

    __shared__ double shared_c[
        SUNOPT_MAX_AFFINE_THREADS
    ];

    __shared__ double shared_max_input[
        SUNOPT_MAX_AFFINE_THREADS
    ];

    const double reserve =
        sunopt_max_affine_clamp(
            minimum_reserve_energy,
            0.0,
            target_storage_energy
        );

    const bool supported =
        affine_supported &&
        heat_loss_retention >
            0.0 &&
        data_length >
            0 &&
        horizon >
            0 &&
        horizon <=
            SUNOPT_MAX_AFFINE_THREADS;

    /*
     * reserve <= epsilon contains the zero-energy heat-loss clipping
     * branch and is intentionally reported unsupported.
     */
    if (
        !supported ||
        reserve <=
            SUNOPT_MAX_AFFINE_EPSILON
    ) {
        if (
            tid ==
            0
        ) {
            survival[
                check_index
            ] =
                2u;
        }

        return;
    }

    /*
     * thread 0 corresponds to future offset 1.
     * Unused lanes are exact identity transforms.
     */
    const size_t future_offset =
        static_cast<size_t>(
            tid
        ) +
        1;

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

        const double export_value =
            export_power[
                index
            ];

        const double demand =
            fmax(
                0.0,
                demand_energy[
                    index
                ]
            );

        const double hp_power =
            fmin(
                maximum_hp_power,
                fmax(
                    0.0,
                    export_value -
                    cut_level_power
                )
            );

        const double charge =
            hp_power *
            hp_thermal_factor;

        const double inverse_retention =
            1.0 /
            heat_loss_retention;

        shared_a[
            tid
        ] =
            inverse_retention;

        shared_b[
            tid
        ] =
            (
                heat_loss_intercept +
                demand -
                charge
            ) *
            inverse_retention;

        shared_c[
            tid
        ] =
            (
                heat_loss_intercept +
                demand +
                reserve -
                SUNOPT_MAX_AFFINE_EPSILON
            ) *
            inverse_retention;

        shared_max_input[
            tid
        ] =
            target_storage_energy;
    } else {
        shared_a[
            tid
        ] =
            1.0;

        shared_b[
            tid
        ] =
            0.0;

        shared_c[
            tid
        ] =
            -CUDART_INF;

        shared_max_input[
            tid
        ] =
            CUDART_INF;
    }

    __syncthreads();

    /*
     * Ordered associative tree:
     *   T0(T1(x)), T2(T3(x)), ...
     * then adjacent pair summaries.
     */
    for (
        unsigned int stride = 1;
        stride <
            SUNOPT_MAX_AFFINE_THREADS;
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
                SUNOPT_MAX_AFFINE_THREADS
            ) {
                double out_a;
                double out_b;
                double out_c;
                double out_max_input;

                sunopt_max_affine_compose(
                    shared_a[
                        tid
                    ],
                    shared_b[
                        tid
                    ],
                    shared_c[
                        tid
                    ],
                    shared_max_input[
                        tid
                    ],
                    shared_a[
                        right
                    ],
                    shared_b[
                        right
                    ],
                    shared_c[
                        right
                    ],
                    shared_max_input[
                        right
                    ],
                    &out_a,
                    &out_b,
                    &out_c,
                    &out_max_input
                );

                shared_a[
                    tid
                ] =
                    out_a;

                shared_b[
                    tid
                ] =
                    out_b;

                shared_c[
                    tid
                ] =
                    out_c;

                shared_max_input[
                    tid
                ] =
                    out_max_input;
            }
        }

        __syncthreads();
    }

    if (
        tid !=
        0
    ) {
        return;
    }

    bool survives =
        true;

    if (
        0.0 >
        shared_max_input[
            0
        ] +
        1.0e-12
    ) {
        survives =
            false;
    }

    double required_after_current_bucket =
        0.0;

    if (
        survives
    ) {
        /*
         * summary.eval(0) = max(c,b)
         */
        required_after_current_bucket =
            fmax(
                shared_c[
                    0
                ],
                shared_b[
                    0
                ]
            );

        if (
            required_after_current_bucket >
            target_storage_energy +
            1.0e-9
        ) {
            survives =
                false;
        }
    }

    if (
        survives
    ) {
        const size_t current_wrapped =
            current_index %
            data_length;

        const double current_export =
            export_power[
                current_wrapped
            ];

        const double hp_power =
            fmin(
                maximum_hp_power,
                fmax(
                    0.0,
                    current_export -
                    cut_level_power
                )
            );

        const double current_charge =
            hp_power *
            hp_thermal_factor;

        const double current =
            sunopt_max_affine_clamp(
                current_storage_energy,
                0.0,
                target_storage_energy
            );

        const double after_current_charge =
            fmin(
                target_storage_energy,
                current +
                current_charge
            );

        survives =
            after_current_charge +
            1.0e-12 >=
            required_after_current_bucket;
    }

    survival[
        check_index
    ] =
        survives ?
            1u :
            0u;
}

} // namespace


extern "C" int sunopt_cuda_run_max_affine_tree_benchmark(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    const double* host_cut_levels,
    size_t step_count,
    uint32_t repetitions,
    uint32_t* host_survival,
    double* out_h2d_ms,
    double* out_kernel_total_ms,
    double* out_d2h_ms,
    double* out_wall_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        host_export_power ==
            nullptr ||
        host_demand_energy ==
            nullptr ||
        host_tank_parameters ==
            nullptr ||
        host_current_indices ==
            nullptr ||
        host_horizon_steps ==
            nullptr ||
        host_dynamic_values ==
            nullptr ||
        host_cut_levels ==
            nullptr ||
        host_survival ==
            nullptr ||
        out_h2d_ms ==
            nullptr ||
        out_kernel_total_ms ==
            nullptr ||
        out_d2h_ms ==
            nullptr ||
        out_wall_ms ==
            nullptr ||
        data_length ==
            0 ||
        tank_count ==
            0 ||
        step_count ==
            0 ||
        repetitions ==
            0
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

    cudaEvent_t kernel_start =
        nullptr;

    cudaEvent_t kernel_stop =
        nullptr;

    double* d_export =
        nullptr;

    double* d_demand =
        nullptr;

    double* d_tank =
        nullptr;

    uint32_t* d_current =
        nullptr;

    uint32_t* d_horizon =
        nullptr;

    double* d_dynamic =
        nullptr;

    double* d_cuts =
        nullptr;

    uint32_t* d_survival =
        nullptr;

    const size_t job_count =
        step_count *
        tank_count;

    const size_t check_count =
        job_count *
        2;

    const size_t annual_bytes =
        data_length *
        sizeof(
            double
        );

    const size_t tank_bytes =
        tank_count *
        6 *
        sizeof(
            double
        );

    const size_t step_u32_bytes =
        step_count *
        sizeof(
            uint32_t
        );

    const size_t dynamic_bytes =
        job_count *
        2 *
        sizeof(
            double
        );

    const size_t cut_bytes =
        check_count *
        sizeof(
            double
        );

    const size_t survival_bytes =
        check_count *
        sizeof(
            uint32_t
        );

    auto cleanup =
        [&]() {
            if (
                d_survival !=
                nullptr
            ) {
                cudaFree(
                    d_survival
                );
            }

            if (
                d_cuts !=
                nullptr
            ) {
                cudaFree(
                    d_cuts
                );
            }

            if (
                d_dynamic !=
                nullptr
            ) {
                cudaFree(
                    d_dynamic
                );
            }

            if (
                d_horizon !=
                nullptr
            ) {
                cudaFree(
                    d_horizon
                );
            }

            if (
                d_current !=
                nullptr
            ) {
                cudaFree(
                    d_current
                );
            }

            if (
                d_tank !=
                nullptr
            ) {
                cudaFree(
                    d_tank
                );
            }

            if (
                d_demand !=
                nullptr
            ) {
                cudaFree(
                    d_demand
                );
            }

            if (
                d_export !=
                nullptr
            ) {
                cudaFree(
                    d_export
                );
            }

            if (
                kernel_stop !=
                nullptr
            ) {
                cudaEventDestroy(
                    kernel_stop
                );
            }

            if (
                kernel_start !=
                nullptr
            ) {
                cudaEventDestroy(
                    kernel_start
                );
            }

            if (
                stream !=
                nullptr
            ) {
                cudaStreamDestroy(
                    stream
                );
            }
        };

    err =
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    err =
        cudaEventCreate(
            &kernel_start
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    err =
        cudaEventCreate(
            &kernel_stop
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

#define SUNOPT_MAX_AFFINE_MALLOC(pointer, bytes) \
    do { \
        err = cudaMalloc( \
            reinterpret_cast<void**>( \
                &(pointer) \
            ), \
            (bytes) \
        ); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

    SUNOPT_MAX_AFFINE_MALLOC(
        d_export,
        annual_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_demand,
        annual_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_tank,
        tank_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_current,
        step_u32_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_horizon,
        step_u32_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_dynamic,
        dynamic_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_cuts,
        cut_bytes
    );

    SUNOPT_MAX_AFFINE_MALLOC(
        d_survival,
        survival_bytes
    );

#undef SUNOPT_MAX_AFFINE_MALLOC

    const auto wall_start =
        Clock::now();

    const auto h2d_start =
        Clock::now();

#define SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(dst, src, bytes) \
    do { \
        err = cudaMemcpy( \
            (dst), \
            (src), \
            (bytes), \
            cudaMemcpyHostToDevice \
        ); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_export,
        host_export_power,
        annual_bytes
    );

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_demand,
        host_demand_energy,
        annual_bytes
    );

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_tank,
        host_tank_parameters,
        tank_bytes
    );

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_current,
        host_current_indices,
        step_u32_bytes
    );

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_horizon,
        host_horizon_steps,
        step_u32_bytes
    );

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_dynamic,
        host_dynamic_values,
        dynamic_bytes
    );

    SUNOPT_MAX_AFFINE_COPY_TO_DEVICE(
        d_cuts,
        host_cut_levels,
        cut_bytes
    );

#undef SUNOPT_MAX_AFFINE_COPY_TO_DEVICE

    const auto h2d_stop =
        Clock::now();

    *out_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            h2d_stop -
            h2d_start
        ).count();

    err =
        cudaEventRecord(
            kernel_start,
            stream
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    const unsigned int blocks =
        static_cast<unsigned int>(
            check_count
        );

    for (
        uint32_t repetition = 0;
        repetition <
            repetitions;
        ++repetition
    ) {
        sunopt_max_affine_tree_kernel<<<
            blocks,
            SUNOPT_MAX_AFFINE_THREADS,
            0,
            stream
        >>>(
            d_export,
            d_demand,
            data_length,
            d_tank,
            tank_count,
            d_current,
            d_horizon,
            d_dynamic,
            d_cuts,
            step_count,
            d_survival
        );

        err =
            cudaGetLastError();

        if (
            err !=
            cudaSuccess
        ) {
            cleanup();

            return static_cast<int>(
                err
            );
        }
    }

    err =
        cudaEventRecord(
            kernel_stop,
            stream
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    err =
        cudaEventSynchronize(
            kernel_stop
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    float kernel_ms =
        0.0f;

    err =
        cudaEventElapsedTime(
            &kernel_ms,
            kernel_start,
            kernel_stop
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    *out_kernel_total_ms =
        static_cast<double>(
            kernel_ms
        );

    const auto d2h_start =
        Clock::now();

    err =
        cudaMemcpy(
            host_survival,
            d_survival,
            survival_bytes,
            cudaMemcpyDeviceToHost
        );

    if (
        err !=
        cudaSuccess
    ) {
        cleanup();

        return static_cast<int>(
            err
        );
    }

    const auto d2h_stop =
        Clock::now();

    *out_d2h_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            d2h_stop -
            d2h_start
        ).count();

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

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_END
'''.strip()


RS_FFI_BLOCK = r'''
    // SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_BEGIN
    fn sunopt_cuda_run_max_affine_tree_benchmark(
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

        host_cut_levels:
            *const f64,

        step_count:
            usize,

        repetitions:
            u32,

        host_survival:
            *mut u32,

        out_h2d_ms:
            *mut f64,

        out_kernel_total_ms:
            *mut f64,

        out_d2h_ms:
            *mut f64,

        out_wall_ms:
            *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_END
'''.strip()


RS_RESULT_BLOCK = r'''
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellMaxAffineCudaTreeResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,

    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub survival_check_count: u32,

    pub supported_jobs: u32,
    pub unsupported_jobs: u32,

    pub reference_full_causal_gpu_ms: f64,

    pub candidate_h2d_ms: f64,
    pub candidate_kernel_total_ms: f64,
    pub candidate_kernel_ms: f64,
    pub candidate_d2h_ms: f64,
    pub candidate_wall_ms: f64,

    pub candidate_ns_per_survival_check: f64,

    pub normal_survival_mismatches: u32,
    pub maxpv_survival_mismatches: u32,
    pub unexpected_gpu_unsupported_checks: u32,
}
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_END
'''.strip()


RS_FN_BLOCK = r'''
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_BEGIN
#[napi]
pub fn run_blackwell_max_affine_cuda_tree(
    snapshot_path:
        Option<String>,

    sample_steps:
        Option<u32>,

    repetitions:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<BlackwellMaxAffineCudaTreeResult> {
    let path =
        snapshot_path
            .map(
                PathBuf::from,
            )
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

    let requested_samples =
        sample_steps
            .unwrap_or(
                192,
            )
            .clamp(
                2,
                snapshot.steps.len()
                    as u32,
            )
            as usize;

    let mut sample_positions =
        Vec::<usize>::with_capacity(
            requested_samples,
        );

    for sample in
        0..
        requested_samples
    {
        let position =
            sample *
            (
                snapshot.steps.len() -
                1
            ) /
            (
                requested_samples -
                1
            );

        if (
            sample_positions
                .last()
                .copied() !=
            Some(
                position,
            )
        ) {
            sample_positions.push(
                position,
            );
        }
    }

    let selected_device =
        device_id
            .unwrap_or(
                0,
            );

    let repetitions =
        repetitions
            .unwrap_or(
                20,
            )
            .clamp(
                1,
                10_000,
            );

    let tank_parameters =
        (
            0..
            snapshot.tank_count
        )
        .map(
            |tank_index| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank_index,
                )
            },
        )
        .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count *
            6,
        );

    for tank in
        &tank_parameters
    {
        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            tank.maximum_heat_pump_electrical_power,
        );

        packed_tanks.push(
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
    }

    let step_count =
        sample_positions.len();

    let job_count =
        step_count *
        snapshot.tank_count;

    let check_count =
        job_count *
        2;

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
            2,
        );

    let mut cut_levels =
        Vec::<f64>::with_capacity(
            check_count,
        );

    let mut expected_survival =
        Vec::<u32>::with_capacity(
            check_count,
        );

    let mut supported_jobs =
        0_u32;

    for &position in
        &sample_positions
    {
        let step =
            &snapshot.steps[
                position
            ];

        let horizon =
            step.horizon_steps
                as usize;

        if (
            horizon == 0 ||
            horizon > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Max-affine CUDA tree benchmark supports horizon 1..=256; snapshot contains horizon {horizon}.",
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
                step.current_index
                    as usize,
                horizon,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index
                    as usize,
                horizon,
            );

        for tank_index in
            0..
            snapshot.tank_count
        {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let current_storage_energy =
                step.dynamic[
                    base +
                    0
                ];

            let minimum_reserve_energy =
                step.dynamic[
                    base +
                    1
                ];

            let requested_electrical_energy =
                step.dynamic[
                    base +
                    2
                ];

            let tank =
                tank_parameters[
                    tank_index
                ];

            dynamic_values.push(
                current_storage_energy,
            );

            dynamic_values.push(
                minimum_reserve_energy,
            );

            let energy_cut =
                research_energy_cut_level(
                    &export_window,
                    requested_electrical_energy,
                    tank,
                );

            cut_levels.push(
                energy_cut,
            );

            cut_levels.push(
                0.0,
            );

            let reserve =
                minimum_reserve_energy
                    .max(
                        0.0,
                    )
                    .min(
                        tank.target_storage_energy,
                    );

            let exact_supported =
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                reserve >
                    1.0e-6;

            if exact_supported {
                supported_jobs +=
                    1;

                expected_survival.push(
                    if research_scalar_survives(
                        &export_window,
                        &demand_window,
                        energy_cut,
                        current_storage_energy,
                        minimum_reserve_energy,
                        tank,
                    ) {
                        1
                    } else {
                        0
                    },
                );

                expected_survival.push(
                    if research_scalar_survives(
                        &export_window,
                        &demand_window,
                        0.0,
                        current_storage_energy,
                        minimum_reserve_energy,
                        tank,
                    ) {
                        1
                    } else {
                        0
                    },
                );
            } else {
                expected_survival.push(
                    2,
                );

                expected_survival.push(
                    2,
                );
            }
        }
    }

    /*
     * Existing complete causal GPU reference on the same sampled steps.
     * This is context only; it performs the whole causal cut search.
     */
    let cuda_parameters =
        blackwell_parameters_from_packed(
            &snapshot.packed_config,
            snapshot.data_length,
        )?;

    let mut reference_context =
        CudaSimulationContext::new(
            selected_device,
        )?;

    reference_context
        .upload_causal_arrays(
            &snapshot.export_power,
            &snapshot.demand_energy,
        )?;

    reference_context
        .configure_causal_static_parameters(
            &cuda_parameters,
        )?;

    let mut reference_dynamic =
        vec![
            SunoptCausalDynamic::default();
            snapshot.tank_count
        ];

    let mut reference_output =
        vec![
            0.0_f64;
            snapshot.tank_count
        ];

    let reference_started =
        Instant::now();

    for &position in
        &sample_positions
    {
        let step =
            &snapshot.steps[
                position
            ];

        blackwell_dynamic_from_values(
            &step.dynamic,
            snapshot.tank_count,
            &mut reference_dynamic,
        )?;

        reference_context
            .run_causal_dynamic_batch_into(
                step.current_index
                    as usize,
                step.horizon_steps
                    as usize,
                &reference_dynamic,
                &mut reference_output,
            )?;
    }

    let reference_full_causal_gpu_ms =
        reference_started
            .elapsed()
            .as_secs_f64() *
        1000.0;

    let mut gpu_survival =
        vec![
            0_u32;
            check_count
        ];

    let mut h2d_ms =
        0.0_f64;

    let mut kernel_total_ms =
        0.0_f64;

    let mut d2h_ms =
        0.0_f64;

    let mut wall_ms =
        0.0_f64;

    let error =
        unsafe {
            sunopt_cuda_run_max_affine_tree_benchmark(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                snapshot.tank_count,
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_values
                    .as_ptr(),
                cut_levels
                    .as_ptr(),
                step_count,
                repetitions,
                gpu_survival
                    .as_mut_ptr(),
                &mut h2d_ms,
                &mut kernel_total_ms,
                &mut d2h_ms,
                &mut wall_ms,
            )
        };

    if (
        error !=
        0
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "CUDA max-affine tree benchmark failed with CUDA error code {error}.",
                ),
            ),
        );
    }

    let mut normal_survival_mismatches =
        0_u32;

    let mut maxpv_survival_mismatches =
        0_u32;

    let mut unexpected_gpu_unsupported_checks =
        0_u32;

    for job in
        0..
        job_count
    {
        let normal_index =
            job *
            2;

        let maxpv_index =
            normal_index +
            1;

        let expected_normal =
            expected_survival[
                normal_index
            ];

        let expected_maxpv =
            expected_survival[
                maxpv_index
            ];

        let gpu_normal =
            gpu_survival[
                normal_index
            ];

        let gpu_maxpv =
            gpu_survival[
                maxpv_index
            ];

        if (
            expected_normal != 2 &&
            gpu_normal == 2
        ) {
            unexpected_gpu_unsupported_checks +=
                1;
        }

        if (
            expected_maxpv != 2 &&
            gpu_maxpv == 2
        ) {
            unexpected_gpu_unsupported_checks +=
                1;
        }

        if (
            expected_normal != 2 &&
            gpu_normal !=
                expected_normal
        ) {
            normal_survival_mismatches +=
                1;
        }

        if (
            expected_maxpv != 2 &&
            gpu_maxpv !=
                expected_maxpv
        ) {
            maxpv_survival_mismatches +=
                1;
        }
    }

    let candidate_kernel_ms =
        kernel_total_ms /
        repetitions
            as f64;

    let supported_check_count =
        supported_jobs
            as f64 *
        2.0;

    let candidate_ns_per_survival_check =
        if (
            supported_check_count >
            0.0
        ) {
            candidate_kernel_ms *
            1_000_000.0 /
            supported_check_count
        } else {
            0.0
        };

    Ok(
        BlackwellMaxAffineCudaTreeResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions,

            sample_step_count:
                step_count
                    as u32,

            sample_job_count:
                job_count
                    as u32,

            survival_check_count:
                check_count
                    as u32,

            supported_jobs,

            unsupported_jobs:
                job_count
                    as u32 -
                supported_jobs,

            reference_full_causal_gpu_ms,

            candidate_h2d_ms:
                h2d_ms,

            candidate_kernel_total_ms:
                kernel_total_ms,

            candidate_kernel_ms,

            candidate_d2h_ms:
                d2h_ms,

            candidate_wall_ms:
                wall_ms,

            candidate_ns_per_survival_check,

            normal_survival_mismatches,

            maxpv_survival_mismatches,

            unexpected_gpu_unsupported_checks,
        },
    )
}
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_END
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

if (
  !Number.isFinite(sampleSteps) ||
  sampleSteps < 2
) {
  throw new Error(
    "sampleSteps must be >= 2",
  );
}

if (
  !Number.isFinite(repetitions) ||
  repetitions < 1
) {
  throw new Error(
    "repetitions must be >= 1",
  );
}

const result =
  addon.runBlackwellMaxAffineCudaTree(
    snapshot,
    sampleSteps,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_MAX_AFFINE_CUDA_TREE_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- BLACKWELL MAX-AFFINE CUDA TREE ---",
);

console.log(
  `snapshot=${result.snapshotPath}`,
);

console.log(
  `device=${result.deviceId}`,
);

console.log(
  `sampleSteps=${result.sampleStepCount}`,
);

console.log(
  `sampleJobs=${result.sampleJobCount}`,
);

console.log(
  `survivalChecks=${result.survivalCheckCount}`,
);

console.log(
  `supportedJobs=${result.supportedJobs}/${result.sampleJobCount}`,
);

console.log(
  `unsupportedJobs=${result.unsupportedJobs}`,
);

console.log();
console.log(
  "REFERENCE (existing complete causal kernel)",
);

console.log(
  `referenceFullCausalGpuMs=${result.referenceFullCausalGpuMs.toFixed(4)}`,
);

console.log();
console.log(
  "CANDIDATE (exact max-affine survival tree)",
);

console.log(
  `repetitions=${result.repetitions}`,
);

console.log(
  `h2dMs=${result.candidateH2dMs.toFixed(4)}`,
);

console.log(
  `kernelTotalMs=${result.candidateKernelTotalMs.toFixed(4)}`,
);

console.log(
  `kernelMsPerRun=${result.candidateKernelMs.toFixed(6)}`,
);

console.log(
  `d2hMs=${result.candidateD2hMs.toFixed(4)}`,
);

console.log(
  `wallMs=${result.candidateWallMs.toFixed(4)}`,
);

console.log(
  `nsPerSupportedSurvivalCheck=${result.candidateNsPerSurvivalCheck.toFixed(2)}`,
);

console.log();
console.log(
  "PARITY",
);

console.log(
  `normalSurvivalMismatches=${result.normalSurvivalMismatches}`,
);

console.log(
  `maxPvSurvivalMismatches=${result.maxpvSurvivalMismatches}`,
);

console.log(
  `unexpectedGpuUnsupportedChecks=${result.unexpectedGpuUnsupportedChecks}`,
);

const reportPath =
  path.join(
    root,
    "blackwell-max-affine-cuda-tree.json",
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
            f"Invalid hash in {HASH_FILE}",
        )

    if RELEASE_FILE.is_file():
        state = json.loads(
            RELEASE_FILE.read_text(
                encoding="utf-8",
            ),
        )

        value = str(
            state.get(
                "sha256",
                "",
            ),
        ).strip().lower()

        if HASH_RE.fullmatch(value):
            return value

        raise RuntimeError(
            f"Invalid sha256 in {RELEASE_FILE}",
        )

    return GENESIS


def replace_between_markers(
    source: str,
    begin: str,
    end: str,
    block: str,
) -> str:
    has_begin = begin in source
    has_end = end in source

    if has_begin != has_end:
        raise RuntimeError(
            f"Only one marker present: {begin!r} / {end!r}",
        )

    if not has_begin:
        raise RuntimeError(
            f"Marker pair not present: {begin!r} / {end!r}",
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


def insert_or_replace_before(
    source: str,
    begin: str,
    end: str,
    block: str,
    anchors: list[str],
) -> str:
    if begin in source or end in source:
        return replace_between_markers(
            source,
            begin,
            end,
            block,
        )

    position = -1

    for anchor in anchors:
        position = source.find(
            anchor,
        )

        if position >= 0:
            break

    if position < 0:
        raise RuntimeError(
            "Required insertion anchor not found. Tried: " +
            ", ".join(
                repr(x)
                for x in anchors
            ),
        )

    return (
        source[:position] +
        block +
        "\n\n\n" +
        source[position:]
    )


def patch_header(source: str) -> str:
    for token in (
        "SunoptCudaContext",
        "SUNOPT_CUDA_EXPORT",
    ):
        if token not in source:
            raise RuntimeError(
                f"Current cuda_context.h is unexpected; missing {token!r}.",
            )

    return insert_or_replace_before(
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
                f"Current cuda_context.cu is unexpected; missing {token!r}.",
            )

    return insert_or_replace_before(
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
        "fn blackwell_parameters_from_packed(",
        "fn blackwell_dynamic_from_values(",
        "BLACKWELL_CONFIG_STRIDE",
        "BLACKWELL_DYNAMIC_STRIDE",
        "struct SunoptCausalDynamic",
        "struct ResearchTankParameters",
        "fn research_tank_from_packed(",
        "fn research_window(",
        "fn research_energy_cut_level(",
        "fn research_scalar_survives(",
        "run_causal_dynamic_batch_into(",
    ]

    for token in required:
        if token not in source:
            raise RuntimeError(
                "Current cuda_context.rs is not the expected "
                "Blackwell physics-research generation; "
                f"missing {token!r}.",
            )

    source = insert_or_replace_before(
        source,
        RS_FFI_BEGIN,
        RS_FFI_END,
        RS_FFI_BLOCK,
        [
            "    fn sunopt_cuda_get_profile(",
            "    fn sunopt_cuda_error_string(",
        ],
    )

    source = insert_or_replace_before(
        source,
        RS_RESULT_BEGIN,
        RS_RESULT_END,
        RS_RESULT_BLOCK,
        [
            "#[napi]\npub struct CudaCausalWorker {",
        ],
    )

    source = insert_or_replace_before(
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
                f"Required current project file missing: {path}",
            )

    h = patch_header(
        H_PATH.read_text(
            encoding="utf-8",
        ),
    )

    cu = patch_cuda(
        CU_PATH.read_text(
            encoding="utf-8",
        ),
    )

    rs = patch_rust(
        RS_PATH.read_text(
            encoding="utf-8",
        ),
    )

    payloads = {
        H_REL:
            h.encode(
                "utf-8",
            ),

        CU_REL:
            cu.encode(
                "utf-8",
            ),

        RS_REL:
            rs.encode(
                "utf-8",
            ),

        MJS_REL:
            (
                MJS +
                "\n"
            ).encode(
                "utf-8",
            ),
    }

    parent = read_parent_hash()

    files = {
        name:
            sha256_bytes(
                data,
            )
        for name, data in
        payloads.items()
    }

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": files,
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) +
        "\n"
    ).encode(
        "utf-8",
    )

    release_hash = sha256_bytes(
        manifest_bytes,
    )

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = INBOX / (
        "20261109-blackwell-max-affine-cuda-tree.zip"
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
                release_hash +
                "\n"
            ).encode(
                "ascii",
            ),
        )

        for name, data in payloads.items():
            archive.writestr(
                name,
                data,
            )

    print(
        "SunOpt Blackwell max-affine CUDA-tree package created",
    )

    print(
        f"  package: {output}",
    )

    print(
        f"  parent : {parent}",
    )

    print(
        f"  release: {release_hash}",
    )

    print(
        "  production lockstep: UNCHANGED",
    )

    print(
        "  candidate: exact 256-thread ordered max-affine tree",
    )

    print()
    print(
        "Next:",
    )

    print(
        r"  python .\tools\update_watcher.py --once",
    )

    print(
        r"  cd .\rust\sunopt-node",
    )

    print(
        r"  npm run build",
    )

    print(
        r"  node .\scripts\blackwell-max-affine-cuda-tree.mjs .\blackwell-benchmark.bin 192 20 0",
    )

    return output


if __name__ == "__main__":
    build_package()
