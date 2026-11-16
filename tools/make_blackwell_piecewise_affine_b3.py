#!/usr/bin/env python3
# tools/make_blackwell_piecewise_affine_b3.py
#
# Run from:
#   C:\NextJS\SunOpt\sunopt
#
# Creates a verified format-2 research update package in updates/inbox.
# Production lockstep remains unchanged.
#
# B3 research idea:
#   - exact max-affine tree evaluates the causal reserve constraint
#   - each tree node also propagates the local derivative wrt cut
#   - a safeguarded piecewise-linear Newton/secant correction searches the
#     surviving cut with 1/2/3/4/6/8 correction iterations
#   - benchmark compares each limit to the canonical scalar 20-step search

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
MJS_REL = "rust/sunopt-node/scripts/blackwell-piecewise-affine-b3.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_BEGIN"
H_END = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_END"
CU_BEGIN = H_BEGIN
CU_END = H_END
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FN_END"

H_BLOCK = r'''
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_BEGIN
/*
 * Benchmark-only B3 piecewise-affine cut solver.
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
 * energy_cut_levels: 1 double per job
 *
 * status per job:
 *   0 normal energy-cut survives
 *   1 maximum-PV cut=0 still fails
 *   2 corrected cut found
 *   3 exact max-affine branch unsupported
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_piecewise_affine_b3_benchmark(
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
    uint32_t max_correction_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_iterations,
    uint32_t* host_evaluations,
    double* host_bracket_width,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_END
'''.strip()

CU_BLOCK = r'''
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_BEGIN
#include <chrono>
#include <math_constants.h>

namespace {

constexpr unsigned int SUNOPT_B3_THREADS = 256;
constexpr double SUNOPT_B3_EPSILON = 1.0e-6;
constexpr double SUNOPT_B3_BIG = 1.0e100;

struct SunoptB3Eval {
    double margin;
    double derivative;
    bool survives;
};

__device__ __forceinline__ double sunopt_b3_clamp(
    double value,
    double lower,
    double upper
) {
    return fmin(fmax(value, lower), upper);
}

__device__ __forceinline__ void sunopt_b3_compose(
    double left_a,
    double left_b,
    double left_c,
    double left_m,
    double left_db,
    double left_dc,
    double left_dm,
    double right_a,
    double right_b,
    double right_c,
    double right_m,
    double right_db,
    double right_dc,
    double right_dm,
    double* out_a,
    double* out_b,
    double* out_c,
    double* out_m,
    double* out_db,
    double* out_dc,
    double* out_dm
) {
    *out_a = left_a * right_a;
    *out_b = left_a * right_b + left_b;
    *out_db = left_a * right_db + left_db;

    const double c_from_right =
        left_a * right_c + left_b;
    const double dc_from_right =
        left_a * right_dc + left_db;

    if (c_from_right > left_c + 1.0e-13) {
        *out_c = c_from_right;
        *out_dc = dc_from_right;
    } else if (left_c > c_from_right + 1.0e-13) {
        *out_c = left_c;
        *out_dc = left_dc;
    } else {
        *out_c = fmax(left_c, c_from_right);
        /* right-hand derivative for an increasing cut */
        *out_dc = fmax(left_dc, dc_from_right);
    }

    if (right_c > left_m + 1.0e-12) {
        *out_m = -SUNOPT_B3_BIG;
        *out_dm = 0.0;
        return;
    }

    const double bound =
        (left_m - right_b) / right_a;
    const double d_bound =
        (left_dm - right_db) / right_a;

    if (bound < right_m - 1.0e-13) {
        *out_m = bound;
        *out_dm = d_bound;
    } else if (right_m < bound - 1.0e-13) {
        *out_m = right_m;
        *out_dm = right_dm;
    } else {
        *out_m = fmin(right_m, bound);
        /* right-hand derivative of min() */
        *out_dm = fmin(right_dm, d_bound);
    }
}

__device__ SunoptB3Eval sunopt_b3_evaluate(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    double target_storage_energy,
    double maximum_hp_power,
    double hp_thermal_factor,
    double heat_loss_retention,
    double heat_loss_intercept,
    double current_storage_energy,
    double reserve,
    size_t current_index,
    size_t horizon,
    double cut,
    double* shared_a,
    double* shared_b,
    double* shared_c,
    double* shared_m,
    double* shared_db,
    double* shared_dc,
    double* shared_dm,
    double* shared_eval_margin,
    double* shared_eval_derivative,
    int* shared_eval_survives
) {
    const unsigned int tid = threadIdx.x;
    const size_t future_offset = static_cast<size_t>(tid) + 1;

    if (future_offset < horizon) {
        const size_t index = (current_index + future_offset) % data_length;
        const double export_value = export_power[index];
        const double demand = fmax(0.0, demand_energy[index]);
        const double raw = export_value - cut;
        const double hp_power = fmin(maximum_hp_power, fmax(0.0, raw));
        const double charge = hp_power * hp_thermal_factor;

        /* right-hand derivative as cut increases */
        const double d_charge =
            (raw > 0.0 && raw <= maximum_hp_power)
                ? -hp_thermal_factor
                : 0.0;

        const double inverse_retention = 1.0 / heat_loss_retention;

        shared_a[tid] = inverse_retention;
        shared_b[tid] =
            (heat_loss_intercept + demand - charge) * inverse_retention;
        shared_db[tid] = -d_charge * inverse_retention;

        shared_c[tid] =
            (heat_loss_intercept + demand + reserve - SUNOPT_B3_EPSILON) *
            inverse_retention;
        shared_dc[tid] = 0.0;

        shared_m[tid] = target_storage_energy;
        shared_dm[tid] = 0.0;
    } else {
        shared_a[tid] = 1.0;
        shared_b[tid] = 0.0;
        shared_c[tid] = -SUNOPT_B3_BIG;
        shared_m[tid] = SUNOPT_B3_BIG;
        shared_db[tid] = 0.0;
        shared_dc[tid] = 0.0;
        shared_dm[tid] = 0.0;
    }

    __syncthreads();

    for (unsigned int stride = 1; stride < SUNOPT_B3_THREADS; stride <<= 1) {
        const unsigned int group = stride << 1;

        if ((tid & (group - 1)) == 0) {
            const unsigned int right = tid + stride;
            if (right < SUNOPT_B3_THREADS) {
                double oa, ob, oc, om, odb, odc, odm;
                sunopt_b3_compose(
                    shared_a[tid], shared_b[tid], shared_c[tid], shared_m[tid],
                    shared_db[tid], shared_dc[tid], shared_dm[tid],
                    shared_a[right], shared_b[right], shared_c[right], shared_m[right],
                    shared_db[right], shared_dc[right], shared_dm[right],
                    &oa, &ob, &oc, &om, &odb, &odc, &odm
                );
                shared_a[tid] = oa;
                shared_b[tid] = ob;
                shared_c[tid] = oc;
                shared_m[tid] = om;
                shared_db[tid] = odb;
                shared_dc[tid] = odc;
                shared_dm[tid] = odm;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const bool c_active =
            shared_c[0] > shared_b[0] + 1.0e-13;
        const bool b_active =
            shared_b[0] > shared_c[0] + 1.0e-13;

        const double required = fmax(shared_c[0], shared_b[0]);
        const double d_required =
            c_active ? shared_dc[0] :
            b_active ? shared_db[0] :
            fmax(shared_dc[0], shared_db[0]);

        const size_t current_wrapped = current_index % data_length;
        const double raw0 = export_power[current_wrapped] - cut;
        const double hp0 = fmin(maximum_hp_power, fmax(0.0, raw0));
        const double charge0 = hp0 * hp_thermal_factor;
        const double d_charge0 =
            (raw0 > 0.0 && raw0 <= maximum_hp_power)
                ? -hp_thermal_factor
                : 0.0;

        const double current =
            sunopt_b3_clamp(current_storage_energy, 0.0, target_storage_energy);
        const double before_cap = current + charge0;
        const double after_current = fmin(target_storage_energy, before_cap);
        const double d_after_current =
            before_cap <= target_storage_energy + 1.0e-12
                ? d_charge0
                : 0.0;

        const double margin_current =
            after_current + 1.0e-12 - required;
        const double d_margin_current =
            d_after_current - d_required;

        const double margin_target =
            target_storage_energy + 1.0e-9 - required;
        const double d_margin_target = -d_required;

        const double margin_domain =
            shared_m[0] + 1.0e-12;
        const double d_margin_domain = shared_dm[0];

        double margin = margin_current;
        double derivative = d_margin_current;

        if (margin_target < margin - 1.0e-13) {
            margin = margin_target;
            derivative = d_margin_target;
        } else if (fabs(margin_target - margin) <= 1.0e-13) {
            derivative = fmin(derivative, d_margin_target);
        }

        if (margin_domain < margin - 1.0e-13) {
            margin = margin_domain;
            derivative = d_margin_domain;
        } else if (fabs(margin_domain - margin) <= 1.0e-13) {
            derivative = fmin(derivative, d_margin_domain);
        }

        *shared_eval_margin = margin;
        *shared_eval_derivative = derivative;
        *shared_eval_survives = margin >= 0.0 ? 1 : 0;
    }

    __syncthreads();

    SunoptB3Eval result{};
    result.margin = *shared_eval_margin;
    result.derivative = *shared_eval_derivative;
    result.survives = *shared_eval_survives != 0;
    return result;
}

__global__ void sunopt_piecewise_affine_b3_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    const double* __restrict__ energy_cut_levels,
    size_t step_count,
    uint32_t max_correction_iterations,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ iterations,
    uint32_t* __restrict__ evaluations,
    double* __restrict__ bracket_width
) {
    const size_t job = static_cast<size_t>(blockIdx.x);
    const size_t total_jobs = step_count * tank_count;
    if (job >= total_jobs) return;

    const unsigned int tid = threadIdx.x;
    const size_t step = job / tank_count;
    const size_t tank = job - step * tank_count;
    const size_t tb = tank * 6;
    const size_t db = job * 2;

    const double target = tank_parameters[tb + 0];
    const double max_hp = tank_parameters[tb + 1];
    const double hp_factor = tank_parameters[tb + 2];
    const double retention = tank_parameters[tb + 3];
    const double loss_intercept = tank_parameters[tb + 4];
    const bool affine_supported = tank_parameters[tb + 5] > 0.5;

    const double current = dynamic_values[db + 0];
    const double reserve = sunopt_b3_clamp(dynamic_values[db + 1], 0.0, target);
    const size_t current_index = static_cast<size_t>(current_indices[step]);
    const size_t horizon = static_cast<size_t>(horizon_steps[step]);
    const double energy_cut = energy_cut_levels[job];

    __shared__ double sa[SUNOPT_B3_THREADS];
    __shared__ double sb[SUNOPT_B3_THREADS];
    __shared__ double sc[SUNOPT_B3_THREADS];
    __shared__ double sm[SUNOPT_B3_THREADS];
    __shared__ double sdb[SUNOPT_B3_THREADS];
    __shared__ double sdc[SUNOPT_B3_THREADS];
    __shared__ double sdm[SUNOPT_B3_THREADS];
    __shared__ double eval_margin;
    __shared__ double eval_derivative;
    __shared__ int eval_survives;
    __shared__ double candidate_cut;

    const bool supported =
        affine_supported && retention > 0.0 && reserve > SUNOPT_B3_EPSILON &&
        data_length > 0 && horizon > 0 && horizon <= SUNOPT_B3_THREADS;

    if (!supported) {
        if (tid == 0) {
            cut_results[job] = energy_cut;
            status[job] = 3u;
            iterations[job] = 0u;
            evaluations[job] = 0u;
            bracket_width[job] = 0.0;
        }
        return;
    }

    SunoptB3Eval hi_eval = sunopt_b3_evaluate(
        export_power, demand_energy, data_length,
        target, max_hp, hp_factor, retention, loss_intercept,
        current, reserve, current_index, horizon, energy_cut,
        sa, sb, sc, sm, sdb, sdc, sdm,
        &eval_margin, &eval_derivative, &eval_survives
    );

    if (hi_eval.survives) {
        if (tid == 0) {
            cut_results[job] = energy_cut;
            status[job] = 0u;
            iterations[job] = 0u;
            evaluations[job] = 1u;
            bracket_width[job] = 0.0;
        }
        return;
    }

    SunoptB3Eval lo_eval = sunopt_b3_evaluate(
        export_power, demand_energy, data_length,
        target, max_hp, hp_factor, retention, loss_intercept,
        current, reserve, current_index, horizon, 0.0,
        sa, sb, sc, sm, sdb, sdc, sdm,
        &eval_margin, &eval_derivative, &eval_survives
    );

    if (!lo_eval.survives) {
        if (tid == 0) {
            cut_results[job] = 0.0;
            status[job] = 1u;
            iterations[job] = 0u;
            evaluations[job] = 2u;
            bracket_width[job] = 0.0;
        }
        return;
    }

    double lo = 0.0;
    double hi = energy_cut;
    uint32_t used_iterations = 0u;

    for (uint32_t iteration = 0; iteration < max_correction_iterations; ++iteration) {
        if (tid == 0) {
            double candidate = CUDART_NAN;

            if (lo_eval.derivative < -1.0e-12 && lo_eval.margin >= 0.0) {
                const double root = lo - lo_eval.margin / lo_eval.derivative;
                if (root > lo + 1.0e-12 && root < hi - 1.0e-12) {
                    candidate = root;
                }
            }

            if (!isfinite(candidate) && hi_eval.derivative < -1.0e-12 && hi_eval.margin <= 0.0) {
                const double root = hi - hi_eval.margin / hi_eval.derivative;
                if (root > lo + 1.0e-12 && root < hi - 1.0e-12) {
                    candidate = root;
                }
            }

            if (!isfinite(candidate)) {
                const double denominator = hi_eval.margin - lo_eval.margin;
                if (fabs(denominator) > 1.0e-18) {
                    const double secant =
                        lo - lo_eval.margin * (hi - lo) / denominator;
                    if (secant > lo + 1.0e-12 && secant < hi - 1.0e-12) {
                        candidate = secant;
                    }
                }
            }

            if (!isfinite(candidate)) {
                candidate = 0.5 * (lo + hi);
            }

            /* Guard against a tangent/secant landing numerically on an endpoint. */
            const double guard = fmax(1.0e-10, (hi - lo) * 1.0e-6);
            if (candidate <= lo + guard || candidate >= hi - guard) {
                candidate = 0.5 * (lo + hi);
            }

            candidate_cut = candidate;
        }
        __syncthreads();

        SunoptB3Eval candidate_eval = sunopt_b3_evaluate(
            export_power, demand_energy, data_length,
            target, max_hp, hp_factor, retention, loss_intercept,
            current, reserve, current_index, horizon, candidate_cut,
            sa, sb, sc, sm, sdb, sdc, sdm,
            &eval_margin, &eval_derivative, &eval_survives
        );

        if (candidate_eval.survives) {
            lo = candidate_cut;
            lo_eval = candidate_eval;
        } else {
            hi = candidate_cut;
            hi_eval = candidate_eval;
        }

        used_iterations = iteration + 1u;

        if (hi - lo <= 1.0e-9) break;
    }

    if (tid == 0) {
        cut_results[job] = lo;
        status[job] = 2u;
        iterations[job] = used_iterations;
        evaluations[job] = 2u + used_iterations;
        bracket_width[job] = hi - lo;
    }
}

} // namespace

extern "C" int sunopt_cuda_run_piecewise_affine_b3_benchmark(
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
    uint32_t max_correction_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_iterations,
    uint32_t* host_evaluations,
    double* host_bracket_width,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results || !host_status ||
        !host_iterations || !host_evaluations || !host_bracket_width ||
        !out_upload_ms || !out_kernel_total_ms || !out_download_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0 || repetitions == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double *d_export=nullptr, *d_demand=nullptr, *d_tanks=nullptr,
           *d_dynamic=nullptr, *d_energy_cut=nullptr, *d_cut_results=nullptr,
           *d_bracket_width=nullptr;
    uint32_t *d_current=nullptr, *d_horizon=nullptr, *d_status=nullptr,
             *d_iterations=nullptr, *d_evaluations=nullptr;

    const size_t jobs = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t step_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = jobs * 2 * sizeof(double);
    const size_t job_double_bytes = jobs * sizeof(double);
    const size_t job_u32_bytes = jobs * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_iterations) cudaFree(d_iterations);
        if (d_status) cudaFree(d_status);
        if (d_bracket_width) cudaFree(d_bracket_width);
        if (d_cut_results) cudaFree(d_cut_results);
        if (d_energy_cut) cudaFree(d_energy_cut);
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

    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }
    err = cudaEventCreate(&start_event);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }
    err = cudaEventCreate(&stop_event);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

#define B3_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B3_MALLOC(d_export, annual_bytes);
    B3_MALLOC(d_demand, annual_bytes);
    B3_MALLOC(d_tanks, tank_bytes);
    B3_MALLOC(d_current, step_bytes);
    B3_MALLOC(d_horizon, step_bytes);
    B3_MALLOC(d_dynamic, dynamic_bytes);
    B3_MALLOC(d_energy_cut, job_double_bytes);
    B3_MALLOC(d_cut_results, job_double_bytes);
    B3_MALLOC(d_status, job_u32_bytes);
    B3_MALLOC(d_iterations, job_u32_bytes);
    B3_MALLOC(d_evaluations, job_u32_bytes);
    B3_MALLOC(d_bracket_width, job_double_bytes);
#undef B3_MALLOC

    const auto wall_start = Clock::now();
    const auto upload_start = Clock::now();

#define B3_COPY_H2D(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B3_COPY_H2D(d_export, host_export_power, annual_bytes);
    B3_COPY_H2D(d_demand, host_demand_energy, annual_bytes);
    B3_COPY_H2D(d_tanks, host_tank_parameters, tank_bytes);
    B3_COPY_H2D(d_current, host_current_indices, step_bytes);
    B3_COPY_H2D(d_horizon, host_horizon_steps, step_bytes);
    B3_COPY_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);
    B3_COPY_H2D(d_energy_cut, host_energy_cut_levels, job_double_bytes);
#undef B3_COPY_H2D

    const auto upload_stop = Clock::now();
    *out_upload_ms = std::chrono::duration<double,std::milli>(upload_stop-upload_start).count();

    err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    const unsigned int blocks = static_cast<unsigned int>(jobs);
    for (uint32_t r=0; r<repetitions; ++r) {
        sunopt_piecewise_affine_b3_kernel<<<blocks,SUNOPT_B3_THREADS,0,stream>>>(
            d_export,d_demand,data_length,d_tanks,tank_count,d_current,d_horizon,
            d_dynamic,d_energy_cut,step_count,max_correction_iterations,
            d_cut_results,d_status,d_iterations,d_evaluations,d_bracket_width
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }
    }

    err = cudaEventRecord(stop_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }
    err = cudaEventSynchronize(stop_event);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    float kernel_ms = 0.0f;
    err = cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }
    *out_kernel_total_ms = static_cast<double>(kernel_ms);

    const auto download_start = Clock::now();
#define B3_COPY_D2H(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)
    B3_COPY_D2H(host_cut_results, d_cut_results, job_double_bytes);
    B3_COPY_D2H(host_status, d_status, job_u32_bytes);
    B3_COPY_D2H(host_iterations, d_iterations, job_u32_bytes);
    B3_COPY_D2H(host_evaluations, d_evaluations, job_u32_bytes);
    B3_COPY_D2H(host_bracket_width, d_bracket_width, job_double_bytes);
#undef B3_COPY_D2H
    const auto download_stop = Clock::now();
    *out_download_ms = std::chrono::duration<double,std::milli>(download_stop-download_start).count();
    *out_wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_start).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_END
'''.strip()

RS_FFI_BLOCK = r'''
    // SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FFI_BEGIN
    fn sunopt_cuda_run_piecewise_affine_b3_benchmark(
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
        max_correction_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_iterations: *mut u32,
        host_evaluations: *mut u32,
        host_bracket_width: *mut f64,
        out_upload_ms: *mut f64,
        out_kernel_total_ms: *mut f64,
        out_download_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FFI_END
'''.strip()

RS_RESULT_BLOCK = r'''
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellPiecewiseAffineB3Result {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub annual_step_count: u32,
    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub supported_jobs: u32,
    pub unsupported_jobs: u32,
    pub correction_search_jobs: u32,
    pub reference_full_causal_gpu_ms: f64,

    pub iteration_limits: Vec<u32>,
    pub upload_ms: Vec<f64>,
    pub kernel_ms: Vec<f64>,
    pub download_ms: Vec<f64>,
    pub wall_ms: Vec<f64>,
    pub annual_projected_kernel_ms: Vec<f64>,

    pub status_mismatches: Vec<u32>,
    pub max_abs_cut_error: Vec<f64>,
    pub mean_abs_cut_error: Vec<f64>,
    pub cut_error_over_1e6: Vec<u32>,
    pub cut_error_over_1e5: Vec<u32>,
    pub cut_error_over_1e4: Vec<u32>,

    pub average_correction_iterations: Vec<f64>,
    pub average_evaluations_supported: Vec<f64>,
    pub maximum_bracket_width: Vec<f64>,
}
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_RESULT_END
'''.strip()

RS_FN_BLOCK = r'''
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FN_BEGIN
#[napi]
pub fn run_blackwell_piecewise_affine_b3(
    snapshot_path: Option<String>,
    sample_steps: Option<u32>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellPiecewiseAffineB3Result> {
    let path = snapshot_path
        .map(PathBuf::from)
        .unwrap_or_else(blackwell_default_snapshot_path);

    let snapshot = read_blackwell_snapshot(&path).map_err(|error| {
        Error::from_reason(format!(
            "Unable to read Blackwell benchmark snapshot {}: {error}",
            path.display(),
        ))
    })?;

    if snapshot.steps.is_empty() || snapshot.tank_count == 0 ||
        snapshot.export_power.is_empty() ||
        snapshot.demand_energy.len() != snapshot.export_power.len()
    {
        return Err(Error::from_reason("Blackwell snapshot is empty or inconsistent."));
    }

    let requested_samples = sample_steps
        .unwrap_or(192)
        .clamp(2, snapshot.steps.len() as u32) as usize;

    let mut sample_positions = Vec::<usize>::with_capacity(requested_samples);
    for sample in 0..requested_samples {
        let position = sample * (snapshot.steps.len() - 1) / (requested_samples - 1);
        if sample_positions.last().copied() != Some(position) {
            sample_positions.push(position);
        }
    }

    let selected_device = device_id.unwrap_or(0);
    let repetitions = repetitions.unwrap_or(20).clamp(1, 10_000);

    let tank_parameters = (0..snapshot.tank_count)
        .map(|tank| research_tank_from_packed(&snapshot.packed_config, tank))
        .collect::<Vec<_>>();

    let mut packed_tanks = Vec::<f64>::with_capacity(snapshot.tank_count * 6);
    for tank in &tank_parameters {
        packed_tanks.push(tank.target_storage_energy);
        packed_tanks.push(tank.maximum_heat_pump_electrical_power);
        packed_tanks.push(tank.cop * tank.time_step_hours);
        packed_tanks.push(tank.heat_loss_retention);
        packed_tanks.push(tank.heat_loss_intercept);
        packed_tanks.push(if tank.affine_supported { 1.0 } else { 0.0 });
    }

    let step_count = sample_positions.len();
    let job_count = step_count * snapshot.tank_count;
    let mut current_indices = Vec::<u32>::with_capacity(step_count);
    let mut horizon_steps = Vec::<u32>::with_capacity(step_count);
    let mut dynamic_values = Vec::<f64>::with_capacity(job_count * 2);
    let mut energy_cut_levels = Vec::<f64>::with_capacity(job_count);
    let mut reference_results = Vec::<ResearchCausalResult>::with_capacity(job_count);
    let mut supported_mask = Vec::<bool>::with_capacity(job_count);

    let mut supported_jobs = 0_u32;
    let mut correction_search_jobs = 0_u32;

    for &position in &sample_positions {
        let step = &snapshot.steps[position];
        let horizon = step.horizon_steps as usize;
        if horizon == 0 || horizon > 256 {
            return Err(Error::from_reason(format!(
                "B3 benchmark supports horizon 1..=256; snapshot contains {horizon}."
            )));
        }

        current_indices.push(step.current_index);
        horizon_steps.push(step.horizon_steps);

        let export_window = research_window(
            &snapshot.export_power,
            step.current_index as usize,
            horizon,
        );
        let demand_window = research_window(
            &snapshot.demand_energy,
            step.current_index as usize,
            horizon,
        );

        for tank_index in 0..snapshot.tank_count {
            let base = tank_index * BLACKWELL_DYNAMIC_STRIDE;
            let current = step.dynamic[base + 0];
            let reserve_input = step.dynamic[base + 1];
            let requested = step.dynamic[base + 2];
            let tank = tank_parameters[tank_index];

            dynamic_values.push(current);
            dynamic_values.push(reserve_input);

            let energy_cut = research_energy_cut_level(&export_window, requested, tank);
            energy_cut_levels.push(energy_cut);

            let reference = research_run_scalar(
                &export_window,
                &demand_window,
                current,
                reserve_input,
                requested,
                tank,
            );

            let reserve = reserve_input.max(0.0).min(tank.target_storage_energy);
            let supported = tank.affine_supported &&
                tank.heat_loss_retention > 0.0 &&
                reserve > 1.0e-6;

            if supported {
                supported_jobs += 1;
                if !reference.normal_plan_survives && reference.maximum_pv_plan_survives {
                    correction_search_jobs += 1;
                }
            }

            supported_mask.push(supported);
            reference_results.push(reference);
        }
    }

    /* Existing complete causal GPU reference on the same sampled steps. */
    let cuda_parameters = blackwell_parameters_from_packed(
        &snapshot.packed_config,
        snapshot.data_length,
    )?;
    let mut reference_context = CudaSimulationContext::new(selected_device)?;
    reference_context.upload_causal_arrays(&snapshot.export_power, &snapshot.demand_energy)?;
    reference_context.configure_causal_static_parameters(&cuda_parameters)?;
    let mut reference_dynamic = vec![SunoptCausalDynamic::default(); snapshot.tank_count];
    let mut reference_output = vec![0.0_f64; snapshot.tank_count];

    let reference_started = Instant::now();
    for &position in &sample_positions {
        let step = &snapshot.steps[position];
        blackwell_dynamic_from_values(
            &step.dynamic,
            snapshot.tank_count,
            &mut reference_dynamic,
        )?;
        reference_context.run_causal_dynamic_batch_into(
            step.current_index as usize,
            step.horizon_steps as usize,
            &reference_dynamic,
            &mut reference_output,
        )?;
    }
    let reference_full_causal_gpu_ms = reference_started.elapsed().as_secs_f64() * 1000.0;

    let iteration_limits = vec![1_u32, 2, 3, 4, 6, 8];
    let mut upload_ms = Vec::<f64>::new();
    let mut kernel_ms = Vec::<f64>::new();
    let mut download_ms = Vec::<f64>::new();
    let mut wall_ms = Vec::<f64>::new();
    let mut annual_projected_kernel_ms = Vec::<f64>::new();
    let mut status_mismatches = Vec::<u32>::new();
    let mut max_abs_cut_error = Vec::<f64>::new();
    let mut mean_abs_cut_error = Vec::<f64>::new();
    let mut cut_error_over_1e6 = Vec::<u32>::new();
    let mut cut_error_over_1e5 = Vec::<u32>::new();
    let mut cut_error_over_1e4 = Vec::<u32>::new();
    let mut average_correction_iterations = Vec::<f64>::new();
    let mut average_evaluations_supported = Vec::<f64>::new();
    let mut maximum_bracket_width = Vec::<f64>::new();

    for &limit in &iteration_limits {
        let mut cuts = vec![0.0_f64; job_count];
        let mut status = vec![0_u32; job_count];
        let mut iterations = vec![0_u32; job_count];
        let mut evaluations = vec![0_u32; job_count];
        let mut widths = vec![0.0_f64; job_count];
        let mut this_upload = 0.0_f64;
        let mut kernel_total = 0.0_f64;
        let mut this_download = 0.0_f64;
        let mut this_wall = 0.0_f64;

        let error = unsafe {
            sunopt_cuda_run_piecewise_affine_b3_benchmark(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                snapshot.tank_count,
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_values.as_ptr(),
                energy_cut_levels.as_ptr(),
                step_count,
                limit,
                repetitions,
                cuts.as_mut_ptr(),
                status.as_mut_ptr(),
                iterations.as_mut_ptr(),
                evaluations.as_mut_ptr(),
                widths.as_mut_ptr(),
                &mut this_upload,
                &mut kernel_total,
                &mut this_download,
                &mut this_wall,
            )
        };

        if error != 0 {
            return Err(Error::from_reason(format!(
                "CUDA B3 piecewise-affine benchmark failed with CUDA error code {error}."
            )));
        }

        let this_kernel = kernel_total / repetitions as f64;
        let mut mismatch_count = 0_u32;
        let mut max_error = 0.0_f64;
        let mut sum_error = 0.0_f64;
        let mut count_error = 0_u32;
        let mut over_1e6 = 0_u32;
        let mut over_1e5 = 0_u32;
        let mut over_1e4 = 0_u32;
        let mut correction_iteration_sum = 0_u64;
        let mut correction_count = 0_u32;
        let mut evaluation_sum = 0_u64;
        let mut max_width = 0.0_f64;

        for job in 0..job_count {
            if !supported_mask[job] {
                if status[job] != 3 { mismatch_count += 1; }
                continue;
            }

            let reference = reference_results[job];
            let expected_status = if reference.normal_plan_survives {
                0_u32
            } else if !reference.maximum_pv_plan_survives {
                1_u32
            } else {
                2_u32
            };

            if status[job] != expected_status { mismatch_count += 1; }

            let cut_error = (cuts[job] - reference.cut_level_power).abs();
            max_error = max_error.max(cut_error);
            sum_error += cut_error;
            count_error += 1;
            if cut_error > 1.0e-6 { over_1e6 += 1; }
            if cut_error > 1.0e-5 { over_1e5 += 1; }
            if cut_error > 1.0e-4 { over_1e4 += 1; }

            evaluation_sum += evaluations[job] as u64;
            max_width = max_width.max(widths[job]);

            if expected_status == 2 {
                correction_iteration_sum += iterations[job] as u64;
                correction_count += 1;
            }
        }

        upload_ms.push(this_upload);
        kernel_ms.push(this_kernel);
        download_ms.push(this_download);
        wall_ms.push(this_wall);
        annual_projected_kernel_ms.push(
            this_kernel * snapshot.steps.len() as f64 / step_count as f64
        );
        status_mismatches.push(mismatch_count);
        max_abs_cut_error.push(max_error);
        mean_abs_cut_error.push(if count_error > 0 { sum_error / count_error as f64 } else { 0.0 });
        cut_error_over_1e6.push(over_1e6);
        cut_error_over_1e5.push(over_1e5);
        cut_error_over_1e4.push(over_1e4);
        average_correction_iterations.push(
            if correction_count > 0 {
                correction_iteration_sum as f64 / correction_count as f64
            } else { 0.0 }
        );
        average_evaluations_supported.push(
            if supported_jobs > 0 {
                evaluation_sum as f64 / supported_jobs as f64
            } else { 0.0 }
        );
        maximum_bracket_width.push(max_width);
    }

    Ok(BlackwellPiecewiseAffineB3Result {
        snapshot_path: path.display().to_string(),
        device_id: selected_device,
        repetitions,
        annual_step_count: snapshot.steps.len() as u32,
        sample_step_count: step_count as u32,
        sample_job_count: job_count as u32,
        supported_jobs,
        unsupported_jobs: job_count as u32 - supported_jobs,
        correction_search_jobs,
        reference_full_causal_gpu_ms,
        iteration_limits,
        upload_ms,
        kernel_ms,
        download_ms,
        wall_ms,
        annual_projected_kernel_ms,
        status_mismatches,
        max_abs_cut_error,
        mean_abs_cut_error,
        cut_error_over_1e6,
        cut_error_over_1e5,
        cut_error_over_1e4,
        average_correction_iterations,
        average_evaluations_supported,
        maximum_bracket_width,
    })
}
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FN_END
'''.strip()

MJS = r'''import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");
const sampleSteps = Number.parseInt(process.argv[3] ?? "192", 10);
const repetitions = Number.parseInt(process.argv[4] ?? "20", 10);
const deviceId = Number.parseInt(process.argv[5] ?? "0", 10);

const result = addon.runBlackwellPiecewiseAffineB3(
  snapshot,
  sampleSteps,
  repetitions,
  deviceId,
);

console.log(`SUNOPT_PIECEWISE_AFFINE_B3_JSON=${JSON.stringify(result)}`);
console.log();
console.log("--- BLACKWELL B3 PIECEWISE-AFFINE CUT SOLVER ---");
console.log(`snapshot=${result.snapshotPath}`);
console.log(`sampleSteps=${result.sampleStepCount}`);
console.log(`sampleJobs=${result.sampleJobCount}`);
console.log(`supportedJobs=${result.supportedJobs}/${result.sampleJobCount}`);
console.log(`correctionSearchJobs=${result.correctionSearchJobs}`);
console.log(`referenceFullCausalGpuMs=${result.referenceFullCausalGpuMs.toFixed(4)}`);
console.log();
console.log([
  "iters","kernelMs","annualProjectedMs","avgEvals","maxCutErr",
  ">1e-6",">1e-5",">1e-4","statusMis","maxBracket"
].join("\t"));

for (let i = 0; i < result.iterationLimits.length; i += 1) {
  console.log([
    result.iterationLimits[i],
    result.kernelMs[i].toFixed(6),
    result.annualProjectedKernelMs[i].toFixed(3),
    result.averageEvaluationsSupported[i].toFixed(3),
    result.maxAbsCutError[i].toExponential(6),
    result.cutErrorOver1e6[i],
    result.cutErrorOver1e5[i],
    result.cutErrorOver1e4[i],
    result.statusMismatches[i],
    result.maximumBracketWidth[i].toExponential(6),
  ].join("\t"));
}

const report = path.join(root, "blackwell-piecewise-affine-b3.json");
writeFileSync(report, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log();
console.log(`report=${report}`);
'''.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="ascii").strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid hash in {HASH_FILE}")
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
        raise RuntimeError(f"Only one marker present: {begin} / {end}")
    if not has_begin:
        raise RuntimeError(f"Marker pair missing: {begin} / {end}")
    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[:start] + block + source[finish:]


def insert_or_replace(source: str, begin: str, end: str, block: str, anchors: list[str]) -> str:
    if begin in source or end in source:
        return replace_between(source, begin, end, block)
    for anchor in anchors:
        pos = source.find(anchor)
        if pos >= 0:
            return source[:pos] + block + "\n\n\n" + source[pos:]
    raise RuntimeError(f"No insertion anchor found for {begin}: {anchors}")


def patch_header(source: str) -> str:
    for token in ("SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN", "SUNOPT_CUDA_EXPORT"):
        if token not in source:
            raise RuntimeError(f"Unexpected cuda_context.h; missing {token!r}")
    return insert_or_replace(
        source, H_BEGIN, H_END, H_BLOCK,
        ["// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN", "SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile("],
    )


def patch_cuda(source: str) -> str:
    for token in ("SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN", "sunopt_max_affine_tree_kernel"):
        if token not in source:
            raise RuntimeError(f"Unexpected cuda_context.cu; missing {token!r}")
    return insert_or_replace(
        source, CU_BEGIN, CU_END, CU_BLOCK,
        ["// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN", 'extern "C" int sunopt_cuda_get_profile('],
    )


def patch_rust(source: str) -> str:
    required = [
        "SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_BEGIN",
        "SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_BEGIN",
        "SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_BEGIN",
        "struct ResearchTankParameters",
        "fn research_tank_from_packed(",
        "fn research_window(",
        "fn research_energy_cut_level(",
        "fn research_run_scalar(",
        "struct BlackwellReplaySnapshot",
        "run_causal_dynamic_batch_into(",
    ]
    for token in required:
        if token not in source:
            raise RuntimeError(f"Unexpected cuda_context.rs; missing {token!r}")

    source = insert_or_replace(
        source, RS_FFI_BEGIN, RS_FFI_END, RS_FFI_BLOCK,
        ["    // SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_BEGIN", "    fn sunopt_cuda_get_profile("],
    )
    source = insert_or_replace(
        source, RS_RESULT_BEGIN, RS_RESULT_END, RS_RESULT_BLOCK,
        ["// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_BEGIN", "#[napi]\npub struct CudaCausalWorker {"],
    )
    source = insert_or_replace(
        source, RS_FN_BEGIN, RS_FN_END, RS_FN_BLOCK,
        ["// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_BEGIN", "#[napi]\npub fn run_blackwell_replay("],
    )
    return source


def build_package() -> Path:
    for path in (H_PATH, CU_PATH, RS_PATH):
        if not path.is_file():
            raise RuntimeError(f"Required current project file missing: {path}")

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
        "files": {name: sha256_bytes(data) for name, data in payloads.items()},
        "delete": [],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    release = sha256_bytes(manifest_bytes)

    INBOX.mkdir(parents=True, exist_ok=True)
    output = INBOX / "20261109-blackwell-piecewise-affine-b3.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("manifest.sha256", (release + "\n").encode("ascii"))
        for name, data in payloads.items():
            archive.writestr(name, data)

    print("SunOpt B3 piecewise-affine benchmark package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print("  production lockstep: UNCHANGED")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(r"  cd .\rust\sunopt-node")
    print(r"  npm run build")
    print(r"  node .\scripts\blackwell-piecewise-affine-b3.mjs .\blackwell-benchmark.bin 192 20 0")
    return output


if __name__ == "__main__":
    build_package()
