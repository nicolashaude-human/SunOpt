#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define SUNOPT_CUDA_EXPORT __declspec(dllexport)
#else
#define SUNOPT_CUDA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SunoptCudaContext SunoptCudaContext;

typedef struct SunoptCausalParameters {
    size_t data_length;
    size_t current_index;
    size_t horizon_steps;

    double current_storage_energy;
    double minimum_reserve_energy;
    double target_storage_energy;
    double maximum_heat_pump_electrical_power;
    double cop;

    /*
     * >= 0:
     *   CUDA computes the canonical 32-iteration energy Waterline
     *   before causal survival testing.
     *
     * < 0:
     *   use host-provided energy_cut_level (legacy/debug path).
     */
    double requested_electrical_energy;

    double energy_cut_level;
    double time_step_hours;
    double storage_energy_per_kelvin;
    double storage_conductance_w_per_kelvin;
    double storage_min_temperature;
    double ambient_temperature;
    double daily_heat_loss;
} SunoptCausalParameters;

typedef struct SunoptCausalResult {
    double cut_level_power;
    int normal_plan_survives;
    int maximum_pv_plan_survives;
    unsigned int survival_checks;
    unsigned int binary_search_iterations;
} SunoptCausalResult;


typedef struct SunoptNoSunshineTankParameters {
    size_t horizon_steps;
    size_t protected_steps;

    double time_step_hours;

    double maximum_heat_pump_electrical_power;
    double cop;

    double initial_waterline_power;

    double target_storage_energy;

    double storage_energy_per_kelvin;
    double storage_conductance_w_per_kelvin;
    double storage_min_temperature;
    double ambient_temperature;
    double daily_heat_loss;
} SunoptNoSunshineTankParameters;

typedef struct SunoptCudaProfile {
    uint64_t causal_calls;
    uint64_t causal_jobs;

    double parameter_h2d_ms;
    double kernel_ms;
    double result_d2h_ms;
    double gpu_total_ms;

    double host_wait_ms;
    double wall_ms;

    uint64_t profiled_search_jobs;

    uint64_t energy_phase_cycles;
    uint64_t boundary_phase_cycles;
    uint64_t radix_round_0_cycles;
    uint64_t radix_round_1_cycles;
    uint64_t radix_round_2_cycles;
    uint64_t radix_round_3_cycles;

    uint64_t boundary_storage_steps;
    uint64_t radix_round_0_storage_steps;
    uint64_t radix_round_1_storage_steps;
    uint64_t radix_round_2_storage_steps;
    uint64_t radix_round_3_storage_steps;
} SunoptCudaProfile;

SUNOPT_CUDA_EXPORT int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_destroy(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_reserve(
    SunoptCudaContext* context,
    size_t input_bytes,
    size_t workspace_bytes,
    size_t result_bytes
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_capacities(
    const SunoptCudaContext* context,
    size_t* out_input_bytes,
    size_t* out_workspace_bytes,
    size_t* out_result_bytes
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_upload_causal_arrays(
    SunoptCudaContext* context,
    const double* export_power,
    const double* demand_energy,
    size_t length
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal(
    SunoptCudaContext* context,
    const SunoptCausalParameters* parameters,
    SunoptCausalResult* host_result
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_batch(
    SunoptCudaContext* context,
    const SunoptCausalParameters* host_parameters,
    size_t job_count,
    SunoptCausalResult* host_results
);


SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_cut_batch(
    SunoptCudaContext* context,
    const SunoptCausalParameters* host_parameters,
    size_t job_count,
    double* host_cut_level_results
);


SUNOPT_CUDA_EXPORT int sunopt_cuda_precompute_no_sunshine_waterlines(
    SunoptCudaContext* context,
    const SunoptNoSunshineTankParameters* host_parameters,
    size_t tank_count,
    double* host_waterlines
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile(
    const SunoptCudaContext* context,
    SunoptCudaProfile* out_profile
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_reset_profile(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT const char* sunopt_cuda_error_string(
    int error_code
);

#ifdef __cplusplus
}
#endif
