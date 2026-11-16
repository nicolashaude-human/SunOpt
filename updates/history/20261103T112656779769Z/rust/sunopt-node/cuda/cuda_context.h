#pragma once

#include <stddef.h>

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

SUNOPT_CUDA_EXPORT int sunopt_cuda_upload_input(
    SunoptCudaContext* context,
    const void* host_data,
    size_t bytes
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_download_result(
    SunoptCudaContext* context,
    void* host_data,
    size_t bytes
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

SUNOPT_CUDA_EXPORT const char* sunopt_cuda_error_string(
    int error_code
);

#ifdef __cplusplus
}
#endif
