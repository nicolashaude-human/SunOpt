#include "cuda_context.h"

#include <cuda_runtime.h>

#include <new>

struct SunoptCudaContext {
    int device_id;
    cudaStream_t stream;

    void* input_buffer;
    size_t input_capacity;

    void* workspace_buffer;
    size_t workspace_capacity;

    void* result_buffer;
    size_t result_capacity;

    size_t causal_length;
};

static cudaError_t reserve_buffer(
    void** buffer,
    size_t* capacity,
    size_t required_bytes
) {
    if (required_bytes <= *capacity) {
        return cudaSuccess;
    }

    void* replacement = nullptr;

    cudaError_t err = cudaMalloc(
        &replacement,
        required_bytes
    );

    if (err != cudaSuccess) {
        return err;
    }

    if (*buffer != nullptr) {
        err = cudaFree(*buffer);

        if (err != cudaSuccess) {
            cudaFree(replacement);
            return err;
        }
    }

    *buffer = replacement;
    *capacity = required_bytes;

    return cudaSuccess;
}

__device__ static inline size_t wrap_index(
    size_t index,
    size_t length
) {
    return index % length;
}

__device__ static inline double hp_power_at_cut_level(
    double export_power,
    double cut_level_power,
    double maximum_heat_pump_electrical_power
) {
    double above =
        export_power - cut_level_power;

    if (above < 0.0) {
        above = 0.0;
    }

    return above <
        maximum_heat_pump_electrical_power
            ? above
            : maximum_heat_pump_electrical_power;
}

__device__ static inline double calculate_heat_loss(
    double stored_energy,
    const SunoptCausalParameters& p
) {
    if (p.daily_heat_loss >= 0.0) {
        double loss =
            p.daily_heat_loss *
            p.time_step_hours /
            24.0;

        if (loss > stored_energy) {
            loss = stored_energy;
        }

        return loss > 0.0 ? loss : 0.0;
    }

    const double storage_temperature =
        p.storage_min_temperature +
        stored_energy /
        p.storage_energy_per_kelvin;

    double delta_temperature =
        storage_temperature -
        p.ambient_temperature;

    if (delta_temperature < 0.0) {
        delta_temperature = 0.0;
    }

    double requested_heat_loss =
        p.storage_conductance_w_per_kelvin *
        delta_temperature *
        p.time_step_hours /
        1000.0;

    if (requested_heat_loss > stored_energy) {
        requested_heat_loss = stored_energy;
    }

    return requested_heat_loss > 0.0
        ? requested_heat_loss
        : 0.0;
}

__device__ static bool survives_at_cut_level(
    const double* export_power,
    const double* demand_energy,
    double cut_level_power,
    const SunoptCausalParameters& p,
    unsigned int* survival_checks
) {
    constexpr double ENERGY_EPSILON = 1.0e-6;

    *survival_checks += 1;

    double reserve_floor =
        p.minimum_reserve_energy <
        p.target_storage_energy
            ? p.minimum_reserve_energy
            : p.target_storage_energy;

    double simulated_energy =
        p.current_storage_energy;

    if (simulated_energy < 0.0) {
        simulated_energy = 0.0;
    }

    if (
        simulated_energy >
        p.target_storage_energy
    ) {
        simulated_energy =
            p.target_storage_energy;
    }

    for (
        size_t offset = 0;
        offset < p.horizon_steps;
        ++offset
    ) {
        const size_t index =
            wrap_index(
                p.current_index + offset,
                p.data_length
            );

        double export_value =
            export_power[index];

        if (export_value < 0.0) {
            export_value = 0.0;
        }

        const double hp_power =
            hp_power_at_cut_level(
                export_value,
                cut_level_power,
                p.maximum_heat_pump_electrical_power
            );

        const double hp_thermal_energy =
            hp_power *
            p.cop *
            p.time_step_hours;

        if (offset == 0) {
            simulated_energy +=
                hp_thermal_energy;

            if (
                simulated_energy >
                p.target_storage_energy
            ) {
                simulated_energy =
                    p.target_storage_energy;
            }

            continue;
        }

        const double heat_loss =
            calculate_heat_loss(
                simulated_energy,
                p
            );

        double after_loss =
            simulated_energy -
            heat_loss;

        if (after_loss < 0.0) {
            after_loss = 0.0;
        }

        double demand =
            demand_energy[index];

        if (demand < 0.0) {
            demand = 0.0;
        }

        const double discharged_energy =
            demand < after_loss
                ? demand
                : after_loss;

        const double energy_before_forecast_charge =
            after_loss -
            discharged_energy;

        double unmet_demand =
            demand -
            discharged_energy;

        if (unmet_demand < 0.0) {
            unmet_demand = 0.0;
        }

        if (unmet_demand > ENERGY_EPSILON) {
            return false;
        }

        if (
            reserve_floor -
            energy_before_forecast_charge >
            ENERGY_EPSILON
        ) {
            return false;
        }

        simulated_energy =
            energy_before_forecast_charge +
            hp_thermal_energy;

        if (
            simulated_energy >
            p.target_storage_energy
        ) {
            simulated_energy =
                p.target_storage_energy;
        }
    }

    return true;
}

__global__ static void causal_kernel(
    const double* export_power,
    const double* demand_energy,
    SunoptCausalParameters parameters,
    SunoptCausalResult* result
) {
    if (
        blockIdx.x != 0 ||
        threadIdx.x != 0
    ) {
        return;
    }

    unsigned int survival_checks = 0;
    unsigned int binary_iterations = 0;

    const bool normal_plan_survives =
        survives_at_cut_level(
            export_power,
            demand_energy,
            parameters.energy_cut_level,
            parameters,
            &survival_checks
        );

    bool maximum_pv_plan_survives =
        normal_plan_survives;

    double cut_level_power =
        parameters.energy_cut_level;

    if (!normal_plan_survives) {
        maximum_pv_plan_survives =
            survives_at_cut_level(
                export_power,
                demand_energy,
                0.0,
                parameters,
                &survival_checks
            );

        if (!maximum_pv_plan_survives) {
            cut_level_power = 0.0;
        } else {
            double lower_cut = 0.0;
            double upper_cut =
                parameters.energy_cut_level;

            for (
                int iteration = 0;
                iteration < 20;
                ++iteration
            ) {
                ++binary_iterations;

                const double middle_cut =
                    (
                        lower_cut +
                        upper_cut
                    ) /
                    2.0;

                if (
                    survives_at_cut_level(
                        export_power,
                        demand_energy,
                        middle_cut,
                        parameters,
                        &survival_checks
                    )
                ) {
                    lower_cut =
                        middle_cut;
                } else {
                    upper_cut =
                        middle_cut;
                }
            }

            cut_level_power =
                lower_cut;
        }
    }

    result->cut_level_power =
        cut_level_power;

    result->normal_plan_survives =
        normal_plan_survives ? 1 : 0;

    result->maximum_pv_plan_survives =
        maximum_pv_plan_survives ? 1 : 0;

    result->survival_checks =
        survival_checks;

    result->binary_search_iterations =
        binary_iterations;
}

extern "C" int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
) {
    if (out_context == nullptr) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    *out_context = nullptr;

    cudaError_t err =
        cudaSetDevice(device_id);

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    SunoptCudaContext* context =
        new (std::nothrow)
            SunoptCudaContext{};

    if (context == nullptr) {
        return static_cast<int>(
            cudaErrorMemoryAllocation
        );
    }

    context->device_id = device_id;
    context->stream = nullptr;

    context->input_buffer = nullptr;
    context->input_capacity = 0;

    context->workspace_buffer = nullptr;
    context->workspace_capacity = 0;

    context->result_buffer = nullptr;
    context->result_capacity = 0;

    context->causal_length = 0;

    err = cudaStreamCreateWithFlags(
        &context->stream,
        cudaStreamNonBlocking
    );

    if (err != cudaSuccess) {
        delete context;
        return static_cast<int>(err);
    }

    *out_context = context;

    return static_cast<int>(
        cudaSuccess
    );
}

extern "C" int sunopt_cuda_destroy(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(
            cudaSuccess
        );
    }

    cudaError_t first_error =
        cudaSuccess;

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        first_error = err;
    }

    if (context->stream != nullptr) {
        err = cudaStreamSynchronize(
            context->stream
        );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }

    if (context->input_buffer != nullptr) {
        err = cudaFree(
            context->input_buffer
        );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }

    if (
        context->workspace_buffer !=
        nullptr
    ) {
        err = cudaFree(
            context->workspace_buffer
        );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }

    if (
        context->result_buffer !=
        nullptr
    ) {
        err = cudaFree(
            context->result_buffer
        );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }

    if (context->stream != nullptr) {
        err = cudaStreamDestroy(
            context->stream
        );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }

    delete context;

    return static_cast<int>(
        first_error
    );
}

extern "C" int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
) {
    if (
        context == nullptr ||
        out_device_id == nullptr
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    *out_device_id =
        context->device_id;

    return static_cast<int>(
        cudaSuccess
    );
}

extern "C" int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(
        cudaStreamSynchronize(
            context->stream
        )
    );
}

extern "C" int sunopt_cuda_reserve(
    SunoptCudaContext* context,
    size_t input_bytes,
    size_t workspace_bytes,
    size_t result_bytes
) {
    if (context == nullptr) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->input_buffer,
        &context->input_capacity,
        input_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->workspace_buffer,
        &context->workspace_capacity,
        workspace_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->result_buffer,
        &context->result_capacity,
        result_bytes
    );

    return static_cast<int>(err);
}

extern "C" int sunopt_cuda_capacities(
    const SunoptCudaContext* context,
    size_t* out_input_bytes,
    size_t* out_workspace_bytes,
    size_t* out_result_bytes
) {
    if (
        context == nullptr ||
        out_input_bytes == nullptr ||
        out_workspace_bytes == nullptr ||
        out_result_bytes == nullptr
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    *out_input_bytes =
        context->input_capacity;

    *out_workspace_bytes =
        context->workspace_capacity;

    *out_result_bytes =
        context->result_capacity;

    return static_cast<int>(
        cudaSuccess
    );
}

extern "C" int sunopt_cuda_upload_input(
    SunoptCudaContext* context,
    const void* host_data,
    size_t bytes
) {
    if (
        context == nullptr ||
        host_data == nullptr ||
        bytes >
            context->input_capacity
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(
        cudaMemcpyAsync(
            context->input_buffer,
            host_data,
            bytes,
            cudaMemcpyHostToDevice,
            context->stream
        )
    );
}

extern "C" int sunopt_cuda_download_result(
    SunoptCudaContext* context,
    void* host_data,
    size_t bytes
) {
    if (
        context == nullptr ||
        host_data == nullptr ||
        bytes >
            context->result_capacity
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(
        cudaMemcpyAsync(
            host_data,
            context->result_buffer,
            bytes,
            cudaMemcpyDeviceToHost,
            context->stream
        )
    );
}

extern "C" int sunopt_cuda_upload_causal_arrays(
    SunoptCudaContext* context,
    const double* export_power,
    const double* demand_energy,
    size_t length
) {
    if (
        context == nullptr ||
        export_power == nullptr ||
        demand_energy == nullptr ||
        length == 0
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    const size_t one_array_bytes =
        length *
        sizeof(double);

    const size_t total_bytes =
        one_array_bytes * 2;

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->input_buffer,
        &context->input_capacity,
        total_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->result_buffer,
        &context->result_capacity,
        sizeof(SunoptCausalResult)
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    unsigned char* base =
        static_cast<unsigned char*>(
            context->input_buffer
        );

    err = cudaMemcpyAsync(
        base,
        export_power,
        one_array_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        base + one_array_bytes,
        demand_energy,
        one_array_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    context->causal_length =
        length;

    return static_cast<int>(
        cudaStreamSynchronize(
            context->stream
        )
    );
}

extern "C" int sunopt_cuda_run_causal(
    SunoptCudaContext* context,
    const SunoptCausalParameters* parameters,
    SunoptCausalResult* host_result
) {
    if (
        context == nullptr ||
        parameters == nullptr ||
        host_result == nullptr ||
        context->causal_length == 0 ||
        parameters->data_length !=
            context->causal_length ||
        parameters->current_index >=
            parameters->data_length ||
        parameters->horizon_steps == 0 ||
        parameters->horizon_steps >
            parameters->data_length
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t one_array_bytes =
        context->causal_length *
        sizeof(double);

    unsigned char* base =
        static_cast<unsigned char*>(
            context->input_buffer
        );

    const double* export_power =
        reinterpret_cast<const double*>(
            base
        );

    const double* demand_energy =
        reinterpret_cast<const double*>(
            base + one_array_bytes
        );

    SunoptCausalResult* device_result =
        static_cast<SunoptCausalResult*>(
            context->result_buffer
        );

    causal_kernel<<<1, 1, 0, context->stream>>>(
        export_power,
        demand_energy,
        *parameters,
        device_result
    );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        host_result,
        device_result,
        sizeof(SunoptCausalResult),
        cudaMemcpyDeviceToHost,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(
        cudaStreamSynchronize(
            context->stream
        )
    );
}

extern "C" const char* sunopt_cuda_error_string(
    int error_code
) {
    return cudaGetErrorString(
        static_cast<cudaError_t>(
            error_code
        )
    );
}
