#include "cuda_context.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
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

    /*
     * Production causal parameter residency.
     *
     * Static tank/controller parameters are uploaded once.
     * Per annual step only SunoptCausalDynamic is transferred.
     */
    void* causal_static_parameter_buffer;
    size_t causal_static_parameter_capacity;
    size_t causal_static_job_count;

    /*
     * Production cut-only output can be written directly by the GPU
     * into mapped pinned host memory. This removes the per-step
     * cudaMemcpyAsync(D2H) command for the 28 cut-level doubles.
     */
    void* causal_mapped_cut_host;
    double* causal_mapped_cut_device;
    size_t causal_mapped_cut_capacity;

    size_t causal_length;

    cudaEvent_t profile_start;
    cudaEvent_t profile_after_h2d;
    cudaEvent_t profile_after_kernel;
    cudaEvent_t profile_after_d2h;

    SunoptCudaProfile profile;
};


static void reset_profile(
    SunoptCudaProfile* profile
) {
    if (profile == nullptr) {
        return;
    }

    *profile = SunoptCudaProfile{};
}


static void destroy_profile_events(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return;
    }

    if (context->profile_start != nullptr) {
        cudaEventDestroy(context->profile_start);
        context->profile_start = nullptr;
    }

    if (context->profile_after_h2d != nullptr) {
        cudaEventDestroy(context->profile_after_h2d);
        context->profile_after_h2d = nullptr;
    }

    if (context->profile_after_kernel != nullptr) {
        cudaEventDestroy(context->profile_after_kernel);
        context->profile_after_kernel = nullptr;
    }

    if (context->profile_after_d2h != nullptr) {
        cudaEventDestroy(context->profile_after_d2h);
        context->profile_after_d2h = nullptr;
    }
}


static cudaError_t create_profile_events(
    SunoptCudaContext* context
) {
    cudaError_t err =
        cudaEventCreate(&context->profile_start);

    if (err != cudaSuccess) {
        return err;
    }

    err = cudaEventCreate(
        &context->profile_after_h2d
    );

    if (err != cudaSuccess) {
        destroy_profile_events(context);
        return err;
    }

    err = cudaEventCreate(
        &context->profile_after_kernel
    );

    if (err != cudaSuccess) {
        destroy_profile_events(context);
        return err;
    }

    err = cudaEventCreate(
        &context->profile_after_d2h
    );

    if (err != cudaSuccess) {
        destroy_profile_events(context);
        return err;
    }

    return cudaSuccess;
}

static cudaError_t reserve_buffer(
    void** buffer,
    size_t* capacity,
    size_t required_bytes
) {
    if (required_bytes <= *capacity) {
        return cudaSuccess;
    }

    void* replacement = nullptr;

    cudaError_t err =
        cudaMalloc(
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


static cudaError_t reserve_mapped_cut_buffer(
    SunoptCudaContext* context,
    size_t required_bytes
) {
    if (
        required_bytes <=
        context->causal_mapped_cut_capacity
    ) {
        return cudaSuccess;
    }


    void* replacement_host =
        nullptr;


    cudaError_t err =
        cudaHostAlloc(
            &replacement_host,
            required_bytes,
            cudaHostAllocMapped
        );


    if (
        err !=
        cudaSuccess
    ) {
        return err;
    }


    void* replacement_device =
        nullptr;


    err =
        cudaHostGetDevicePointer(
            &replacement_device,
            replacement_host,
            0
        );


    if (
        err !=
        cudaSuccess
    ) {
        cudaFreeHost(
            replacement_host
        );

        return err;
    }


    if (
        context->causal_mapped_cut_host !=
        nullptr
    ) {
        err =
            cudaFreeHost(
                context->causal_mapped_cut_host
            );


        if (
            err !=
            cudaSuccess
        ) {
            cudaFreeHost(
                replacement_host
            );

            return err;
        }
    }


    context->causal_mapped_cut_host =
        replacement_host;

    context->causal_mapped_cut_device =
        static_cast<double*>(
            replacement_device
        );

    context->causal_mapped_cut_capacity =
        required_bytes;


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
        export_power -
        cut_level_power;

    if (above < 0.0) {
        above = 0.0;
    }

    return above <
        maximum_heat_pump_electrical_power
            ? above
            : maximum_heat_pump_electrical_power;
}

struct SunoptHeatLossModel {
    /*
     * Unified affine loss model:
     *
     *   requestedLoss =
     *       heat_loss_slope * storedEnergy
     *       + heat_loss_intercept
     *
     * Legacy fixed daily loss:
     *   slope     = 0
     *   intercept = fixed loss per timestep
     *
     * Newton storage loss:
     *   slope/intercept are the precombined physical coefficients.
     *
     * calculate_heat_loss() therefore has no per-step mode branch.
     */
    double heat_loss_slope;

    double heat_loss_intercept;

    /*
     * Affine storage retention used by the production shared-window
     * fast path:
     *
     *   retention =
     *       1 - heat_loss_slope
     *
     * Together with a staged:
     *
     *   thermalLoad[offset] =
     *       heat_loss_intercept + demand[offset]
     *
     * the chronological recurrence becomes:
     *
     *   net =
     *       storedEnergy * retention
     *       - thermalLoad[offset]
     *
     * instead of:
     *
     *   loss  = storedEnergy*slope + intercept
     *   after = storedEnergy - loss
     *   net   = after - demand
     */
    double heat_loss_retention;

    /*
     * HP thermal energy produced by one kW_el during one timestep.
     */
    double hp_thermal_energy_per_electrical_power;

    /*
     * Minimum allowed value of:
     *
     *   afterLoss - demand
     *
     * that simultaneously satisfies both canonical conditions:
     *
     *   unmetDemand <= epsilon
     *   reserveFloor - energyAfterDemand <= epsilon
     *
     * for finite, nonnegative resident demand data.
     */
    double minimum_net_energy_after_demand;

    double initial_storage_energy;

    /*
     * Exact affine recurrence fast path.
     *
     * When:
     *
     *   heat_loss_slope     >= 0
     *   heat_loss_intercept >= 0
     *   reserve floor       > epsilon
     *
     * every surviving chronological step necessarily has:
     *
     *   requestedHeatLoss < storedEnergy
     *   netEnergyAfterDemand > 0
     *
     * Therefore the heat-loss clamp and post-demand max() are
     * mathematically irrelevant on every path that can survive.
     *
     * Failing candidates are still detected by the same threshold.
     */
    int affine_survival_fast_path;
};


__device__ static inline SunoptHeatLossModel
prepare_heat_loss_model(
    const SunoptCausalParameters& p
) {
    constexpr double ENERGY_EPSILON =
        1.0e-6;

    SunoptHeatLossModel model{};


    if (
        p.daily_heat_loss >=
        0.0
    ) {
        /*
         * Legacy fixed loss expressed through the same affine clamp
         * as the Newton model.
         */
        model.heat_loss_slope =
            0.0;

        model.heat_loss_intercept =
            p.daily_heat_loss *
            p.time_step_hours /
            24.0;

    } else {

        const double inverse_storage_energy_per_kelvin =
            1.0 /
            p.storage_energy_per_kelvin;

        const double temperature_offset_kelvin =
            p.storage_min_temperature -
            p.ambient_temperature;

        const double conductance_step_scale =
            p.storage_conductance_w_per_kelvin *
            p.time_step_hours /
            1000.0;

        model.heat_loss_slope =
            conductance_step_scale *
            inverse_storage_energy_per_kelvin;

        model.heat_loss_intercept =
            conductance_step_scale *
            temperature_offset_kelvin;
    }


    model.heat_loss_retention =
        1.0 -
        model.heat_loss_slope;


    model.hp_thermal_energy_per_electrical_power =
        p.cop *
        p.time_step_hours;


    const double reserve_floor =
        p.minimum_reserve_energy <
        p.target_storage_energy
            ? p.minimum_reserve_energy
            : p.target_storage_energy;


    /*
     * Exact collapse of the two canonical post-demand checks:
     *
     *   unmetDemand <= epsilon
     *   reserveFloor - energyAfterDemand <= epsilon
     *
     * Let net = afterLoss - demand.
     */
    model.minimum_net_energy_after_demand =
        reserve_floor >
        ENERGY_EPSILON
            ? reserve_floor -
              ENERGY_EPSILON
            : -ENERGY_EPSILON;


    model.affine_survival_fast_path =
        (
            model.heat_loss_slope >=
                0.0 &&
            model.heat_loss_intercept >=
                0.0 &&
            reserve_floor >
                ENERGY_EPSILON
        )
            ? 1
            : 0;


    model.initial_storage_energy =
        p.current_storage_energy;

    if (
        model.initial_storage_energy <
        0.0
    ) {
        model.initial_storage_energy =
            0.0;
    }

    if (
        model.initial_storage_energy >
        p.target_storage_energy
    ) {
        model.initial_storage_energy =
            p.target_storage_energy;
    }


    return model;
}


__device__ static inline double calculate_heat_loss(
    double stored_energy,
    const SunoptHeatLossModel& model
) {
    double requested_heat_loss =
        stored_energy *
        model.heat_loss_slope +
        model.heat_loss_intercept;


    if (
        requested_heat_loss <
        0.0
    ) {
        requested_heat_loss =
            0.0;
    }


    if (
        requested_heat_loss >
        stored_energy
    ) {
        requested_heat_loss =
            stored_energy;
    }


    return requested_heat_loss;
}


constexpr unsigned int SUNOPT_SHARED_CAUSAL_WINDOW_STEPS =
    256u;


/*
 * Forward declarations.
 *
 * The shared-window helpers are intentionally placed before the
 * existing warp reduction definitions, so NVCC must know these two
 * device functions before they are referenced.
 */
__device__ static inline double warp_sum_double(
    double value
);

__device__ static inline double warp_max_double(
    double value
);


/*
 * Shared-window survival path.
 *
 * export_window[] and demand_window[] contain the chronological
 * horizon starting at currentIndex. They are staged once per block.
 *
 * The recurrence is intentionally identical to survives_at_cut_level()
 * except that:
 *
 *   - no annual ring index exists inside the candidate loop;
 *   - no global annual-array loads occur inside the candidate loop.
 */
__device__ static bool survives_window_at_cut_level(
    const double* export_window,
    const double* demand_window,
    double cut_level_power,
    const SunoptCausalParameters& p,
    const SunoptHeatLossModel& heat_loss_model,
    unsigned int* survival_checks
) {
    *survival_checks += 1;

    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );

    const double maximum_heat_pump_electrical_power =
        p.maximum_heat_pump_electrical_power;

    const double target_storage_energy =
        p.target_storage_energy;

    const double minimum_net_energy_after_demand =
        heat_loss_model
          .minimum_net_energy_after_demand;

    const double hp_thermal_energy_per_electrical_power =
        heat_loss_model
          .hp_thermal_energy_per_electrical_power;

    double simulated_energy =
        heat_loss_model
          .initial_storage_energy;


    if (
        horizon_steps >
        0u
    ) {
        const double export_value =
            export_window[
                0
            ];

        const double hp_power =
            hp_power_at_cut_level(
                export_value,
                cut_level_power,
                maximum_heat_pump_electrical_power
            );

        const double hp_thermal_energy =
            hp_power *
            hp_thermal_energy_per_electrical_power;

        simulated_energy +=
            hp_thermal_energy;

        if (
            simulated_energy >
            target_storage_energy
        ) {
            simulated_energy =
                target_storage_energy;
        }
    }


    /*
     * --------------------------------------------------------------
     * EXACT POSITIVE-RESERVE AFFINE FAST PATH
     * --------------------------------------------------------------
     *
     * With nonnegative loss coefficients and reserveFloor > epsilon:
     *
     *   requestedLoss = E * slope + intercept >= 0
     *
     * If requestedLoss >= E, canonical afterLoss is zero and the
     * positive reserve threshold necessarily fails.
     *
     * Therefore any surviving step is exactly equivalent to:
     *
     *   requestedLoss = E * slope + intercept
     *   net = E - requestedLoss - demand
     *   if net < threshold: fail
     *   E = min(target, net + hpEnergy)
     *
     * No heat-loss clamp and no max(net, 0) are needed.
     */
    if (
        heat_loss_model
          .affine_survival_fast_path
    ) {
        const double heat_loss_retention =
            heat_loss_model
              .heat_loss_retention;


        /*
         * Production specialization for the 48 h causal horizon:
         * 192 quarter-hour samples.
         *
         * The fixed bound lets NVCC partially unroll the dominant
         * chronological recurrence while preserving the exact scalar
         * dependency and every physics comparison.
         *
         * Every other supported horizon uses the generic loop below.
         */
        if (
            horizon_steps ==
            192u
        ) {
            #pragma unroll 4
            for (
                unsigned int offset = 1u;
                offset < 192u;
                ++offset
            ) {
            const double export_value =
                export_window[
                    offset
                ];

            const double hp_power =
                hp_power_at_cut_level(
                    export_value,
                    cut_level_power,
                    maximum_heat_pump_electrical_power
                );

            const double hp_thermal_energy =
                hp_power *
                hp_thermal_energy_per_electrical_power;


            const double net_energy_after_demand =
                simulated_energy *
                heat_loss_retention -
                demand_window[
                    offset
                ];


            if (
                net_energy_after_demand <
                minimum_net_energy_after_demand
            ) {
                return false;
            }


            simulated_energy =
                net_energy_after_demand +
                hp_thermal_energy;


            if (
                simulated_energy >
                target_storage_energy
            ) {
                simulated_energy =
                    target_storage_energy;
            }
            }

            return true;
        }


        for (
            unsigned int offset = 1u;
            offset < horizon_steps;
            ++offset
        ) {
            const double export_value =
                export_window[
                    offset
                ];

            const double hp_power =
                hp_power_at_cut_level(
                    export_value,
                    cut_level_power,
                    maximum_heat_pump_electrical_power
                );

            const double hp_thermal_energy =
                hp_power *
                hp_thermal_energy_per_electrical_power;


            const double net_energy_after_demand =
                simulated_energy *
                heat_loss_retention -
                demand_window[
                    offset
                ];


            if (
                net_energy_after_demand <
                minimum_net_energy_after_demand
            ) {
                return false;
            }


            simulated_energy =
                net_energy_after_demand +
                hp_thermal_energy;


            if (
                simulated_energy >
                target_storage_energy
            ) {
                simulated_energy =
                    target_storage_energy;
            }
        }

        return true;
    }


    /*
     * Generic exact fallback for zero reserve or unusual thermal
     * coefficients.
     */
    for (
        unsigned int offset = 1u;
        offset < horizon_steps;
        ++offset
    ) {
        const double export_value =
            export_window[
                offset
            ];

        const double hp_power =
            hp_power_at_cut_level(
                export_value,
                cut_level_power,
                maximum_heat_pump_electrical_power
            );

        const double hp_thermal_energy =
            hp_power *
            hp_thermal_energy_per_electrical_power;

        const double heat_loss =
            calculate_heat_loss(
                simulated_energy,
                heat_loss_model
            );

        const double after_loss =
            simulated_energy -
            heat_loss;

        const double net_energy_after_demand =
            after_loss -
            demand_window[
                offset
            ];


        if (
            net_energy_after_demand <
            minimum_net_energy_after_demand
        ) {
            return false;
        }


        const double energy_before_forecast_charge =
            net_energy_after_demand >
            0.0
                ? net_energy_after_demand
                : 0.0;


        simulated_energy =
            energy_before_forecast_charge +
            hp_thermal_energy;


        if (
            simulated_energy >
            target_storage_energy
        ) {
            simulated_energy =
                target_storage_energy;
        }
    }

    return true;
}


__device__ static inline double
energy_at_cut_level_window_warp(
    const double* export_window,
    double cut_level_power,
    const SunoptCausalParameters& p,
    unsigned int lane
) {
    double local_energy =
        0.0;

    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );

    for (
        unsigned int offset = lane;
        offset < horizon_steps;
        offset += 32u
    ) {
        const double export_value =
            export_window[
                offset
            ];

        double available_power =
            export_value -
            cut_level_power;

        if (
            available_power <
            0.0
        ) {
            available_power =
                0.0;
        }

        const double hp_power =
            available_power <
                p.maximum_heat_pump_electrical_power
                ? available_power
                : p.maximum_heat_pump_electrical_power;

        local_energy +=
            hp_power *
            p.time_step_hours;
    }

    return warp_sum_double(
        local_energy
    );
}


__device__ static inline double
maximum_export_power_window_warp(
    const double* export_window,
    const SunoptCausalParameters& p,
    unsigned int lane
) {
    double local_max =
        0.0;

    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );

    for (
        unsigned int offset = lane;
        offset < horizon_steps;
        offset += 32u
    ) {
        const double export_value =
            export_window[
                offset
            ];

        local_max =
            local_max >
            export_value
                ? local_max
                : export_value;
    }

    return warp_max_double(
        local_max
    );
}


__device__ static inline double
calculate_energy_cut_level_window_warp(
    const double* export_window,
    const SunoptCausalParameters& p,
    unsigned int lane
) {
    const double maximum_export_power =
        maximum_export_power_window_warp(
            export_window,
            p,
            lane
        );

    const double maximum_schedulable_electrical_energy =
        energy_at_cut_level_window_warp(
            export_window,
            0.0,
            p,
            lane
        );

    double requested =
        p.requested_electrical_energy;

    if (
        requested <
        0.0
    ) {
        requested =
            0.0;
    }

    if (
        requested >
        maximum_schedulable_electrical_energy
    ) {
        requested =
            maximum_schedulable_electrical_energy;
    }

    double lower_cut =
        0.0;

    double upper_cut =
        maximum_export_power;

    for (
        int iteration = 0;
        iteration < 32;
        ++iteration
    ) {
        const double middle_cut =
            (
                lower_cut +
                upper_cut
            ) /
            2.0;

        const double energy =
            energy_at_cut_level_window_warp(
                export_window,
                middle_cut,
                p,
                lane
            );

        if (
            lane ==
            0
        ) {
            if (
                energy >
                requested
            ) {
                lower_cut =
                    middle_cut;
            } else {
                upper_cut =
                    middle_cut;
            }
        }

        lower_cut =
            __shfl_sync(
                0xffffffffu,
                lower_cut,
                0
            );

        upper_cut =
            __shfl_sync(
                0xffffffffu,
                upper_cut,
                0
            );
    }

    return upper_cut;
}


__device__ static bool survives_at_cut_level(
    const double* export_power,
    const double* demand_energy,
    double cut_level_power,
    const SunoptCausalParameters& p,
    const SunoptHeatLossModel& heat_loss_model,
    unsigned int* survival_checks
) {
    *survival_checks += 1;

    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );

    const unsigned int data_length =
        static_cast<unsigned int>(
            p.data_length
        );

    const unsigned int current_index =
        static_cast<unsigned int>(
            p.current_index
        );

    const double maximum_heat_pump_electrical_power =
        p.maximum_heat_pump_electrical_power;

    const double target_storage_energy =
        p.target_storage_energy;

    const double minimum_net_energy_after_demand =
        heat_loss_model
          .minimum_net_energy_after_demand;

    const double hp_thermal_energy_per_electrical_power =
        heat_loss_model
          .hp_thermal_energy_per_electrical_power;

    double simulated_energy =
        heat_loss_model
          .initial_storage_energy;

    unsigned int index =
        current_index;


    if (
        horizon_steps >
        0u
    ) {
        const double export_value =
            export_power[
                index
            ];

        const double hp_power =
            hp_power_at_cut_level(
                export_value,
                cut_level_power,
                maximum_heat_pump_electrical_power
            );

        simulated_energy +=
            hp_power *
            hp_thermal_energy_per_electrical_power;

        if (
            simulated_energy >
            target_storage_energy
        ) {
            simulated_energy =
                target_storage_energy;
        }

        ++index;

        if (
            index >=
            data_length
        ) {
            index =
                0u;
        }
    }


    if (
        heat_loss_model
          .affine_survival_fast_path
    ) {
        const double heat_loss_slope =
            heat_loss_model
              .heat_loss_slope;

        const double heat_loss_intercept =
            heat_loss_model
              .heat_loss_intercept;


        for (
            unsigned int offset = 1u;
            offset < horizon_steps;
            ++offset
        ) {
            const double export_value =
                export_power[
                    index
                ];

            const double hp_power =
                hp_power_at_cut_level(
                    export_value,
                    cut_level_power,
                    maximum_heat_pump_electrical_power
                );

            const double hp_thermal_energy =
                hp_power *
                hp_thermal_energy_per_electrical_power;

            const double requested_heat_loss =
                simulated_energy *
                heat_loss_slope +
                heat_loss_intercept;

            const double after_loss =
                simulated_energy -
                requested_heat_loss;

            const double net_energy_after_demand =
                after_loss -
                demand_energy[
                    index
                ];


            if (
                net_energy_after_demand <
                minimum_net_energy_after_demand
            ) {
                return false;
            }


            simulated_energy =
                net_energy_after_demand +
                hp_thermal_energy;


            if (
                simulated_energy >
                target_storage_energy
            ) {
                simulated_energy =
                    target_storage_energy;
            }


            ++index;

            if (
                index >=
                data_length
            ) {
                index =
                    0u;
            }
        }

        return true;
    }


    for (
        unsigned int offset = 1u;
        offset < horizon_steps;
        ++offset
    ) {
        const double export_value =
            export_power[
                index
            ];

        const double hp_power =
            hp_power_at_cut_level(
                export_value,
                cut_level_power,
                maximum_heat_pump_electrical_power
            );

        const double hp_thermal_energy =
            hp_power *
            hp_thermal_energy_per_electrical_power;

        const double heat_loss =
            calculate_heat_loss(
                simulated_energy,
                heat_loss_model
            );

        const double after_loss =
            simulated_energy -
            heat_loss;

        const double net_energy_after_demand =
            after_loss -
            demand_energy[
                index
            ];


        if (
            net_energy_after_demand <
            minimum_net_energy_after_demand
        ) {
            return false;
        }


        const double energy_before_forecast_charge =
            net_energy_after_demand >
            0.0
                ? net_energy_after_demand
                : 0.0;


        simulated_energy =
            energy_before_forecast_charge +
            hp_thermal_energy;


        if (
            simulated_energy >
            target_storage_energy
        ) {
            simulated_energy =
                target_storage_energy;
        }


        ++index;

        if (
            index >=
            data_length
        ) {
            index =
                0u;
        }
    }

    return true;
}


__device__ static SunoptCausalResult run_one_causal(
    const double* export_power,
    const double* demand_energy,
    const SunoptCausalParameters& parameters
) {
    SunoptCausalResult result{};

    unsigned int survival_checks = 0;
    unsigned int binary_iterations = 0;

    /*
     * Reference/legacy serial causal path.
     *
     * The fast warp32 kernel already prepares this model once per
     * block. run_one_causal() must do the same so its calls match the
     * new survives_at_cut_level() signature.
     */
    const SunoptHeatLossModel heat_loss_model =
        prepare_heat_loss_model(
            parameters
        );

    const bool normal_plan_survives =
        survives_at_cut_level(
            export_power,
            demand_energy,
            parameters.energy_cut_level,
            parameters,
            heat_loss_model,
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
                heat_loss_model,
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
                        heat_loss_model,
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

    result.cut_level_power =
        cut_level_power;

    result.normal_plan_survives =
        normal_plan_survives
            ? 1
            : 0;

    result.maximum_pv_plan_survives =
        maximum_pv_plan_survives
            ? 1
            : 0;

    result.survival_checks =
        survival_checks;

    result.binary_search_iterations =
        binary_iterations;

    return result;
}


/*
 * ================================================================
 * FULL-YEAR NO-SUNSHINE MATRIX
 * ================================================================
 *
 * The no-sunshine waterline used by the lockstep controller does not
 * depend on currentStorageEnergy. It depends only on:
 *
 *   - annual export/load arrays
 *   - current annual index
 *   - tank/storage constants
 *
 * Therefore every (tank, currentIndex) pair is independent.
 *
 * Grid:
 *
 *   tank_count * annual_length threads
 *
 * For 28 tanks and 35,040 annual steps:
 *
 *   981,120 independent jobs
 *
 * Each thread walks the 672-step seven-day horizon backwards and
 * writes exactly one currentWaterlinePower value.
 *
 * This removes 981,120 JS/native no-sunshine calls from the annual
 * lockstep hot path.
 */

__device__ static inline double nosunshine_heat_loss(
    double stored_energy,
    const SunoptNoSunshineTankParameters& p
) {
    if (stored_energy <= 0.0) {
        return 0.0;
    }

    if (p.daily_heat_loss >= 0.0) {
        double loss =
            p.daily_heat_loss *
            p.time_step_hours /
            24.0;

        if (loss < 0.0) {
            loss = 0.0;
        }

        if (loss > stored_energy) {
            loss = stored_energy;
        }

        return loss;
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

    double loss =
        p.storage_conductance_w_per_kelvin *
        delta_temperature *
        p.time_step_hours /
        1000.0;

    if (loss < 0.0) {
        loss = 0.0;
    }

    if (loss > stored_energy) {
        loss = stored_energy;
    }

    return loss;
}


__device__ static inline double nosunshine_after_step_energy(
    double current_energy,
    double demand_energy,
    const SunoptNoSunshineTankParameters& p
) {
    if (current_energy < 0.0) {
        current_energy = 0.0;
    }

    if (
        current_energy >
        p.target_storage_energy
    ) {
        current_energy =
            p.target_storage_energy;
    }

    const double heat_loss =
        nosunshine_heat_loss(
            current_energy,
            p
        );

    double after_loss =
        current_energy -
        heat_loss;

    if (after_loss < 0.0) {
        after_loss = 0.0;
    }

    if (demand_energy < 0.0) {
        demand_energy = 0.0;
    }

    const double discharged =
        demand_energy <
        after_loss
            ? demand_energy
            : after_loss;

    double stored =
        after_loss -
        discharged;

    if (stored < 0.0) {
        stored = 0.0;
    }

    return stored;
}


__device__ static inline double nosunshine_unmet_demand(
    double current_energy,
    double demand_energy,
    const SunoptNoSunshineTankParameters& p
) {
    if (current_energy < 0.0) {
        current_energy = 0.0;
    }

    if (
        current_energy >
        p.target_storage_energy
    ) {
        current_energy =
            p.target_storage_energy;
    }

    const double heat_loss =
        nosunshine_heat_loss(
            current_energy,
            p
        );

    double after_loss =
        current_energy -
        heat_loss;

    if (after_loss < 0.0) {
        after_loss = 0.0;
    }

    if (demand_energy < 0.0) {
        demand_energy = 0.0;
    }

    const double discharged =
        demand_energy <
        after_loss
            ? demand_energy
            : after_loss;

    double unmet =
        demand_energy -
        discharged;

    if (unmet < 0.0) {
        unmet = 0.0;
    }

    return unmet;
}


__device__ static inline double nosunshine_required_before_step(
    double required_energy_after_step,
    double demand_energy,
    const SunoptNoSunshineTankParameters& p
) {
    constexpr double ENERGY_EPSILON =
        1.0e-6;

    constexpr double BINARY_GRID =
        4294967296.0;

    /*
     * Preserve canonical physical-maximum behavior.
     */
    const double maximum_after =
        nosunshine_after_step_energy(
            p.target_storage_energy,
            demand_energy,
            p
        );

    const double maximum_unmet =
        nosunshine_unmet_demand(
            p.target_storage_energy,
            demand_energy,
            p
        );

    const bool maximum_is_sufficient =
        maximum_unmet <=
            ENERGY_EPSILON &&
        maximum_after +
            ENERGY_EPSILON >=
            required_energy_after_step;

    if (!maximum_is_sufficient) {
        return p.target_storage_energy;
    }


    /*
     * Minimal energy that must exist immediately AFTER passive loss.
     *
     * This is the closed-form equivalent of the two canonical
     * isSufficient checks used by the previous 32-iteration binary
     * reverse search.
     */
    double required_after_loss = 0.0;

    if (
        required_energy_after_step <=
        ENERGY_EPSILON
    ) {
        required_after_loss =
            demand_energy -
            ENERGY_EPSILON;

        if (required_after_loss < 0.0) {
            required_after_loss = 0.0;
        }

    } else {
        required_after_loss =
            demand_energy +
            required_energy_after_step -
            ENERGY_EPSILON;

        if (required_after_loss < 0.0) {
            required_after_loss = 0.0;
        }
    }


    double exact_required_energy =
        0.0;

    if (required_after_loss > 0.0) {
        if (p.daily_heat_loss >= 0.0) {
            const double fixed_loss =
                p.daily_heat_loss *
                p.time_step_hours /
                24.0;

            exact_required_energy =
                required_after_loss +
                fixed_loss;

        } else {
            const double alpha =
                p.storage_conductance_w_per_kelvin *
                p.time_step_hours /
                1000.0;

            const double linear_factor =
                alpha /
                p.storage_energy_per_kelvin;

            double zero_loss_threshold_energy =
                (
                    p.ambient_temperature -
                    p.storage_min_temperature
                ) *
                p.storage_energy_per_kelvin;

            if (
                zero_loss_threshold_energy <
                0.0
            ) {
                zero_loss_threshold_energy =
                    0.0;
            }

            if (
                required_after_loss <=
                zero_loss_threshold_energy
            ) {
                exact_required_energy =
                    required_after_loss;

            } else {
                const double denominator =
                    1.0 -
                    linear_factor;

                if (denominator <= 0.0) {
                    return p.target_storage_energy;
                }

                exact_required_energy =
                    (
                        required_after_loss +
                        alpha *
                            (
                                p.storage_min_temperature -
                                p.ambient_temperature
                            )
                    ) /
                    denominator;
            }
        }
    }

    if (exact_required_energy < 0.0) {
        exact_required_energy = 0.0;
    }

    if (
        exact_required_energy >
        p.target_storage_energy
    ) {
        exact_required_energy =
            p.target_storage_energy;
    }


    /*
     * Quantise to the exact upper-bound grid produced by 32 binary
     * iterations over [0, targetStorageEnergy].
     */
    const double quantum =
        p.target_storage_energy /
        BINARY_GRID;

    if (quantum <= 0.0) {
        return 0.0;
    }

    double quantised =
        ceil(
            exact_required_energy /
            quantum
        ) *
        quantum;

    if (quantised < quantum) {
        quantised = quantum;
    }

    if (
        quantised >
        p.target_storage_energy
    ) {
        quantised =
            p.target_storage_energy;
    }

    return quantised;
}


__global__ static void no_sunshine_year_matrix_kernel(
    const double* export_power,
    const double* demand_energy,
    const SunoptNoSunshineTankParameters* tank_parameters,
    size_t tank_count,
    size_t data_length,
    double* waterlines
) {
    const size_t job =
        static_cast<size_t>(
            blockIdx.x
        ) *
        blockDim.x +
        threadIdx.x;

    const size_t total_jobs =
        tank_count *
        data_length;

    if (job >= total_jobs) {
        return;
    }

    const size_t tank_index =
        job /
        data_length;

    const size_t current_index =
        job -
        tank_index *
        data_length;

    const SunoptNoSunshineTankParameters p =
        tank_parameters[
            tank_index
        ];

    double required_energy_after_step =
        0.0;

    double current_waterline =
        p.initial_waterline_power;


    for (
        size_t reverse_offset =
            p.horizon_steps;

        reverse_offset > 0;

        --reverse_offset
    ) {
        const size_t offset =
            reverse_offset -
            1;

        const size_t data_index =
            (
                current_index +
                offset
            ) %
            data_length;

        double bucket_demand_energy =
            0.0;

        if (
            offset >=
            p.protected_steps
        ) {
            bucket_demand_energy =
                demand_energy[
                    data_index
                ];
        }

        const double current_export_power =
            export_power[
                data_index
            ];

        const double available_hp_electrical_power =
            current_export_power <
                p.maximum_heat_pump_electrical_power
                ? current_export_power
                : p.maximum_heat_pump_electrical_power;

        const double available_pv_hp_thermal_energy =
            available_hp_electrical_power *
            p.cop *
            p.time_step_hours;

        const double allocated_hp_thermal_energy =
            required_energy_after_step <
                available_pv_hp_thermal_energy
                ? required_energy_after_step
                : available_pv_hp_thermal_energy;

        double allocated_hp_electrical_power =
            0.0;

        if (
            allocated_hp_thermal_energy >
            0.0
        ) {
            allocated_hp_electrical_power =
                allocated_hp_thermal_energy /
                (
                    p.cop *
                    p.time_step_hours
                );
        }

        const double inactive_waterline_power =
            p.initial_waterline_power >
                current_export_power
                ? p.initial_waterline_power
                : current_export_power;

        double waterline_power =
            inactive_waterline_power;

        if (
            allocated_hp_electrical_power >
            1.0e-6
        ) {
            waterline_power =
                current_export_power -
                allocated_hp_electrical_power;

            if (waterline_power < 0.0) {
                waterline_power = 0.0;
            }
        }

        double remaining_required_after_charge =
            required_energy_after_step -
            allocated_hp_thermal_energy;

        if (
            remaining_required_after_charge <
            0.0
        ) {
            remaining_required_after_charge =
                0.0;
        }

        required_energy_after_step =
            nosunshine_required_before_step(
                remaining_required_after_charge,
                bucket_demand_energy,
                p
            );

        if (offset == 0) {
            current_waterline =
                waterline_power;
        }
    }

    waterlines[job] =
        current_waterline;
}


/*
 * ================================================================
 * WARP-PARALLEL CAUSAL BATCH KERNEL
 * ================================================================
 *
 * One CUDA block = one tank/job.
 * One block contains exactly one warp (32 lanes).
 *
 * The previous kernel mapped one CUDA thread to one complete causal
 * job. With 28 tanks this produced only one partially occupied warp:
 *
 *   1 block x 256 threads
 *   -> only lanes 0..27 active
 *   -> all 28 tanks executed on one SM
 *
 * This kernel instead launches:
 *
 *   job_count blocks x 32 threads
 *
 * For the 28-tank sweep this is 28 independent warps which CUDA may
 * distribute across the RTX 5060 SMs.
 *
 * Search semantics:
 *
 * - normal Waterline and Waterline 0 are tested concurrently by
 *   lanes 0 and 1.
 *
 * - when causal correction is required, all 32 lanes test 32
 *   monotonically ordered cut candidates at once.
 *
 * - four base-32 refinement rounds provide exactly:
 *
 *       32^4 = 2^20
 *
 *   subdivisions, i.e. the same 20-bit cut-level resolution as the
 *   previous 20-step binary search.
 *
 * The result remains the lower/surviving cut boundary.
 *
 * Logical profiling counters intentionally preserve the canonical
 * algorithm:
 *
 *   1 or 2 initial survival checks
 *   + 20 binary-search iterations when correction is required.
 *
 * Physical GPU evaluations are larger (up to 2 + 4*32), but are
 * executed warp-parallel instead of sequentially.
 */

/*
 * ================================================================
 * WARP-PARALLEL ENERGY WATERLINE
 * ================================================================
 *
 * The canonical host implementation performed:
 *
 *   32 binary-search iterations
 *     x 192 horizon buckets
 *
 * for every tank and quarter hour.
 *
 * One block already represents one tank and contains one warp.
 * Therefore the 192 horizon buckets are distributed over 32 lanes:
 *
 *   lane 0 -> offsets 0, 32, 64, ...
 *   lane 1 -> offsets 1, 33, 65, ...
 *   ...
 *
 * Each binary-search iteration performs one warp reduction.
 *
 * The 32 binary iterations remain sequential because every next
 * interval depends on the preceding comparison. Only the independent
 * bucket sum is parallelised.
 */

__device__ static inline double warp_sum_double(
    double value
) {
    constexpr unsigned int MASK =
        0xffffffffu;

    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        value +=
            __shfl_down_sync(
                MASK,
                value,
                offset
            );
    }

    return __shfl_sync(
        MASK,
        value,
        0
    );
}


__device__ static inline double warp_max_double(
    double value
) {
    constexpr unsigned int MASK =
        0xffffffffu;

    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        const double other =
            __shfl_down_sync(
                MASK,
                value,
                offset
            );

        value =
            value > other
                ? value
                : other;
    }

    return __shfl_sync(
        MASK,
        value,
        0
    );
}


__device__ static inline double
energy_at_cut_level_warp(
    const double* export_power,
    double cut_level_power,
    const SunoptCausalParameters& p,
    unsigned int lane
) {
    double local_energy =
        0.0;


    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );

    const unsigned int data_length =
        static_cast<unsigned int>(
            p.data_length
        );

    const unsigned int current_index =
        static_cast<unsigned int>(
            p.current_index
        );


    if (
        lane >=
        horizon_steps
    ) {
        return warp_sum_double(
            local_energy
        );
    }


    /*
     * lane < horizon_steps <= data_length, therefore:
     *
     *   current_index + lane < 2 * data_length
     *
     * and at most one subtraction is required.
     */
    unsigned int index =
        current_index +
        lane;


    if (
        index >=
        data_length
    ) {
        index -=
            data_length;
    }


    for (
        unsigned int offset = lane;
        offset < horizon_steps;
        offset += 32u
    ) {

        const double export_value =
            export_power[
                index
            ];


        double available_power =
            export_value -
            cut_level_power;


        if (
            available_power <
            0.0
        ) {
            available_power =
                0.0;
        }


        const double hp_power =
            available_power <
                p.maximum_heat_pump_electrical_power
                ? available_power
                : p.maximum_heat_pump_electrical_power;


        local_energy +=
            hp_power *
            p.time_step_hours;


        /*
         * Only advance when another lane-strided point exists.
         *
         * If another point exists, data_length > 32, so a single
         * subtraction is sufficient after adding the stride.
         */
        if (
            offset +
            32u <
            horizon_steps
        ) {

            index +=
                32u;


            if (
                index >=
                data_length
            ) {
                index -=
                    data_length;
            }
        }
    }


    return warp_sum_double(
        local_energy
    );
}


__device__ static inline double
maximum_export_power_warp(
    const double* export_power,
    const SunoptCausalParameters& p,
    unsigned int lane
) {
    double local_max =
        0.0;


    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );

    const unsigned int data_length =
        static_cast<unsigned int>(
            p.data_length
        );

    const unsigned int current_index =
        static_cast<unsigned int>(
            p.current_index
        );


    if (
        lane >=
        horizon_steps
    ) {
        return warp_max_double(
            local_max
        );
    }


    unsigned int index =
        current_index +
        lane;


    if (
        index >=
        data_length
    ) {
        index -=
            data_length;
    }


    for (
        unsigned int offset = lane;
        offset < horizon_steps;
        offset += 32u
    ) {

        const double export_value =
            export_power[
                index
            ];


        local_max =
            local_max >
            export_value
                ? local_max
                : export_value;


        if (
            offset +
            32u <
            horizon_steps
        ) {

            index +=
                32u;


            if (
                index >=
                data_length
            ) {
                index -=
                    data_length;
            }
        }
    }


    return warp_max_double(
        local_max
    );
}


__device__ static inline unsigned int
warp_sum_u32(
    unsigned int value
) {
    constexpr unsigned int MASK =
        0xffffffffu;

    for (
        int offset = 16;
        offset > 0;
        offset >>= 1
    ) {
        value +=
            __shfl_down_sync(
                MASK,
                value,
                offset
            );
    }

    return __shfl_sync(
        MASK,
        value,
        0
    );
}


__device__ static inline double
calculate_energy_cut_level_warp(
    const double* export_power,
    const SunoptCausalParameters& p,
    unsigned int lane
) {
    const double maximum_export_power =
        maximum_export_power_warp(
            export_power,
            p,
            lane
        );

    const double maximum_schedulable_electrical_energy =
        energy_at_cut_level_warp(
            export_power,
            0.0,
            p,
            lane
        );

    double requested =
        p.requested_electrical_energy;

    if (requested < 0.0) {
        requested = 0.0;
    }

    const double target_electrical_energy =
        requested <
            maximum_schedulable_electrical_energy
            ? requested
            : maximum_schedulable_electrical_energy;

    double energy_cut_level =
        maximum_export_power;

    if (
        target_electrical_energy >
            0.0 &&
        maximum_export_power >
            0.0
    ) {
        if (
            target_electrical_energy >=
            maximum_schedulable_electrical_energy
        ) {
            energy_cut_level =
                0.0;

        } else {
            double lower_cut =
                0.0;

            double upper_cut =
                maximum_export_power;

            for (
                int iteration = 0;
                iteration < 32;
                ++iteration
            ) {
                const double middle_cut =
                    (
                        lower_cut +
                        upper_cut
                    ) /
                    2.0;

                const double energy_at_middle =
                    energy_at_cut_level_warp(
                        export_power,
                        middle_cut,
                        p,
                        lane
                    );

                if (
                    energy_at_middle >
                    target_electrical_energy
                ) {
                    lower_cut =
                        middle_cut;
                } else {
                    upper_cut =
                        middle_cut;
                }
            }

            energy_cut_level =
                (
                    lower_cut +
                    upper_cut
                ) /
                2.0;
        }
    }

    return energy_cut_level;
}


struct SunoptCausalDynamicBatch {
    SunoptCausalDynamic values[
        32
    ];
};


/*
 * Blackwell compiler benchmark switch.
 * Mode 0 is the production reference and changes normal builds by nothing.
 * The benchmark runner supplies the macro through NVCC_APPEND_FLAGS.
 */
#ifndef SUNOPT_LAUNCH_BOUNDS_MODE
#define SUNOPT_LAUNCH_BOUNDS_MODE 0
#endif

#if SUNOPT_LAUNCH_BOUNDS_MODE == 1
#define SUNOPT_CAUSAL_LAUNCH_BOUNDS __launch_bounds__(32)
#elif SUNOPT_LAUNCH_BOUNDS_MODE == 2
#define SUNOPT_CAUSAL_LAUNCH_BOUNDS __launch_bounds__(32, 1)
#else
#define SUNOPT_CAUSAL_LAUNCH_BOUNDS
#endif


__global__ static void SUNOPT_CAUSAL_LAUNCH_BOUNDS causal_batch_kernel(
    const double* export_power,
    const double* demand_energy,
    const SunoptCausalParameters* parameters,
    SunoptCausalDynamicBatch dynamic_batch,
    int use_dynamic_parameters,
    size_t dynamic_current_index,
    size_t dynamic_horizon_steps,
    SunoptCausalResult* results,
    double* cut_level_results,
    size_t job_count
) {
    const size_t job =
        static_cast<size_t>(
            blockIdx.x
        );

    const unsigned int lane =
        threadIdx.x;

    if (
        job >= job_count ||
        lane >= 32
    ) {
        return;
    }

    __shared__
    SunoptCausalParameters p;

    __shared__
    SunoptHeatLossModel heat_loss_model_shared;

    __shared__
    int normal_plan_survives_shared;

    __shared__
    int maximum_pv_plan_survives_shared;

    __shared__
    int search_required_shared;

    __shared__
    double lower_cut_shared;

    __shared__
    double upper_cut_shared;

    __shared__
    double export_window[
        SUNOPT_SHARED_CAUSAL_WINDOW_STEPS
    ];

    __shared__
    double demand_window[
        SUNOPT_SHARED_CAUSAL_WINDOW_STEPS
    ];


    if (lane == 0) {
        p =
            parameters[job];


        if (
            use_dynamic_parameters !=
            0
        ) {
            const SunoptCausalDynamic dynamic =
                dynamic_batch
                  .values[
                      job
                  ];


            p.current_index =
                dynamic_current_index;

            p.horizon_steps =
                dynamic_horizon_steps;

            p.current_storage_energy =
                dynamic
                  .current_storage_energy;

            p.minimum_reserve_energy =
                dynamic
                  .minimum_reserve_energy;

            p.requested_electrical_energy =
                dynamic
                  .requested_electrical_energy;

            /*
             * The fused energy search replaces this before causal
             * survival testing.
             */
            p.energy_cut_level =
                0.0;
        }


        heat_loss_model_shared =
            prepare_heat_loss_model(
                p
            );

        normal_plan_survives_shared =
            0;

        maximum_pv_plan_survives_shared =
            0;

        search_required_shared =
            0;

        lower_cut_shared =
            0.0;

        upper_cut_shared =
            p.energy_cut_level;
    }

    __syncthreads();


    const bool use_shared_window =
        p.horizon_steps <=
        static_cast<size_t>(
            SUNOPT_SHARED_CAUSAL_WINDOW_STEPS
        );


    if (
        use_shared_window
    ) {
        const unsigned int horizon_steps =
            static_cast<unsigned int>(
                p.horizon_steps
            );

        for (
            unsigned int offset = lane;
            offset < horizon_steps;
            offset += 32u
        ) {
            /*
             * This ring-index work happens exactly once per block,
             * not once per survival candidate.
             */
            size_t annual_index =
                p.current_index +
                static_cast<size_t>(
                    offset
                );

            if (
                annual_index >=
                p.data_length
            ) {
                annual_index -=
                    p.data_length;
            }

            export_window[
                offset
            ] =
                export_power[
                    annual_index
                ];

            const double demand =
                demand_energy[
                    annual_index
                ];


            demand_window[
                offset
            ] =
                heat_loss_model_shared
                  .affine_survival_fast_path
                    ? demand +
                      heat_loss_model_shared
                        .heat_loss_intercept
                    : demand;
        }
    }

    __syncthreads();


    const SunoptHeatLossModel heat_loss_model =
        heat_loss_model_shared;


    if (
        p.requested_electrical_energy >=
        0.0
    ) {
        const double energy_cut_level =
            use_shared_window
                ? calculate_energy_cut_level_window_warp(
                      export_window,
                      p,
                      lane
                  )
                : calculate_energy_cut_level_warp(
                      export_power,
                      p,
                      lane
                  );

        if (lane == 0) {
            p.energy_cut_level =
                energy_cut_level;
        }

        __syncthreads();
    }


    bool initial_survives =
        false;

    if (
        lane == 0 ||
        lane == 1
    ) {
        unsigned int ignored_checks =
            0;

        const double cut =
            lane == 0
                ? p.energy_cut_level
                : 0.0;

        initial_survives =
            use_shared_window
                ? survives_window_at_cut_level(
                      export_window,
                      demand_window,
                      cut,
                      p,
                      heat_loss_model,
                      &ignored_checks
                  )
                : survives_at_cut_level(
                      export_power,
                      demand_energy,
                      cut,
                      p,
                      heat_loss_model,
                      &ignored_checks
                  );
    }


    if (lane == 0) {
        normal_plan_survives_shared =
            initial_survives
                ? 1
                : 0;
    }

    if (lane == 1) {
        maximum_pv_plan_survives_shared =
            initial_survives
                ? 1
                : 0;
    }

    __syncthreads();


    if (lane == 0) {
        if (
            normal_plan_survives_shared
        ) {
            maximum_pv_plan_survives_shared =
                1;

            search_required_shared =
                0;

        } else if (
            !maximum_pv_plan_survives_shared
        ) {
            search_required_shared =
                0;

            lower_cut_shared =
                0.0;

            upper_cut_shared =
                0.0;

        } else {
            search_required_shared =
                1;

            lower_cut_shared =
                0.0;

            upper_cut_shared =
                p.energy_cut_level;
        }
    }

    __syncthreads();


    if (
        search_required_shared
    ) {
        for (
            int round = 0;
            round < 4;
            ++round
        ) {
            const double round_lower =
                lower_cut_shared;

            const double round_upper =
                upper_cut_shared;

            const double width =
                (
                    round_upper -
                    round_lower
                ) /
                32.0;

            const double candidate =
                round_lower +
                width *
                static_cast<double>(
                    lane + 1
                );

            unsigned int ignored_checks =
                0;

            const bool survives =
                use_shared_window
                    ? survives_window_at_cut_level(
                          export_window,
                          demand_window,
                          candidate,
                          p,
                          heat_loss_model,
                          &ignored_checks
                      )
                    : survives_at_cut_level(
                          export_power,
                          demand_energy,
                          candidate,
                          p,
                          heat_loss_model,
                          &ignored_checks
                      );

            const unsigned int surviving_mask =
                __ballot_sync(
                    0xffffffffu,
                    survives
                );


            if (lane == 0) {
                if (
                    surviving_mask ==
                    0u
                ) {
                    upper_cut_shared =
                        round_lower +
                        width;

                } else {
                    const int highest_surviving_lane =
                        31 -
                        __clz(
                            surviving_mask
                        );

                    const double new_lower =
                        round_lower +
                        width *
                        static_cast<double>(
                            highest_surviving_lane +
                            1
                        );

                    lower_cut_shared =
                        new_lower;

                    if (
                        highest_surviving_lane >=
                        31
                    ) {
                        upper_cut_shared =
                            new_lower;
                    } else {
                        upper_cut_shared =
                            round_lower +
                            width *
                            static_cast<double>(
                                highest_surviving_lane +
                                2
                            );
                    }
                }
            }

            __syncthreads();
        }
    }


    if (lane == 0) {
        double cut_level_power =
            0.0;


        if (
            normal_plan_survives_shared
        ) {
            cut_level_power =
                p.energy_cut_level;

        } else if (
            !maximum_pv_plan_survives_shared
        ) {
            cut_level_power =
                0.0;

        } else {
            cut_level_power =
                lower_cut_shared;
        }


        if (
            cut_level_results !=
            nullptr
        ) {
            cut_level_results[
                job
            ] =
                cut_level_power;
        }


        if (
            results !=
            nullptr
        ) {
            SunoptCausalResult result{};

            result.cut_level_power =
                cut_level_power;

            result.normal_plan_survives =
                normal_plan_survives_shared;

            result.maximum_pv_plan_survives =
                maximum_pv_plan_survives_shared;


            if (
                normal_plan_survives_shared
            ) {
                result.survival_checks =
                    1;

                result.binary_search_iterations =
                    0;

            } else if (
                !maximum_pv_plan_survives_shared
            ) {
                result.survival_checks =
                    2;

                result.binary_search_iterations =
                    0;

            } else {
                result.survival_checks =
                    22;

                result.binary_search_iterations =
                    20;
            }


            results[
                job
            ] =
                result;
        }
    }
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

    context->causal_static_parameter_buffer =
        nullptr;

    context->causal_static_parameter_capacity =
        0;

    context->causal_static_job_count =
        0;

    context->causal_mapped_cut_host =
        nullptr;

    context->causal_mapped_cut_device =
        nullptr;

    context->causal_mapped_cut_capacity =
        0;

    context->causal_length = 0;

    context->profile_start = nullptr;
    context->profile_after_h2d = nullptr;
    context->profile_after_kernel = nullptr;
    context->profile_after_d2h = nullptr;

    reset_profile(&context->profile);

    err = cudaStreamCreateWithFlags(
        &context->stream,
        cudaStreamNonBlocking
    );

    if (err != cudaSuccess) {
        delete context;
        return static_cast<int>(err);
    }

    err = create_profile_events(context);

    if (err != cudaSuccess) {
        cudaStreamDestroy(context->stream);
        context->stream = nullptr;
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

    if (context->workspace_buffer != nullptr) {
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

    if (context->result_buffer != nullptr) {
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

    if (
        context->causal_static_parameter_buffer !=
        nullptr
    ) {
        err = cudaFree(
            context->causal_static_parameter_buffer
        );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }


    if (
        context->causal_mapped_cut_host !=
        nullptr
    ) {
        err =
            cudaFreeHost(
                context->causal_mapped_cut_host
            );

        if (
            first_error == cudaSuccess &&
            err != cudaSuccess
        ) {
            first_error = err;
        }
    }


    destroy_profile_events(context);

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


extern "C" int sunopt_cuda_precompute_no_sunshine_waterlines(
    SunoptCudaContext* context,
    const SunoptNoSunshineTankParameters* host_parameters,
    size_t tank_count,
    double* host_waterlines
) {
    if (
        context == nullptr ||
        host_parameters == nullptr ||
        host_waterlines == nullptr ||
        context->causal_length == 0 ||
        tank_count == 0
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    for (
        size_t tank = 0;
        tank < tank_count;
        ++tank
    ) {
        const auto& p =
            host_parameters[
                tank
            ];

        if (
            p.horizon_steps == 0 ||
            p.horizon_steps >
                context->causal_length ||
            p.protected_steps >
                p.horizon_steps ||
            p.time_step_hours <= 0.0 ||
            p.maximum_heat_pump_electrical_power <
                0.0 ||
            p.cop <= 0.0 ||
            p.target_storage_energy <= 0.0 ||
            p.storage_energy_per_kelvin <= 0.0
        ) {
            return static_cast<int>(
                cudaErrorInvalidValue
            );
        }
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t parameter_bytes =
        tank_count *
        sizeof(
            SunoptNoSunshineTankParameters
        );

    const size_t total_jobs =
        tank_count *
        context->causal_length;

    if (
        context->causal_length != 0 &&
        total_jobs /
            context->causal_length !=
            tank_count
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    const size_t result_bytes =
        total_jobs *
        sizeof(double);

    err = reserve_buffer(
        &context->workspace_buffer,
        &context->workspace_capacity,
        parameter_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->result_buffer,
        &context->result_capacity,
        result_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        context->workspace_buffer,
        host_parameters,
        parameter_bytes,
        cudaMemcpyHostToDevice,
        context->stream
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
        reinterpret_cast<
            const double*
        >(
            base
        );

    const double* demand_energy =
        reinterpret_cast<
            const double*
        >(
            base +
            one_array_bytes
        );

    const auto* device_parameters =
        static_cast<
            const SunoptNoSunshineTankParameters*
        >(
            context->workspace_buffer
        );

    double* device_waterlines =
        static_cast<double*>(
            context->result_buffer
        );

    constexpr unsigned int THREADS =
        256;

    const unsigned int blocks =
        static_cast<unsigned int>(
            (
                total_jobs +
                THREADS -
                1
            ) /
            THREADS
        );

    no_sunshine_year_matrix_kernel<<<
        blocks,
        THREADS,
        0,
        context->stream
    >>>(
        export_power,
        demand_energy,
        device_parameters,
        tank_count,
        context->causal_length,
        device_waterlines
    );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        host_waterlines,
        device_waterlines,
        result_bytes,
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


extern "C" int sunopt_cuda_run_causal_batch(
    SunoptCudaContext* context,
    const SunoptCausalParameters* host_parameters,
    size_t job_count,
    SunoptCausalResult* host_results
) {
    using Clock =
        std::chrono::steady_clock;

    const auto wall_start =
        Clock::now();

    /*
     * Detailed CUDA events are useful for benchmarking, but recording
     * four events plus querying four elapsed intervals on every one
     * of the 35,040 annual lockstep calls is itself production work.
     *
     * Sample every 128th call instead. Exact causalCalls, causalJobs,
     * hostWaitMs and wallMs remain fully accumulated. The GPU phase
     * counters are extrapolated from the samples.
     */
    constexpr uint64_t PROFILE_SAMPLE_INTERVAL =
        128u;

    const uint64_t next_causal_call =
        context != nullptr
            ? context->profile.causal_calls +
              1u
            : 0u;

    const bool detailed_profile_sample =
        (
            next_causal_call !=
                0u &&
            next_causal_call %
                PROFILE_SAMPLE_INTERVAL ==
                0u
        );

    if (
        context == nullptr ||
        host_parameters == nullptr ||
        host_results == nullptr ||
        context->causal_length == 0 ||
        job_count == 0
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    for (
        size_t i = 0;
        i < job_count;
        ++i
    ) {
        if (
            host_parameters[i].data_length !=
                context->causal_length ||
            host_parameters[i].current_index >=
                host_parameters[i].data_length ||
            host_parameters[i].horizon_steps == 0 ||
            host_parameters[i].horizon_steps >
                host_parameters[i].data_length ||
            host_parameters[i].data_length >
                static_cast<size_t>(
                    UINT32_MAX -
                    32u
                )
        ) {
            return static_cast<int>(
                cudaErrorInvalidValue
            );
        }
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t parameter_bytes =
        job_count *
        sizeof(SunoptCausalParameters);

    const size_t result_bytes =
        job_count *
        sizeof(SunoptCausalResult);

    err = reserve_buffer(
        &context->workspace_buffer,
        &context->workspace_capacity,
        parameter_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = reserve_buffer(
        &context->result_buffer,
        &context->result_capacity,
        result_bytes
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    if (
        detailed_profile_sample
    ) {
        err = cudaEventRecord(
            context->profile_start,
            context->stream
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
    }

    err = cudaMemcpyAsync(
        context->workspace_buffer,
        host_parameters,
        parameter_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    if (
        detailed_profile_sample
    ) {
        err = cudaEventRecord(
            context->profile_after_h2d,
            context->stream
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
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

    const SunoptCausalParameters* device_parameters =
        static_cast<const SunoptCausalParameters*>(
            context->workspace_buffer
        );

    SunoptCausalResult* device_results =
        static_cast<SunoptCausalResult*>(
            context->result_buffer
        );

    /*
     * One complete warp per causal job/tank.
     *
     * 28 tanks:
     *
     *   28 blocks x 32 threads
     *   = 28 independent warps
     *
     * instead of one partially occupied 256-thread block.
     */
    constexpr unsigned int THREADS =
        32;

    const unsigned int blocks =
        static_cast<unsigned int>(
            job_count
        );

    const SunoptCausalDynamicBatch empty_dynamic_batch{};


    causal_batch_kernel<<<
        blocks,
        THREADS,
        0,
        context->stream
    >>>(
        export_power,
        demand_energy,
        device_parameters,
        empty_dynamic_batch,
        0,
        0,
        0,
        device_results,
        nullptr,
        job_count
    );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    if (
        detailed_profile_sample
    ) {
        err = cudaEventRecord(
            context->profile_after_kernel,
            context->stream
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
    }

    err = cudaMemcpyAsync(
        host_results,
        device_results,
        result_bytes,
        cudaMemcpyDeviceToHost,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    if (
        detailed_profile_sample
    ) {
        err = cudaEventRecord(
            context->profile_after_d2h,
            context->stream
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
    }

    const auto wait_start =
        Clock::now();

    if (
        detailed_profile_sample
    ) {
        err = cudaEventSynchronize(
            context->profile_after_d2h
        );

    } else {
        err = cudaStreamSynchronize(
            context->stream
        );
    }

    const auto wait_end =
        Clock::now();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    float parameter_h2d_ms =
        0.0f;

    float kernel_ms =
        0.0f;

    float result_d2h_ms =
        0.0f;

    float gpu_total_ms =
        0.0f;


    if (
        detailed_profile_sample
    ) {
        err = cudaEventElapsedTime(
            &parameter_h2d_ms,
            context->profile_start,
            context->profile_after_h2d
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }


        err = cudaEventElapsedTime(
            &kernel_ms,
            context->profile_after_h2d,
            context->profile_after_kernel
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }


        err = cudaEventElapsedTime(
            &result_d2h_ms,
            context->profile_after_kernel,
            context->profile_after_d2h
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }


        err = cudaEventElapsedTime(
            &gpu_total_ms,
            context->profile_start,
            context->profile_after_d2h
        );

        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
    }


    const auto wall_end =
        Clock::now();

    const double host_wait_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            wait_end -
            wait_start
        ).count();

    const double wall_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            wall_end -
            wall_start
        ).count();

    context->profile.causal_calls +=
        1;

    context->profile.causal_jobs +=
        static_cast<uint64_t>(
            job_count
        );


    if (
        detailed_profile_sample
    ) {
        const double sample_scale =
            static_cast<double>(
                PROFILE_SAMPLE_INTERVAL
            );


        context->profile.parameter_h2d_ms +=
            static_cast<double>(
                parameter_h2d_ms
            ) *
            sample_scale;


        context->profile.kernel_ms +=
            static_cast<double>(
                kernel_ms
            ) *
            sample_scale;


        context->profile.result_d2h_ms +=
            static_cast<double>(
                result_d2h_ms
            ) *
            sample_scale;


        context->profile.gpu_total_ms +=
            static_cast<double>(
                gpu_total_ms
            ) *
            sample_scale;
    }


    /*
     * These two values are cheap host-side measurements and stay
     * exact rather than sampled.
     */
    context->profile.host_wait_ms +=
        host_wait_ms;

    context->profile.wall_ms +=
        wall_ms;



    return static_cast<int>(
        cudaSuccess
    );
}


extern "C" int sunopt_cuda_configure_causal_static_parameters(
    SunoptCudaContext* context,
    const SunoptCausalParameters* host_parameters,
    size_t job_count
) {
    if (
        context == nullptr ||
        host_parameters == nullptr ||
        context->causal_length == 0 ||
        job_count == 0
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }


    for (
        size_t i = 0;
        i < job_count;
        ++i
    ) {
        if (
            host_parameters[i].data_length !=
                context->causal_length ||
            host_parameters[i].horizon_steps == 0 ||
            host_parameters[i].horizon_steps >
                host_parameters[i].data_length ||
            host_parameters[i].data_length >
                static_cast<size_t>(
                    UINT32_MAX -
                    32u
                )
        ) {
            return static_cast<int>(
                cudaErrorInvalidValue
            );
        }
    }


    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }


    const size_t parameter_bytes =
        job_count *
        sizeof(
            SunoptCausalParameters
        );


    err = reserve_buffer(
        &context->causal_static_parameter_buffer,
        &context->causal_static_parameter_capacity,
        parameter_bytes
    );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }


    err = cudaMemcpyAsync(
        context->causal_static_parameter_buffer,
        host_parameters,
        parameter_bytes,
        cudaMemcpyHostToDevice,
        context->stream
    );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }


    err =
        cudaStreamSynchronize(
            context->stream
        );


    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }


    context->causal_static_job_count =
        job_count;


    return static_cast<int>(
        cudaSuccess
    );
}


extern "C" int sunopt_cuda_run_causal_dynamic_batch(
    SunoptCudaContext* context,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
) {
    using Clock =
        std::chrono::steady_clock;

    constexpr uint64_t PROFILE_SAMPLE_INTERVAL =
        128u;

    const uint64_t next_causal_call =
        context != nullptr
            ? context->profile.causal_calls +
              1u
            : 0u;

    const bool detailed_profile_sample =
        (
            next_causal_call !=
                0u &&
            next_causal_call %
                PROFILE_SAMPLE_INTERVAL ==
                0u
        );

    Clock::time_point wall_start{};
    Clock::time_point wait_start{};
    Clock::time_point wait_end{};

    if (
        detailed_profile_sample
    ) {
        wall_start =
            Clock::now();
    }

    if (
        context == nullptr ||
        host_dynamic_parameters == nullptr ||
        host_cut_level_results == nullptr ||
        context->causal_length == 0 ||
        context->causal_static_parameter_buffer ==
            nullptr ||
        context->causal_static_job_count !=
            job_count ||
        job_count == 0 ||
        job_count > 32 ||
        current_index >=
            context->causal_length ||
        horizon_steps == 0 ||
        horizon_steps >
            context->causal_length
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    SunoptCausalDynamicBatch dynamic_batch{};

    for (
        size_t i = 0;
        i < job_count;
        ++i
    ) {
        dynamic_batch.values[i] =
            host_dynamic_parameters[i];
    }

    const size_t result_bytes =
        job_count *
        sizeof(double);

    err = reserve_mapped_cut_buffer(
        context,
        result_bytes
    );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    if (
        detailed_profile_sample
    ) {
        err = cudaEventRecord(
            context->profile_start,
            context->stream
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        err = cudaEventRecord(
            context->profile_after_h2d,
            context->stream
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }
    }

    const size_t one_array_bytes =
        context->causal_length *
        sizeof(double);

    unsigned char* base =
        static_cast<unsigned char*>(
            context->input_buffer
        );

    const double* export_power =
        reinterpret_cast<
            const double*
        >(
            base
        );

    const double* demand_energy =
        reinterpret_cast<
            const double*
        >(
            base +
            one_array_bytes
        );

    const auto* device_parameters =
        static_cast<
            const SunoptCausalParameters*
        >(
            context->causal_static_parameter_buffer
        );

    double* device_cut_level_results =
        context->causal_mapped_cut_device;

    constexpr unsigned int THREADS =
        32;

    const unsigned int blocks =
        static_cast<unsigned int>(
            job_count
        );

    causal_batch_kernel<<<
        blocks,
        THREADS,
        0,
        context->stream
    >>>(
        export_power,
        demand_energy,
        device_parameters,
        dynamic_batch,
        1,
        current_index,
        horizon_steps,
        nullptr,
        device_cut_level_results,
        job_count
    );

    err =
        cudaGetLastError();

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    if (
        detailed_profile_sample
    ) {
        err = cudaEventRecord(
            context->profile_after_kernel,
            context->stream
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        err = cudaEventRecord(
            context->profile_after_d2h,
            context->stream
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        wait_start =
            Clock::now();

        err = cudaEventSynchronize(
            context->profile_after_d2h
        );

        wait_end =
            Clock::now();

    } else {
        err = cudaStreamSynchronize(
            context->stream
        );
    }

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    std::memcpy(
        host_cut_level_results,
        context->causal_mapped_cut_host,
        result_bytes
    );

    context->profile.causal_calls +=
        1;

    context->profile.causal_jobs +=
        static_cast<uint64_t>(
            job_count
        );

    if (
        detailed_profile_sample
    ) {
        float parameter_h2d_ms =
            0.0f;

        float kernel_ms =
            0.0f;

        float result_d2h_ms =
            0.0f;

        float gpu_total_ms =
            0.0f;

        err = cudaEventElapsedTime(
            &parameter_h2d_ms,
            context->profile_start,
            context->profile_after_h2d
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        err = cudaEventElapsedTime(
            &kernel_ms,
            context->profile_after_h2d,
            context->profile_after_kernel
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        err = cudaEventElapsedTime(
            &result_d2h_ms,
            context->profile_after_kernel,
            context->profile_after_d2h
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        err = cudaEventElapsedTime(
            &gpu_total_ms,
            context->profile_start,
            context->profile_after_d2h
        );

        if (
            err !=
            cudaSuccess
        ) {
            return static_cast<int>(
                err
            );
        }

        const auto wall_end =
            Clock::now();

        const double host_wait_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                wait_end -
                wait_start
            ).count();

        const double wall_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                wall_end -
                wall_start
            ).count();

        const double sample_scale =
            static_cast<double>(
                PROFILE_SAMPLE_INTERVAL
            );

        context->profile.parameter_h2d_ms +=
            static_cast<double>(
                parameter_h2d_ms
            ) *
            sample_scale;

        context->profile.kernel_ms +=
            static_cast<double>(
                kernel_ms
            ) *
            sample_scale;

        context->profile.result_d2h_ms +=
            static_cast<double>(
                result_d2h_ms
            ) *
            sample_scale;

        context->profile.gpu_total_ms +=
            static_cast<double>(
                gpu_total_ms
            ) *
            sample_scale;

        context->profile.host_wait_ms +=
            host_wait_ms *
            sample_scale;

        context->profile.wall_ms +=
            wall_ms *
            sample_scale;
    }

    return static_cast<int>(
        cudaSuccess
    );
}


extern "C" int sunopt_cuda_run_causal(
    SunoptCudaContext* context,
    const SunoptCausalParameters* parameters,
    SunoptCausalResult* host_result
) {
    return sunopt_cuda_run_causal_batch(
        context,
        parameters,
        1,
        host_result
    );
}

// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_BEGIN
extern "C" int sunopt_cuda_run_causal_graph_replay(
    SunoptCudaContext* context,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t step_count,
    size_t tank_count,
    uint32_t repetitions,
    double* host_results,
    double* host_wall_ms,
    double* host_gpu_ms,
    double* out_graph_build_ms,
    double* out_result_d2h_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        context == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_parameters == nullptr ||
        host_results == nullptr ||
        host_wall_ms == nullptr ||
        host_gpu_ms == nullptr ||
        out_graph_build_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        step_count == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        repetitions == 0 ||
        context->causal_length == 0 ||
        context->causal_static_parameter_buffer == nullptr ||
        context->causal_static_job_count != tank_count
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    if (
        step_count >
        SIZE_MAX /
        tank_count
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    for (
        size_t step = 0;
        step < step_count;
        ++step
    ) {
        if (
            static_cast<size_t>(
                host_current_indices[step]
            ) >= context->causal_length ||
            host_horizon_steps[step] == 0 ||
            static_cast<size_t>(
                host_horizon_steps[step]
            ) > context->causal_length
        ) {
            return static_cast<int>(
                cudaErrorInvalidValue
            );
        }
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    const size_t result_count =
        step_count *
        tank_count;

    const size_t result_bytes =
        result_count *
        sizeof(double);

    double* device_results =
        nullptr;

    cudaGraph_t graph =
        nullptr;

    cudaGraphExec_t graph_exec =
        nullptr;

    cudaEvent_t replay_start =
        nullptr;

    cudaEvent_t replay_end =
        nullptr;

    Clock::time_point graph_build_start{};
    Clock::time_point graph_build_end{};
    Clock::time_point d2h_start{};
    Clock::time_point d2h_end{};

    *out_graph_build_ms =
        0.0;

    *out_result_d2h_ms =
        0.0;

    err =
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_results
            ),
            result_bytes
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
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
            base +
            one_array_bytes
        );

    const auto* device_parameters =
        static_cast<
            const SunoptCausalParameters*
        >(
            context
              ->causal_static_parameter_buffer
        );

    constexpr unsigned int THREADS =
        32;

    const unsigned int blocks =
        static_cast<unsigned int>(
            tank_count
        );

    graph_build_start =
        Clock::now();

    err =
        cudaStreamBeginCapture(
            context->stream,
            cudaStreamCaptureModeThreadLocal
        );

    if (
        err !=
        cudaSuccess
    ) {
        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    for (
        size_t step = 0;
        step < step_count;
        ++step
    ) {
        SunoptCausalDynamicBatch dynamic_batch{};

        const size_t dynamic_offset =
            step *
            tank_count;

        for (
            size_t tank = 0;
            tank < tank_count;
            ++tank
        ) {
            dynamic_batch.values[tank] =
                host_dynamic_parameters[
                    dynamic_offset +
                    tank
                ];
        }

        causal_batch_kernel<<<
            blocks,
            THREADS,
            0,
            context->stream
        >>>(
            export_power,
            demand_energy,
            device_parameters,
            dynamic_batch,
            1,
            static_cast<size_t>(
                host_current_indices[step]
            ),
            static_cast<size_t>(
                host_horizon_steps[step]
            ),
            nullptr,
            device_results +
                dynamic_offset,
            tank_count
        );

        err =
            cudaGetLastError();

        if (
            err !=
            cudaSuccess
        ) {
            break;
        }
    }

    {
        cudaError_t capture_end_error =
            cudaStreamEndCapture(
                context->stream,
                &graph
            );

        if (
            err ==
            cudaSuccess
        ) {
            err =
                capture_end_error;
        }
    }

    if (
        err !=
        cudaSuccess
    ) {
        if (
            graph !=
            nullptr
        ) {
            cudaGraphDestroy(
                graph
            );
        }

        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    err =
        cudaGraphInstantiate(
            &graph_exec,
            graph,
            nullptr,
            nullptr,
            0
        );

    {
        cudaError_t graph_destroy_error =
            cudaGraphDestroy(
                graph
            );

        graph =
            nullptr;

        if (
            err ==
            cudaSuccess &&
            graph_destroy_error !=
            cudaSuccess
        ) {
            err =
                graph_destroy_error;
        }
    }

    if (
        err !=
        cudaSuccess
    ) {
        if (
            graph_exec !=
            nullptr
        ) {
            cudaGraphExecDestroy(
                graph_exec
            );
        }

        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    graph_build_end =
        Clock::now();

    *out_graph_build_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            graph_build_end -
            graph_build_start
        ).count();

    err =
        cudaEventCreate(
            &replay_start
        );

    if (
        err ==
        cudaSuccess
    ) {
        err =
            cudaEventCreate(
                &replay_end
            );
    }

    if (
        err !=
        cudaSuccess
    ) {
        if (
            replay_start !=
            nullptr
        ) {
            cudaEventDestroy(
                replay_start
            );
        }

        if (
            replay_end !=
            nullptr
        ) {
            cudaEventDestroy(
                replay_end
            );
        }

        cudaGraphExecDestroy(
            graph_exec
        );

        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    /*
     * Untimed warm-up: graph execution objects and code paths are
     * resident before the measured repetitions begin.
     */
    err =
        cudaGraphLaunch(
            graph_exec,
            context->stream
        );

    if (
        err ==
        cudaSuccess
    ) {
        err =
            cudaStreamSynchronize(
                context->stream
            );
    }

    if (
        err ==
        cudaSuccess
    ) {
        for (
            uint32_t repetition = 0;
            repetition < repetitions;
            ++repetition
        ) {
            const auto wall_start =
                Clock::now();

            err =
                cudaEventRecord(
                    replay_start,
                    context->stream
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            err =
                cudaGraphLaunch(
                    graph_exec,
                    context->stream
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            err =
                cudaEventRecord(
                    replay_end,
                    context->stream
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            err =
                cudaEventSynchronize(
                    replay_end
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            const auto wall_end =
                Clock::now();

            float gpu_ms =
                0.0f;

            err =
                cudaEventElapsedTime(
                    &gpu_ms,
                    replay_start,
                    replay_end
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            host_gpu_ms[
                repetition
            ] =
                static_cast<double>(
                    gpu_ms
                );

            host_wall_ms[
                repetition
            ] =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    wall_end -
                    wall_start
                ).count();
        }
    }

    if (
        err ==
        cudaSuccess
    ) {
        d2h_start =
            Clock::now();

        err =
            cudaMemcpyAsync(
                host_results,
                device_results,
                result_bytes,
                cudaMemcpyDeviceToHost,
                context->stream
            );

        if (
            err ==
            cudaSuccess
        ) {
            err =
                cudaStreamSynchronize(
                    context->stream
                );
        }

        d2h_end =
            Clock::now();

        if (
            err ==
            cudaSuccess
        ) {
            *out_result_d2h_ms =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    d2h_end -
                    d2h_start
                ).count();
        }
    }

    cudaEventDestroy(
        replay_start
    );

    cudaEventDestroy(
        replay_end
    );

    cudaGraphExecDestroy(
        graph_exec
    );

    {
        cudaError_t free_error =
            cudaFree(
                device_results
            );

        if (
            err ==
            cudaSuccess &&
            free_error !=
            cudaSuccess
        ) {
            err =
                free_error;
        }
    }

    return static_cast<int>(
        err
    );
}
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_END


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


extern "C" int sunopt_cuda_get_profile(
    const SunoptCudaContext* context,
    SunoptCudaProfile* out_profile
) {
    if (
        context == nullptr ||
        out_profile == nullptr
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    *out_profile =
        context->profile;

    return static_cast<int>(
        cudaSuccess
    );
}


extern "C" int sunopt_cuda_reset_profile(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    reset_profile(&context->profile);

    return static_cast<int>(
        cudaSuccess
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
