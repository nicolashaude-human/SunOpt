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


// SUNOPT_BLACKWELL_B14C_WARP32_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14C_THREADS = 32;
constexpr unsigned int SUNOPT_B14C_FINAL_SEGMENTS = 32;
constexpr unsigned int SUNOPT_B14C_SCRATCH_SEGMENTS = 96;
constexpr unsigned int SUNOPT_B14C_BOUNDARIES = 128;
constexpr double SUNOPT_B14C_EPSILON = 1.0e-6;

struct SunoptB14CSegment {
    double lo;
    double hi;
    double m;
    double b;
};

__device__ __forceinline__ double sunopt_b14c_clamp(
    double value,
    double lower,
    double upper
) {
    return fmin(fmax(value, lower), upper);
}

__device__ bool sunopt_b14c_scalar_survives(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    double cut,
    double current_storage_energy,
    double reserve,
    double target,
    double max_hp,
    double hp_factor,
    double retention,
    double loss_intercept
) {
    double stored = sunopt_b14c_clamp(current_storage_energy, 0.0, target);

    for (size_t offset = 0; offset < horizon; ++offset) {
        const size_t index = (current_index + offset) % data_length;
        const double raw = export_power[index] - cut;
        const double hp = fmin(max_hp, fmax(0.0, raw));
        const double charge = hp * hp_factor;

        if (offset == 0) {
            stored = fmin(target, stored + charge);
            continue;
        }

        const double requested_loss =
            fmax((1.0 - retention) * stored + loss_intercept, 0.0);
        const double heat_loss = fmin(requested_loss, stored);
        const double after_loss = fmax(stored - heat_loss, 0.0);
        const double demand = fmax(demand_energy[index], 0.0);
        const double discharged = fmin(demand, after_loss);
        const double before_charge = after_loss - discharged;
        const double unmet = fmax(demand - discharged, 0.0);

        if (unmet > SUNOPT_B14C_EPSILON) return false;
        if (reserve - before_charge > SUNOPT_B14C_EPSILON) return false;

        stored = fmin(target, before_charge + charge);
    }

    return true;
}

__device__ int sunopt_b14c_merge(
    SunoptB14CSegment* segments,
    int count
) {
    int write = 0;

    for (int read = 0; read < count; ++read) {
        const SunoptB14CSegment segment = segments[read];
        if (segment.hi <= segment.lo + 1.0e-13) continue;

        if (write > 0) {
            SunoptB14CSegment& last = segments[write - 1];
            const double scale =
                1.0 + fmax(fmax(fabs(last.m), fabs(segment.m)),
                           fmax(fabs(last.b), fabs(segment.b)));

            if (fabs(last.hi - segment.lo) <= 1.0e-11 &&
                fabs(last.m - segment.m) <= 1.0e-11 * scale &&
                fabs(last.b - segment.b) <= 1.0e-10 * scale)
            {
                last.hi = segment.hi;
                continue;
            }
        }

        segments[write++] = segment;
    }

    return write;
}

__device__ void sunopt_b14c_sort_dedup(
    double* values,
    int* count
) {
    int n = *count;

    for (int i = 1; i < n; ++i) {
        const double value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }

    int write = 0;
    for (int read = 0; read < n; ++read) {
        if (write == 0 || fabs(values[read] - values[write - 1]) >= 1.0e-12) {
            values[write++] = values[read];
        }
    }

    *count = write;
}

__device__ int sunopt_b14c_charge(
    double export_value,
    double maximum_cut,
    double max_hp,
    double hp_factor,
    SunoptB14CSegment* output
) {
    double boundaries[4];
    int boundary_count = 0;
    boundaries[boundary_count++] = 0.0;
    boundaries[boundary_count++] = maximum_cut;

    const double b0 = export_value - max_hp;
    const double b1 = export_value;
    if (b0 > 0.0 && b0 < maximum_cut) boundaries[boundary_count++] = b0;
    if (b1 > 0.0 && b1 < maximum_cut) boundaries[boundary_count++] = b1;

    sunopt_b14c_sort_dedup(boundaries, &boundary_count);

    int count = 0;
    for (int i = 0; i + 1 < boundary_count; ++i) {
        const double lo = boundaries[i];
        const double hi = boundaries[i + 1];
        if (hi <= lo + 1.0e-13) continue;

        const double middle = (lo + hi) * 0.5;
        const double raw = export_value - middle;
        SunoptB14CSegment segment{};
        segment.lo = lo;
        segment.hi = hi;

        if (raw <= 0.0) {
            segment.m = 0.0;
            segment.b = 0.0;
        } else if (raw >= max_hp) {
            segment.m = 0.0;
            segment.b = max_hp * hp_factor;
        } else {
            segment.m = -hp_factor;
            segment.b = export_value * hp_factor;
        }

        output[count++] = segment;
    }

    return sunopt_b14c_merge(output, count);
}

__device__ int sunopt_b14c_find_segment(
    const SunoptB14CSegment* segments,
    int count,
    double value
) {
    for (int i = 0; i < count; ++i) {
        if (value >= segments[i].lo - 1.0e-12 &&
            value <= segments[i].hi + 1.0e-12)
        {
            return i;
        }
    }
    return -1;
}

__device__ int sunopt_b14c_add(
    const SunoptB14CSegment* left,
    int left_count,
    const SunoptB14CSegment* right,
    int right_count,
    double maximum_cut,
    SunoptB14CSegment* output,
    double* boundaries,
    bool* scratch_overflow
) {
    int boundary_count = 0;
    boundaries[boundary_count++] = 0.0;
    boundaries[boundary_count++] = maximum_cut;

    for (int i = 0; i < left_count; ++i) {
        const double values[2] = { left[i].lo, left[i].hi };
        for (int k = 0; k < 2; ++k) {
            const double value = values[k];
            if (value > 0.0 && value < maximum_cut) {
                if (boundary_count >= static_cast<int>(SUNOPT_B14C_BOUNDARIES)) {
                    *scratch_overflow = true;
                    return 0;
                }
                boundaries[boundary_count++] = value;
            }
        }
    }

    for (int i = 0; i < right_count; ++i) {
        const double values[2] = { right[i].lo, right[i].hi };
        for (int k = 0; k < 2; ++k) {
            const double value = values[k];
            if (value > 0.0 && value < maximum_cut) {
                if (boundary_count >= static_cast<int>(SUNOPT_B14C_BOUNDARIES)) {
                    *scratch_overflow = true;
                    return 0;
                }
                boundaries[boundary_count++] = value;
            }
        }
    }

    sunopt_b14c_sort_dedup(boundaries, &boundary_count);

    int out_count = 0;
    for (int i = 0; i + 1 < boundary_count; ++i) {
        const double lo = boundaries[i];
        const double hi = boundaries[i + 1];
        if (hi <= lo + 1.0e-13) continue;

        if (out_count >= static_cast<int>(SUNOPT_B14C_SCRATCH_SEGMENTS)) {
            *scratch_overflow = true;
            return 0;
        }

        const double middle = (lo + hi) * 0.5;
        const int li = sunopt_b14c_find_segment(left, left_count, middle);
        const int ri = sunopt_b14c_find_segment(right, right_count, middle);
        if (li < 0 || ri < 0) {
            *scratch_overflow = true;
            return 0;
        }

        output[out_count++] = SunoptB14CSegment{
            lo,
            hi,
            left[li].m + right[ri].m,
            left[li].b + right[ri].b,
        };
    }

    return sunopt_b14c_merge(output, out_count);
}

__device__ bool sunopt_b14c_push_segment(
    SunoptB14CSegment* output,
    int* count,
    SunoptB14CSegment value,
    bool* scratch_overflow
) {
    if (*count >= static_cast<int>(SUNOPT_B14C_SCRATCH_SEGMENTS)) {
        *scratch_overflow = true;
        return false;
    }
    output[(*count)++] = value;
    return true;
}

__device__ int sunopt_b14c_clip_capacity(
    const SunoptB14CSegment* input,
    int input_count,
    double capacity,
    SunoptB14CSegment* output,
    bool* scratch_overflow
) {
    int out_count = 0;

    for (int i = 0; i < input_count; ++i) {
        const SunoptB14CSegment segment = input[i];
        const double y0 = segment.m * segment.lo + segment.b;
        const double y1 = segment.m * segment.hi + segment.b;
        const double d0 = y0 - capacity;
        const double d1 = y1 - capacity;

        if (fabs(segment.m) < 1.0e-15 || d0 * d1 >= 0.0) {
            if (y0 <= capacity && y1 <= capacity) {
                if (!sunopt_b14c_push_segment(output, &out_count, segment, scratch_overflow)) return 0;
            } else {
                if (!sunopt_b14c_push_segment(output, &out_count, SunoptB14CSegment{segment.lo, segment.hi, 0.0, capacity}, scratch_overflow)) return 0;
            }
            continue;
        }

        const double intersection =
            fmin(fmax((capacity - segment.b) / segment.m, segment.lo), segment.hi);
        const bool left_is_affine = y0 <= capacity;

        if (intersection > segment.lo + 1.0e-13) {
            if (left_is_affine) {
                if (!sunopt_b14c_push_segment(output, &out_count, SunoptB14CSegment{segment.lo, intersection, segment.m, segment.b}, scratch_overflow)) return 0;
            } else {
                if (!sunopt_b14c_push_segment(output, &out_count, SunoptB14CSegment{segment.lo, intersection, 0.0, capacity}, scratch_overflow)) return 0;
            }
        }

        if (segment.hi > intersection + 1.0e-13) {
            if (left_is_affine) {
                if (!sunopt_b14c_push_segment(output, &out_count, SunoptB14CSegment{intersection, segment.hi, 0.0, capacity}, scratch_overflow)) return 0;
            } else {
                if (!sunopt_b14c_push_segment(output, &out_count, SunoptB14CSegment{intersection, segment.hi, segment.m, segment.b}, scratch_overflow)) return 0;
            }
        }
    }

    return sunopt_b14c_merge(output, out_count);
}

__device__ double sunopt_b14c_truncate_feasible(
    const SunoptB14CSegment* input,
    int input_count,
    double threshold,
    double current_hi
) {
    for (int i = 0; i < input_count; ++i) {
        const SunoptB14CSegment segment = input[i];
        if (segment.lo >= current_hi) break;

        const double segment_hi = fmin(segment.hi, current_hi);
        const double y_lo = segment.m * segment.lo + segment.b;
        const double y_hi = segment.m * segment_hi + segment.b;

        if (y_lo + 1.0e-12 < threshold) {
            return fmin(segment.lo, current_hi);
        }

        if (y_hi + 1.0e-12 < threshold && segment.m < 0.0) {
            return fmin(
                fmin(fmax((threshold - segment.b) / segment.m, segment.lo), segment_hi),
                current_hi
            );
        }
    }

    return current_hi;
}

__global__ void sunopt_b14c_warp32_kernel(
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
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ max_segments,
    uint32_t* __restrict__ max_work_segments
) {
    const size_t job = static_cast<size_t>(blockIdx.x);
    const size_t total_jobs = step_count * tank_count;
    if (job >= total_jobs) return;

    __shared__ SunoptB14CSegment buffer_a[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_b[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_c[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment charge_segments[3];
    __shared__ double boundaries[SUNOPT_B14C_BOUNDARIES];

    if (threadIdx.x != 0) return;

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
    const double reserve = sunopt_b14c_clamp(dynamic_values[db + 1], 0.0, target);
    const size_t current_index = static_cast<size_t>(current_indices[step]);
    const size_t horizon = static_cast<size_t>(horizon_steps[step]);
    const double energy_cut = energy_cut_levels[job];

    cut_results[job] = energy_cut;
    status[job] = 3u;
    max_segments[job] = 0u;
    max_work_segments[job] = 0u;

    if (!affine_supported || retention <= 0.0 || reserve <= SUNOPT_B14C_EPSILON ||
        data_length == 0 || horizon == 0 || horizon > data_length)
    {
        return;
    }

    if (sunopt_b14c_scalar_survives(
            export_power, demand_energy, data_length, current_index, horizon,
            energy_cut, current, reserve, target, max_hp, hp_factor,
            retention, loss_intercept))
    {
        status[job] = 0u;
        max_segments[job] = 1u;
        max_work_segments[job] = 1u;
        return;
    }

    if (!sunopt_b14c_scalar_survives(
            export_power, demand_energy, data_length, current_index, horizon,
            0.0, current, reserve, target, max_hp, hp_factor,
            retention, loss_intercept))
    {
        cut_results[job] = 0.0;
        status[job] = 1u;
        max_segments[job] = 1u;
        max_work_segments[job] = 1u;
        return;
    }

    bool scratch_overflow = false;
    bool final_overflow = false;
    uint32_t maximum_final_segments = 0u;
    uint32_t maximum_work_segments = 0u;

    int charge_count = sunopt_b14c_charge(
        export_power[current_index % data_length], energy_cut,
        max_hp, hp_factor, charge_segments);

    buffer_a[0] = SunoptB14CSegment{
        0.0,
        energy_cut,
        0.0,
        sunopt_b14c_clamp(current, 0.0, target),
    };
    int stored_count = 1;

    int added_count = sunopt_b14c_add(
        buffer_a, stored_count,
        charge_segments, charge_count,
        energy_cut, buffer_b, boundaries, &scratch_overflow);
    maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count) ? maximum_work_segments : static_cast<uint32_t>(added_count);
    if (scratch_overflow) {
        status[job] = 5u;
        return;
    }

    stored_count = sunopt_b14c_clip_capacity(
        buffer_b, added_count, target, buffer_a, &scratch_overflow);
    maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count) ? maximum_work_segments : static_cast<uint32_t>(stored_count);
    if (scratch_overflow) {
        status[job] = 5u;
        return;
    }

    maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count) ? maximum_final_segments : static_cast<uint32_t>(stored_count);
    if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;

    double feasible_hi = energy_cut;

    for (size_t offset = 1; offset < horizon; ++offset) {
        const size_t index = (current_index + offset) % data_length;

        for (int i = 0; i < stored_count; ++i) {
            buffer_b[i] = SunoptB14CSegment{
                buffer_a[i].lo,
                buffer_a[i].hi,
                retention * buffer_a[i].m,
                retention * buffer_a[i].b - loss_intercept - fmax(demand_energy[index], 0.0),
            };
        }
        int before_count = sunopt_b14c_merge(buffer_b, stored_count);
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(before_count) ? maximum_work_segments : static_cast<uint32_t>(before_count);
        maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(before_count) ? maximum_final_segments : static_cast<uint32_t>(before_count);
        if (before_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;

        feasible_hi = sunopt_b14c_truncate_feasible(
            buffer_b, before_count, reserve - SUNOPT_B14C_EPSILON, feasible_hi);

        if (feasible_hi <= 0.0) break;

        charge_count = sunopt_b14c_charge(
            export_power[index], energy_cut, max_hp, hp_factor, charge_segments);

        added_count = sunopt_b14c_add(
            buffer_b, before_count,
            charge_segments, charge_count,
            energy_cut, buffer_c, boundaries, &scratch_overflow);
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count) ? maximum_work_segments : static_cast<uint32_t>(added_count);
        if (scratch_overflow) {
            status[job] = 5u;
            max_segments[job] = maximum_final_segments;
            max_work_segments[job] = maximum_work_segments;
            return;
        }

        stored_count = sunopt_b14c_clip_capacity(
            buffer_c, added_count, target, buffer_a, &scratch_overflow);
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count) ? maximum_work_segments : static_cast<uint32_t>(stored_count);
        if (scratch_overflow) {
            status[job] = 5u;
            max_segments[job] = maximum_final_segments;
            max_work_segments[job] = maximum_work_segments;
            return;
        }

        maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count) ? maximum_final_segments : static_cast<uint32_t>(stored_count);
        if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
    }

    const double mathematical_cut = fmin(fmax(feasible_hi, 0.0), energy_cut);
    double lower = 0.0;
    double upper = energy_cut;

    for (int iteration = 0; iteration < 20; ++iteration) {
        const double middle = (lower + upper) * 0.5;
        if (middle <= mathematical_cut + 1.0e-12) lower = middle;
        else upper = middle;
    }

    cut_results[job] = lower;
    max_segments[job] = maximum_final_segments;
    max_work_segments[job] = maximum_work_segments;
    status[job] = final_overflow ? 4u : 2u;
}

} // namespace

extern "C" int sunopt_cuda_run_b14c_warp32_benchmark(
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
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_max_segments,
    uint32_t* host_max_work_segments,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results || !host_status ||
        !host_max_segments || !host_max_work_segments ||
        !out_upload_ms || !out_kernel_total_ms || !out_download_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0 || repetitions == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double *d_export=nullptr, *d_demand=nullptr, *d_tanks=nullptr,
           *d_dynamic=nullptr, *d_energy_cut=nullptr, *d_cut_results=nullptr;
    uint32_t *d_current=nullptr, *d_horizon=nullptr, *d_status=nullptr,
             *d_max_segments=nullptr, *d_max_work_segments=nullptr;

    const size_t jobs = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t step_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = jobs * 2 * sizeof(double);
    const size_t job_double_bytes = jobs * sizeof(double);
    const size_t job_u32_bytes = jobs * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
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

#define B14C_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14C_MALLOC(d_export, annual_bytes);
    B14C_MALLOC(d_demand, annual_bytes);
    B14C_MALLOC(d_tanks, tank_bytes);
    B14C_MALLOC(d_current, step_bytes);
    B14C_MALLOC(d_horizon, step_bytes);
    B14C_MALLOC(d_dynamic, dynamic_bytes);
    B14C_MALLOC(d_energy_cut, job_double_bytes);
    B14C_MALLOC(d_cut_results, job_double_bytes);
    B14C_MALLOC(d_status, job_u32_bytes);
    B14C_MALLOC(d_max_segments, job_u32_bytes);
    B14C_MALLOC(d_max_work_segments, job_u32_bytes);
#undef B14C_MALLOC

    const auto wall_start = Clock::now();
    const auto upload_start = Clock::now();

#define B14C_COPY_H2D(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14C_COPY_H2D(d_export, host_export_power, annual_bytes);
    B14C_COPY_H2D(d_demand, host_demand_energy, annual_bytes);
    B14C_COPY_H2D(d_tanks, host_tank_parameters, tank_bytes);
    B14C_COPY_H2D(d_current, host_current_indices, step_bytes);
    B14C_COPY_H2D(d_horizon, host_horizon_steps, step_bytes);
    B14C_COPY_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);
    B14C_COPY_H2D(d_energy_cut, host_energy_cut_levels, job_double_bytes);
#undef B14C_COPY_H2D

    const auto upload_stop = Clock::now();
    *out_upload_ms = std::chrono::duration<double,std::milli>(upload_stop-upload_start).count();

    err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    const unsigned int blocks = static_cast<unsigned int>(jobs);
    for (uint32_t r = 0; r < repetitions; ++r) {
        sunopt_b14c_warp32_kernel<<<blocks,SUNOPT_B14C_THREADS,0,stream>>>(
            d_export,d_demand,data_length,d_tanks,tank_count,d_current,d_horizon,
            d_dynamic,d_energy_cut,step_count,d_cut_results,d_status,
            d_max_segments,d_max_work_segments
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
#define B14C_COPY_D2H(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)
    B14C_COPY_D2H(host_cut_results, d_cut_results, job_double_bytes);
    B14C_COPY_D2H(host_status, d_status, job_u32_bytes);
    B14C_COPY_D2H(host_max_segments, d_max_segments, job_u32_bytes);
    B14C_COPY_D2H(host_max_work_segments, d_max_work_segments, job_u32_bytes);
#undef B14C_COPY_D2H
    const auto download_stop = Clock::now();

    *out_download_ms = std::chrono::duration<double,std::milli>(download_stop-download_start).count();
    *out_wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_start).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14C_WARP32_END



// SUNOPT_BLACKWELL_B14D_SORTED_MERGE_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14D_THREADS = 32;

// Exact O(n) sum of two already ordered, gap-free PWA segment lists.
// B14C rebuilt a combined boundary vector, insertion-sorted it, then
// linearly searched both lists for every interval.  The mathematical
// result is identical; this walks both ordered lists once.
__device__ int sunopt_b14d_add_sorted(
    const SunoptB14CSegment* left,
    int left_count,
    const SunoptB14CSegment* right,
    int right_count,
    double maximum_cut,
    SunoptB14CSegment* output,
    bool* scratch_overflow
) {
    int li = 0;
    int ri = 0;
    int out_count = 0;

    while (li < left_count && ri < right_count) {
        const SunoptB14CSegment l = left[li];
        const SunoptB14CSegment r = right[ri];

        const double lo = fmax(0.0, fmax(l.lo, r.lo));
        const double hi = fmin(maximum_cut, fmin(l.hi, r.hi));

        if (hi > lo + 1.0e-13) {
            if (out_count >= static_cast<int>(SUNOPT_B14C_SCRATCH_SEGMENTS)) {
                *scratch_overflow = true;
                return 0;
            }

            output[out_count++] = SunoptB14CSegment{
                lo,
                hi,
                l.m + r.m,
                l.b + r.b,
            };
        }

        const bool advance_left = l.hi <= r.hi + 1.0e-12;
        const bool advance_right = r.hi <= l.hi + 1.0e-12;

        if (advance_left) ++li;
        if (advance_right) ++ri;

        // Defensive progress guard for malformed/non-overlapping lists.
        if (!advance_left && !advance_right) {
            if (l.hi < r.hi) ++li;
            else ++ri;
        }
    }

    return sunopt_b14c_merge(output, out_count);
}

__global__ void sunopt_b14d_warp32_merge_kernel(
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
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ max_segments,
    uint32_t* __restrict__ max_work_segments
) {
    const size_t job = static_cast<size_t>(blockIdx.x);
    const size_t total_jobs = step_count * tank_count;
    if (job >= total_jobs) return;

    __shared__ SunoptB14CSegment buffer_a[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_b[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_c[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment charge_segments[3];

    // B14D deliberately changes one thing at a time: the exact PWA-add
    // operation is now O(n), while lane 0 still owns the ordered algebra.
    // This keeps B14C/B14D parity attributable to the merge replacement.
    if (threadIdx.x != 0) return;

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
    const double reserve = sunopt_b14c_clamp(dynamic_values[db + 1], 0.0, target);
    const size_t current_index = static_cast<size_t>(current_indices[step]);
    const size_t horizon = static_cast<size_t>(horizon_steps[step]);
    const double energy_cut = energy_cut_levels[job];

    cut_results[job] = energy_cut;
    status[job] = 3u;
    max_segments[job] = 0u;
    max_work_segments[job] = 0u;

    if (!affine_supported || retention <= 0.0 || reserve <= SUNOPT_B14C_EPSILON ||
        data_length == 0 || horizon == 0 || horizon > data_length)
    {
        return;
    }

    if (sunopt_b14c_scalar_survives(
            export_power, demand_energy, data_length, current_index, horizon,
            energy_cut, current, reserve, target, max_hp, hp_factor,
            retention, loss_intercept))
    {
        status[job] = 0u;
        max_segments[job] = 1u;
        max_work_segments[job] = 1u;
        return;
    }

    if (!sunopt_b14c_scalar_survives(
            export_power, demand_energy, data_length, current_index, horizon,
            0.0, current, reserve, target, max_hp, hp_factor,
            retention, loss_intercept))
    {
        cut_results[job] = 0.0;
        status[job] = 1u;
        max_segments[job] = 1u;
        max_work_segments[job] = 1u;
        return;
    }

    bool scratch_overflow = false;
    bool final_overflow = false;
    uint32_t maximum_final_segments = 0u;
    uint32_t maximum_work_segments = 0u;

    int charge_count = sunopt_b14c_charge(
        export_power[current_index % data_length], energy_cut,
        max_hp, hp_factor, charge_segments);

    buffer_a[0] = SunoptB14CSegment{
        0.0,
        energy_cut,
        0.0,
        sunopt_b14c_clamp(current, 0.0, target),
    };
    int stored_count = 1;

    int added_count = sunopt_b14d_add_sorted(
        buffer_a, stored_count,
        charge_segments, charge_count,
        energy_cut, buffer_b, &scratch_overflow);
    maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count) ? maximum_work_segments : static_cast<uint32_t>(added_count);
    if (scratch_overflow) {
        status[job] = 5u;
        return;
    }

    stored_count = sunopt_b14c_clip_capacity(
        buffer_b, added_count, target, buffer_a, &scratch_overflow);
    maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count) ? maximum_work_segments : static_cast<uint32_t>(stored_count);
    if (scratch_overflow) {
        status[job] = 5u;
        return;
    }

    maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count) ? maximum_final_segments : static_cast<uint32_t>(stored_count);
    if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;

    double feasible_hi = energy_cut;

    for (size_t offset = 1; offset < horizon; ++offset) {
        const size_t index = (current_index + offset) % data_length;

        for (int i = 0; i < stored_count; ++i) {
            buffer_b[i] = SunoptB14CSegment{
                buffer_a[i].lo,
                buffer_a[i].hi,
                retention * buffer_a[i].m,
                retention * buffer_a[i].b - loss_intercept - fmax(demand_energy[index], 0.0),
            };
        }
        int before_count = sunopt_b14c_merge(buffer_b, stored_count);
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(before_count) ? maximum_work_segments : static_cast<uint32_t>(before_count);
        maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(before_count) ? maximum_final_segments : static_cast<uint32_t>(before_count);
        if (before_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;

        feasible_hi = sunopt_b14c_truncate_feasible(
            buffer_b, before_count, reserve - SUNOPT_B14C_EPSILON, feasible_hi);

        if (feasible_hi <= 0.0) break;

        charge_count = sunopt_b14c_charge(
            export_power[index], energy_cut, max_hp, hp_factor, charge_segments);

        added_count = sunopt_b14d_add_sorted(
            buffer_b, before_count,
            charge_segments, charge_count,
            energy_cut, buffer_c, &scratch_overflow);
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count) ? maximum_work_segments : static_cast<uint32_t>(added_count);
        if (scratch_overflow) {
            status[job] = 5u;
            max_segments[job] = maximum_final_segments;
            max_work_segments[job] = maximum_work_segments;
            return;
        }

        stored_count = sunopt_b14c_clip_capacity(
            buffer_c, added_count, target, buffer_a, &scratch_overflow);
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count) ? maximum_work_segments : static_cast<uint32_t>(stored_count);
        if (scratch_overflow) {
            status[job] = 5u;
            max_segments[job] = maximum_final_segments;
            max_work_segments[job] = maximum_work_segments;
            return;
        }

        maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count) ? maximum_final_segments : static_cast<uint32_t>(stored_count);
        if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
    }

    const double mathematical_cut = fmin(fmax(feasible_hi, 0.0), energy_cut);
    double lower = 0.0;
    double upper = energy_cut;

    for (int iteration = 0; iteration < 20; ++iteration) {
        const double middle = (lower + upper) * 0.5;
        if (middle <= mathematical_cut + 1.0e-12) lower = middle;
        else upper = middle;
    }

    cut_results[job] = lower;
    max_segments[job] = maximum_final_segments;
    max_work_segments[job] = maximum_work_segments;
    status[job] = final_overflow ? 4u : 2u;
}

} // namespace

extern "C" int sunopt_cuda_run_b14d_warp32_merge_benchmark(
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
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_max_segments,
    uint32_t* host_max_work_segments,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results || !host_status ||
        !host_max_segments || !host_max_work_segments ||
        !out_upload_ms || !out_kernel_total_ms || !out_download_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0 || repetitions == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double *d_export=nullptr, *d_demand=nullptr, *d_tanks=nullptr,
           *d_dynamic=nullptr, *d_energy_cut=nullptr, *d_cut_results=nullptr;
    uint32_t *d_current=nullptr, *d_horizon=nullptr, *d_status=nullptr,
             *d_max_segments=nullptr, *d_max_work_segments=nullptr;

    const size_t jobs = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t step_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = jobs * 2 * sizeof(double);
    const size_t job_double_bytes = jobs * sizeof(double);
    const size_t job_u32_bytes = jobs * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
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

#define B14D_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14D_MALLOC(d_export, annual_bytes);
    B14D_MALLOC(d_demand, annual_bytes);
    B14D_MALLOC(d_tanks, tank_bytes);
    B14D_MALLOC(d_current, step_bytes);
    B14D_MALLOC(d_horizon, step_bytes);
    B14D_MALLOC(d_dynamic, dynamic_bytes);
    B14D_MALLOC(d_energy_cut, job_double_bytes);
    B14D_MALLOC(d_cut_results, job_double_bytes);
    B14D_MALLOC(d_status, job_u32_bytes);
    B14D_MALLOC(d_max_segments, job_u32_bytes);
    B14D_MALLOC(d_max_work_segments, job_u32_bytes);
#undef B14D_MALLOC

    const auto wall_start = Clock::now();
    const auto upload_start = Clock::now();

#define B14D_COPY_H2D(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14D_COPY_H2D(d_export, host_export_power, annual_bytes);
    B14D_COPY_H2D(d_demand, host_demand_energy, annual_bytes);
    B14D_COPY_H2D(d_tanks, host_tank_parameters, tank_bytes);
    B14D_COPY_H2D(d_current, host_current_indices, step_bytes);
    B14D_COPY_H2D(d_horizon, host_horizon_steps, step_bytes);
    B14D_COPY_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);
    B14D_COPY_H2D(d_energy_cut, host_energy_cut_levels, job_double_bytes);
#undef B14D_COPY_H2D

    const auto upload_stop = Clock::now();
    *out_upload_ms = std::chrono::duration<double,std::milli>(upload_stop-upload_start).count();

    err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    const unsigned int blocks = static_cast<unsigned int>(jobs);
    for (uint32_t r = 0; r < repetitions; ++r) {
        sunopt_b14d_warp32_merge_kernel<<<blocks,SUNOPT_B14D_THREADS,0,stream>>>(
            d_export,d_demand,data_length,d_tanks,tank_count,d_current,d_horizon,
            d_dynamic,d_energy_cut,step_count,d_cut_results,d_status,
            d_max_segments,d_max_work_segments
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
#define B14D_COPY_D2H(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)
    B14D_COPY_D2H(host_cut_results, d_cut_results, job_double_bytes);
    B14D_COPY_D2H(host_status, d_status, job_u32_bytes);
    B14D_COPY_D2H(host_max_segments, d_max_segments, job_u32_bytes);
    B14D_COPY_D2H(host_max_work_segments, d_max_work_segments, job_u32_bytes);
#undef B14D_COPY_D2H
    const auto download_stop = Clock::now();

    *out_download_ms = std::chrono::duration<double,std::milli>(download_stop-download_start).count();
    *out_wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_start).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14D_SORTED_MERGE_END



// SUNOPT_BLACKWELL_B14E_WARP_PARALLEL_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14E_THREADS = 32;
constexpr unsigned int SUNOPT_B14E_FULL_MASK = 0xffffffffu;

// Parallel specialization of B14D's ordered PWA add. The right operand is
// always the heat-pump charge PWA (<=3 segments). One lane owns one left
// segment, computes its ordered intersections, then a warp prefix sum writes
// them in the exact same order as the serial merge. If the left list ever
// exceeds one warp, correctness wins: lane 0 falls back to B14D.
__device__ int sunopt_b14e_add_charge_warp(
    const SunoptB14CSegment* left,
    int left_count,
    const SunoptB14CSegment* charge,
    int charge_count,
    double maximum_cut,
    SunoptB14CSegment* output,
    int lane,
    int* shared_count,
    int* shared_overflow
) {
    if (left_count > 32 || charge_count > 3) {
        if (lane == 0) {
            bool overflow = false;
            *shared_count = sunopt_b14d_add_sorted(
                left, left_count, charge, charge_count,
                maximum_cut, output, &overflow);
            *shared_overflow = overflow ? 1 : 0;
        }
        __syncwarp(SUNOPT_B14E_FULL_MASK);
        return *shared_count;
    }

    SunoptB14CSegment local[3];
    int local_count = 0;

    if (lane < left_count) {
        const SunoptB14CSegment l = left[lane];
        for (int ri = 0; ri < charge_count; ++ri) {
            const SunoptB14CSegment r = charge[ri];
            const double lo = fmax(0.0, fmax(l.lo, r.lo));
            const double hi = fmin(maximum_cut, fmin(l.hi, r.hi));
            if (hi > lo + 1.0e-13) {
                local[local_count++] = SunoptB14CSegment{
                    lo,
                    hi,
                    l.m + r.m,
                    l.b + r.b,
                };
            }
        }
    }

    int inclusive = local_count;
    for (int delta = 1; delta < 32; delta <<= 1) {
        const int value = __shfl_up_sync(
            SUNOPT_B14E_FULL_MASK, inclusive, delta);
        if (lane >= delta) inclusive += value;
    }

    const int total = __shfl_sync(
        SUNOPT_B14E_FULL_MASK, inclusive, 31);
    const int start = inclusive - local_count;

    if (lane == 0) {
        *shared_overflow =
            total > static_cast<int>(SUNOPT_B14C_SCRATCH_SEGMENTS) ? 1 : 0;
    }
    __syncwarp(SUNOPT_B14E_FULL_MASK);

    if (*shared_overflow == 0) {
        for (int i = 0; i < local_count; ++i) {
            output[start + i] = local[i];
        }
    }
    __syncwarp(SUNOPT_B14E_FULL_MASK);

    if (lane == 0) {
        *shared_count =
            *shared_overflow == 0
                ? sunopt_b14c_merge(output, total)
                : 0;
    }
    __syncwarp(SUNOPT_B14E_FULL_MASK);
    return *shared_count;
}

// Capacity clipping is independent per input segment. Each lane clips one
// segment into one or two ordered pieces; a warp prefix sum compacts them.
// Again, >32 input segments use the exact B14C serial helper as a fallback.
__device__ int sunopt_b14e_clip_capacity_warp(
    const SunoptB14CSegment* input,
    int input_count,
    double capacity,
    SunoptB14CSegment* output,
    int lane,
    int* shared_count,
    int* shared_overflow
) {
    if (input_count > 32) {
        if (lane == 0) {
            bool overflow = false;
            *shared_count = sunopt_b14c_clip_capacity(
                input, input_count, capacity, output, &overflow);
            *shared_overflow = overflow ? 1 : 0;
        }
        __syncwarp(SUNOPT_B14E_FULL_MASK);
        return *shared_count;
    }

    SunoptB14CSegment local[2];
    int local_count = 0;

    if (lane < input_count) {
        const SunoptB14CSegment segment = input[lane];
        const double y0 = segment.m * segment.lo + segment.b;
        const double y1 = segment.m * segment.hi + segment.b;
        const double d0 = y0 - capacity;
        const double d1 = y1 - capacity;

        if (fabs(segment.m) < 1.0e-15 || d0 * d1 >= 0.0) {
            if (y0 <= capacity && y1 <= capacity) {
                local[local_count++] = segment;
            } else {
                local[local_count++] = SunoptB14CSegment{
                    segment.lo, segment.hi, 0.0, capacity};
            }
        } else {
            const double intersection = fmin(
                fmax((capacity - segment.b) / segment.m, segment.lo),
                segment.hi);
            const bool left_is_affine = y0 <= capacity;

            if (intersection > segment.lo + 1.0e-13) {
                local[local_count++] = left_is_affine
                    ? SunoptB14CSegment{
                        segment.lo, intersection, segment.m, segment.b}
                    : SunoptB14CSegment{
                        segment.lo, intersection, 0.0, capacity};
            }

            if (segment.hi > intersection + 1.0e-13) {
                local[local_count++] = left_is_affine
                    ? SunoptB14CSegment{
                        intersection, segment.hi, 0.0, capacity}
                    : SunoptB14CSegment{
                        intersection, segment.hi, segment.m, segment.b};
            }
        }
    }

    int inclusive = local_count;
    for (int delta = 1; delta < 32; delta <<= 1) {
        const int value = __shfl_up_sync(
            SUNOPT_B14E_FULL_MASK, inclusive, delta);
        if (lane >= delta) inclusive += value;
    }

    const int total = __shfl_sync(
        SUNOPT_B14E_FULL_MASK, inclusive, 31);
    const int start = inclusive - local_count;

    if (lane == 0) {
        *shared_overflow =
            total > static_cast<int>(SUNOPT_B14C_SCRATCH_SEGMENTS) ? 1 : 0;
    }
    __syncwarp(SUNOPT_B14E_FULL_MASK);

    if (*shared_overflow == 0) {
        for (int i = 0; i < local_count; ++i) {
            output[start + i] = local[i];
        }
    }
    __syncwarp(SUNOPT_B14E_FULL_MASK);

    if (lane == 0) {
        *shared_count =
            *shared_overflow == 0
                ? sunopt_b14c_merge(output, total)
                : 0;
    }
    __syncwarp(SUNOPT_B14E_FULL_MASK);
    return *shared_count;
}

// Exact parallel form of B14C's ordered feasibility scan. Each lane computes
// the earliest infeasible cut in its segment(s); a warp min reduction returns
// the first global violation because segment domains are ordered/disjoint.
__device__ double sunopt_b14e_truncate_feasible_warp(
    const SunoptB14CSegment* input,
    int input_count,
    double threshold,
    double current_hi,
    int lane
) {
    double candidate = current_hi;

    for (int i = lane; i < input_count; i += 32) {
        const SunoptB14CSegment segment = input[i];
        if (segment.lo >= current_hi) continue;

        const double segment_hi = fmin(segment.hi, current_hi);
        const double y_lo = segment.m * segment.lo + segment.b;
        const double y_hi = segment.m * segment_hi + segment.b;

        if (y_lo + 1.0e-12 < threshold) {
            candidate = fmin(candidate, fmin(segment.lo, current_hi));
        } else if (y_hi + 1.0e-12 < threshold && segment.m < 0.0) {
            const double crossing = fmin(
                fmin(
                    fmax((threshold - segment.b) / segment.m, segment.lo),
                    segment_hi),
                current_hi);
            candidate = fmin(candidate, crossing);
        }
    }

    for (int delta = 16; delta > 0; delta >>= 1) {
        candidate = fmin(
            candidate,
            __shfl_down_sync(
                SUNOPT_B14E_FULL_MASK, candidate, delta));
    }

    return __shfl_sync(SUNOPT_B14E_FULL_MASK, candidate, 0);
}

__global__ void sunopt_b14e_warp_parallel_kernel(
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
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ max_segments,
    uint32_t* __restrict__ max_work_segments
) {
    const size_t job = static_cast<size_t>(blockIdx.x);
    const size_t total_jobs = step_count * tank_count;
    if (job >= total_jobs) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;

    __shared__ SunoptB14CSegment buffer_a[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_b[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_c[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment charge_segments[3];
    __shared__ int shared_count;
    __shared__ int shared_overflow;

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
    const double reserve = sunopt_b14c_clamp(dynamic_values[db + 1], 0.0, target);
    const size_t current_index = static_cast<size_t>(current_indices[step]);
    const size_t horizon = static_cast<size_t>(horizon_steps[step]);
    const double energy_cut = energy_cut_levels[job];

    int early_status = -1;
    if (lane == 0) {
        cut_results[job] = energy_cut;
        status[job] = 3u;
        max_segments[job] = 0u;
        max_work_segments[job] = 0u;

        if (!affine_supported || retention <= 0.0 || reserve <= SUNOPT_B14C_EPSILON ||
            data_length == 0 || horizon == 0 || horizon > data_length)
        {
            early_status = 3;
        } else if (sunopt_b14c_scalar_survives(
                export_power, demand_energy, data_length, current_index, horizon,
                energy_cut, current, reserve, target, max_hp, hp_factor,
                retention, loss_intercept))
        {
            early_status = 0;
        } else if (!sunopt_b14c_scalar_survives(
                export_power, demand_energy, data_length, current_index, horizon,
                0.0, current, reserve, target, max_hp, hp_factor,
                retention, loss_intercept))
        {
            early_status = 1;
        }
    }

    early_status = __shfl_sync(
        SUNOPT_B14E_FULL_MASK, early_status, 0);

    if (early_status >= 0) {
        if (lane == 0) {
            status[job] = static_cast<uint32_t>(early_status);
            if (early_status == 1) cut_results[job] = 0.0;
            if (early_status <= 1) {
                max_segments[job] = 1u;
                max_work_segments[job] = 1u;
            }
        }
        return;
    }

    bool final_overflow = false;
    uint32_t maximum_final_segments = 0u;
    uint32_t maximum_work_segments = 0u;

    int charge_count = 0;
    if (lane == 0) {
        charge_count = sunopt_b14c_charge(
            export_power[current_index % data_length], energy_cut,
            max_hp, hp_factor, charge_segments);
        buffer_a[0] = SunoptB14CSegment{
            0.0,
            energy_cut,
            0.0,
            sunopt_b14c_clamp(current, 0.0, target),
        };
    }
    charge_count = __shfl_sync(
        SUNOPT_B14E_FULL_MASK, charge_count, 0);
    __syncwarp(SUNOPT_B14E_FULL_MASK);

    int stored_count = 1;
    int added_count = sunopt_b14e_add_charge_warp(
        buffer_a, stored_count,
        charge_segments, charge_count,
        energy_cut, buffer_b, lane,
        &shared_count, &shared_overflow);

    if (lane == 0) {
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count)
            ? maximum_work_segments : static_cast<uint32_t>(added_count);
    }
    if (shared_overflow != 0) {
        if (lane == 0) status[job] = 5u;
        return;
    }

    stored_count = sunopt_b14e_clip_capacity_warp(
        buffer_b, added_count, target, buffer_a,
        lane, &shared_count, &shared_overflow);

    if (lane == 0) {
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count)
            ? maximum_work_segments : static_cast<uint32_t>(stored_count);
        maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count)
            ? maximum_final_segments : static_cast<uint32_t>(stored_count);
        if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
    }
    if (shared_overflow != 0) {
        if (lane == 0) status[job] = 5u;
        return;
    }

    double feasible_hi = energy_cut;

    for (size_t offset = 1; offset < horizon; ++offset) {
        const size_t index = (current_index + offset) % data_length;
        const double demand = fmax(demand_energy[index], 0.0);

        for (int i = lane; i < stored_count; i += 32) {
            const SunoptB14CSegment source = buffer_a[i];
            buffer_b[i] = SunoptB14CSegment{
                source.lo,
                source.hi,
                retention * source.m,
                retention * source.b - loss_intercept - demand,
            };
        }
        __syncwarp(SUNOPT_B14E_FULL_MASK);

        if (lane == 0) {
            shared_count = sunopt_b14c_merge(buffer_b, stored_count);
        }
        __syncwarp(SUNOPT_B14E_FULL_MASK);
        const int before_count = shared_count;

        if (lane == 0) {
            maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(before_count)
                ? maximum_work_segments : static_cast<uint32_t>(before_count);
            maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(before_count)
                ? maximum_final_segments : static_cast<uint32_t>(before_count);
            if (before_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
        }

        feasible_hi = sunopt_b14e_truncate_feasible_warp(
            buffer_b, before_count,
            reserve - SUNOPT_B14C_EPSILON,
            feasible_hi, lane);

        if (feasible_hi <= 0.0) break;

        if (lane == 0) {
            charge_count = sunopt_b14c_charge(
                export_power[index], energy_cut,
                max_hp, hp_factor, charge_segments);
        }
        charge_count = __shfl_sync(
            SUNOPT_B14E_FULL_MASK, charge_count, 0);
        __syncwarp(SUNOPT_B14E_FULL_MASK);

        added_count = sunopt_b14e_add_charge_warp(
            buffer_b, before_count,
            charge_segments, charge_count,
            energy_cut, buffer_c, lane,
            &shared_count, &shared_overflow);

        if (lane == 0) {
            maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count)
                ? maximum_work_segments : static_cast<uint32_t>(added_count);
        }
        if (shared_overflow != 0) {
            if (lane == 0) {
                status[job] = 5u;
                max_segments[job] = maximum_final_segments;
                max_work_segments[job] = maximum_work_segments;
            }
            return;
        }

        stored_count = sunopt_b14e_clip_capacity_warp(
            buffer_c, added_count, target, buffer_a,
            lane, &shared_count, &shared_overflow);

        if (lane == 0) {
            maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count)
                ? maximum_work_segments : static_cast<uint32_t>(stored_count);
            maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count)
                ? maximum_final_segments : static_cast<uint32_t>(stored_count);
            if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
        }
        if (shared_overflow != 0) {
            if (lane == 0) {
                status[job] = 5u;
                max_segments[job] = maximum_final_segments;
                max_work_segments[job] = maximum_work_segments;
            }
            return;
        }
    }

    if (lane == 0) {
        const double mathematical_cut = fmin(fmax(feasible_hi, 0.0), energy_cut);
        double lower = 0.0;
        double upper = energy_cut;

        for (int iteration = 0; iteration < 20; ++iteration) {
            const double middle = (lower + upper) * 0.5;
            if (middle <= mathematical_cut + 1.0e-12) lower = middle;
            else upper = middle;
        }

        cut_results[job] = lower;
        max_segments[job] = maximum_final_segments;
        max_work_segments[job] = maximum_work_segments;
        status[job] = final_overflow ? 4u : 2u;
    }
}

} // namespace

extern "C" int sunopt_cuda_run_b14e_warp_parallel_benchmark(
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
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_max_segments,
    uint32_t* host_max_work_segments,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results || !host_status ||
        !host_max_segments || !host_max_work_segments ||
        !out_upload_ms || !out_kernel_total_ms || !out_download_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0 || repetitions == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double *d_export=nullptr, *d_demand=nullptr, *d_tanks=nullptr,
           *d_dynamic=nullptr, *d_energy_cut=nullptr, *d_cut_results=nullptr;
    uint32_t *d_current=nullptr, *d_horizon=nullptr, *d_status=nullptr,
             *d_max_segments=nullptr, *d_max_work_segments=nullptr;

    const size_t jobs = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t step_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = jobs * 2 * sizeof(double);
    const size_t job_double_bytes = jobs * sizeof(double);
    const size_t job_u32_bytes = jobs * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
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

#define B14E_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14E_MALLOC(d_export, annual_bytes);
    B14E_MALLOC(d_demand, annual_bytes);
    B14E_MALLOC(d_tanks, tank_bytes);
    B14E_MALLOC(d_current, step_bytes);
    B14E_MALLOC(d_horizon, step_bytes);
    B14E_MALLOC(d_dynamic, dynamic_bytes);
    B14E_MALLOC(d_energy_cut, job_double_bytes);
    B14E_MALLOC(d_cut_results, job_double_bytes);
    B14E_MALLOC(d_status, job_u32_bytes);
    B14E_MALLOC(d_max_segments, job_u32_bytes);
    B14E_MALLOC(d_max_work_segments, job_u32_bytes);
#undef B14E_MALLOC

    const auto wall_start = Clock::now();
    const auto upload_start = Clock::now();

#define B14E_COPY_H2D(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14E_COPY_H2D(d_export, host_export_power, annual_bytes);
    B14E_COPY_H2D(d_demand, host_demand_energy, annual_bytes);
    B14E_COPY_H2D(d_tanks, host_tank_parameters, tank_bytes);
    B14E_COPY_H2D(d_current, host_current_indices, step_bytes);
    B14E_COPY_H2D(d_horizon, host_horizon_steps, step_bytes);
    B14E_COPY_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);
    B14E_COPY_H2D(d_energy_cut, host_energy_cut_levels, job_double_bytes);
#undef B14E_COPY_H2D

    const auto upload_stop = Clock::now();
    *out_upload_ms = std::chrono::duration<double,std::milli>(upload_stop-upload_start).count();

    err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    const unsigned int blocks = static_cast<unsigned int>(jobs);
    for (uint32_t r = 0; r < repetitions; ++r) {
        sunopt_b14e_warp_parallel_kernel<<<blocks,SUNOPT_B14E_THREADS,0,stream>>>(
            d_export,d_demand,data_length,d_tanks,tank_count,d_current,d_horizon,
            d_dynamic,d_energy_cut,step_count,d_cut_results,d_status,
            d_max_segments,d_max_work_segments
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
#define B14E_COPY_D2H(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)
    B14E_COPY_D2H(host_cut_results, d_cut_results, job_double_bytes);
    B14E_COPY_D2H(host_status, d_status, job_u32_bytes);
    B14E_COPY_D2H(host_max_segments, d_max_segments, job_u32_bytes);
    B14E_COPY_D2H(host_max_work_segments, d_max_work_segments, job_u32_bytes);
#undef B14E_COPY_D2H
    const auto download_stop = Clock::now();

    *out_download_ms = std::chrono::duration<double,std::milli>(download_stop-download_start).count();
    *out_wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_start).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14E_WARP_PARALLEL_HOST_END
// SUNOPT_BLACKWELL_B14E_WARP_PARALLEL_END


// SUNOPT_BLACKWELL_B14F_WARP_PRECHECK_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14F_THREADS = 32;
constexpr unsigned int SUNOPT_B14F_FULL_MASK = 0xffffffffu;

struct SunoptB14FTransform {
    double a;
    double b;
    double c;
    double max_input;
};

__device__ __forceinline__ SunoptB14FTransform
sunopt_b14f_identity() {
    SunoptB14FTransform t{};
    t.a = 1.0;
    t.b = 0.0;
    t.c = -CUDART_INF;
    t.max_input = CUDART_INF;
    return t;
}

// left(right(x)); identical max-affine/domain semantics to the validated
// continuous B13/B14 fixed-cut survival formulation.
__device__ __forceinline__ SunoptB14FTransform
sunopt_b14f_compose(
    const SunoptB14FTransform left,
    const SunoptB14FTransform right
) {
    SunoptB14FTransform out{};

    double max_input = right.max_input;
    if (right.c > left.max_input + 1.0e-12) {
        max_input = -CUDART_INF;
    } else if (right.a > 0.0) {
        const double bound = (left.max_input - right.b) / right.a;
        max_input = fmin(max_input, bound);
    } else if (right.b > left.max_input + 1.0e-12) {
        max_input = -CUDART_INF;
    }

    out.a = left.a * right.a;
    out.b = left.a * right.b + left.b;
    out.c = fmax(left.c, left.a * right.c + left.b);
    out.max_input = max_input;
    return out;
}

__device__ __forceinline__ SunoptB14FTransform
sunopt_b14f_shuffle_down(
    const SunoptB14FTransform value,
    unsigned int delta
) {
    SunoptB14FTransform out{};
    out.a = __shfl_down_sync(SUNOPT_B14F_FULL_MASK, value.a, delta);
    out.b = __shfl_down_sync(SUNOPT_B14F_FULL_MASK, value.b, delta);
    out.c = __shfl_down_sync(SUNOPT_B14F_FULL_MASK, value.c, delta);
    out.max_input = __shfl_down_sync(
        SUNOPT_B14F_FULL_MASK, value.max_input, delta);
    return out;
}

/*
 * Exact fixed-cut survival in one warp.
 *
 * Offsets 1..horizon-1 are divided into 32 contiguous chunks. A lane
 * composes at most ceil(191/32)=6 transforms for the production horizon.
 * The warp then reduces those contiguous chunk summaries with the same
 * ordered associative composition. The current bucket is handled exactly
 * like Production after the backward requirement has been summarized.
 */
__device__ __forceinline__ bool
sunopt_b14f_fixed_survives_warp(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    double cut,
    double current_storage_energy,
    double reserve,
    double target,
    double max_hp,
    double hp_factor,
    double retention,
    double loss_intercept,
    int lane
) {
    const size_t future_count = horizon > 1 ? horizon - 1 : 0;
    const size_t chunk_size = (future_count + 31u) / 32u;
    const size_t chunk_begin = 1u + static_cast<size_t>(lane) * chunk_size;
    const size_t chunk_end = chunk_size == 0
        ? 1u
        : ((chunk_begin + chunk_size) < horizon
            ? (chunk_begin + chunk_size)
            : horizon);

    SunoptB14FTransform local = sunopt_b14f_identity();
    const double inverse_retention = 1.0 / retention;

    for (size_t offset = chunk_begin;
         offset < static_cast<size_t>(chunk_end) && offset < horizon;
         ++offset)
    {
        const size_t index = (current_index + offset) % data_length;
        const double demand = fmax(demand_energy[index], 0.0);
        const double raw = export_power[index] - cut;
        const double hp = fmin(max_hp, fmax(0.0, raw));
        const double charge = hp * hp_factor;

        SunoptB14FTransform step{};
        step.a = inverse_retention;
        step.b = (loss_intercept + demand - charge) * inverse_retention;
        step.c = (
            loss_intercept + demand + reserve - SUNOPT_B14C_EPSILON
        ) * inverse_retention;
        step.max_input = target;

        local = sunopt_b14f_compose(local, step);
    }

    for (unsigned int stride = 1; stride < 32u; stride <<= 1) {
        const SunoptB14FTransform right =
            sunopt_b14f_shuffle_down(local, stride);
        const unsigned int group = stride << 1;
        if (((static_cast<unsigned int>(lane) & (group - 1u)) == 0u) &&
            static_cast<unsigned int>(lane) + stride < 32u)
        {
            local = sunopt_b14f_compose(local, right);
        }
    }

    int survives = 0;
    if (lane == 0) {
        const SunoptB14FTransform summary = local;
        bool feasible = 0.0 <= summary.max_input + 1.0e-12;
        const double required = fmax(summary.c, summary.b);
        if (required > target + 1.0e-9) feasible = false;

        const double raw0 = export_power[current_index % data_length] - cut;
        const double hp0 = fmin(max_hp, fmax(0.0, raw0));
        const double charge0 = hp0 * hp_factor;
        const double current = sunopt_b14c_clamp(
            current_storage_energy, 0.0, target);
        const double available = fmin(target, current + charge0);

        survives = feasible &&
            available + 1.0e-12 >= required ? 1 : 0;
    }

    return __shfl_sync(SUNOPT_B14F_FULL_MASK, survives, 0) != 0;
}

__global__ void sunopt_b14f_warp_precheck_kernel(
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
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ max_segments,
    uint32_t* __restrict__ max_work_segments
) {
    const size_t job = static_cast<size_t>(blockIdx.x);
    const size_t total_jobs = step_count * tank_count;
    if (job >= total_jobs) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;

    __shared__ SunoptB14CSegment buffer_a[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_b[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment buffer_c[SUNOPT_B14C_SCRATCH_SEGMENTS];
    __shared__ SunoptB14CSegment charge_segments[3];
    __shared__ int shared_count;
    __shared__ int shared_overflow;

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
    const double reserve = sunopt_b14c_clamp(dynamic_values[db + 1], 0.0, target);
    const size_t current_index = static_cast<size_t>(current_indices[step]);
    const size_t horizon = static_cast<size_t>(horizon_steps[step]);
    const double energy_cut = energy_cut_levels[job];

    if (lane == 0) {
        cut_results[job] = energy_cut;
        status[job] = 3u;
        max_segments[job] = 0u;
        max_work_segments[job] = 0u;
    }

    int supported =
        affine_supported &&
        retention > 0.0 &&
        reserve > SUNOPT_B14C_EPSILON &&
        data_length > 0 &&
        horizon > 0 &&
        horizon <= data_length;
    supported = __shfl_sync(
        SUNOPT_B14F_FULL_MASK, supported, 0);

    if (!supported) {
        return;
    }

    // B14F: replace the two serial Lane-0 48 h recurrence scans with an
    // exact ordered max-affine reduction. Every lane owns a contiguous
    // chunk (<=6 buckets for the production 192-step horizon), composes it
    // locally, then the warp combines the 32 chunks left-to-right.
    const bool normal_survives = sunopt_b14f_fixed_survives_warp(
        export_power, demand_energy, data_length, current_index, horizon,
        energy_cut, current, reserve, target, max_hp, hp_factor,
        retention, loss_intercept, lane);

    if (normal_survives) {
        if (lane == 0) {
            status[job] = 0u;
            max_segments[job] = 1u;
            max_work_segments[job] = 1u;
        }
        return;
    }

    const bool maximum_pv_survives = sunopt_b14f_fixed_survives_warp(
        export_power, demand_energy, data_length, current_index, horizon,
        0.0, current, reserve, target, max_hp, hp_factor,
        retention, loss_intercept, lane);

    if (!maximum_pv_survives) {
        if (lane == 0) {
            cut_results[job] = 0.0;
            status[job] = 1u;
            max_segments[job] = 1u;
            max_work_segments[job] = 1u;
        }
        return;
    }

    bool final_overflow = false;
    uint32_t maximum_final_segments = 0u;
    uint32_t maximum_work_segments = 0u;

    int charge_count = 0;
    if (lane == 0) {
        charge_count = sunopt_b14c_charge(
            export_power[current_index % data_length], energy_cut,
            max_hp, hp_factor, charge_segments);
        buffer_a[0] = SunoptB14CSegment{
            0.0,
            energy_cut,
            0.0,
            sunopt_b14c_clamp(current, 0.0, target),
        };
    }
    charge_count = __shfl_sync(
        SUNOPT_B14E_FULL_MASK, charge_count, 0);
    __syncwarp(SUNOPT_B14E_FULL_MASK);

    int stored_count = 1;
    int added_count = sunopt_b14e_add_charge_warp(
        buffer_a, stored_count,
        charge_segments, charge_count,
        energy_cut, buffer_b, lane,
        &shared_count, &shared_overflow);

    if (lane == 0) {
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count)
            ? maximum_work_segments : static_cast<uint32_t>(added_count);
    }
    if (shared_overflow != 0) {
        if (lane == 0) status[job] = 5u;
        return;
    }

    stored_count = sunopt_b14e_clip_capacity_warp(
        buffer_b, added_count, target, buffer_a,
        lane, &shared_count, &shared_overflow);

    if (lane == 0) {
        maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count)
            ? maximum_work_segments : static_cast<uint32_t>(stored_count);
        maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count)
            ? maximum_final_segments : static_cast<uint32_t>(stored_count);
        if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
    }
    if (shared_overflow != 0) {
        if (lane == 0) status[job] = 5u;
        return;
    }

    double feasible_hi = energy_cut;

    for (size_t offset = 1; offset < horizon; ++offset) {
        const size_t index = (current_index + offset) % data_length;
        const double demand = fmax(demand_energy[index], 0.0);

        for (int i = lane; i < stored_count; i += 32) {
            const SunoptB14CSegment source = buffer_a[i];
            buffer_b[i] = SunoptB14CSegment{
                source.lo,
                source.hi,
                retention * source.m,
                retention * source.b - loss_intercept - demand,
            };
        }
        __syncwarp(SUNOPT_B14E_FULL_MASK);

        if (lane == 0) {
            shared_count = sunopt_b14c_merge(buffer_b, stored_count);
        }
        __syncwarp(SUNOPT_B14E_FULL_MASK);
        const int before_count = shared_count;

        if (lane == 0) {
            maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(before_count)
                ? maximum_work_segments : static_cast<uint32_t>(before_count);
            maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(before_count)
                ? maximum_final_segments : static_cast<uint32_t>(before_count);
            if (before_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
        }

        feasible_hi = sunopt_b14e_truncate_feasible_warp(
            buffer_b, before_count,
            reserve - SUNOPT_B14C_EPSILON,
            feasible_hi, lane);

        if (feasible_hi <= 0.0) break;

        if (lane == 0) {
            charge_count = sunopt_b14c_charge(
                export_power[index], energy_cut,
                max_hp, hp_factor, charge_segments);
        }
        charge_count = __shfl_sync(
            SUNOPT_B14E_FULL_MASK, charge_count, 0);
        __syncwarp(SUNOPT_B14E_FULL_MASK);

        added_count = sunopt_b14e_add_charge_warp(
            buffer_b, before_count,
            charge_segments, charge_count,
            energy_cut, buffer_c, lane,
            &shared_count, &shared_overflow);

        if (lane == 0) {
            maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(added_count)
                ? maximum_work_segments : static_cast<uint32_t>(added_count);
        }
        if (shared_overflow != 0) {
            if (lane == 0) {
                status[job] = 5u;
                max_segments[job] = maximum_final_segments;
                max_work_segments[job] = maximum_work_segments;
            }
            return;
        }

        stored_count = sunopt_b14e_clip_capacity_warp(
            buffer_c, added_count, target, buffer_a,
            lane, &shared_count, &shared_overflow);

        if (lane == 0) {
            maximum_work_segments = maximum_work_segments > static_cast<uint32_t>(stored_count)
                ? maximum_work_segments : static_cast<uint32_t>(stored_count);
            maximum_final_segments = maximum_final_segments > static_cast<uint32_t>(stored_count)
                ? maximum_final_segments : static_cast<uint32_t>(stored_count);
            if (stored_count > static_cast<int>(SUNOPT_B14C_FINAL_SEGMENTS)) final_overflow = true;
        }
        if (shared_overflow != 0) {
            if (lane == 0) {
                status[job] = 5u;
                max_segments[job] = maximum_final_segments;
                max_work_segments[job] = maximum_work_segments;
            }
            return;
        }
    }

    if (lane == 0) {
        const double mathematical_cut = fmin(fmax(feasible_hi, 0.0), energy_cut);
        double lower = 0.0;
        double upper = energy_cut;

        for (int iteration = 0; iteration < 20; ++iteration) {
            const double middle = (lower + upper) * 0.5;
            if (middle <= mathematical_cut + 1.0e-12) lower = middle;
            else upper = middle;
        }

        cut_results[job] = lower;
        max_segments[job] = maximum_final_segments;
        max_work_segments[job] = maximum_work_segments;
        status[job] = final_overflow ? 4u : 2u;
    }
}


} // namespace

extern "C" int sunopt_cuda_run_b14f_warp_precheck_benchmark(
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
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_max_segments,
    uint32_t* host_max_work_segments,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results || !host_status ||
        !host_max_segments || !host_max_work_segments ||
        !out_upload_ms || !out_kernel_total_ms || !out_download_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0 || repetitions == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double *d_export=nullptr, *d_demand=nullptr, *d_tanks=nullptr,
           *d_dynamic=nullptr, *d_energy_cut=nullptr, *d_cut_results=nullptr;
    uint32_t *d_current=nullptr, *d_horizon=nullptr, *d_status=nullptr,
             *d_max_segments=nullptr, *d_max_work_segments=nullptr;

    const size_t jobs = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t step_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = jobs * 2 * sizeof(double);
    const size_t job_double_bytes = jobs * sizeof(double);
    const size_t job_u32_bytes = jobs * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
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

#define B14F_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14F_MALLOC(d_export, annual_bytes);
    B14F_MALLOC(d_demand, annual_bytes);
    B14F_MALLOC(d_tanks, tank_bytes);
    B14F_MALLOC(d_current, step_bytes);
    B14F_MALLOC(d_horizon, step_bytes);
    B14F_MALLOC(d_dynamic, dynamic_bytes);
    B14F_MALLOC(d_energy_cut, job_double_bytes);
    B14F_MALLOC(d_cut_results, job_double_bytes);
    B14F_MALLOC(d_status, job_u32_bytes);
    B14F_MALLOC(d_max_segments, job_u32_bytes);
    B14F_MALLOC(d_max_work_segments, job_u32_bytes);
#undef B14F_MALLOC

    const auto wall_start = Clock::now();
    const auto upload_start = Clock::now();

#define B14F_COPY_H2D(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14F_COPY_H2D(d_export, host_export_power, annual_bytes);
    B14F_COPY_H2D(d_demand, host_demand_energy, annual_bytes);
    B14F_COPY_H2D(d_tanks, host_tank_parameters, tank_bytes);
    B14F_COPY_H2D(d_current, host_current_indices, step_bytes);
    B14F_COPY_H2D(d_horizon, host_horizon_steps, step_bytes);
    B14F_COPY_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);
    B14F_COPY_H2D(d_energy_cut, host_energy_cut_levels, job_double_bytes);
#undef B14F_COPY_H2D

    const auto upload_stop = Clock::now();
    *out_upload_ms = std::chrono::duration<double,std::milli>(upload_stop-upload_start).count();

    err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    const unsigned int blocks = static_cast<unsigned int>(jobs);
    for (uint32_t r = 0; r < repetitions; ++r) {
        sunopt_b14f_warp_precheck_kernel<<<blocks,SUNOPT_B14F_THREADS,0,stream>>>(
            d_export,d_demand,data_length,d_tanks,tank_count,d_current,d_horizon,
            d_dynamic,d_energy_cut,step_count,d_cut_results,d_status,
            d_max_segments,d_max_work_segments
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
#define B14F_COPY_D2H(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)
    B14F_COPY_D2H(host_cut_results, d_cut_results, job_double_bytes);
    B14F_COPY_D2H(host_status, d_status, job_u32_bytes);
    B14F_COPY_D2H(host_max_segments, d_max_segments, job_u32_bytes);
    B14F_COPY_D2H(host_max_work_segments, d_max_work_segments, job_u32_bytes);
#undef B14F_COPY_D2H
    const auto download_stop = Clock::now();

    *out_download_ms = std::chrono::duration<double,std::milli>(download_stop-download_start).count();
    *out_wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_start).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14F_WARP_PRECHECK_HOST_END
// SUNOPT_BLACKWELL_B14F_WARP_PRECHECK_END


// SUNOPT_BLACKWELL_B14G_WARP_BINARY_BEGIN
namespace {

constexpr unsigned int SUNOPT_B14G_THREADS = 32;
constexpr unsigned int SUNOPT_B14G_FULL_MASK = 0xffffffffu;

/*
 * B14G control experiment: remove the PWA-envelope construction entirely.
 * It preserves Production semantics directly:
 *   1. test the requested energy cut,
 *   2. test cut=0,
 *   3. run exactly 20 lower-bound binary-search iterations.
 *
 * Every fixed-cut survival evaluation uses B14F's already validated ordered
 * max-affine warp reduction. Thus the only sequential dependency left is the
 * Production-required 20 binary-search decisions, not the 192-bucket horizon.
 */
__global__ void sunopt_b14g_warp_binary_kernel(
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
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ max_segments,
    uint32_t* __restrict__ max_work_segments
) {
    const size_t job = static_cast<size_t>(blockIdx.x);
    const size_t total_jobs = step_count * tank_count;
    if (job >= total_jobs) return;

    const int lane = static_cast<int>(threadIdx.x) & 31;
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
    const double reserve = sunopt_b14c_clamp(dynamic_values[db + 1], 0.0, target);
    const size_t current_index = static_cast<size_t>(current_indices[step]);
    const size_t horizon = static_cast<size_t>(horizon_steps[step]);
    const double energy_cut = energy_cut_levels[job];

    if (lane == 0) {
        cut_results[job] = energy_cut;
        status[job] = 3u;
        max_segments[job] = 0u;
        max_work_segments[job] = 0u;
    }

    int supported =
        affine_supported &&
        retention > 0.0 &&
        reserve > SUNOPT_B14C_EPSILON &&
        data_length > 0 &&
        horizon > 0 &&
        horizon <= data_length;
    supported = __shfl_sync(SUNOPT_B14G_FULL_MASK, supported, 0);

    if (!supported) return;

    const bool normal_survives = sunopt_b14f_fixed_survives_warp(
        export_power, demand_energy, data_length, current_index, horizon,
        energy_cut, current, reserve, target, max_hp, hp_factor,
        retention, loss_intercept, lane);

    if (normal_survives) {
        if (lane == 0) {
            status[job] = 0u;
            max_segments[job] = 1u;
            max_work_segments[job] = 1u;
        }
        return;
    }

    const bool maximum_pv_survives = sunopt_b14f_fixed_survives_warp(
        export_power, demand_energy, data_length, current_index, horizon,
        0.0, current, reserve, target, max_hp, hp_factor,
        retention, loss_intercept, lane);

    if (!maximum_pv_survives) {
        if (lane == 0) {
            cut_results[job] = 0.0;
            status[job] = 1u;
            max_segments[job] = 1u;
            max_work_segments[job] = 1u;
        }
        return;
    }

    double lower = 0.0;
    double upper = energy_cut;

    // Production-equivalent lower-bound binary search: exactly 20 iterations.
    for (int iteration = 0; iteration < 20; ++iteration) {
        const double middle = (lower + upper) * 0.5;
        const bool survives = sunopt_b14f_fixed_survives_warp(
            export_power, demand_energy, data_length, current_index, horizon,
            middle, current, reserve, target, max_hp, hp_factor,
            retention, loss_intercept, lane);

        if (lane == 0) {
            if (survives) lower = middle;
            else upper = middle;
        }

        lower = __shfl_sync(SUNOPT_B14G_FULL_MASK, lower, 0);
        upper = __shfl_sync(SUNOPT_B14G_FULL_MASK, upper, 0);
    }

    if (lane == 0) {
        cut_results[job] = lower;
        status[job] = 2u;
        max_segments[job] = 1u;
        max_work_segments[job] = 1u;
    }
}

} // namespace

extern "C" int sunopt_cuda_run_b14g_warp_binary_benchmark(
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
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_max_segments,
    uint32_t* host_max_work_segments,
    double* out_upload_ms,
    double* out_kernel_total_ms,
    double* out_download_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results || !host_status ||
        !host_max_segments || !host_max_work_segments ||
        !out_upload_ms || !out_kernel_total_ms || !out_download_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0 || repetitions == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double *d_export=nullptr, *d_demand=nullptr, *d_tanks=nullptr,
           *d_dynamic=nullptr, *d_energy_cut=nullptr, *d_cut_results=nullptr;
    uint32_t *d_current=nullptr, *d_horizon=nullptr, *d_status=nullptr,
             *d_max_segments=nullptr, *d_max_work_segments=nullptr;

    const size_t jobs = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t step_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = jobs * 2 * sizeof(double);
    const size_t job_double_bytes = jobs * sizeof(double);
    const size_t job_u32_bytes = jobs * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
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

#define B14G_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14G_MALLOC(d_export, annual_bytes);
    B14G_MALLOC(d_demand, annual_bytes);
    B14G_MALLOC(d_tanks, tank_bytes);
    B14G_MALLOC(d_current, step_bytes);
    B14G_MALLOC(d_horizon, step_bytes);
    B14G_MALLOC(d_dynamic, dynamic_bytes);
    B14G_MALLOC(d_energy_cut, job_double_bytes);
    B14G_MALLOC(d_cut_results, job_double_bytes);
    B14G_MALLOC(d_status, job_u32_bytes);
    B14G_MALLOC(d_max_segments, job_u32_bytes);
    B14G_MALLOC(d_max_work_segments, job_u32_bytes);
#undef B14G_MALLOC

    const auto wall_start = Clock::now();
    const auto upload_start = Clock::now();

#define B14G_COPY_H2D(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyHostToDevice); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14G_COPY_H2D(d_export, host_export_power, annual_bytes);
    B14G_COPY_H2D(d_demand, host_demand_energy, annual_bytes);
    B14G_COPY_H2D(d_tanks, host_tank_parameters, tank_bytes);
    B14G_COPY_H2D(d_current, host_current_indices, step_bytes);
    B14G_COPY_H2D(d_horizon, host_horizon_steps, step_bytes);
    B14G_COPY_H2D(d_dynamic, host_dynamic_values, dynamic_bytes);
    B14G_COPY_H2D(d_energy_cut, host_energy_cut_levels, job_double_bytes);
#undef B14G_COPY_H2D

    const auto upload_stop = Clock::now();
    *out_upload_ms = std::chrono::duration<double,std::milli>(upload_stop-upload_start).count();

    err = cudaEventRecord(start_event, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    const unsigned int blocks = static_cast<unsigned int>(jobs);
    for (uint32_t r = 0; r < repetitions; ++r) {
        sunopt_b14g_warp_binary_kernel<<<blocks,SUNOPT_B14G_THREADS,0,stream>>>(
            d_export,d_demand,data_length,d_tanks,tank_count,d_current,d_horizon,
            d_dynamic,d_energy_cut,step_count,d_cut_results,d_status,
            d_max_segments,d_max_work_segments
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
#define B14G_COPY_D2H(dst, src, bytes) do { \
    err = cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)
    B14G_COPY_D2H(host_cut_results, d_cut_results, job_double_bytes);
    B14G_COPY_D2H(host_status, d_status, job_u32_bytes);
    B14G_COPY_D2H(host_max_segments, d_max_segments, job_u32_bytes);
    B14G_COPY_D2H(host_max_work_segments, d_max_work_segments, job_u32_bytes);
#undef B14G_COPY_D2H
    const auto download_stop = Clock::now();

    *out_download_ms = std::chrono::duration<double,std::milli>(download_stop-download_start).count();
    *out_wall_ms = std::chrono::duration<double,std::milli>(Clock::now()-wall_start).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14G_WARP_BINARY_END

// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_BEGIN
extern "C" int sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark(
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
    double* host_cut_results,
    double* out_static_h2d_ms,
    double* out_dynamic_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results ||
        !out_static_h2d_ms || !out_dynamic_h2d_ms || !out_kernel_ms ||
        !out_result_d2h_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    double *d_export = nullptr;
    double *d_demand = nullptr;
    double *d_tanks = nullptr;
    double *d_dynamic = nullptr;
    double *d_energy_cut = nullptr;
    double *d_cut_results = nullptr;

    uint32_t *d_current = nullptr;
    uint32_t *d_horizon = nullptr;
    uint32_t *d_status = nullptr;
    uint32_t *d_max_segments = nullptr;
    uint32_t *d_max_work_segments = nullptr;

    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t dynamic_step_bytes = tank_count * 2 * sizeof(double);
    const size_t cut_step_bytes = tank_count * sizeof(double);
    const size_t u32_step_bytes = tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
        if (d_cut_results) cudaFree(d_cut_results);
        if (d_energy_cut) cudaFree(d_energy_cut);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (kernel_stop) cudaEventDestroy(kernel_stop);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (stream) cudaStreamDestroy(stream);
    };

    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaEventCreate(&kernel_start);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaEventCreate(&kernel_stop);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

#define B14I1_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14I1_MALLOC(d_export, annual_bytes);
    B14I1_MALLOC(d_demand, annual_bytes);
    B14I1_MALLOC(d_tanks, tank_bytes);
    B14I1_MALLOC(d_current, sizeof(uint32_t));
    B14I1_MALLOC(d_horizon, sizeof(uint32_t));
    B14I1_MALLOC(d_dynamic, dynamic_step_bytes);
    B14I1_MALLOC(d_energy_cut, cut_step_bytes);
    B14I1_MALLOC(d_cut_results, cut_step_bytes);
    B14I1_MALLOC(d_status, u32_step_bytes);
    B14I1_MALLOC(d_max_segments, u32_step_bytes);
    B14I1_MALLOC(d_max_work_segments, u32_step_bytes);

#undef B14I1_MALLOC

    const auto wall_start = Clock::now();
    const auto static_start = Clock::now();

    err = cudaMemcpyAsync(
        d_export, host_export_power, annual_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaMemcpyAsync(
        d_demand, host_demand_energy, annual_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaMemcpyAsync(
        d_tanks, host_tank_parameters, tank_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    *out_static_h2d_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - static_start
        ).count();

    double dynamic_h2d_ms = 0.0;
    double result_d2h_ms = 0.0;
    double kernel_ms_total = 0.0;

    for (size_t step = 0; step < step_count; ++step) {
        const auto h2d_start = Clock::now();

        err = cudaMemcpyAsync(
            d_current,
            host_current_indices + step,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaMemcpyAsync(
            d_horizon,
            host_horizon_steps + step,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaMemcpyAsync(
            d_dynamic,
            host_dynamic_values + step * tank_count * 2,
            dynamic_step_bytes,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaMemcpyAsync(
            d_energy_cut,
            host_energy_cut_levels + step * tank_count,
            cut_step_bytes,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        dynamic_h2d_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - h2d_start
            ).count();

        err = cudaEventRecord(kernel_start, stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        sunopt_b14g_warp_binary_kernel<<<
            static_cast<unsigned int>(tank_count),
            SUNOPT_B14G_THREADS,
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
            d_energy_cut,
            1,
            d_cut_results,
            d_status,
            d_max_segments,
            d_max_work_segments
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaEventRecord(kernel_stop, stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        const auto d2h_start = Clock::now();

        err = cudaMemcpyAsync(
            host_cut_results + step * tank_count,
            d_cut_results,
            cut_step_bytes,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        result_d2h_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - d2h_start
            ).count();

        float step_kernel_ms = 0.0f;
        err = cudaEventElapsedTime(
            &step_kernel_ms,
            kernel_start,
            kernel_stop
        );
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        kernel_ms_total += static_cast<double>(step_kernel_ms);
    }

    *out_dynamic_h2d_ms = dynamic_h2d_ms;
    *out_kernel_ms = kernel_ms_total;
    *out_result_d2h_ms = result_d2h_ms;
    *out_wall_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - wall_start
        ).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_END

// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_BEGIN
namespace {

template <unsigned int BLOCK_THREADS>
__device__ __forceinline__ bool
sunopt_b14j_fixed_survives_block(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    size_t current_index,
    size_t horizon,
    double cut,
    double current_storage_energy,
    double reserve,
    double target,
    double max_hp,
    double hp_factor,
    double retention,
    double loss_intercept,
    SunoptB14FTransform* shared_transform,
    int* shared_survives
) {
    const unsigned int tid = static_cast<unsigned int>(threadIdx.x);
    const size_t future_count = horizon > 1 ? horizon - 1 : 0;
    const size_t chunk_size =
        (future_count + static_cast<size_t>(BLOCK_THREADS) - 1u) /
        static_cast<size_t>(BLOCK_THREADS);

    const size_t chunk_begin =
        1u + static_cast<size_t>(tid) * chunk_size;

    const size_t chunk_end =
        chunk_size == 0
            ? 1u
            : ((chunk_begin + chunk_size) < horizon
                ? (chunk_begin + chunk_size)
                : horizon);

    SunoptB14FTransform local =
        sunopt_b14f_identity();

    const double inverse_retention =
        1.0 / retention;

    for (
        size_t offset = chunk_begin;
        offset < chunk_end && offset < horizon;
        ++offset
    ) {
        const size_t index =
            (current_index + offset) % data_length;

        const double demand =
            fmax(demand_energy[index], 0.0);

        const double raw =
            export_power[index] - cut;

        const double hp =
            fmin(max_hp, fmax(0.0, raw));

        const double charge =
            hp * hp_factor;

        SunoptB14FTransform step{};
        step.a = inverse_retention;
        step.b =
            (loss_intercept + demand - charge) *
            inverse_retention;
        step.c =
            (
                loss_intercept +
                demand +
                reserve -
                SUNOPT_B14C_EPSILON
            ) *
            inverse_retention;
        step.max_input = target;

        local =
            sunopt_b14f_compose(
                local,
                step
            );
    }

    shared_transform[tid] = local;
    __syncthreads();

    for (
        unsigned int stride = 1u;
        stride < BLOCK_THREADS;
        stride <<= 1u
    ) {
        const unsigned int group =
            stride << 1u;

        if (
            (tid & (group - 1u)) == 0u &&
            tid + stride < BLOCK_THREADS
        ) {
            shared_transform[tid] =
                sunopt_b14f_compose(
                    shared_transform[tid],
                    shared_transform[tid + stride]
                );
        }

        __syncthreads();
    }

    if (tid == 0u) {
        const SunoptB14FTransform summary =
            shared_transform[0];

        bool feasible =
            0.0 <= summary.max_input + 1.0e-12;

        const double required =
            fmax(summary.c, summary.b);

        if (required > target + 1.0e-9) {
            feasible = false;
        }

        const double raw0 =
            export_power[current_index % data_length] -
            cut;

        const double hp0 =
            fmin(max_hp, fmax(0.0, raw0));

        const double charge0 =
            hp0 * hp_factor;

        const double current =
            sunopt_b14c_clamp(
                current_storage_energy,
                0.0,
                target
            );

        const double available =
            fmin(
                target,
                current + charge0
            );

        *shared_survives =
            feasible &&
            available + 1.0e-12 >= required
                ? 1
                : 0;
    }

    __syncthreads();
    return *shared_survives != 0;
}

template <unsigned int BLOCK_THREADS>
__global__ void
sunopt_b14j_small_batch_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    uint32_t current_index_u32,
    uint32_t horizon_u32,
    const double* __restrict__ dynamic_values,
    const double* __restrict__ energy_cut_levels,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status
) {
    const size_t tank =
        static_cast<size_t>(blockIdx.x);

    if (tank >= tank_count) {
        return;
    }

    const unsigned int tid =
        static_cast<unsigned int>(threadIdx.x);

    __shared__ SunoptB14FTransform
        shared_transform[BLOCK_THREADS];

    __shared__ int shared_survives;
    __shared__ double shared_lower;
    __shared__ double shared_upper;

    const size_t tb = tank * 6u;
    const size_t db = tank * 2u;

    const double target =
        tank_parameters[tb + 0];

    const double max_hp =
        tank_parameters[tb + 1];

    const double hp_factor =
        tank_parameters[tb + 2];

    const double retention =
        tank_parameters[tb + 3];

    const double loss_intercept =
        tank_parameters[tb + 4];

    const bool affine_supported =
        tank_parameters[tb + 5] > 0.5;

    const double current =
        dynamic_values[db + 0];

    const double reserve =
        sunopt_b14c_clamp(
            dynamic_values[db + 1],
            0.0,
            target
        );

    const size_t current_index =
        static_cast<size_t>(current_index_u32);

    const size_t horizon =
        static_cast<size_t>(horizon_u32);

    const double energy_cut =
        energy_cut_levels[tank];

    if (tid == 0u) {
        cut_results[tank] = energy_cut;
        status[tank] = 3u;
    }

    __syncthreads();

    const bool supported =
        affine_supported &&
        retention > 0.0 &&
        reserve > SUNOPT_B14C_EPSILON &&
        data_length > 0 &&
        horizon > 0 &&
        horizon <= data_length;

    if (!supported) {
        return;
    }

    const bool normal_survives =
        sunopt_b14j_fixed_survives_block<BLOCK_THREADS>(
            export_power,
            demand_energy,
            data_length,
            current_index,
            horizon,
            energy_cut,
            current,
            reserve,
            target,
            max_hp,
            hp_factor,
            retention,
            loss_intercept,
            shared_transform,
            &shared_survives
        );

    if (normal_survives) {
        if (tid == 0u) {
            status[tank] = 0u;
        }
        return;
    }

    const bool maximum_pv_survives =
        sunopt_b14j_fixed_survives_block<BLOCK_THREADS>(
            export_power,
            demand_energy,
            data_length,
            current_index,
            horizon,
            0.0,
            current,
            reserve,
            target,
            max_hp,
            hp_factor,
            retention,
            loss_intercept,
            shared_transform,
            &shared_survives
        );

    if (!maximum_pv_survives) {
        if (tid == 0u) {
            cut_results[tank] = 0.0;
            status[tank] = 1u;
        }
        return;
    }

    if (tid == 0u) {
        shared_lower = 0.0;
        shared_upper = energy_cut;
    }

    __syncthreads();

    for (
        int iteration = 0;
        iteration < 20;
        ++iteration
    ) {
        const double middle =
            (shared_lower + shared_upper) * 0.5;

        const bool survives =
            sunopt_b14j_fixed_survives_block<BLOCK_THREADS>(
                export_power,
                demand_energy,
                data_length,
                current_index,
                horizon,
                middle,
                current,
                reserve,
                target,
                max_hp,
                hp_factor,
                retention,
                loss_intercept,
                shared_transform,
                &shared_survives
            );

        if (tid == 0u) {
            if (survives) {
                shared_lower = middle;
            } else {
                shared_upper = middle;
            }
        }

        __syncthreads();
    }

    if (tid == 0u) {
        cut_results[tank] = shared_lower;
        status[tank] = 2u;
    }
}

template <unsigned int BLOCK_THREADS>
cudaError_t sunopt_b14j_launch(
    cudaStream_t stream,
    const double* d_export,
    const double* d_demand,
    size_t data_length,
    const double* d_tanks,
    size_t tank_count,
    uint32_t current_index,
    uint32_t horizon,
    const double* d_dynamic,
    const double* d_energy_cut,
    double* d_cut_results,
    uint32_t* d_status
) {
    sunopt_b14j_small_batch_kernel<BLOCK_THREADS>
        <<<static_cast<unsigned int>(tank_count),
           BLOCK_THREADS,
           0,
           stream>>>(
            d_export,
            d_demand,
            data_length,
            d_tanks,
            tank_count,
            current_index,
            horizon,
            d_dynamic,
            d_energy_cut,
            d_cut_results,
            d_status
        );

    return cudaGetLastError();
}

} // namespace

extern "C" int
sunopt_cuda_run_b14j_small_batch_benchmark(
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
    uint32_t block_threads,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_dynamic_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (
        !host_export_power ||
        !host_demand_energy ||
        !host_tank_parameters ||
        !host_current_indices ||
        !host_horizon_steps ||
        !host_dynamic_values ||
        !host_energy_cut_levels ||
        !host_cut_results ||
        !host_status ||
        !out_static_h2d_ms ||
        !out_dynamic_enqueue_ms ||
        !out_kernel_ms ||
        !out_result_wait_ms ||
        !out_wall_ms ||
        data_length == 0 ||
        tank_count == 0 ||
        step_count == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    if (
        block_threads != 32u &&
        block_threads != 64u &&
        block_threads != 128u &&
        block_threads != 256u &&
        block_threads != 512u
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err =
        cudaSetDevice(device_id);

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_dynamic = nullptr;
    double* d_energy_cut = nullptr;
    double* d_cut_results = nullptr;
    uint32_t* d_status = nullptr;

    const size_t annual_bytes =
        data_length * sizeof(double);

    const size_t tank_bytes =
        tank_count * 6u * sizeof(double);

    const size_t dynamic_step_bytes =
        tank_count * 2u * sizeof(double);

    const size_t cut_step_bytes =
        tank_count * sizeof(double);

    const size_t status_step_bytes =
        tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_status) cudaFree(d_status);
        if (d_cut_results) cudaFree(d_cut_results);
        if (d_energy_cut) cudaFree(d_energy_cut);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);

        if (kernel_stop) {
            cudaEventDestroy(kernel_stop);
        }

        if (kernel_start) {
            cudaEventDestroy(kernel_start);
        }

        if (stream) {
            cudaStreamDestroy(stream);
        }
    };

    err =
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaEventCreate(&kernel_start);
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaEventCreate(&kernel_stop);
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

#define B14J_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14J_MALLOC(d_export, annual_bytes);
    B14J_MALLOC(d_demand, annual_bytes);
    B14J_MALLOC(d_tanks, tank_bytes);
    B14J_MALLOC(d_dynamic, dynamic_step_bytes);
    B14J_MALLOC(d_energy_cut, cut_step_bytes);
    B14J_MALLOC(d_cut_results, cut_step_bytes);
    B14J_MALLOC(d_status, status_step_bytes);

#undef B14J_MALLOC

    const auto wall_start = Clock::now();
    const auto static_start = Clock::now();

    err = cudaMemcpyAsync(
        d_export,
        host_export_power,
        annual_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        d_demand,
        host_demand_energy,
        annual_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(
        d_tanks,
        host_tank_parameters,
        tank_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    *out_static_h2d_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - static_start
        ).count();

    double dynamic_enqueue_ms = 0.0;
    double kernel_ms_total = 0.0;
    double result_wait_ms = 0.0;

    for (
        size_t step = 0;
        step < step_count;
        ++step
    ) {
        const auto enqueue_start =
            Clock::now();

        err = cudaMemcpyAsync(
            d_dynamic,
            host_dynamic_values +
                step * tank_count * 2u,
            dynamic_step_bytes,
            cudaMemcpyHostToDevice,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = cudaMemcpyAsync(
            d_energy_cut,
            host_energy_cut_levels +
                step * tank_count,
            cut_step_bytes,
            cudaMemcpyHostToDevice,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        dynamic_enqueue_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - enqueue_start
            ).count();

        err =
            cudaEventRecord(
                kernel_start,
                stream
            );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        switch (block_threads) {
            case 32u:
                err = sunopt_b14j_launch<32u>(
                    stream,
                    d_export,
                    d_demand,
                    data_length,
                    d_tanks,
                    tank_count,
                    host_current_indices[step],
                    host_horizon_steps[step],
                    d_dynamic,
                    d_energy_cut,
                    d_cut_results,
                    d_status
                );
                break;

            case 64u:
                err = sunopt_b14j_launch<64u>(
                    stream,
                    d_export,
                    d_demand,
                    data_length,
                    d_tanks,
                    tank_count,
                    host_current_indices[step],
                    host_horizon_steps[step],
                    d_dynamic,
                    d_energy_cut,
                    d_cut_results,
                    d_status
                );
                break;

            case 128u:
                err = sunopt_b14j_launch<128u>(
                    stream,
                    d_export,
                    d_demand,
                    data_length,
                    d_tanks,
                    tank_count,
                    host_current_indices[step],
                    host_horizon_steps[step],
                    d_dynamic,
                    d_energy_cut,
                    d_cut_results,
                    d_status
                );
                break;

            case 256u:
                err = sunopt_b14j_launch<256u>(
                    stream,
                    d_export,
                    d_demand,
                    data_length,
                    d_tanks,
                    tank_count,
                    host_current_indices[step],
                    host_horizon_steps[step],
                    d_dynamic,
                    d_energy_cut,
                    d_cut_results,
                    d_status
                );
                break;

            case 512u:
                err = sunopt_b14j_launch<512u>(
                    stream,
                    d_export,
                    d_demand,
                    data_length,
                    d_tanks,
                    tank_count,
                    host_current_indices[step],
                    host_horizon_steps[step],
                    d_dynamic,
                    d_energy_cut,
                    d_cut_results,
                    d_status
                );
                break;

            default:
                err = cudaErrorInvalidValue;
                break;
        }

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err =
            cudaEventRecord(
                kernel_stop,
                stream
            );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        const auto wait_start =
            Clock::now();

        err = cudaMemcpyAsync(
            host_cut_results +
                step * tank_count,
            d_cut_results,
            cut_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err = cudaMemcpyAsync(
            host_status +
                step * tank_count,
            d_status,
            status_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        err =
            cudaStreamSynchronize(stream);

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        result_wait_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - wait_start
            ).count();

        float step_kernel_ms = 0.0f;

        err =
            cudaEventElapsedTime(
                &step_kernel_ms,
                kernel_start,
                kernel_stop
            );

        if (err != cudaSuccess) {
            cleanup();
            return static_cast<int>(err);
        }

        kernel_ms_total +=
            static_cast<double>(
                step_kernel_ms
            );
    }

    *out_dynamic_enqueue_ms =
        dynamic_enqueue_ms;

    *out_kernel_ms =
        kernel_ms_total;

    *out_result_wait_ms =
        result_wait_ms;

    *out_wall_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - wall_start
        ).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_END

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


// SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN
/*
 * Exact serial digital-HP survival evaluator.
 *
 * This deliberately does not use the max-affine reduction. It mirrors the
 * scalar digital reference step-by-step and therefore handles reserve == 0
 * and the empty-storage clipping boundary exactly.
 */
__device__ __forceinline__ bool
sunopt_digital_survives_serial_exact(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    const double* tank_parameters,
    size_t tank_index,
    size_t current_index,
    size_t horizon,
    double current_storage_energy,
    double minimum_reserve_energy,
    double cut
) {
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

    const double reserve =
        fmin(
            target,
            fmax(
                0.0,
                minimum_reserve_energy
            )
        );

    double stored =
        fmin(
            target,
            fmax(
                0.0,
                current_storage_energy
            )
        );

    for (
        size_t offset = 0;
        offset < horizon;
        ++offset
    ) {
        const size_t index =
            (
                current_index +
                offset
            ) %
            data_length;

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

        /* Current demand/loss already happened before currentStorageEnergy. */
        if (
            offset == 0
        ) {
            stored =
                fmin(
                    target,
                    stored +
                    charge
                );

            continue;
        }

        /*
         * Mirrors digital_survives_scalar():
         * 1. passive loss, clipped at zero and at stored energy
         * 2. thermal demand
         * 3. reserve check BEFORE forecast HP charging
         * 4. all-or-nothing HP charge
         */
        const double requested_loss =
            fmax(
                0.0,
                (
                    1.0 -
                    retention
                ) *
                stored +
                loss_intercept
            );

        const double heat_loss =
            fmin(
                requested_loss,
                stored
            );

        const double after_loss =
            fmax(
                0.0,
                stored -
                heat_loss
            );

        const double demand =
            fmax(
                0.0,
                demand_energy[
                    index
                ]
            );

        const double discharged =
            fmin(
                demand,
                after_loss
            );

        const double before_charge =
            after_loss -
            discharged;

        const double unmet =
            fmax(
                0.0,
                demand -
                discharged
            );

        if (
            unmet >
            SUNOPT_DIGITAL_EPSILON
        ) {
            return false;
        }

        if (
            reserve -
            before_charge >
            SUNOPT_DIGITAL_EPSILON
        ) {
            return false;
        }

        stored =
            fmin(
                target,
                before_charge +
                charge
            );
    }

    return true;
}
// SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_END


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
        if (
            !shared_supported
        ) {
            /*
             * Exact GPU fallback for the max-affine edge cases.
             * The breakpoints are already sorted in shared memory.
             */
            eval_count =
                1;

            const bool serial_normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                serial_normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1;

                const bool serial_maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !serial_maxpv_survives
                ) {
                    final_status =
                        1;

                    final_cut =
                        0.0;
                } else {
                    final_status =
                        2;

                    double best_cut =
                        0.0;

                    int left =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            valid_count
                        );

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    for (
                        uint32_t iteration = 0;
                        iteration <
                            search_iterations &&
                        left <=
                            right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                breakpoints[
                                    middle
                                ]
                            );

                        eval_count +=
                            1;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best_cut =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_cut =
                        best_cut;
                }
            }
        } else {
            eval_count +=
                1;

            if (
                normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            }
        }
    }

    __syncthreads();

    /*
     * Max-affine-unsupported jobs are already fully solved by the serial GPU
     * branch. No status=3 leaves this kernel.
     */
    if (
        final_status ==
            0 ||
        !shared_supported
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


// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN
extern "C" int sunopt_cuda_run_digital_lockstep_candidate(
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
    double* out_static_h2d_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
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
        out_static_h2d_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        repetitions == 0 ||
        search_iterations == 0
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

    cudaEvent_t gpu_start =
        nullptr;

    cudaEvent_t gpu_stop =
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

    const size_t annual_bytes =
        data_length *
        sizeof(double);

    const size_t tank_bytes =
        tank_count *
        7 *
        sizeof(double);

    const size_t dynamic_step_bytes =
        tank_count *
        3 *
        sizeof(double);

    const size_t cut_step_bytes =
        tank_count *
        sizeof(double);

    const size_t u32_step_bytes =
        tank_count *
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
            if (gpu_stop) cudaEventDestroy(gpu_stop);
            if (gpu_start) cudaEventDestroy(gpu_start);
            if (stream) cudaStreamDestroy(stream);
        };

#define SUNOPT_LOCKSTEP_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_LOCKSTEP_MALLOC(ptr, bytes) \
    SUNOPT_LOCKSTEP_CHECK( \
        cudaMalloc( \
            reinterpret_cast<void**>(&(ptr)), \
            (bytes) \
        ) \
    )

    SUNOPT_LOCKSTEP_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventCreate(
            &gpu_start
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventCreate(
            &gpu_stop
        )
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_export,
        annual_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_demand,
        annual_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_tanks,
        tank_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_current,
        sizeof(uint32_t)
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_horizon,
        sizeof(uint32_t)
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_dynamic,
        dynamic_step_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_cuts,
        cut_step_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_status,
        u32_step_bytes
    );

    SUNOPT_LOCKSTEP_MALLOC(
        d_evaluations,
        u32_step_bytes
    );

#undef SUNOPT_LOCKSTEP_MALLOC

    const auto static_h2d_start =
        Clock::now();

    SUNOPT_LOCKSTEP_CHECK(
        cudaMemcpy(
            d_export,
            host_export_power,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaMemcpy(
            d_demand,
            host_demand_energy,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaMemcpy(
            d_tanks,
            host_tank_parameters,
            tank_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const auto static_h2d_stop =
        Clock::now();

    *out_static_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            static_h2d_stop -
            static_h2d_start
        ).count();

    double host_wait_ms =
        0.0;

    const auto wall_start =
        Clock::now();

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventRecord(
            gpu_start,
            stream
        )
    );

    const unsigned int blocks =
        static_cast<unsigned int>(
            tank_count
        );

    for (
        uint32_t repetition = 0;
        repetition <
            repetitions;
        ++repetition
    ) {
        for (
            size_t step = 0;
            step <
                step_count;
            ++step
        ) {
            const size_t dynamic_offset =
                step *
                tank_count *
                3;

            const size_t result_offset =
                step *
                tank_count;

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    d_current,
                    host_current_indices +
                        step,
                    sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    d_horizon,
                    host_horizon_steps +
                        step,
                    sizeof(uint32_t),
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    d_dynamic,
                    host_dynamic_values +
                        dynamic_offset,
                    dynamic_step_bytes,
                    cudaMemcpyHostToDevice,
                    stream
                )
            );

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
                1,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaGetLastError()
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    host_cut_results +
                        result_offset,
                    d_cuts,
                    cut_step_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    host_status +
                        result_offset,
                    d_status,
                    u32_step_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                )
            );

            SUNOPT_LOCKSTEP_CHECK(
                cudaMemcpyAsync(
                    host_survival_evaluations +
                        result_offset,
                    d_evaluations,
                    u32_step_bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                )
            );

            const auto wait_start =
                Clock::now();

            SUNOPT_LOCKSTEP_CHECK(
                cudaStreamSynchronize(
                    stream
                )
            );

            const auto wait_stop =
                Clock::now();

            host_wait_ms +=
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    wait_stop -
                    wait_start
                ).count();
        }
    }

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventRecord(
            gpu_stop,
            stream
        )
    );

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventSynchronize(
            gpu_stop
        )
    );

    float gpu_total_ms =
        0.0f;

    SUNOPT_LOCKSTEP_CHECK(
        cudaEventElapsedTime(
            &gpu_total_ms,
            gpu_start,
            gpu_stop
        )
    );

    const auto wall_stop =
        Clock::now();

    *out_gpu_total_ms =
        static_cast<double>(
            gpu_total_ms
        );

    *out_host_wait_ms =
        host_wait_ms;

    *out_wall_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            wall_stop -
            wall_start
        ).count();

#undef SUNOPT_LOCKSTEP_CHECK

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_END


// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_BEGIN
extern "C" int sunopt_cuda_run_digital_lockstep_graph_candidate(
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
    double* out_static_h2d_ms,
    double* out_graph_build_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

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
        out_static_h2d_ms == nullptr ||
        out_graph_build_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        search_iterations == 0 ||
        repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;

    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    uint32_t* h_current = nullptr;
    uint32_t* h_horizon = nullptr;
    double* h_dynamic = nullptr;
    double* h_cuts = nullptr;
    uint32_t* h_status = nullptr;
    uint32_t* h_evaluations = nullptr;

    const size_t annual_bytes =
        data_length * sizeof(double);

    const size_t tank_bytes =
        tank_count * 7 * sizeof(double);

    const size_t dynamic_step_count =
        tank_count * 3;

    const size_t dynamic_step_bytes =
        dynamic_step_count * sizeof(double);

    const size_t cut_step_bytes =
        tank_count * sizeof(double);

    const size_t u32_step_bytes =
        tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);

        if (h_evaluations) cudaFreeHost(h_evaluations);
        if (h_status) cudaFreeHost(h_status);
        if (h_cuts) cudaFreeHost(h_cuts);
        if (h_dynamic) cudaFreeHost(h_dynamic);
        if (h_horizon) cudaFreeHost(h_horizon);
        if (h_current) cudaFreeHost(h_current);

        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);

        if (gpu_stop) cudaEventDestroy(gpu_stop);
        if (gpu_start) cudaEventDestroy(gpu_start);
        if (stream) cudaStreamDestroy(stream);
    };

#define SUNOPT_GRAPH_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_GRAPH_MALLOC(ptr, bytes) \
    SUNOPT_GRAPH_CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)))

#define SUNOPT_GRAPH_HOST_MALLOC(ptr, bytes) \
    SUNOPT_GRAPH_CHECK(cudaMallocHost(reinterpret_cast<void**>(&(ptr)), (bytes)))

    SUNOPT_GRAPH_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_GRAPH_CHECK(cudaEventCreate(&gpu_start));
    SUNOPT_GRAPH_CHECK(cudaEventCreate(&gpu_stop));

    SUNOPT_GRAPH_MALLOC(d_export, annual_bytes);
    SUNOPT_GRAPH_MALLOC(d_demand, annual_bytes);
    SUNOPT_GRAPH_MALLOC(d_tanks, tank_bytes);
    SUNOPT_GRAPH_MALLOC(d_current, sizeof(uint32_t));
    SUNOPT_GRAPH_MALLOC(d_horizon, sizeof(uint32_t));
    SUNOPT_GRAPH_MALLOC(d_dynamic, dynamic_step_bytes);
    SUNOPT_GRAPH_MALLOC(d_cuts, cut_step_bytes);
    SUNOPT_GRAPH_MALLOC(d_status, u32_step_bytes);
    SUNOPT_GRAPH_MALLOC(d_evaluations, u32_step_bytes);

    SUNOPT_GRAPH_HOST_MALLOC(h_current, sizeof(uint32_t));
    SUNOPT_GRAPH_HOST_MALLOC(h_horizon, sizeof(uint32_t));
    SUNOPT_GRAPH_HOST_MALLOC(h_dynamic, dynamic_step_bytes);
    SUNOPT_GRAPH_HOST_MALLOC(h_cuts, cut_step_bytes);
    SUNOPT_GRAPH_HOST_MALLOC(h_status, u32_step_bytes);
    SUNOPT_GRAPH_HOST_MALLOC(h_evaluations, u32_step_bytes);

#undef SUNOPT_GRAPH_MALLOC
#undef SUNOPT_GRAPH_HOST_MALLOC

    const auto static_h2d_start = Clock::now();

    SUNOPT_GRAPH_CHECK(
        cudaMemcpy(
            d_export,
            host_export_power,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpy(
            d_demand,
            host_demand_energy,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpy(
            d_tanks,
            host_tank_parameters,
            tank_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const auto static_h2d_stop = Clock::now();

    *out_static_h2d_ms =
        std::chrono::duration<double, std::milli>(
            static_h2d_stop - static_h2d_start
        ).count();

    const auto graph_build_start = Clock::now();

    SUNOPT_GRAPH_CHECK(
        cudaStreamBeginCapture(
            stream,
            cudaStreamCaptureModeThreadLocal
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpyAsync(
            d_current,
            h_current,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpyAsync(
            d_horizon,
            h_horizon,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpyAsync(
            d_dynamic,
            h_dynamic,
            dynamic_step_bytes,
            cudaMemcpyHostToDevice,
            stream
        )
    );

    sunopt_digital_hp_breakpoint_kernel<<<
        static_cast<unsigned int>(tank_count),
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
        1,
        search_iterations,
        d_cuts,
        d_status,
        d_evaluations
    );

    SUNOPT_GRAPH_CHECK(cudaGetLastError());

    SUNOPT_GRAPH_CHECK(
        cudaMemcpyAsync(
            h_cuts,
            d_cuts,
            cut_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpyAsync(
            h_status,
            d_status,
            u32_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaMemcpyAsync(
            h_evaluations,
            d_evaluations,
            u32_step_bytes,
            cudaMemcpyDeviceToHost,
            stream
        )
    );

    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    err = cudaGraphInstantiate(
        &graph_exec,
        graph,
        nullptr,
        nullptr,
        0
    );

    if (err != cudaSuccess) {
        cleanup();
        return static_cast<int>(err);
    }

    SUNOPT_GRAPH_CHECK(cudaGraphDestroy(graph));
    graph = nullptr;

    const auto graph_build_stop = Clock::now();

    *out_graph_build_ms =
        std::chrono::duration<double, std::milli>(
            graph_build_stop - graph_build_start
        ).count();

    double host_wait_ms = 0.0;

    const auto wall_start = Clock::now();

    SUNOPT_GRAPH_CHECK(
        cudaEventRecord(
            gpu_start,
            stream
        )
    );

    for (
        uint32_t repetition = 0;
        repetition < repetitions;
        ++repetition
    ) {
        for (
            size_t step = 0;
            step < step_count;
            ++step
        ) {
            const size_t dynamic_offset =
                step * dynamic_step_count;

            const size_t result_offset =
                step * tank_count;

            *h_current =
                host_current_indices[step];

            *h_horizon =
                host_horizon_steps[step];

            for (
                size_t i = 0;
                i < dynamic_step_count;
                ++i
            ) {
                h_dynamic[i] =
                    host_dynamic_values[
                        dynamic_offset + i
                    ];
            }

            SUNOPT_GRAPH_CHECK(
                cudaGraphLaunch(
                    graph_exec,
                    stream
                )
            );

            const auto wait_start =
                Clock::now();

            SUNOPT_GRAPH_CHECK(
                cudaStreamSynchronize(
                    stream
                )
            );

            const auto wait_stop =
                Clock::now();

            host_wait_ms +=
                std::chrono::duration<double, std::milli>(
                    wait_stop - wait_start
                ).count();

            for (
                size_t tank = 0;
                tank < tank_count;
                ++tank
            ) {
                host_cut_results[
                    result_offset + tank
                ] = h_cuts[tank];

                host_status[
                    result_offset + tank
                ] = h_status[tank];

                host_survival_evaluations[
                    result_offset + tank
                ] = h_evaluations[tank];
            }
        }
    }

    SUNOPT_GRAPH_CHECK(
        cudaEventRecord(
            gpu_stop,
            stream
        )
    );

    SUNOPT_GRAPH_CHECK(
        cudaEventSynchronize(
            gpu_stop
        )
    );

    float gpu_total_ms = 0.0f;

    SUNOPT_GRAPH_CHECK(
        cudaEventElapsedTime(
            &gpu_total_ms,
            gpu_start,
            gpu_stop
        )
    );

    const auto wall_stop = Clock::now();

    *out_gpu_total_ms =
        static_cast<double>(gpu_total_ms);

    *out_host_wait_ms =
        host_wait_ms;

    *out_wall_ms =
        std::chrono::duration<double, std::milli>(
            wall_stop - wall_start
        ).count();

#undef SUNOPT_GRAPH_CHECK

    cleanup();

    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_END


// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_BEGIN
__global__ void
sunopt_digital_hp_presorted_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_window,
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

    /* B5: shared sorted export order, reused by every tank block. */
    const double sorted_export_value = presorted_export_window[tid];

    breakpoints[tid] =
        (
            maximum_hp_power > 0.0 &&
            sorted_export_value >= maximum_hp_power
        ) ?
            (sorted_export_value - maximum_hp_power) :
            -CUDART_INF;

    __syncthreads();

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
        if (
            !shared_supported
        ) {
            /*
             * Exact GPU fallback for the max-affine edge cases.
             * The breakpoints are already sorted in shared memory.
             */
            eval_count =
                1;

            const bool serial_normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                serial_normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1;

                const bool serial_maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !serial_maxpv_survives
                ) {
                    final_status =
                        1;

                    final_cut =
                        0.0;
                } else {
                    final_status =
                        2;

                    double best_cut =
                        0.0;

                    int left =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            valid_count
                        );

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    for (
                        uint32_t iteration = 0;
                        iteration <
                            search_iterations &&
                        left <=
                            right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                breakpoints[
                                    middle
                                ]
                            );

                        eval_count +=
                            1;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best_cut =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_cut =
                        best_cut;
                }
            }
        } else {
            eval_count +=
                1;

            if (
                normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            }
        }
    }

    __syncthreads();

    /*
     * Max-affine-unsupported jobs are already fully solved by the serial GPU
     * branch. No status=3 leaves this kernel.
     */
    if (
        final_status ==
            0 ||
        !shared_supported
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

extern "C" int sunopt_cuda_run_digital_lockstep_presorted_candidate(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_static_h2d_ms,
    double* out_presorted_h2d_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_static_h2d_ms == nullptr ||
        out_presorted_h2d_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 || tank_count == 0 || tank_count > 32 ||
        step_count == 0 || search_iterations == 0 || repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_presorted = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 7 * sizeof(double);
    const size_t presorted_bytes = step_count * SUNOPT_DIGITAL_THREADS * sizeof(double);
    const size_t dynamic_step_bytes = tank_count * 3 * sizeof(double);
    const size_t cut_step_bytes = tank_count * sizeof(double);
    const size_t u32_step_bytes = tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_presorted) cudaFree(d_presorted);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (gpu_stop) cudaEventDestroy(gpu_stop);
        if (gpu_start) cudaEventDestroy(gpu_start);
        if (stream) cudaStreamDestroy(stream);
    };

#define CHECK(call) do { err = (call); if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } } while(false)
#define MALLOC(ptr, bytes) CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)))

    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&gpu_start));
    CHECK(cudaEventCreate(&gpu_stop));
    MALLOC(d_export, annual_bytes);
    MALLOC(d_demand, annual_bytes);
    MALLOC(d_tanks, tank_bytes);
    MALLOC(d_presorted, presorted_bytes);
    MALLOC(d_current, sizeof(uint32_t));
    MALLOC(d_horizon, sizeof(uint32_t));
    MALLOC(d_dynamic, dynamic_step_bytes);
    MALLOC(d_cuts, cut_step_bytes);
    MALLOC(d_status, u32_step_bytes);
    MALLOC(d_evaluations, u32_step_bytes);
#undef MALLOC

    const auto static_start = Clock::now();
    CHECK(cudaMemcpy(d_export, host_export_power, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_demand, host_demand_energy, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tanks, host_tank_parameters, tank_bytes, cudaMemcpyHostToDevice));
    const auto static_stop = Clock::now();
    *out_static_h2d_ms = std::chrono::duration<double, std::milli>(static_stop-static_start).count();

    const auto presort_start = Clock::now();
    CHECK(cudaMemcpy(d_presorted, host_presorted_export_windows, presorted_bytes, cudaMemcpyHostToDevice));
    const auto presort_stop = Clock::now();
    *out_presorted_h2d_ms = std::chrono::duration<double, std::milli>(presort_stop-presort_start).count();

    double host_wait_ms = 0.0;
    const auto wall_start = Clock::now();
    CHECK(cudaEventRecord(gpu_start, stream));

    const unsigned int blocks = static_cast<unsigned int>(tank_count);

    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t step = 0; step < step_count; ++step) {
            const size_t dynamic_offset = step * tank_count * 3;
            const size_t result_offset = step * tank_count;

            CHECK(cudaMemcpyAsync(d_current, host_current_indices + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_horizon, host_horizon_steps + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_dynamic, host_dynamic_values + dynamic_offset, dynamic_step_bytes, cudaMemcpyHostToDevice, stream));

            sunopt_digital_hp_presorted_kernel<<<blocks, SUNOPT_DIGITAL_THREADS, 0, stream>>>(
                d_export,
                d_demand,
                d_presorted + step * SUNOPT_DIGITAL_THREADS,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                1,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );
            CHECK(cudaGetLastError());

            CHECK(cudaMemcpyAsync(host_cut_results + result_offset, d_cuts, cut_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_status + result_offset, d_status, u32_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_survival_evaluations + result_offset, d_evaluations, u32_step_bytes, cudaMemcpyDeviceToHost, stream));

            const auto wait_start = Clock::now();
            CHECK(cudaStreamSynchronize(stream));
            const auto wait_stop = Clock::now();
            host_wait_ms += std::chrono::duration<double, std::milli>(wait_stop-wait_start).count();
        }
    }

    CHECK(cudaEventRecord(gpu_stop, stream));
    CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    const auto wall_stop = Clock::now();

    *out_gpu_total_ms = static_cast<double>(gpu_ms);
    *out_host_wait_ms = host_wait_ms;
    *out_wall_ms = std::chrono::duration<double, std::milli>(wall_stop-wall_start).count();

#undef CHECK
    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_END


// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_BEGIN
__device__ __forceinline__ SunoptDigitalTransform
sunopt_digital_shuffle_down_transform(
    SunoptDigitalTransform value,
    unsigned int delta
) {
    constexpr unsigned int FULL_MASK = 0xffffffffu;

    SunoptDigitalTransform result{};

    result.a =
        __shfl_down_sync(
            FULL_MASK,
            value.a,
            delta
        );

    result.b =
        __shfl_down_sync(
            FULL_MASK,
            value.b,
            delta
        );

    result.c =
        __shfl_down_sync(
            FULL_MASK,
            value.c,
            delta
        );

    result.max_input =
        __shfl_down_sync(
            FULL_MASK,
            value.max_input,
            delta
        );

    return result;
}


/*
 * Exact ordered max-affine survival tree using warp shuffles.
 *
 * The operation is associative but not commutative. Therefore each warp
 * reduces contiguous transforms strictly left-to-right. Warp 0 then reduces
 * the eight warp partials in the same order.
 */
__device__ __forceinline__ bool
sunopt_digital_survives_warp(
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

    const unsigned int lane =
        tid & 31u;

    const unsigned int warp_id =
        tid >> 5;

    const size_t tank_base =
        tank_index * 7;

    const double target =
        tank_parameters[tank_base + 0];

    const double maximum_hp_power =
        tank_parameters[tank_base + 1];

    const double full_charge =
        tank_parameters[tank_base + 2];

    const double retention =
        tank_parameters[tank_base + 3];

    const double loss_intercept =
        tank_parameters[tank_base + 4];

    const bool affine_supported =
        tank_parameters[tank_base + 5] >
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
        retention > 0.0 &&
        data_length > 0 &&
        horizon > 0 &&
        horizon <= SUNOPT_DIGITAL_THREADS &&
        reserve > SUNOPT_DIGITAL_EPSILON;

    __shared__ bool warp_survives;

    if (
        !supported
    ) {
        if (
            tid == 0
        ) {
            *shared_supported =
                false;

            warp_survives =
                false;
        }

        __syncthreads();

        return false;
    }

    const size_t future_offset =
        static_cast<size_t>(tid) + 1;

    SunoptDigitalTransform local =
        sunopt_digital_identity();

    if (
        future_offset < horizon
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
                demand_energy[index]
            );

        const double charge =
            sunopt_digital_hp_on(
                export_power[index],
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

    /*
     * 32 contiguous transforms -> one transform per warp.
     * No block-wide barriers are needed in these five levels.
     */
    for (
        unsigned int stride = 1;
        stride < 32u;
        stride <<= 1
    ) {
        const SunoptDigitalTransform right =
            sunopt_digital_shuffle_down_transform(
                local,
                stride
            );

        const unsigned int group =
            stride << 1;

        if (
            (
                lane &
                (
                    group -
                    1
                )
            ) ==
            0
        ) {
            local =
                sunopt_digital_compose(
                    local,
                    right
                );
        }
    }

    if (
        lane == 0
    ) {
        shared_transform[warp_id] =
            local;
    }

    /*
     * First and only transform-tree block barrier.
     */
    __syncthreads();

    if (
        warp_id == 0
    ) {
        SunoptDigitalTransform warp_local =
            lane < 8u ?
                shared_transform[lane] :
                sunopt_digital_identity();

        for (
            unsigned int stride = 1;
            stride < 8u;
            stride <<= 1
        ) {
            const SunoptDigitalTransform right =
                sunopt_digital_shuffle_down_transform(
                    warp_local,
                    stride
                );

            const unsigned int group =
                stride << 1;

            if (
                (
                    lane &
                    (
                        group -
                        1
                    )
                ) ==
                0
            ) {
                warp_local =
                    sunopt_digital_compose(
                        warp_local,
                        right
                    );
            }
        }

        if (
            lane == 0
        ) {
            const SunoptDigitalTransform summary =
                warp_local;

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

            warp_survives =
                feasible &&
                available +
                    1.0e-12 >=
                    required;

            *shared_supported =
                true;
        }
    }

    /*
     * Result publication barrier.
     */
    __syncthreads();

    return warp_survives;
}


__global__ void
sunopt_digital_hp_presorted_warp_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_window,
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

    /* B5: shared sorted export order, reused by every tank block. */
    const double sorted_export_value = presorted_export_window[tid];

    breakpoints[tid] =
        (
            maximum_hp_power > 0.0 &&
            sorted_export_value >= maximum_hp_power
        ) ?
            (sorted_export_value - maximum_hp_power) :
            -CUDART_INF;

    __syncthreads();

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
        sunopt_digital_survives_warp(
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
        if (
            !shared_supported
        ) {
            /*
             * Exact GPU fallback for the max-affine edge cases.
             * The breakpoints are already sorted in shared memory.
             */
            eval_count =
                1;

            const bool serial_normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                serial_normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1;

                const bool serial_maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !serial_maxpv_survives
                ) {
                    final_status =
                        1;

                    final_cut =
                        0.0;
                } else {
                    final_status =
                        2;

                    double best_cut =
                        0.0;

                    int left =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            valid_count
                        );

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    for (
                        uint32_t iteration = 0;
                        iteration <
                            search_iterations &&
                        left <=
                            right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                breakpoints[
                                    middle
                                ]
                            );

                        eval_count +=
                            1;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best_cut =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_cut =
                        best_cut;
                }
            }
        } else {
            eval_count +=
                1;

            if (
                normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            }
        }
    }

    __syncthreads();

    /*
     * Max-affine-unsupported jobs are already fully solved by the serial GPU
     * branch. No status=3 leaves this kernel.
     */
    if (
        final_status ==
            0 ||
        !shared_supported
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
        sunopt_digital_survives_warp(
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
            sunopt_digital_survives_warp(
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


extern "C" int sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_static_h2d_ms,
    double* out_presorted_h2d_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_static_h2d_ms == nullptr ||
        out_presorted_h2d_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 || tank_count == 0 || tank_count > 32 ||
        step_count == 0 || search_iterations == 0 || repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_presorted = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 7 * sizeof(double);
    const size_t presorted_bytes = step_count * SUNOPT_DIGITAL_THREADS * sizeof(double);
    const size_t dynamic_step_bytes = tank_count * 3 * sizeof(double);
    const size_t cut_step_bytes = tank_count * sizeof(double);
    const size_t u32_step_bytes = tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_presorted) cudaFree(d_presorted);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (gpu_stop) cudaEventDestroy(gpu_stop);
        if (gpu_start) cudaEventDestroy(gpu_start);
        if (stream) cudaStreamDestroy(stream);
    };

#define CHECK(call) do { err = (call); if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } } while(false)
#define MALLOC(ptr, bytes) CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)))

    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&gpu_start));
    CHECK(cudaEventCreate(&gpu_stop));
    MALLOC(d_export, annual_bytes);
    MALLOC(d_demand, annual_bytes);
    MALLOC(d_tanks, tank_bytes);
    MALLOC(d_presorted, presorted_bytes);
    MALLOC(d_current, sizeof(uint32_t));
    MALLOC(d_horizon, sizeof(uint32_t));
    MALLOC(d_dynamic, dynamic_step_bytes);
    MALLOC(d_cuts, cut_step_bytes);
    MALLOC(d_status, u32_step_bytes);
    MALLOC(d_evaluations, u32_step_bytes);
#undef MALLOC

    const auto static_start = Clock::now();
    CHECK(cudaMemcpy(d_export, host_export_power, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_demand, host_demand_energy, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tanks, host_tank_parameters, tank_bytes, cudaMemcpyHostToDevice));
    const auto static_stop = Clock::now();
    *out_static_h2d_ms = std::chrono::duration<double, std::milli>(static_stop-static_start).count();

    const auto presort_start = Clock::now();
    CHECK(cudaMemcpy(d_presorted, host_presorted_export_windows, presorted_bytes, cudaMemcpyHostToDevice));
    const auto presort_stop = Clock::now();
    *out_presorted_h2d_ms = std::chrono::duration<double, std::milli>(presort_stop-presort_start).count();

    double host_wait_ms = 0.0;
    const auto wall_start = Clock::now();
    CHECK(cudaEventRecord(gpu_start, stream));

    const unsigned int blocks = static_cast<unsigned int>(tank_count);

    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t step = 0; step < step_count; ++step) {
            const size_t dynamic_offset = step * tank_count * 3;
            const size_t result_offset = step * tank_count;

            CHECK(cudaMemcpyAsync(d_current, host_current_indices + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_horizon, host_horizon_steps + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_dynamic, host_dynamic_values + dynamic_offset, dynamic_step_bytes, cudaMemcpyHostToDevice, stream));

            sunopt_digital_hp_presorted_warp_kernel<<<blocks, SUNOPT_DIGITAL_THREADS, 0, stream>>>(
                d_export,
                d_demand,
                d_presorted + step * SUNOPT_DIGITAL_THREADS,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                1,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );
            CHECK(cudaGetLastError());

            CHECK(cudaMemcpyAsync(host_cut_results + result_offset, d_cuts, cut_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_status + result_offset, d_status, u32_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_survival_evaluations + result_offset, d_evaluations, u32_step_bytes, cudaMemcpyDeviceToHost, stream));

            const auto wait_start = Clock::now();
            CHECK(cudaStreamSynchronize(stream));
            const auto wait_stop = Clock::now();
            host_wait_ms += std::chrono::duration<double, std::milli>(wait_stop-wait_start).count();
        }
    }

    CHECK(cudaEventRecord(gpu_stop, stream));
    CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    const auto wall_stop = Clock::now();

    *out_gpu_total_ms = static_cast<double>(gpu_ms);
    *out_host_wait_ms = host_wait_ms;
    *out_wall_ms = std::chrono::duration<double, std::milli>(wall_stop-wall_start).count();

#undef CHECK
    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_END


// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_BEGIN
struct SunoptDigitalPiecewiseTransform {
    double a;
    double b;
    double c;
    double threshold;
    double max_input;
};


__device__ __forceinline__ SunoptDigitalPiecewiseTransform
sunopt_digital_piecewise_identity() {
    SunoptDigitalPiecewiseTransform t{};

    t.a = 1.0;
    t.b = 0.0;
    t.c = -CUDART_INF;
    t.threshold = -CUDART_INF;
    t.max_input = CUDART_INF;

    return t;
}


__device__ __forceinline__ double
sunopt_digital_piecewise_eval(
    const SunoptDigitalPiecewiseTransform t,
    double x
) {
    return (
        x <= t.threshold
    ) ?
        t.c :
        (
            t.a *
            x +
            t.b
        );
}


/*
 * Exact composition of monotone two-piece transforms:
 *
 *   F(x) = c              for x <= threshold
 *          a*x + b        for x >  threshold
 *
 * The high branch is monotone and begins at or above the low constant.
 * This family is closed under composition.
 */
__device__ __forceinline__ SunoptDigitalPiecewiseTransform
sunopt_digital_piecewise_compose(
    const SunoptDigitalPiecewiseTransform left,
    const SunoptDigitalPiecewiseTransform right
) {
    SunoptDigitalPiecewiseTransform out{};

    out.a =
        left.a *
        right.a;

    out.b =
        left.a *
        right.b +
        left.b;

    out.c =
        sunopt_digital_piecewise_eval(
            left,
            right.c
        );

    if (
        right.c >
        left.threshold
    ) {
        out.threshold =
            right.threshold;
    } else {
        const double crossing =
            (
                left.threshold -
                right.b
            ) /
            right.a;

        out.threshold =
            fmax(
                right.threshold,
                crossing
            );
    }

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
        /*
         * right() uses the low constant at x == threshold.
         * If its high branch jumps above left.max_input immediately after
         * threshold, the exact last feasible x is the threshold itself.
         */
        const double high_at_threshold =
            right.a *
            right.threshold +
            right.b;

        if (
            high_at_threshold >
            left.max_input +
            1.0e-12
        ) {
            max_input =
                fmin(
                    max_input,
                    right.threshold
                );
        } else {
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
        }
    }

    out.max_input =
        max_input;

    return out;
}


__device__ __forceinline__ SunoptDigitalPiecewiseTransform
sunopt_digital_piecewise_shuffle_down(
    SunoptDigitalPiecewiseTransform value,
    unsigned int delta
) {
    constexpr unsigned int FULL_MASK =
        0xffffffffu;

    SunoptDigitalPiecewiseTransform result{};

    result.a =
        __shfl_down_sync(
            FULL_MASK,
            value.a,
            delta
        );

    result.b =
        __shfl_down_sync(
            FULL_MASK,
            value.b,
            delta
        );

    result.c =
        __shfl_down_sync(
            FULL_MASK,
            value.c,
            delta
        );

    result.threshold =
        __shfl_down_sync(
            FULL_MASK,
            value.threshold,
            delta
        );

    result.max_input =
        __shfl_down_sync(
            FULL_MASK,
            value.max_input,
            delta
        );

    return result;
}


/*
 * Exact digital survival tree INCLUDING reserve ~= 0.
 *
 * For one future bucket let:
 *
 *   y = max(0, retention*E - loss_intercept)
 *
 * Scalar survival requires:
 *   unmet demand <= EPSILON
 *   before_charge >= reserve - EPSILON
 *   before_charge + charge >= suffix_required
 *
 * The exact minimal input energy therefore has the closed two-piece form:
 *
 *   F(x) = c              for x <= threshold
 *          a*x + b        for x >  threshold
 *
 * with:
 *   reserve_req = max(0, reserve - EPSILON)
 *   low_y =
 *       demand + reserve_req             if reserve_req > 0
 *       max(0, demand - EPSILON)         otherwise
 *   c =
 *       0                                if low_y == 0
 *       (loss_intercept + low_y)/r       otherwise
 *   threshold = charge + reserve_req
 *   a = 1/r
 *   b = (loss_intercept + demand - charge)/r
 *
 * This is the missing exact zero-reserve branch: no serial GPU replay is
 * required merely because reserve is zero.
 */
__device__ __forceinline__ bool
sunopt_digital_survives_piecewise_warp(
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
    SunoptDigitalPiecewiseTransform* shared_transform,
    bool* shared_supported
) {
    const unsigned int tid =
        threadIdx.x;

    const unsigned int lane =
        tid &
        31u;

    const unsigned int warp_id =
        tid >>
        5;

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
            SUNOPT_DIGITAL_THREADS;

    __shared__ bool piecewise_survives;

    if (
        !supported
    ) {
        if (
            tid ==
            0
        ) {
            *shared_supported =
                false;

            piecewise_survives =
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

    SunoptDigitalPiecewiseTransform local =
        sunopt_digital_piecewise_identity();

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

        const double reserve_required =
            fmax(
                0.0,
                reserve -
                SUNOPT_DIGITAL_EPSILON
            );

        const double demand_required =
            fmax(
                0.0,
                demand -
                SUNOPT_DIGITAL_EPSILON
            );

        const double low_after_loss =
            reserve_required >
                0.0 ?
                (
                    demand +
                    reserve_required
                ) :
                demand_required;

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
            low_after_loss >
                0.0 ?
                (
                    loss_intercept +
                    low_after_loss
                ) *
                inverse_retention :
                0.0;

        local.threshold =
            charge +
            reserve_required;

        local.max_input =
            target;
    }

    for (
        unsigned int stride = 1;
        stride <
            32u;
        stride <<=
            1
    ) {
        const SunoptDigitalPiecewiseTransform right =
            sunopt_digital_piecewise_shuffle_down(
                local,
                stride
            );

        const unsigned int group =
            stride <<
            1;

        if (
            (
                lane &
                (
                    group -
                    1
                )
            ) ==
            0
        ) {
            local =
                sunopt_digital_piecewise_compose(
                    local,
                    right
                );
        }
    }

    if (
        lane ==
        0
    ) {
        shared_transform[
            warp_id
        ] =
            local;
    }

    __syncthreads();

    if (
        warp_id ==
        0
    ) {
        SunoptDigitalPiecewiseTransform warp_local =
            lane <
                8u ?
                shared_transform[
                    lane
                ] :
                sunopt_digital_piecewise_identity();

        for (
            unsigned int stride = 1;
            stride <
                8u;
            stride <<=
                1
        ) {
            const SunoptDigitalPiecewiseTransform right =
                sunopt_digital_piecewise_shuffle_down(
                    warp_local,
                    stride
                );

            const unsigned int group =
                stride <<
                1;

            if (
                (
                    lane &
                    (
                        group -
                        1
                    )
                ) ==
                0
            ) {
                warp_local =
                    sunopt_digital_piecewise_compose(
                        warp_local,
                        right
                    );
            }
        }

        if (
            lane ==
            0
        ) {
            const SunoptDigitalPiecewiseTransform summary =
                warp_local;

            bool feasible =
                0.0 <=
                summary.max_input +
                1.0e-12;

            const double required =
                sunopt_digital_piecewise_eval(
                    summary,
                    0.0
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

            piecewise_survives =
                feasible &&
                available +
                    1.0e-12 >=
                    required;

            *shared_supported =
                true;
        }
    }

    __syncthreads();

    return piecewise_survives;
}


__global__ void
sunopt_digital_hp_piecewise_zero_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_window,
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

    __shared__ SunoptDigitalPiecewiseTransform transforms[
        SUNOPT_DIGITAL_THREADS
    ];

    __shared__ bool shared_supported;
    __shared__ double normal_cut;
    __shared__ unsigned int valid_count;
    __shared__ unsigned int eval_count;
    __shared__ double final_cut;
    __shared__ uint32_t final_status;

    /* B5: shared sorted export order, reused by every tank block. */
    const double sorted_export_value = presorted_export_window[tid];

    breakpoints[tid] =
        (
            maximum_hp_power > 0.0 &&
            sorted_export_value >= maximum_hp_power
        ) ?
            (sorted_export_value - maximum_hp_power) :
            -CUDART_INF;

    __syncthreads();

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
        sunopt_digital_survives_piecewise_warp(
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
        if (
            !shared_supported
        ) {
            /*
             * Exact GPU fallback for the max-affine edge cases.
             * The breakpoints are already sorted in shared memory.
             */
            eval_count =
                1;

            const bool serial_normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                serial_normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1;

                const bool serial_maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !serial_maxpv_survives
                ) {
                    final_status =
                        1;

                    final_cut =
                        0.0;
                } else {
                    final_status =
                        2;

                    double best_cut =
                        0.0;

                    int left =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            valid_count
                        );

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    for (
                        uint32_t iteration = 0;
                        iteration <
                            search_iterations &&
                        left <=
                            right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                breakpoints[
                                    middle
                                ]
                            );

                        eval_count +=
                            1;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best_cut =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_cut =
                        best_cut;
                }
            }
        } else {
            eval_count +=
                1;

            if (
                normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            }
        }
    }

    __syncthreads();

    /*
     * Max-affine-unsupported jobs are already fully solved by the serial GPU
     * branch. No status=3 leaves this kernel.
     */
    if (
        final_status ==
            0 ||
        !shared_supported
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
        sunopt_digital_survives_piecewise_warp(
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
            sunopt_digital_survives_piecewise_warp(
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


extern "C" int sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_static_h2d_ms,
    double* out_presorted_h2d_ms,
    double* out_gpu_total_ms,
    double* out_host_wait_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_static_h2d_ms == nullptr ||
        out_presorted_h2d_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 || tank_count == 0 || tank_count > 32 ||
        step_count == 0 || search_iterations == 0 || repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_presorted = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 7 * sizeof(double);
    const size_t presorted_bytes = step_count * SUNOPT_DIGITAL_THREADS * sizeof(double);
    const size_t dynamic_step_bytes = tank_count * 3 * sizeof(double);
    const size_t cut_step_bytes = tank_count * sizeof(double);
    const size_t u32_step_bytes = tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_presorted) cudaFree(d_presorted);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (gpu_stop) cudaEventDestroy(gpu_stop);
        if (gpu_start) cudaEventDestroy(gpu_start);
        if (stream) cudaStreamDestroy(stream);
    };

#define CHECK(call) do { err = (call); if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } } while(false)
#define MALLOC(ptr, bytes) CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)))

    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&gpu_start));
    CHECK(cudaEventCreate(&gpu_stop));
    MALLOC(d_export, annual_bytes);
    MALLOC(d_demand, annual_bytes);
    MALLOC(d_tanks, tank_bytes);
    MALLOC(d_presorted, presorted_bytes);
    MALLOC(d_current, sizeof(uint32_t));
    MALLOC(d_horizon, sizeof(uint32_t));
    MALLOC(d_dynamic, dynamic_step_bytes);
    MALLOC(d_cuts, cut_step_bytes);
    MALLOC(d_status, u32_step_bytes);
    MALLOC(d_evaluations, u32_step_bytes);
#undef MALLOC

    const auto static_start = Clock::now();
    CHECK(cudaMemcpy(d_export, host_export_power, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_demand, host_demand_energy, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tanks, host_tank_parameters, tank_bytes, cudaMemcpyHostToDevice));
    const auto static_stop = Clock::now();
    *out_static_h2d_ms = std::chrono::duration<double, std::milli>(static_stop-static_start).count();

    const auto presort_start = Clock::now();
    CHECK(cudaMemcpy(d_presorted, host_presorted_export_windows, presorted_bytes, cudaMemcpyHostToDevice));
    const auto presort_stop = Clock::now();
    *out_presorted_h2d_ms = std::chrono::duration<double, std::milli>(presort_stop-presort_start).count();

    double host_wait_ms = 0.0;
    const auto wall_start = Clock::now();
    CHECK(cudaEventRecord(gpu_start, stream));

    const unsigned int blocks = static_cast<unsigned int>(tank_count);

    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t step = 0; step < step_count; ++step) {
            const size_t dynamic_offset = step * tank_count * 3;
            const size_t result_offset = step * tank_count;

            CHECK(cudaMemcpyAsync(d_current, host_current_indices + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_horizon, host_horizon_steps + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_dynamic, host_dynamic_values + dynamic_offset, dynamic_step_bytes, cudaMemcpyHostToDevice, stream));

            sunopt_digital_hp_piecewise_zero_kernel<<<blocks, SUNOPT_DIGITAL_THREADS, 0, stream>>>(
                d_export,
                d_demand,
                d_presorted + step * SUNOPT_DIGITAL_THREADS,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                1,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );
            CHECK(cudaGetLastError());

            CHECK(cudaMemcpyAsync(host_cut_results + result_offset, d_cuts, cut_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_status + result_offset, d_status, u32_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_survival_evaluations + result_offset, d_evaluations, u32_step_bytes, cudaMemcpyDeviceToHost, stream));

            const auto wait_start = Clock::now();
            CHECK(cudaStreamSynchronize(stream));
            const auto wait_stop = Clock::now();
            host_wait_ms += std::chrono::duration<double, std::milli>(wait_stop-wait_start).count();
        }
    }

    CHECK(cudaEventRecord(gpu_stop, stream));
    CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    const auto wall_stop = Clock::now();

    *out_gpu_total_ms = static_cast<double>(gpu_ms);
    *out_host_wait_ms = host_wait_ms;
    *out_wall_ms = std::chrono::duration<double, std::milli>(wall_stop-wall_start).count();

#undef CHECK
    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN
__device__ __forceinline__ void
sunopt_digital_hp_presorted_warp_job(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_window,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    size_t forced_job_index,
    uint32_t search_iterations,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    const size_t job_index =
        forced_job_index;

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

    /* B5: shared sorted export order, reused by every tank block. */
    const double sorted_export_value = presorted_export_window[tid];

    breakpoints[tid] =
        (
            maximum_hp_power > 0.0 &&
            sorted_export_value >= maximum_hp_power
        ) ?
            (sorted_export_value - maximum_hp_power) :
            -CUDART_INF;

    __syncthreads();

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
        sunopt_digital_survives_warp(
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
        if (
            !shared_supported
        ) {
            /*
             * Exact GPU fallback for the max-affine edge cases.
             * The breakpoints are already sorted in shared memory.
             */
            eval_count =
                1;

            const bool serial_normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                serial_normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1;

                const bool serial_maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !serial_maxpv_survives
                ) {
                    final_status =
                        1;

                    final_cut =
                        0.0;
                } else {
                    final_status =
                        2;

                    double best_cut =
                        0.0;

                    int left =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            valid_count
                        );

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    for (
                        uint32_t iteration = 0;
                        iteration <
                            search_iterations &&
                        left <=
                            right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                breakpoints[
                                    middle
                                ]
                            );

                        eval_count +=
                            1;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best_cut =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_cut =
                        best_cut;
                }
            }
        } else {
            eval_count +=
                1;

            if (
                normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            }
        }
    }

    __syncthreads();

    /*
     * Max-affine-unsupported jobs are already fully solved by the serial GPU
     * branch. No status=3 leaves this kernel.
     */
    if (
        final_status ==
            0 ||
        !shared_supported
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
        sunopt_digital_survives_warp(
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
            sunopt_digital_survives_warp(
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


__global__ void
sunopt_digital_resident_chunk_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_windows,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    size_t chunk_start,
    size_t chunk_count,
    uint32_t search_iterations,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    const size_t tank_index =
        static_cast<size_t>(blockIdx.x);

    if (tank_index >= tank_count) {
        return;
    }

    const size_t chunk_end =
        min(
            step_count,
            chunk_start + chunk_count
        );

    for (
        size_t step_index = chunk_start;
        step_index < chunk_end;
        ++step_index
    ) {
        const size_t job_index =
            step_index * tank_count + tank_index;

        sunopt_digital_hp_presorted_warp_job(
            export_power,
            demand_energy,
            presorted_export_windows +
                step_index * SUNOPT_DIGITAL_THREADS,
            data_length,
            tank_parameters,
            tank_count,
            current_indices,
            horizon_steps,
            dynamic_values,
            step_count,
            job_index,
            search_iterations,
            cut_results,
            status,
            survival_evaluations
        );

        __syncthreads();
    }
}


extern "C" int sunopt_cuda_run_digital_resident_chunk_replay(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t chunk_steps,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_input_h2d_ms == nullptr ||
        out_kernel_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        chunk_steps == 0 ||
        repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_presorted = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    const size_t job_count = step_count * tank_count;
    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 7 * sizeof(double);
    const size_t presorted_bytes =
        step_count * SUNOPT_DIGITAL_THREADS * sizeof(double);
    const size_t step_u32_bytes = step_count * sizeof(uint32_t);
    const size_t dynamic_bytes = job_count * 3 * sizeof(double);
    const size_t cut_bytes = job_count * sizeof(double);
    const size_t u32_job_bytes = job_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_presorted) cudaFree(d_presorted);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (stop_event) cudaEventDestroy(stop_event);
        if (start_event) cudaEventDestroy(start_event);
        if (stream) cudaStreamDestroy(stream);
    };

#define SUNOPT_CHUNK_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_CHUNK_MALLOC(ptr, bytes) \
    SUNOPT_CHUNK_CHECK( \
        cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)) \
    )

    SUNOPT_CHUNK_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_CHUNK_CHECK(cudaEventCreate(&start_event));
    SUNOPT_CHUNK_CHECK(cudaEventCreate(&stop_event));

    SUNOPT_CHUNK_MALLOC(d_export, annual_bytes);
    SUNOPT_CHUNK_MALLOC(d_demand, annual_bytes);
    SUNOPT_CHUNK_MALLOC(d_tanks, tank_bytes);
    SUNOPT_CHUNK_MALLOC(d_presorted, presorted_bytes);
    SUNOPT_CHUNK_MALLOC(d_current, step_u32_bytes);
    SUNOPT_CHUNK_MALLOC(d_horizon, step_u32_bytes);
    SUNOPT_CHUNK_MALLOC(d_dynamic, dynamic_bytes);
    SUNOPT_CHUNK_MALLOC(d_cuts, cut_bytes);
    SUNOPT_CHUNK_MALLOC(d_status, u32_job_bytes);
    SUNOPT_CHUNK_MALLOC(d_evaluations, u32_job_bytes);

#undef SUNOPT_CHUNK_MALLOC

    const auto h2d_start = Clock::now();

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_export,
        host_export_power,
        annual_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_demand,
        host_demand_energy,
        annual_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_tanks,
        host_tank_parameters,
        tank_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_presorted,
        host_presorted_export_windows,
        presorted_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_current,
        host_current_indices,
        step_u32_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_horizon,
        host_horizon_steps,
        step_u32_bytes,
        cudaMemcpyHostToDevice
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        d_dynamic,
        host_dynamic_values,
        dynamic_bytes,
        cudaMemcpyHostToDevice
    ));

    const auto h2d_stop = Clock::now();

    *out_input_h2d_ms =
        std::chrono::duration<double, std::milli>(
            h2d_stop - h2d_start
        ).count();

    const auto wall_start = Clock::now();

    SUNOPT_CHUNK_CHECK(cudaEventRecord(start_event, stream));

    const size_t chunk = static_cast<size_t>(chunk_steps);

    for (
        uint32_t repetition = 0;
        repetition < repetitions;
        ++repetition
    ) {
        for (
            size_t start = 0;
            start < step_count;
            start += chunk
        ) {
            const size_t count =
                min(
                    chunk,
                    step_count - start
                );

            sunopt_digital_resident_chunk_kernel<<<
                static_cast<unsigned int>(tank_count),
                SUNOPT_DIGITAL_THREADS,
                0,
                stream
            >>>(
                d_export,
                d_demand,
                d_presorted,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                step_count,
                start,
                count,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_CHUNK_CHECK(cudaGetLastError());
        }
    }

    SUNOPT_CHUNK_CHECK(cudaEventRecord(stop_event, stream));
    SUNOPT_CHUNK_CHECK(cudaEventSynchronize(stop_event));

    float kernel_total_ms = 0.0f;

    SUNOPT_CHUNK_CHECK(cudaEventElapsedTime(
        &kernel_total_ms,
        start_event,
        stop_event
    ));

    *out_kernel_ms =
        static_cast<double>(kernel_total_ms) /
        static_cast<double>(repetitions);

    const auto d2h_start = Clock::now();

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        host_cut_results,
        d_cuts,
        cut_bytes,
        cudaMemcpyDeviceToHost
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        host_status,
        d_status,
        u32_job_bytes,
        cudaMemcpyDeviceToHost
    ));

    SUNOPT_CHUNK_CHECK(cudaMemcpy(
        host_survival_evaluations,
        d_evaluations,
        u32_job_bytes,
        cudaMemcpyDeviceToHost
    ));

    const auto d2h_stop = Clock::now();

    *out_result_d2h_ms =
        std::chrono::duration<double, std::milli>(
            d2h_stop - d2h_start
        ).count();

    const auto wall_stop = Clock::now();

    *out_wall_ms =
        std::chrono::duration<double, std::milli>(
            wall_stop - wall_start
        ).count() /
        static_cast<double>(repetitions);

#undef SUNOPT_CHUNK_CHECK

    cleanup();

    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_BEGIN
__global__ void
sunopt_digital_resident_job_batch_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_windows,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    size_t first_job,
    size_t batch_job_count,
    uint32_t search_iterations,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    const size_t local_job =
        static_cast<size_t>(
            blockIdx.x
        );

    if (
        local_job >=
        batch_job_count
    ) {
        return;
    }

    const size_t job_index =
        first_job +
        local_job;

    const size_t total_jobs =
        step_count *
        tank_count;

    if (
        job_index >=
        total_jobs
    ) {
        return;
    }

    const size_t step_index =
        job_index /
        tank_count;

    sunopt_digital_hp_presorted_warp_job(
        export_power,
        demand_energy,
        presorted_export_windows +
            step_index *
            SUNOPT_DIGITAL_THREADS,
        data_length,
        tank_parameters,
        tank_count,
        current_indices,
        horizon_steps,
        dynamic_values,
        step_count,
        job_index,
        search_iterations,
        cut_results,
        status,
        survival_evaluations
    );
}


extern "C" int sunopt_cuda_run_digital_resident_job_batch(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t search_iterations,
    uint32_t jobs_per_launch,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_input_h2d_ms == nullptr ||
        out_kernel_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        jobs_per_launch == 0 ||
        repetitions == 0
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

    const size_t total_jobs =
        step_count *
        tank_count;

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

    double* d_presorted =
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

    const size_t annual_bytes =
        data_length *
        sizeof(double);

    const size_t tank_bytes =
        tank_count *
        7 *
        sizeof(double);

    const size_t presorted_bytes =
        step_count *
        SUNOPT_DIGITAL_THREADS *
        sizeof(double);

    const size_t step_u32_bytes =
        step_count *
        sizeof(uint32_t);

    const size_t dynamic_bytes =
        total_jobs *
        3 *
        sizeof(double);

    const size_t cut_bytes =
        total_jobs *
        sizeof(double);

    const size_t u32_job_bytes =
        total_jobs *
        sizeof(uint32_t);

    auto cleanup =
        [&]() {
            if (d_evaluations) cudaFree(d_evaluations);
            if (d_status) cudaFree(d_status);
            if (d_cuts) cudaFree(d_cuts);
            if (d_dynamic) cudaFree(d_dynamic);
            if (d_horizon) cudaFree(d_horizon);
            if (d_current) cudaFree(d_current);
            if (d_presorted) cudaFree(d_presorted);
            if (d_tanks) cudaFree(d_tanks);
            if (d_demand) cudaFree(d_demand);
            if (d_export) cudaFree(d_export);

            if (stop_event) cudaEventDestroy(stop_event);
            if (start_event) cudaEventDestroy(start_event);
            if (stream) cudaStreamDestroy(stream);
        };

#define SUNOPT_JOB_BATCH_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_JOB_BATCH_MALLOC(ptr, bytes) \
    SUNOPT_JOB_BATCH_CHECK( \
        cudaMalloc( \
            reinterpret_cast<void**>(&(ptr)), \
            (bytes) \
        ) \
    )

    SUNOPT_JOB_BATCH_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaEventCreate(
            &start_event
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_export,
        annual_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_demand,
        annual_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_tanks,
        tank_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_presorted,
        presorted_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_current,
        step_u32_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_horizon,
        step_u32_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_dynamic,
        dynamic_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_cuts,
        cut_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_status,
        u32_job_bytes
    );

    SUNOPT_JOB_BATCH_MALLOC(
        d_evaluations,
        u32_job_bytes
    );

#undef SUNOPT_JOB_BATCH_MALLOC

    const auto h2d_start =
        Clock::now();

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_export,
            host_export_power,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_demand,
            host_demand_energy,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_tanks,
            host_tank_parameters,
            tank_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_presorted,
            host_presorted_export_windows,
            presorted_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_current,
            host_current_indices,
            step_u32_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_horizon,
            host_horizon_steps,
            step_u32_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            d_dynamic,
            host_dynamic_values,
            dynamic_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const auto h2d_stop =
        Clock::now();

    *out_input_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            h2d_stop -
            h2d_start
        ).count();

    const auto wall_start =
        Clock::now();

    SUNOPT_JOB_BATCH_CHECK(
        cudaEventRecord(
            start_event,
            stream
        )
    );

    const size_t launch_jobs =
        static_cast<size_t>(
            jobs_per_launch
        );

    for (
        uint32_t repetition = 0;
        repetition <
            repetitions;
        ++repetition
    ) {
        for (
            size_t first_job = 0;
            first_job <
                total_jobs;
            first_job +=
                launch_jobs
        ) {
            const size_t current_count =
                min(
                    launch_jobs,
                    total_jobs -
                        first_job
                );

            sunopt_digital_resident_job_batch_kernel<<<
                static_cast<unsigned int>(
                    current_count
                ),
                SUNOPT_DIGITAL_THREADS,
                0,
                stream
            >>>(
                d_export,
                d_demand,
                d_presorted,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                step_count,
                first_job,
                current_count,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_JOB_BATCH_CHECK(
                cudaGetLastError()
            );
        }
    }

    SUNOPT_JOB_BATCH_CHECK(
        cudaEventRecord(
            stop_event,
            stream
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaEventSynchronize(
            stop_event
        )
    );

    float kernel_total_ms =
        0.0f;

    SUNOPT_JOB_BATCH_CHECK(
        cudaEventElapsedTime(
            &kernel_total_ms,
            start_event,
            stop_event
        )
    );

    *out_kernel_ms =
        static_cast<double>(
            kernel_total_ms
        ) /
        static_cast<double>(
            repetitions
        );

    const auto d2h_start =
        Clock::now();

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            host_cut_results,
            d_cuts,
            cut_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            host_status,
            d_status,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_JOB_BATCH_CHECK(
        cudaMemcpy(
            host_survival_evaluations,
            d_evaluations,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    const auto d2h_stop =
        Clock::now();

    *out_result_d2h_ms =
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
        ).count() /
        static_cast<double>(
            repetitions
        );

#undef SUNOPT_JOB_BATCH_CHECK

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_END


// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_BEGIN

__device__ __forceinline__ bool
sunopt_digital_survives_candidate_warp(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    const double* tank_parameters,
    size_t tank_index,
    size_t current_index,
    size_t horizon,
    double current_storage_energy,
    double minimum_reserve_energy,
    double cut
) {
    const unsigned int lane =
        threadIdx.x & 31u;

    const size_t tank_base =
        tank_index * 7;

    const double target =
        tank_parameters[tank_base + 0];

    const double maximum_hp_power =
        tank_parameters[tank_base + 1];

    const double full_charge =
        tank_parameters[tank_base + 2];

    const double retention =
        tank_parameters[tank_base + 3];

    const double loss_intercept =
        tank_parameters[tank_base + 4];

    const double reserve =
        fmin(
            target,
            fmax(
                0.0,
                minimum_reserve_energy
            )
        );

    SunoptDigitalTransform local =
        sunopt_digital_identity();

#pragma unroll
    for (
        unsigned int local_offset = 0;
        local_offset < 8u;
        ++local_offset
    ) {
        const size_t future_offset =
            1 +
            static_cast<size_t>(lane) * 8 +
            local_offset;

        if (
            future_offset >= horizon
        ) {
            continue;
        }

        const size_t index =
            (
                current_index +
                future_offset
            ) %
            data_length;

        const double demand =
            fmax(
                0.0,
                demand_energy[index]
            );

        const double charge =
            sunopt_digital_hp_on(
                export_power[index],
                cut,
                maximum_hp_power
            ) ?
                full_charge :
                0.0;

        const double inverse_retention =
            1.0 /
            retention;

        SunoptDigitalTransform step =
            sunopt_digital_identity();

        step.a =
            inverse_retention;

        step.b =
            (
                loss_intercept +
                demand -
                charge
            ) *
            inverse_retention;

        step.c =
            (
                loss_intercept +
                demand +
                reserve -
                SUNOPT_DIGITAL_EPSILON
            ) *
            inverse_retention;

        step.max_input =
            target;

        local =
            sunopt_digital_compose(
                local,
                step
            );
    }

    for (
        unsigned int stride = 1;
        stride < 32u;
        stride <<= 1
    ) {
        const SunoptDigitalTransform right =
            sunopt_digital_shuffle_down_transform(
                local,
                stride
            );

        const unsigned int group =
            stride << 1;

        if (
            (
                lane &
                (
                    group -
                    1
                )
            ) ==
            0
        ) {
            local =
                sunopt_digital_compose(
                    local,
                    right
                );
        }
    }

    unsigned int survives =
        0u;

    if (
        lane ==
        0
    ) {
        const SunoptDigitalTransform summary =
            local;

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
            (
                feasible &&
                available +
                    1.0e-12 >=
                    required
            ) ?
                1u :
                0u;
    }

    survives =
        __shfl_sync(
            0xffffffffu,
            survives,
            0
        );

    return survives != 0u;
}


__global__ void
sunopt_digital_multilane_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_windows,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    size_t step_index,
    uint32_t candidate_warps,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    const size_t tank_index =
        static_cast<size_t>(
            blockIdx.x
        );

    if (
        tank_index >= tank_count
    ) {
        return;
    }

    const unsigned int tid =
        threadIdx.x;

    const unsigned int warp_id =
        tid >> 5;

    const unsigned int lane =
        tid & 31u;

    if (
        warp_id >= candidate_warps
    ) {
        return;
    }

    const size_t job_index =
        step_index *
            tank_count +
        tank_index;

    const size_t tank_base =
        tank_index * 7;

    const double target =
        tank_parameters[tank_base + 0];

    const double maximum_hp_power =
        tank_parameters[tank_base + 1];

    const double electrical_energy_per_on_bucket =
        tank_parameters[tank_base + 6];

    const size_t dynamic_base =
        job_index * 3;

    const double current_storage =
        dynamic_values[dynamic_base + 0];

    const double reserve =
        dynamic_values[dynamic_base + 1];

    const double requested_electrical_energy =
        dynamic_values[dynamic_base + 2];

    const size_t current_index =
        static_cast<size_t>(
            current_indices[step_index]
        );

    const size_t horizon =
        static_cast<size_t>(
            horizon_steps[step_index]
        );

    const double* sorted_export =
        presorted_export_windows +
        step_index *
            SUNOPT_DIGITAL_THREADS;

    __shared__ double candidate_cuts[8];
    __shared__ int candidate_indices[8];
    __shared__ uint32_t candidate_survives[8];

    __shared__ unsigned int active_candidates;
    __shared__ unsigned int valid_count;
    __shared__ int first_valid;
    __shared__ double normal_cut;
    __shared__ bool parallel_supported;
    __shared__ bool correction_needed;
    __shared__ int left_index;
    __shared__ int right_index;
    __shared__ double best_cut;
    __shared__ uint32_t final_status;
    __shared__ double final_cut;
    __shared__ uint32_t eval_count;

    if (
        tid ==
        0
    ) {
        int first =
            static_cast<int>(
                SUNOPT_DIGITAL_THREADS
            );

        for (
            int i = 0;
            i <
                static_cast<int>(
                    SUNOPT_DIGITAL_THREADS
                );
            ++i
        ) {
            if (
                sorted_export[i] >=
                maximum_hp_power
            ) {
                first = i;
                break;
            }
        }

        first_valid =
            first;

        valid_count =
            first <
                static_cast<int>(
                    SUNOPT_DIGITAL_THREADS
                ) ?
                static_cast<unsigned int>(
                    SUNOPT_DIGITAL_THREADS -
                    first
                ) :
                0u;

        const double maximum_schedulable =
            static_cast<double>(
                valid_count
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
            target_electrical <= 0.0 ||
            valid_count == 0 ||
            electrical_energy_per_on_bucket <=
                0.0
        ) {
            normal_cut =
                valid_count > 0 ?
                    sorted_export[
                        SUNOPT_DIGITAL_THREADS -
                        1
                    ] :
                    maximum_hp_power;
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

            if (
                required_on < 1u
            ) {
                required_on = 1u;
            }

            if (
                required_on > valid_count
            ) {
                required_on =
                    valid_count;
            }

            normal_cut =
                fmax(
                    0.0,
                    sorted_export[
                        SUNOPT_DIGITAL_THREADS -
                        required_on
                    ] -
                    maximum_hp_power
                );
        }

        const bool affine_supported =
            tank_parameters[
                tank_base + 5
            ] >
            0.5;

        const double retention =
            tank_parameters[
                tank_base + 3
            ];

        const double clamped_reserve =
            fmin(
                target,
                fmax(
                    0.0,
                    reserve
                )
            );

        parallel_supported =
            affine_supported &&
            retention > 0.0 &&
            horizon > 0 &&
            horizon <=
                SUNOPT_DIGITAL_THREADS &&
            clamped_reserve >
                SUNOPT_DIGITAL_EPSILON;

        correction_needed =
            false;

        final_status =
            2u;

        final_cut =
            0.0;

        eval_count =
            0u;

        candidate_cuts[0] =
            normal_cut;

        candidate_cuts[1] =
            0.0;

        for (
            unsigned int warp = 2;
            warp < candidate_warps;
            ++warp
        ) {
            candidate_cuts[warp] =
                0.0;
        }
    }

    __syncthreads();

    if (
        !parallel_supported
    ) {
        if (
            tid ==
            0
        ) {
            eval_count =
                1u;

            const bool normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                normal_survives
            ) {
                final_status =
                    0u;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1u;

                const bool maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !maxpv_survives
                ) {
                    final_status =
                        1u;

                    final_cut =
                        0.0;
                } else {
                    int left =
                        first_valid;

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    double best =
                        0.0;

                    for (
                        unsigned int iteration = 0;
                        iteration < 8u &&
                        left <= right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                sorted_export[middle] -
                                    maximum_hp_power
                            );

                        eval_count +=
                            1u;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_status =
                        2u;

                    final_cut =
                        best;
                }
            }

            cut_results[job_index] =
                final_cut;

            status[job_index] =
                final_status;

            survival_evaluations[job_index] =
                eval_count;
        }

        return;
    }

    const bool initial_survives =
        sunopt_digital_survives_candidate_warp(
            export_power,
            demand_energy,
            data_length,
            tank_parameters,
            tank_index,
            current_index,
            horizon,
            current_storage,
            reserve,
            candidate_cuts[warp_id]
        );

    if (
        lane ==
        0
    ) {
        candidate_survives[warp_id] =
            initial_survives ?
                1u :
                0u;
    }

    __syncthreads();

    if (
        tid ==
        0
    ) {
        eval_count =
            2u;

        const bool normal_survives =
            candidate_survives[0] !=
            0u;

        const bool maxpv_survives =
            candidate_survives[1] !=
            0u;

        if (
            normal_survives
        ) {
            final_status =
                0u;

            final_cut =
                normal_cut;

            correction_needed =
                false;
        } else if (
            !maxpv_survives
        ) {
            final_status =
                1u;

            final_cut =
                0.0;

            correction_needed =
                false;
        } else {
            correction_needed =
                true;

            left_index =
                first_valid;

            right_index =
                static_cast<int>(
                    SUNOPT_DIGITAL_THREADS -
                    1
                );

            best_cut =
                0.0;
        }
    }

    __syncthreads();

    if (
        !correction_needed
    ) {
        if (
            tid ==
            0
        ) {
            cut_results[job_index] =
                final_cut;

            status[job_index] =
                final_status;

            survival_evaluations[job_index] =
                eval_count;
        }

        return;
    }

    for (
        unsigned int round = 0;
        round < 8u;
        ++round
    ) {
        if (
            tid ==
            0
        ) {
            if (
                left_index >
                right_index
            ) {
                active_candidates =
                    0u;
            } else {
                const int count =
                    right_index -
                    left_index +
                    1;

                active_candidates =
                    static_cast<unsigned int>(
                        count
                    );

                if (
                    active_candidates >
                    candidate_warps
                ) {
                    active_candidates =
                        candidate_warps;
                }

                if (
                    static_cast<unsigned int>(
                        count
                    ) <=
                    candidate_warps
                ) {
                    for (
                        unsigned int warp = 0;
                        warp < candidate_warps;
                        ++warp
                    ) {
                        const int index =
                            warp <
                                active_candidates ?
                                (
                                    left_index +
                                    static_cast<int>(
                                        warp
                                    )
                                ) :
                                right_index;

                        candidate_indices[warp] =
                            index;

                        candidate_cuts[warp] =
                            fmax(
                                0.0,
                                sorted_export[index] -
                                    maximum_hp_power
                            );
                    }
                } else {
                    for (
                        unsigned int warp = 0;
                        warp < candidate_warps;
                        ++warp
                    ) {
                        const long long numerator =
                            static_cast<long long>(
                                warp +
                                1
                            ) *
                            static_cast<long long>(
                                count -
                                1
                            );

                        int index =
                            left_index +
                            static_cast<int>(
                                numerator /
                                static_cast<long long>(
                                    candidate_warps +
                                    1
                                )
                            );

                        if (
                            warp > 0 &&
                            index <=
                                candidate_indices[
                                    warp -
                                    1
                                ]
                        ) {
                            index =
                                candidate_indices[
                                    warp -
                                    1
                                ] +
                                1;
                        }

                        if (
                            index >
                            right_index
                        ) {
                            index =
                                right_index;
                        }

                        candidate_indices[warp] =
                            index;

                        candidate_cuts[warp] =
                            fmax(
                                0.0,
                                sorted_export[index] -
                                    maximum_hp_power
                            );
                    }
                }
            }
        }

        __syncthreads();

        if (
            active_candidates ==
            0u
        ) {
            break;
        }

        const bool survives =
            sunopt_digital_survives_candidate_warp(
                export_power,
                demand_energy,
                data_length,
                tank_parameters,
                tank_index,
                current_index,
                horizon,
                current_storage,
                reserve,
                candidate_cuts[warp_id]
            );

        if (
            lane ==
            0
        ) {
            candidate_survives[warp_id] =
                survives ?
                    1u :
                    0u;
        }

        __syncthreads();

        if (
            tid ==
            0
        ) {
            eval_count +=
                active_candidates;

            int highest_surviving_warp =
                -1;

            for (
                unsigned int warp = 0;
                warp < active_candidates;
                ++warp
            ) {
                if (
                    candidate_survives[warp] !=
                    0u
                ) {
                    highest_surviving_warp =
                        static_cast<int>(
                            warp
                        );
                } else {
                    break;
                }
            }

            if (
                highest_surviving_warp <
                0
            ) {
                right_index =
                    candidate_indices[0] -
                    1;
            } else {
                const unsigned int winner =
                    static_cast<unsigned int>(
                        highest_surviving_warp
                    );

                best_cut =
                    candidate_cuts[winner];

                left_index =
                    candidate_indices[winner] +
                    1;

                if (
                    winner +
                        1 <
                    active_candidates
                ) {
                    right_index =
                        candidate_indices[
                            winner +
                            1
                        ] -
                        1;
                }
            }
        }

        __syncthreads();
    }

    if (
        tid ==
        0
    ) {
        cut_results[job_index] =
            best_cut;

        status[job_index] =
            2u;

        survival_evaluations[job_index] =
            eval_count;
    }
}


extern "C" int sunopt_cuda_run_digital_multilane_lockstep(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t candidate_warps,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_input_h2d_ms == nullptr ||
        out_kernel_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        step_count == 0 ||
        candidate_warps < 2 ||
        candidate_warps > 8 ||
        repetitions == 0
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    const unsigned int threads =
        candidate_warps *
        32u;

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

    double* d_presorted =
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

    const size_t total_jobs =
        step_count *
        tank_count;

    const size_t annual_bytes =
        data_length *
        sizeof(double);

    const size_t tank_bytes =
        tank_count *
        7 *
        sizeof(double);

    const size_t presorted_bytes =
        step_count *
        SUNOPT_DIGITAL_THREADS *
        sizeof(double);

    const size_t step_u32_bytes =
        step_count *
        sizeof(uint32_t);

    const size_t dynamic_bytes =
        total_jobs *
        3 *
        sizeof(double);

    const size_t cut_bytes =
        total_jobs *
        sizeof(double);

    const size_t u32_job_bytes =
        total_jobs *
        sizeof(uint32_t);

    auto cleanup =
        [&]() {
            if (d_evaluations) cudaFree(d_evaluations);
            if (d_status) cudaFree(d_status);
            if (d_cuts) cudaFree(d_cuts);
            if (d_dynamic) cudaFree(d_dynamic);
            if (d_horizon) cudaFree(d_horizon);
            if (d_current) cudaFree(d_current);
            if (d_presorted) cudaFree(d_presorted);
            if (d_tanks) cudaFree(d_tanks);
            if (d_demand) cudaFree(d_demand);
            if (d_export) cudaFree(d_export);

            if (stop_event) cudaEventDestroy(stop_event);
            if (start_event) cudaEventDestroy(start_event);
            if (stream) cudaStreamDestroy(stream);
        };

#define SUNOPT_MULTILANE_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_MULTILANE_MALLOC(ptr, bytes) \
    SUNOPT_MULTILANE_CHECK( \
        cudaMalloc( \
            reinterpret_cast<void**>(&(ptr)), \
            (bytes) \
        ) \
    )

    SUNOPT_MULTILANE_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaEventCreate(
            &start_event
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );

    SUNOPT_MULTILANE_MALLOC(
        d_export,
        annual_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_demand,
        annual_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_tanks,
        tank_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_presorted,
        presorted_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_current,
        step_u32_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_horizon,
        step_u32_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_dynamic,
        dynamic_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_cuts,
        cut_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_status,
        u32_job_bytes
    );

    SUNOPT_MULTILANE_MALLOC(
        d_evaluations,
        u32_job_bytes
    );

#undef SUNOPT_MULTILANE_MALLOC

    const auto h2d_start =
        Clock::now();

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_export,
            host_export_power,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_demand,
            host_demand_energy,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_tanks,
            host_tank_parameters,
            tank_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_presorted,
            host_presorted_export_windows,
            presorted_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_current,
            host_current_indices,
            step_u32_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_horizon,
            host_horizon_steps,
            step_u32_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            d_dynamic,
            host_dynamic_values,
            dynamic_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const auto h2d_stop =
        Clock::now();

    *out_input_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            h2d_stop -
            h2d_start
        ).count();

    const auto wall_start =
        Clock::now();

    SUNOPT_MULTILANE_CHECK(
        cudaEventRecord(
            start_event,
            stream
        )
    );

    for (
        uint32_t repetition = 0;
        repetition < repetitions;
        ++repetition
    ) {
        for (
            size_t step = 0;
            step < step_count;
            ++step
        ) {
            sunopt_digital_multilane_kernel<<<
                static_cast<unsigned int>(
                    tank_count
                ),
                threads,
                0,
                stream
            >>>(
                d_export,
                d_demand,
                d_presorted,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                step_count,
                step,
                candidate_warps,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_MULTILANE_CHECK(
                cudaGetLastError()
            );
        }
    }

    SUNOPT_MULTILANE_CHECK(
        cudaEventRecord(
            stop_event,
            stream
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaEventSynchronize(
            stop_event
        )
    );

    float kernel_total_ms =
        0.0f;

    SUNOPT_MULTILANE_CHECK(
        cudaEventElapsedTime(
            &kernel_total_ms,
            start_event,
            stop_event
        )
    );

    *out_kernel_ms =
        static_cast<double>(
            kernel_total_ms
        ) /
        static_cast<double>(
            repetitions
        );

    const auto d2h_start =
        Clock::now();

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            host_cut_results,
            d_cuts,
            cut_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            host_status,
            d_status,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_MULTILANE_CHECK(
        cudaMemcpy(
            host_survival_evaluations,
            d_evaluations,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    const auto d2h_stop =
        Clock::now();

    *out_result_d2h_ms =
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
        ).count() /
        static_cast<double>(
            repetitions
        );

#undef SUNOPT_MULTILANE_CHECK

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}

// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_END


// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_BEGIN
#include <cooperative_groups.h>

namespace sunopt_cg = cooperative_groups;

struct SunoptDigitalCluster4Control {
    double normal_cut;
    double candidate_cut;
    double best_cut;

    int first_valid;
    int left_index;
    int right_index;

    uint32_t valid_count;
    uint32_t eval_count;
    uint32_t final_status;

    bool parallel_supported;
    bool done;
    bool candidate_survives;
};


__device__ __forceinline__ bool
sunopt_digital_cluster4_survives(
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
    SunoptDigitalTransform* warp_partials,
    SunoptDigitalTransform* block_summary,
    SunoptDigitalCluster4Control* control
) {
    sunopt_cg::cluster_group cluster =
        sunopt_cg::this_cluster();

    const unsigned int cluster_rank =
        cluster.block_rank();

    const unsigned int tid =
        threadIdx.x;

    const unsigned int lane =
        tid & 31u;

    const unsigned int warp_id =
        tid >> 5;

    const size_t tank_base =
        tank_index * 7;

    const double target =
        tank_parameters[tank_base + 0];

    const double maximum_hp_power =
        tank_parameters[tank_base + 1];

    const double full_charge =
        tank_parameters[tank_base + 2];

    const double retention =
        tank_parameters[tank_base + 3];

    const double loss_intercept =
        tank_parameters[tank_base + 4];

    const double reserve =
        fmin(
            target,
            fmax(
                0.0,
                minimum_reserve_energy
            )
        );

    /*
     * Four blocks x 64 threads cover offsets 1..256.
     * The chronological partition is contiguous:
     * block 0 -> 1..64, block 1 -> 65..128, ...
     */
    const size_t future_offset =
        1 +
        static_cast<size_t>(
            cluster_rank
        ) *
            64 +
        static_cast<size_t>(
            tid
        );

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
                demand_energy[index]
            );

        const double charge =
            sunopt_digital_hp_on(
                export_power[index],
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

    /*
     * First reduce each 32-thread warp.
     */
    for (
        unsigned int stride = 1;
        stride < 32u;
        stride <<= 1
    ) {
        const SunoptDigitalTransform right =
            sunopt_digital_shuffle_down_transform(
                local,
                stride
            );

        const unsigned int group =
            stride << 1;

        if (
            (
                lane &
                (
                    group -
                    1
                )
            ) ==
            0
        ) {
            local =
                sunopt_digital_compose(
                    local,
                    right
                );
        }
    }

    if (
        lane ==
        0
    ) {
        warp_partials[
            warp_id
        ] =
            local;
    }

    __syncthreads();

    /*
     * Warp 0 composes the two contiguous 32-bucket summaries.
     */
    if (
        warp_id ==
        0
    ) {
        SunoptDigitalTransform block_local =
            lane < 2u ?
                warp_partials[lane] :
                sunopt_digital_identity();

        const SunoptDigitalTransform right =
            sunopt_digital_shuffle_down_transform(
                block_local,
                1u
            );

        if (
            lane ==
            0
        ) {
            block_local =
                sunopt_digital_compose(
                    block_local,
                    right
                );

            *block_summary =
                block_local;
        }
    }

    /*
     * All four block summaries are now valid and remain alive in DSM.
     */
    cluster.sync();

    if (
        cluster_rank ==
            0u &&
        tid ==
            0u
    ) {
        SunoptDigitalTransform summary =
            sunopt_digital_identity();

        for (
            unsigned int rank = 0;
            rank < 4u;
            ++rank
        ) {
            SunoptDigitalTransform* remote =
                cluster.map_shared_rank(
                    block_summary,
                    rank
                );

            summary =
                sunopt_digital_compose(
                    summary,
                    *remote
                );
        }

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

        control->candidate_survives =
            feasible &&
            available +
                1.0e-12 >=
                required;
    }

    cluster.sync();

    SunoptDigitalCluster4Control* root =
        cluster.map_shared_rank(
            control,
            0
        );

    return root->candidate_survives;
}


__global__ void
__cluster_dims__(4, 1, 1)
sunopt_digital_cluster4_kernel(
    const double* __restrict__ export_power,
    const double* __restrict__ demand_energy,
    const double* __restrict__ presorted_export_windows,
    size_t data_length,
    const double* __restrict__ tank_parameters,
    size_t tank_count,
    const uint32_t* __restrict__ current_indices,
    const uint32_t* __restrict__ horizon_steps,
    const double* __restrict__ dynamic_values,
    size_t step_count,
    size_t step_index,
    double* __restrict__ cut_results,
    uint32_t* __restrict__ status,
    uint32_t* __restrict__ survival_evaluations
) {
    sunopt_cg::cluster_group cluster =
        sunopt_cg::this_cluster();

    const unsigned int cluster_rank =
        cluster.block_rank();

    const unsigned int tid =
        threadIdx.x;

    const size_t tank_index =
        static_cast<size_t>(
            blockIdx.x
        ) /
        4;

    if (
        tank_index >=
        tank_count
    ) {
        return;
    }

    const size_t job_index =
        step_index *
            tank_count +
        tank_index;

    const size_t tank_base =
        tank_index * 7;

    const double target =
        tank_parameters[tank_base + 0];

    const double maximum_hp_power =
        tank_parameters[tank_base + 1];

    const double electrical_energy_per_on_bucket =
        tank_parameters[tank_base + 6];

    const size_t dynamic_base =
        job_index * 3;

    const double current_storage =
        dynamic_values[dynamic_base + 0];

    const double reserve =
        dynamic_values[dynamic_base + 1];

    const double requested_electrical_energy =
        dynamic_values[dynamic_base + 2];

    const size_t current_index =
        static_cast<size_t>(
            current_indices[step_index]
        );

    const size_t horizon =
        static_cast<size_t>(
            horizon_steps[step_index]
        );

    const double* sorted_export =
        presorted_export_windows +
        step_index *
            SUNOPT_DIGITAL_THREADS;

    __shared__ SunoptDigitalTransform warp_partials[2];
    __shared__ SunoptDigitalTransform block_summary;
    __shared__ SunoptDigitalCluster4Control control;

    SunoptDigitalCluster4Control* root =
        cluster.map_shared_rank(
            &control,
            0
        );

    if (
        cluster_rank ==
            0u &&
        tid ==
            0u
    ) {
        int first =
            static_cast<int>(
                SUNOPT_DIGITAL_THREADS
            );

        for (
            int i = 0;
            i <
                static_cast<int>(
                    SUNOPT_DIGITAL_THREADS
                );
            ++i
        ) {
            if (
                sorted_export[i] >=
                maximum_hp_power
            ) {
                first =
                    i;

                break;
            }
        }

        control.first_valid =
            first;

        control.valid_count =
            first <
                static_cast<int>(
                    SUNOPT_DIGITAL_THREADS
                ) ?
                static_cast<uint32_t>(
                    SUNOPT_DIGITAL_THREADS -
                    first
                ) :
                0u;

        const double maximum_schedulable =
            static_cast<double>(
                control.valid_count
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
            target_electrical <= 0.0 ||
            control.valid_count == 0 ||
            electrical_energy_per_on_bucket <=
                0.0
        ) {
            control.normal_cut =
                control.valid_count > 0 ?
                    sorted_export[
                        SUNOPT_DIGITAL_THREADS -
                        1
                    ] :
                    maximum_hp_power;
        } else if (
            target_electrical >=
            maximum_schedulable -
                1.0e-12
        ) {
            control.normal_cut =
                0.0;
        } else {
            uint32_t required_on =
                static_cast<uint32_t>(
                    ceil(
                        target_electrical /
                        electrical_energy_per_on_bucket -
                        1.0e-12
                    )
                );

            if (
                required_on < 1u
            ) {
                required_on =
                    1u;
            }

            if (
                required_on >
                control.valid_count
            ) {
                required_on =
                    control.valid_count;
            }

            control.normal_cut =
                fmax(
                    0.0,
                    sorted_export[
                        SUNOPT_DIGITAL_THREADS -
                        required_on
                    ] -
                    maximum_hp_power
                );
        }

        const bool affine_supported =
            tank_parameters[
                tank_base + 5
            ] >
            0.5;

        const double retention =
            tank_parameters[
                tank_base + 3
            ];

        const double clamped_reserve =
            fmin(
                target,
                fmax(
                    0.0,
                    reserve
                )
            );

        control.parallel_supported =
            affine_supported &&
            retention > 0.0 &&
            horizon > 0 &&
            horizon <=
                SUNOPT_DIGITAL_THREADS &&
            clamped_reserve >
                SUNOPT_DIGITAL_EPSILON;

        control.eval_count =
            0u;

        control.final_status =
            2u;

        control.best_cut =
            0.0;

        control.done =
            false;
    }

    cluster.sync();

    /*
     * Preserve the proven exact serial fallback for reserve ~= 0.
     */
    if (
        !root->parallel_supported
    ) {
        if (
            cluster_rank ==
                0u &&
            tid ==
                0u
        ) {
            control.eval_count =
                1u;

            const bool normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    control.normal_cut
                );

            if (
                normal_survives
            ) {
                control.final_status =
                    0u;

                control.best_cut =
                    control.normal_cut;
            } else {
                control.eval_count +=
                    1u;

                const bool maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !maxpv_survives
                ) {
                    control.final_status =
                        1u;

                    control.best_cut =
                        0.0;
                } else {
                    int left =
                        control.first_valid;

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    double best =
                        0.0;

                    for (
                        uint32_t iteration = 0;
                        iteration < 8u &&
                        left <= right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                sorted_export[middle] -
                                    maximum_hp_power
                            );

                        control.eval_count +=
                            1u;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    control.final_status =
                        2u;

                    control.best_cut =
                        best;
                }
            }

            cut_results[job_index] =
                control.best_cut;

            status[job_index] =
                control.final_status;

            survival_evaluations[job_index] =
                control.eval_count;
        }

        cluster.sync();

        return;
    }

    /*
     * Normal digital cut.
     */
    const bool normal_survives =
        sunopt_digital_cluster4_survives(
            export_power,
            demand_energy,
            data_length,
            tank_parameters,
            tank_index,
            current_index,
            horizon,
            current_storage,
            reserve,
            root->normal_cut,
            warp_partials,
            &block_summary,
            &control
        );

    if (
        cluster_rank ==
            0u &&
        tid ==
            0u
    ) {
        control.eval_count =
            1u;

        if (
            normal_survives
        ) {
            control.final_status =
                0u;

            control.best_cut =
                control.normal_cut;

            control.done =
                true;
        }
    }

    cluster.sync();

    if (
        root->done
    ) {
        if (
            cluster_rank ==
                0u &&
            tid ==
                0u
        ) {
            cut_results[job_index] =
                control.best_cut;

            status[job_index] =
                control.final_status;

            survival_evaluations[job_index] =
                control.eval_count;
        }

        cluster.sync();

        return;
    }

    /*
     * Maximum-PV cut = 0.
     */
    const bool maxpv_survives =
        sunopt_digital_cluster4_survives(
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
            warp_partials,
            &block_summary,
            &control
        );

    if (
        cluster_rank ==
            0u &&
        tid ==
            0u
    ) {
        control.eval_count =
            2u;

        if (
            !maxpv_survives
        ) {
            control.final_status =
                1u;

            control.best_cut =
                0.0;

            control.done =
                true;
        } else {
            control.left_index =
                control.first_valid;

            control.right_index =
                static_cast<int>(
                    SUNOPT_DIGITAL_THREADS -
                    1
                );

            control.best_cut =
                0.0;
        }
    }

    cluster.sync();

    if (
        root->done
    ) {
        if (
            cluster_rank ==
                0u &&
            tid ==
                0u
        ) {
            cut_results[job_index] =
                control.best_cut;

            status[job_index] =
                control.final_status;

            survival_evaluations[job_index] =
                control.eval_count;
        }

        cluster.sync();

        return;
    }

    /*
     * Same exact breakpoint binary search as the B8 reference.
     * Therefore the number and identity of survival evaluations should match.
     */
    for (
        uint32_t iteration = 0;
        iteration < 8u;
        ++iteration
    ) {
        if (
            cluster_rank ==
                0u &&
            tid ==
                0u
        ) {
            if (
                control.left_index >
                control.right_index
            ) {
                control.done =
                    true;
            } else {
                const int middle =
                    control.left_index +
                    (
                        control.right_index -
                        control.left_index
                    ) /
                    2;

                control.candidate_cut =
                    fmax(
                        0.0,
                        sorted_export[middle] -
                            maximum_hp_power
                    );

                /*
                 * Reuse right_index temporarily only after the evaluator:
                 * middle is recoverable by another exact midpoint operation
                 * from the current bounds, so no extra shared field is needed.
                 */
                control.done =
                    false;
            }
        }

        cluster.sync();

        if (
            root->done
        ) {
            break;
        }

        const int middle =
            root->left_index +
            (
                root->right_index -
                root->left_index
            ) /
                2;

        const bool survives =
            sunopt_digital_cluster4_survives(
                export_power,
                demand_energy,
                data_length,
                tank_parameters,
                tank_index,
                current_index,
                horizon,
                current_storage,
                reserve,
                root->candidate_cut,
                warp_partials,
                &block_summary,
                &control
            );

        if (
            cluster_rank ==
                0u &&
            tid ==
                0u
        ) {
            control.eval_count +=
                1u;

            if (
                survives
            ) {
                control.best_cut =
                    control.candidate_cut;

                control.left_index =
                    middle +
                    1;
            } else {
                control.right_index =
                    middle -
                    1;
            }
        }

        cluster.sync();
    }

    if (
        cluster_rank ==
            0u &&
        tid ==
            0u
    ) {
        control.final_status =
            2u;

        cut_results[job_index] =
            control.best_cut;

        status[job_index] =
            control.final_status;

        survival_evaluations[job_index] =
            control.eval_count;
    }

    cluster.sync();
}


extern "C" int sunopt_cuda_run_digital_cluster4_lockstep(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const double* host_presorted_export_windows,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    size_t step_count,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_input_h2d_ms == nullptr ||
        out_kernel_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
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

    double* d_presorted =
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

    const size_t total_jobs =
        step_count *
        tank_count;

    const size_t annual_bytes =
        data_length *
        sizeof(double);

    const size_t tank_bytes =
        tank_count *
        7 *
        sizeof(double);

    const size_t presorted_bytes =
        step_count *
        SUNOPT_DIGITAL_THREADS *
        sizeof(double);

    const size_t step_u32_bytes =
        step_count *
        sizeof(uint32_t);

    const size_t dynamic_bytes =
        total_jobs *
        3 *
        sizeof(double);

    const size_t cut_bytes =
        total_jobs *
        sizeof(double);

    const size_t u32_job_bytes =
        total_jobs *
        sizeof(uint32_t);

    auto cleanup =
        [&]() {
            if (d_evaluations) cudaFree(d_evaluations);
            if (d_status) cudaFree(d_status);
            if (d_cuts) cudaFree(d_cuts);
            if (d_dynamic) cudaFree(d_dynamic);
            if (d_horizon) cudaFree(d_horizon);
            if (d_current) cudaFree(d_current);
            if (d_presorted) cudaFree(d_presorted);
            if (d_tanks) cudaFree(d_tanks);
            if (d_demand) cudaFree(d_demand);
            if (d_export) cudaFree(d_export);

            if (stop_event) cudaEventDestroy(stop_event);
            if (start_event) cudaEventDestroy(start_event);
            if (stream) cudaStreamDestroy(stream);
        };

#define SUNOPT_CLUSTER4_CHECK(call) \
    do { \
        err = (call); \
        if (err != cudaSuccess) { \
            cleanup(); \
            return static_cast<int>(err); \
        } \
    } while (false)

#define SUNOPT_CLUSTER4_MALLOC(ptr, bytes) \
    SUNOPT_CLUSTER4_CHECK( \
        cudaMalloc( \
            reinterpret_cast<void**>(&(ptr)), \
            (bytes) \
        ) \
    )

    SUNOPT_CLUSTER4_CHECK(
        cudaStreamCreateWithFlags(
            &stream,
            cudaStreamNonBlocking
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaEventCreate(
            &start_event
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaEventCreate(
            &stop_event
        )
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_export,
        annual_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_demand,
        annual_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_tanks,
        tank_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_presorted,
        presorted_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_current,
        step_u32_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_horizon,
        step_u32_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_dynamic,
        dynamic_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_cuts,
        cut_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_status,
        u32_job_bytes
    );

    SUNOPT_CLUSTER4_MALLOC(
        d_evaluations,
        u32_job_bytes
    );

#undef SUNOPT_CLUSTER4_MALLOC

    const auto h2d_start =
        Clock::now();

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_export,
            host_export_power,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_demand,
            host_demand_energy,
            annual_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_tanks,
            host_tank_parameters,
            tank_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_presorted,
            host_presorted_export_windows,
            presorted_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_current,
            host_current_indices,
            step_u32_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_horizon,
            host_horizon_steps,
            step_u32_bytes,
            cudaMemcpyHostToDevice
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            d_dynamic,
            host_dynamic_values,
            dynamic_bytes,
            cudaMemcpyHostToDevice
        )
    );

    const auto h2d_stop =
        Clock::now();

    *out_input_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            h2d_stop -
            h2d_start
        ).count();

    const auto wall_start =
        Clock::now();

    SUNOPT_CLUSTER4_CHECK(
        cudaEventRecord(
            start_event,
            stream
        )
    );

    for (
        uint32_t repetition = 0;
        repetition < repetitions;
        ++repetition
    ) {
        for (
            size_t step = 0;
            step < step_count;
            ++step
        ) {
            sunopt_digital_cluster4_kernel<<<
                static_cast<unsigned int>(
                    tank_count *
                    4
                ),
                64,
                0,
                stream
            >>>(
                d_export,
                d_demand,
                d_presorted,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                step_count,
                step,
                d_cuts,
                d_status,
                d_evaluations
            );

            SUNOPT_CLUSTER4_CHECK(
                cudaGetLastError()
            );
        }
    }

    SUNOPT_CLUSTER4_CHECK(
        cudaEventRecord(
            stop_event,
            stream
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaEventSynchronize(
            stop_event
        )
    );

    float kernel_total_ms =
        0.0f;

    SUNOPT_CLUSTER4_CHECK(
        cudaEventElapsedTime(
            &kernel_total_ms,
            start_event,
            stop_event
        )
    );

    *out_kernel_ms =
        static_cast<double>(
            kernel_total_ms
        ) /
        static_cast<double>(
            repetitions
        );

    const auto d2h_start =
        Clock::now();

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            host_cut_results,
            d_cuts,
            cut_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            host_status,
            d_status,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    SUNOPT_CLUSTER4_CHECK(
        cudaMemcpy(
            host_survival_evaluations,
            d_evaluations,
            u32_job_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    const auto d2h_stop =
        Clock::now();

    *out_result_d2h_ms =
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
        ).count() /
        static_cast<double>(
            repetitions
        );

#undef SUNOPT_CLUSTER4_CHECK

    cleanup();

    return static_cast<int>(
        cudaSuccess
    );
}
// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_END


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
