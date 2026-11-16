#include "cuda_context.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cfloat>
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


            /*
             * demand_window[offset] contains the precombined thermal
             * load for this exact fast path:
             *
             *   heatLossIntercept + demand
             *
             * Therefore the complete pre-charge recurrence is only:
             *
             *   net =
             *       storedEnergy * (1 - heatLossSlope)
             *       - thermalLoad
             *
             * This removes two dependent FP64 add/sub operations from
             * every chronological candidate step.
             */
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



/*
 * ================================================================
 * PARALLEL AFFINE CAUSAL SEGMENT COMPOSITION
 * ================================================================
 *
 * Production positive-reserve recurrence for offsets >= 1:
 *
 *   net_t =
 *       retention * E_t
 *       - thermalLoad_t
 *
 *   survive iff:
 *       net_t >= minimumNet
 *
 *   E_(t+1) =
 *       min(
 *           target,
 *           net_t + hpThermal_t
 *       )
 *
 * We only need to know whether a candidate survives, not the final
 * storage state. Propagating the minimum required storage backwards
 * through one timestep gives:
 *
 *   R_before =
 *       max(
 *           C,
 *           S * R_after + B
 *       )
 *
 * valid only while:
 *
 *   R_after <= U
 *
 * where for one timestep:
 *
 *   S = 1 / retention
 *   B = (thermalLoad - hpThermal) / retention
 *   C = (minimumNet + thermalLoad) / retention
 *   U = targetStorageEnergy
 *
 * This function family is closed under composition. Therefore 191
 * chronological steps can be split into 32 contiguous segments,
 * composed locally by 32 lanes, then combined by a five-stage warp
 * scan.
 */

struct SunoptAffineRequirementSegment {
    double slope;
    double intercept;
    double floor;
    double maximum_future_requirement;
    int valid;
};


__device__ static inline SunoptAffineRequirementSegment
identity_affine_requirement_segment() {
    SunoptAffineRequirementSegment result{};

    result.slope =
        1.0;

    result.intercept =
        0.0;

    result.floor =
        -DBL_MAX;

    result.maximum_future_requirement =
        DBL_MAX;

    result.valid =
        1;

    return result;
}


/*
 * Compose:
 *
 *   earlier(
 *       later(
 *           x
 *       )
 *   )
 *
 * The operation is associative for positive slopes.
 */
__device__ static inline SunoptAffineRequirementSegment
compose_affine_requirement_segments(
    const SunoptAffineRequirementSegment& earlier,
    const SunoptAffineRequirementSegment& later
) {
    SunoptAffineRequirementSegment result{};

    if (
        !earlier.valid ||
        !later.valid
    ) {
        result.valid =
            0;

        return result;
    }


    /*
     * later(x) must remain inside the valid input domain of earlier.
     *
     * later(x) =
     *     max(
     *         later.floor,
     *         later.slope*x + later.intercept
     *     )
     */
    if (
        later.floor >
        earlier.maximum_future_requirement
    ) {
        result.valid =
            0;

        return result;
    }


    result.slope =
        earlier.slope *
        later.slope;


    result.intercept =
        earlier.slope *
        later.intercept +
        earlier.intercept;


    const double propagated_floor =
        earlier.slope *
        later.floor +
        earlier.intercept;


    result.floor =
        earlier.floor >
        propagated_floor
            ? earlier.floor
            : propagated_floor;


    const double future_limit_from_earlier =
        (
            earlier.maximum_future_requirement -
            later.intercept
        ) /
        later.slope;


    result.maximum_future_requirement =
        later.maximum_future_requirement <
        future_limit_from_earlier
            ? later.maximum_future_requirement
            : future_limit_from_earlier;


    result.valid =
        1;

    return result;
}


__device__ static inline SunoptAffineRequirementSegment
shuffle_up_affine_requirement_segment(
    const SunoptAffineRequirementSegment& value,
    int delta,
    unsigned int lane
) {
    SunoptAffineRequirementSegment shifted{};

    shifted.slope =
        __shfl_up_sync(
            0xffffffffu,
            value.slope,
            delta
        );

    shifted.intercept =
        __shfl_up_sync(
            0xffffffffu,
            value.intercept,
            delta
        );

    shifted.floor =
        __shfl_up_sync(
            0xffffffffu,
            value.floor,
            delta
        );

    shifted.maximum_future_requirement =
        __shfl_up_sync(
            0xffffffffu,
            value.maximum_future_requirement,
            delta
        );

    shifted.valid =
        __shfl_up_sync(
            0xffffffffu,
            value.valid,
            delta
        );


    if (
        lane <
        static_cast<unsigned int>(
            delta
        )
    ) {
        return identity_affine_requirement_segment();
    }


    return shifted;
}


/*
 * One complete warp evaluates one cut-level candidate.
 *
 * All 32 lanes return the same survival result.
 */
__device__ static bool survives_window_segment_parallel(
    const double* export_window,
    const double* thermal_load_window,
    double cut_level_power,
    const SunoptCausalParameters& p,
    const SunoptHeatLossModel& heat_loss_model,
    unsigned int lane
) {
    const unsigned int horizon_steps =
        static_cast<unsigned int>(
            p.horizon_steps
        );


    const double retention =
        heat_loss_model
          .heat_loss_retention;


    /*
     * Segment composition requires a strictly monotone positive
     * recurrence coefficient. Unusual parameter sets use the exact
     * serial fallback in the caller.
     */
    if (
        retention <=
        0.0
    ) {
        return false;
    }


    const double target_storage_energy =
        p.target_storage_energy;

    const double minimum_net_energy =
        heat_loss_model
          .minimum_net_energy_after_demand;

    const double maximum_heat_pump_electrical_power =
        p.maximum_heat_pump_electrical_power;

    const double hp_thermal_energy_per_electrical_power =
        heat_loss_model
          .hp_thermal_energy_per_electrical_power;


    /*
     * OFFSET 0 remains special exactly as in the serial kernel:
     * charge first, then cap at target; no demand/loss check.
     *
     * Every lane computes this identical scalar. That is cheaper than
     * introducing another shared-memory handoff.
     */
    double initial_energy =
        heat_loss_model
          .initial_storage_energy;


    if (
        horizon_steps >
        0u
    ) {
        const double hp_power =
            hp_power_at_cut_level(
                export_window[
                    0
                ],
                cut_level_power,
                maximum_heat_pump_electrical_power
            );

        initial_energy +=
            hp_power *
            hp_thermal_energy_per_electrical_power;


        if (
            initial_energy >
            target_storage_energy
        ) {
            initial_energy =
                target_storage_energy;
        }
    }


    if (
        horizon_steps <=
        1u
    ) {
        return true;
    }


    const unsigned int chronological_steps =
        horizon_steps -
        1u;


    /*
     * Contiguous chunks are essential: segment composition is
     * associative but not commutative.
     */
    const unsigned int chunk_steps =
        (
            chronological_steps +
            31u
        ) /
        32u;


    const unsigned int begin_offset =
        1u +
        lane *
        chunk_steps;


    unsigned int end_offset =
        begin_offset +
        chunk_steps;


    if (
        end_offset >
        horizon_steps
    ) {
        end_offset =
            horizon_steps;
    }


    SunoptAffineRequirementSegment local =
        identity_affine_requirement_segment();


    const double inverse_retention =
        1.0 /
        retention;


    for (
        unsigned int offset =
            begin_offset;
        offset <
            end_offset;
        ++offset
    ) {
        const double hp_power =
            hp_power_at_cut_level(
                export_window[
                    offset
                ],
                cut_level_power,
                maximum_heat_pump_electrical_power
            );


        const double hp_thermal_energy =
            hp_power *
            hp_thermal_energy_per_electrical_power;


        const double thermal_load =
            thermal_load_window[
                offset
            ];


        SunoptAffineRequirementSegment step{};

        step.slope =
            inverse_retention;

        step.intercept =
            (
                thermal_load -
                hp_thermal_energy
            ) *
            inverse_retention;

        step.floor =
            (
                minimum_net_energy +
                thermal_load
            ) *
            inverse_retention;

        step.maximum_future_requirement =
            target_storage_energy;

        step.valid =
            1;


        local =
            compose_affine_requirement_segments(
                local,
                step
            );
    }


    /*
     * Inclusive ordered warp scan.
     *
     * After delta=16, lane 31 contains the exact composition of all
     * 32 chronological chunks.
     */
    for (
        int delta = 1;
        delta < 32;
        delta <<= 1
    ) {
        const SunoptAffineRequirementSegment earlier =
            shuffle_up_affine_requirement_segment(
                local,
                delta,
                lane
            );


        if (
            lane >=
            static_cast<unsigned int>(
                delta
            )
        ) {
            local =
                compose_affine_requirement_segments(
                    earlier,
                    local
                );
        }
    }


    int survives =
        0;


    if (
        lane ==
        31u
    ) {
        /*
         * There is no requirement after the final horizon step.
         * R_after = 0 is exact here because:
         *
         *   minimumNet > 0
         *   hpThermal  >= 0
         *
         * so the final step's floor dominates any negative
         * no-future-demand term.
         */
        const double required_from_future =
            local.intercept;


        const double required_initial_energy =
            local.floor >
            required_from_future
                ? local.floor
                : required_from_future;


        survives =
            (
                local.valid &&
                local.maximum_future_requirement >=
                    0.0 &&
                initial_energy >=
                    required_initial_energy
            )
                ? 1
                : 0;
    }


    survives =
        __shfl_sync(
            0xffffffffu,
            survives,
            31
        );


    return survives !=
        0;
}


__global__ static void causal_batch_kernel(
    const double* export_power,
    const double* demand_energy,
    const SunoptCausalParameters* parameters,
    SunoptCausalResult* results,
    size_t job_count
) {
    const size_t job =
        static_cast<size_t>(
            blockIdx.x
        );


    if (
        job >=
        job_count
    ) {
        return;
    }


    const unsigned int thread =
        threadIdx.x;

    const unsigned int warp =
        thread >>
        5;

    const unsigned int lane =
        thread &
        31u;


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

    __shared__
    int candidate_survives_shared[
        32
    ];


    if (
        thread ==
        0u
    ) {
        p =
            parameters[
                job
            ];

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


    /*
     * 1024 threads are available, so the complete 192-value horizon
     * is staged in one cooperative pass.
     */
    if (
        use_shared_window &&
        thread <
        static_cast<unsigned int>(
            p.horizon_steps
        )
    ) {
        const unsigned int offset =
            thread;


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


    __syncthreads();


    const SunoptHeatLossModel heat_loss_model =
        heat_loss_model_shared;


    const bool use_parallel_segments =
        (
            use_shared_window &&
            heat_loss_model
              .affine_survival_fast_path &&
            heat_loss_model
              .heat_loss_retention >
                0.0
        );


    /*
     * ENERGY WATERLINE
     *
     * One warp is sufficient. All other warps wait at the block
     * barrier and are then reused for boundary/radix candidates.
     */
    if (
        p.requested_electrical_energy >=
        0.0
    ) {
        if (
            warp ==
            0u
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


            if (
                lane ==
                0u
            ) {
                p.energy_cut_level =
                    energy_cut_level;
            }
        }


        __syncthreads();
    }


    /*
     * INITIAL BOUNDARY TESTS
     *
     * warp 0 = normal energy Waterline
     * warp 1 = maximum-PV Waterline 0
     */
    if (
        warp <
        2u
    ) {
        const double cut =
            warp ==
            0u
                ? p.energy_cut_level
                : 0.0;


        bool survives =
            false;


        if (
            use_parallel_segments
        ) {
            survives =
                survives_window_segment_parallel(
                    export_window,
                    demand_window,
                    cut,
                    p,
                    heat_loss_model,
                    lane
                );

        } else if (
            lane ==
            0u
        ) {
            unsigned int ignored_checks =
                0;


            survives =
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


        if (
            lane ==
            0u
        ) {
            candidate_survives_shared[
                warp
            ] =
                survives
                    ? 1
                    : 0;
        }
    }


    __syncthreads();


    if (
        thread ==
        0u
    ) {
        normal_plan_survives_shared =
            candidate_survives_shared[
                0
            ];

        maximum_pv_plan_survives_shared =
            candidate_survives_shared[
                1
            ];


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


    /*
     * FOUR BASE-32 REFINEMENT ROUNDS
     *
     * One physical warp per cut candidate:
     *
     *   warp  0 -> candidate  1
     *   ...
     *   warp 31 -> candidate 32
     *
     * Inside each warp, 32 lanes parallelize the 191 chronological
     * recurrence steps by associative segment composition.
     */
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
                    warp +
                    1u
                );


            bool survives =
                false;


            if (
                use_parallel_segments
            ) {
                survives =
                    survives_window_segment_parallel(
                        export_window,
                        demand_window,
                        candidate,
                        p,
                        heat_loss_model,
                        lane
                    );

            } else if (
                lane ==
                0u
            ) {
                unsigned int ignored_checks =
                    0;


                survives =
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
            }


            if (
                lane ==
                0u
            ) {
                candidate_survives_shared[
                    warp
                ] =
                    survives
                        ? 1
                        : 0;
            }


            __syncthreads();


            /*
             * Warp 0 turns the 32 per-warp booleans back into the same
             * base-32 survival mask used by the proven 32-thread
             * search geometry.
             */
            if (
                warp ==
                0u
            ) {
                const bool lane_survives =
                    candidate_survives_shared[
                        lane
                    ] !=
                    0;


                const unsigned int surviving_mask =
                    __ballot_sync(
                        0xffffffffu,
                        lane_survives
                    );


                if (
                    lane ==
                    0u
                ) {
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
            }


            __syncthreads();
        }
    }


    if (
        thread ==
        0u
    ) {
        SunoptCausalResult result{};


        result.normal_plan_survives =
            normal_plan_survives_shared;

        result.maximum_pv_plan_survives =
            maximum_pv_plan_survives_shared;


        if (
            normal_plan_survives_shared
        ) {
            result.cut_level_power =
                p.energy_cut_level;

            result.survival_checks =
                1;

            result.binary_search_iterations =
                0;

        } else if (
            !maximum_pv_plan_survives_shared
        ) {
            result.cut_level_power =
                0.0;

            result.survival_checks =
                2;

            result.binary_search_iterations =
                0;

        } else {
            result.cut_level_power =
                lower_cut_shared;

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

    err = cudaEventRecord(
        context->profile_start,
        context->stream
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

    err = cudaEventRecord(
        context->profile_after_h2d,
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
     * One complete 32-warp block per causal job/tank.
     *
     * 28 tanks:
     *
     *   28 blocks x 1024 threads
     *   = one warp per radix candidate
     *
     * Each candidate warp parallelizes the chronological recurrence
     * through exact affine segment composition.
     */
    constexpr unsigned int THREADS =
        1024;

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
        device_results,
        job_count
    );

    err = cudaGetLastError();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaEventRecord(
        context->profile_after_kernel,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
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

    err = cudaEventRecord(
        context->profile_after_d2h,
        context->stream
    );

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const auto wait_start =
        Clock::now();

    err = cudaEventSynchronize(
        context->profile_after_d2h
    );

    const auto wait_end =
        Clock::now();

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    float parameter_h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float result_d2h_ms = 0.0f;
    float gpu_total_ms = 0.0f;

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

    context->profile.causal_calls += 1;

    context->profile.causal_jobs +=
        static_cast<uint64_t>(
            job_count
        );

    context->profile.parameter_h2d_ms +=
        static_cast<double>(
            parameter_h2d_ms
        );

    context->profile.kernel_ms +=
        static_cast<double>(
            kernel_ms
        );

    context->profile.result_d2h_ms +=
        static_cast<double>(
            result_d2h_ms
        );

    context->profile.gpu_total_ms +=
        static_cast<double>(
            gpu_total_ms
        );

    context->profile.host_wait_ms +=
        host_wait_ms;

    context->profile.wall_ms +=
        wall_ms;



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
