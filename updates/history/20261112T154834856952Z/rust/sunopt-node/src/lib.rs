// rust/sunopt-node/src/lib.rs

use std::{
    cell::{
        Cell,
        RefCell,
    },
    collections::HashMap,
    path::Path,
};

use napi::bindgen_prelude::{
    Error,
    Float64Array,
    Result,
    Status,
};

use napi_derive::napi;

use sunopt_core::{
    acceleration_scope_cache_key,
    compile_acceleration_scope_from_typescript,

    CompiledAccelerationScope,

    NoSunshineKernelParameters,

    PeakShavingCausalKernelParameters,
};


/*
 * ================================================================
 * ONE GLOBAL JIT CACHE
 * ================================================================
 *
 * Key:
 *
 *   hash(TypeScript source + rust_* scope ID)
 *
 * JITModule owns executable memory and is intentionally retained
 * inside the process-local cache.
 */


thread_local! {

    static ACCELERATION_SCOPE_CACHE:
        RefCell<
            HashMap<
                u64,
                CompiledAccelerationScope
            >
        > =
        RefCell::new(
            HashMap::new(),
        );

    /*
     * Approach A1:
     *
     * The TypeScript source is registered once. Hot-path calls use a small
     * u32 handle instead of resending and re-hashing the complete source.
     *
     * The existing source-based exports remain available as the reference /
     * backwards-compatible path.
     */
    static REGISTERED_SCOPE_HANDLES:
        RefCell<
            HashMap<
                u32,
                u64
            >
        > =
        RefCell::new(
            HashMap::new(),
        );

    static NEXT_REGISTERED_SCOPE_HANDLE:
        Cell<u32> =
        const {
            Cell::new(1)
        };
}


fn native_error(
    message:
        impl Into<String>,
) -> Error {

    Error::new(
        Status::GenericFailure,
        message.into(),
    )
}


fn invalid_argument(
    message:
        impl Into<String>,
) -> Error {

    Error::new(
        Status::InvalidArg,
        message.into(),
    )
}


fn scope_path()
    -> &'static Path {

    /*
     * Oxc only needs a .ts path so SourceType is TypeScript.
     *
     * The actual source text is passed from the global TypeScript
     * runtime and is not read from this Rust path.
     */
    Path::new(
        "rust-acceleration-scopes.ts",
    )
}


fn ensure_compiled_scope(
    source:
        &str,

    scope_id:
        &str,
) -> Result<u64> {

    if !scope_id.starts_with(
        "rust_",
    ) {

        return Err(
            invalid_argument(
                format!(
                    "Acceleration scope ID must start with rust_: {scope_id}"
                ),
            ),
        );
    }


    /*
     * Cheap cache lookup: NO Oxc parse here.
     *
     * Oxc is invoked only below when this source+scope combination
     * is not already compiled.
     */
    let source_hash =
        acceleration_scope_cache_key(
            source,
            scope_id,
        )
        .map_err(
            native_error,
        )?;


    ACCELERATION_SCOPE_CACHE.with(
        |cache| -> Result<()> {

            if cache
                .borrow()
                .contains_key(
                    &source_hash,
                )
            {

                return Ok(());
            }


            let compiled =
                compile_acceleration_scope_from_typescript(
                    source,
                    scope_path(),
                    scope_id,
                )
                .map_err(
                    native_error,
                )?;


            cache
                .borrow_mut()
                .insert(
                    source_hash,
                    compiled,
                );


            Ok(())
        },
    )?;


    Ok(
        source_hash,
    )
}


/*
 * ================================================================
 * 7-DAY N-API TYPES
 * ================================================================
 */


#[napi(object)]
pub struct NoSunshineParametersInput {

    #[napi(js_name = "currentIndex")]
    pub current_index:
        u32,

    #[napi(js_name = "horizonSteps")]
    pub horizon_steps:
        u32,

    #[napi(js_name = "protectedSteps")]
    pub protected_steps:
        u32,

    #[napi(js_name = "timeStepHours")]
    pub time_step_hours:
        f64,

    #[napi(js_name = "householdReservePower")]
    pub household_reserve_power:
        f64,

    #[napi(js_name = "maximumHeatPumpElectricalPower")]
    pub maximum_heat_pump_electrical_power:
        f64,

    pub cop:
        f64,

    #[napi(js_name = "initialWaterlinePower")]
    pub initial_waterline_power:
        f64,

    #[napi(js_name = "targetStorageEnergy")]
    pub target_storage_energy:
        f64,

    #[napi(js_name = "storageEnergyPerKelvin")]
    pub storage_energy_per_kelvin:
        f64,

    #[napi(js_name = "storageConductanceWPerKelvin")]
    pub storage_conductance_w_per_kelvin:
        f64,

    #[napi(js_name = "storageMinTemperature")]
    pub storage_min_temperature:
        f64,

    #[napi(js_name = "ambientTemperature")]
    pub ambient_temperature:
        f64,

    #[napi(js_name = "dailyHeatLoss")]
    pub daily_heat_loss:
        f64,

    #[napi(js_name = "currentStorageEnergy")]
    pub current_storage_energy:
        f64,
}


#[napi(object)]
pub struct NoSunshineResultOutput {

    #[napi(js_name = "kWhNoSunshine")]
    pub kwh_no_sunshine:
        f64,

    #[napi(js_name = "bucketDemandEnergy")]
    pub bucket_demand_energy:
        f64,

    #[napi(js_name = "bucketCoolingLossEnergy")]
    pub bucket_cooling_loss_energy:
        f64,

    #[napi(js_name = "preBucketHoldingLossEnergy")]
    pub pre_bucket_holding_loss_energy:
        f64,

    #[napi(js_name = "allocatedPvThermalEnergy")]
    pub allocated_pv_thermal_energy:
        f64,

    #[napi(js_name = "unavoidableThermalDeficit")]
    pub unavoidable_thermal_deficit:
        f64,

    #[napi(js_name = "requiredStorageEnergyAtStart")]
    pub required_storage_energy_at_start:
        f64,

    #[napi(js_name = "additionalStorageEnergyNeededAtStart")]
    pub additional_storage_energy_needed_at_start:
        f64,

    #[napi(js_name = "currentWaterlinePower")]
    pub current_waterline_power:
        f64,

    #[napi(js_name = "currentScheduledHpElectricalPower")]
    pub current_scheduled_hp_electrical_power:
        f64,

    #[napi(js_name = "currentScheduledHpThermalPower")]
    pub current_scheduled_hp_thermal_power:
        f64,
}


/*
 * ================================================================
 * 48-HOUR CAUSAL N-API TYPES
 * ================================================================
 */


#[napi(object)]
pub struct PeakShavingCausalParametersInput {

    #[napi(js_name = "currentIndex")]
    pub current_index:
        u32,

    #[napi(js_name = "horizonSteps")]
    pub horizon_steps:
        u32,

    #[napi(js_name = "currentStorageEnergy")]
    pub current_storage_energy:
        f64,

    #[napi(js_name = "minimumReserveEnergy")]
    pub minimum_reserve_energy:
        f64,

    #[napi(js_name = "targetStorageEnergy")]
    pub target_storage_energy:
        f64,

    #[napi(js_name = "maximumHeatPumpElectricalPower")]
    pub maximum_heat_pump_electrical_power:
        f64,

    pub cop:
        f64,

    #[napi(js_name = "energyCutLevel")]
    pub energy_cut_level:
        f64,

    #[napi(js_name = "timeStepHours")]
    pub time_step_hours:
        f64,

    #[napi(js_name = "storageEnergyPerKelvin")]
    pub storage_energy_per_kelvin:
        f64,

    #[napi(js_name = "storageConductanceWPerKelvin")]
    pub storage_conductance_w_per_kelvin:
        f64,

    #[napi(js_name = "storageMinTemperature")]
    pub storage_min_temperature:
        f64,

    #[napi(js_name = "ambientTemperature")]
    pub ambient_temperature:
        f64,

    #[napi(js_name = "dailyHeatLoss")]
    pub daily_heat_loss:
        f64,
}


#[napi(object)]
pub struct PeakShavingCausalResultOutput {

    #[napi(js_name = "cutLevelPower")]
    pub cut_level_power:
        f64,

    #[napi(js_name = "normalPlanSurvives")]
    pub normal_plan_survives:
        bool,

    #[napi(js_name = "maximumPvPlanSurvives")]
    pub maximum_pv_plan_survives:
        bool,

    #[napi(js_name = "survivalChecks")]
    pub survival_checks:
        f64,

    #[napi(js_name = "binarySearchIterations")]
    pub binary_search_iterations:
        f64,
}


/*
 * ================================================================
 * APPROACH A1 — REGISTERED SOURCE HANDLES
 * ================================================================
 *
 * A0 measured a large hot-path cost from repeatedly transporting and hashing
 * the complete TypeScript source even after the compiled scope was cached.
 *
 * Registration performs the existing source validation / compilation exactly
 * once and returns a small process-local handle. Execution by handle still
 * reaches the SAME CompiledAccelerationScope object.
 */


fn registered_cache_key(
    handle:
        u32,
) -> Result<u64> {

    REGISTERED_SCOPE_HANDLES.with(
        |handles| {

            handles
                .borrow()
                .get(
                    &handle,
                )
                .copied()
                .ok_or_else(
                    || {
                        invalid_argument(
                            format!(
                                "Unknown or expired acceleration scope handle: {handle}"
                            ),
                        )
                    },
                )
        },
    )
}


#[napi]
pub fn register_acceleration_scope(
    source:
        String,

    scope_id:
        String,
) -> Result<u32> {

    let cache_key =
        ensure_compiled_scope(
            &source,
            &scope_id,
        )?;


    let handle =
        NEXT_REGISTERED_SCOPE_HANDLE.with(
            |next| {

                let handle =
                    next.get();

                let following =
                    handle
                        .checked_add(1)
                        .filter(
                            |value| {
                                *value !=
                                0
                            },
                        )
                        .unwrap_or(1);

                next.set(
                    following,
                );

                handle
            },
        );


    REGISTERED_SCOPE_HANDLES.with(
        |handles| {

            handles
                .borrow_mut()
                .insert(
                    handle,
                    cache_key,
                );
        },
    );


    Ok(
        handle,
    )
}


/*
 * ================================================================
 * NATIVE EXECUTION
 * ================================================================
 */


fn execute_no_sunshine_from_cache_key(
    cache_key:
        u64,

    pv_power:
        &Float64Array,

    load_energy:
        &Float64Array,

    parameters:
        NoSunshineParametersInput,
) -> Result<
    NoSunshineResultOutput
> {

    let kernel_parameters =
        NoSunshineKernelParameters {

            current_index:
                parameters.current_index
                    as usize,

            horizon_steps:
                parameters.horizon_steps
                    as usize,

            protected_steps:
                parameters.protected_steps
                    as usize,

            time_step_hours:
                parameters.time_step_hours,

            household_reserve_power:
                parameters
                    .household_reserve_power,

            maximum_heat_pump_electrical_power:
                parameters
                    .maximum_heat_pump_electrical_power,

            cop:
                parameters.cop,

            initial_waterline_power:
                parameters
                    .initial_waterline_power,

            target_storage_energy:
                parameters
                    .target_storage_energy,

            storage_energy_per_kelvin:
                parameters
                    .storage_energy_per_kelvin,

            storage_conductance_w_per_kelvin:
                parameters
                    .storage_conductance_w_per_kelvin,

            storage_min_temperature:
                parameters
                    .storage_min_temperature,

            ambient_temperature:
                parameters
                    .ambient_temperature,

            daily_heat_loss:
                parameters
                    .daily_heat_loss,

            current_storage_energy:
                parameters
                    .current_storage_energy,
        };


    let result =
        ACCELERATION_SCOPE_CACHE.with(
            |cache|
                -> Result<
                    sunopt_core::NoSunshineKernelResult
                >
            {

                let cache =
                    cache.borrow();


                let compiled =
                    cache
                        .get(
                            &cache_key,
                        )
                        .ok_or_else(
                            || {
                                native_error(
                                    "Compiled rust_noSunshine scope disappeared from cache.",
                                )
                            },
                        )?;


                match compiled {

                    CompiledAccelerationScope::NoSunshine(
                        kernel,
                    ) =>
                        kernel
                            .execute(
                                pv_power,
                                load_energy,
                                kernel_parameters,
                            )
                            .map_err(
                                native_error,
                            ),

                    _ =>
                        Err(
                            native_error(
                                "Acceleration cache type mismatch for rust_noSunshine.",
                            ),
                        ),
                }
            },
        )?;


    Ok(
        NoSunshineResultOutput {

            kwh_no_sunshine:
                result
                    .kwh_no_sunshine,

            bucket_demand_energy:
                result
                    .bucket_demand_energy,

            bucket_cooling_loss_energy:
                result
                    .bucket_cooling_loss_energy,

            pre_bucket_holding_loss_energy:
                result
                    .pre_bucket_holding_loss_energy,

            allocated_pv_thermal_energy:
                result
                    .allocated_pv_thermal_energy,

            unavoidable_thermal_deficit:
                result
                    .unavoidable_thermal_deficit,

            required_storage_energy_at_start:
                result
                    .required_storage_energy_at_start,

            additional_storage_energy_needed_at_start:
                result
                    .additional_storage_energy_needed_at_start,

            current_waterline_power:
                result
                    .current_waterline_power,

            current_scheduled_hp_electrical_power:
                result
                    .current_scheduled_hp_electrical_power,

            current_scheduled_hp_thermal_power:
                result
                    .current_scheduled_hp_thermal_power,
        },
    )
}


#[napi]
pub fn run_no_sunshine_kernel(
    source:
        String,

    scope_id:
        String,

    pv_power:
        Float64Array,

    load_energy:
        Float64Array,

    parameters:
        NoSunshineParametersInput,
) -> Result<
    NoSunshineResultOutput
> {

    if scope_id !=
        "rust_noSunshine"
    {

        return Err(
            invalid_argument(
                format!(
                    "runNoSunshineKernel received wrong scope ID: {scope_id}"
                ),
            ),
        );
    }


    let cache_key =
        ensure_compiled_scope(
            &source,
            &scope_id,
        )?;


    execute_no_sunshine_from_cache_key(
        cache_key,
        &pv_power,
        &load_energy,
        parameters,
    )
}


#[napi]
pub fn run_no_sunshine_kernel_registered(
    handle:
        u32,

    pv_power:
        Float64Array,

    load_energy:
        Float64Array,

    parameters:
        NoSunshineParametersInput,
) -> Result<
    NoSunshineResultOutput
> {

    let cache_key =
        registered_cache_key(
            handle,
        )?;


    execute_no_sunshine_from_cache_key(
        cache_key,
        &pv_power,
        &load_energy,
        parameters,
    )
}


fn execute_peak_shaving_causal_from_cache_key(
    cache_key:
        u64,

    export_power:
        &Float64Array,

    demand_energy:
        &Float64Array,

    parameters:
        PeakShavingCausalParametersInput,
) -> Result<
    PeakShavingCausalResultOutput
> {

    let kernel_parameters =
        PeakShavingCausalKernelParameters {

            current_index:
                parameters.current_index
                    as usize,

            horizon_steps:
                parameters.horizon_steps
                    as usize,

            current_storage_energy:
                parameters
                    .current_storage_energy,

            minimum_reserve_energy:
                parameters
                    .minimum_reserve_energy,

            target_storage_energy:
                parameters
                    .target_storage_energy,

            maximum_heat_pump_electrical_power:
                parameters
                    .maximum_heat_pump_electrical_power,

            cop:
                parameters.cop,

            energy_cut_level:
                parameters
                    .energy_cut_level,

            time_step_hours:
                parameters
                    .time_step_hours,

            storage_energy_per_kelvin:
                parameters
                    .storage_energy_per_kelvin,

            storage_conductance_w_per_kelvin:
                parameters
                    .storage_conductance_w_per_kelvin,

            storage_min_temperature:
                parameters
                    .storage_min_temperature,

            ambient_temperature:
                parameters
                    .ambient_temperature,

            daily_heat_loss:
                parameters
                    .daily_heat_loss,
        };


    let result =
        ACCELERATION_SCOPE_CACHE.with(
            |cache|
                -> Result<
                    sunopt_core::PeakShavingCausalKernelResult
                >
            {

                let cache =
                    cache.borrow();


                let compiled =
                    cache
                        .get(
                            &cache_key,
                        )
                        .ok_or_else(
                            || {
                                native_error(
                                    "Compiled rust_peakShavingCausal scope disappeared from cache.",
                                )
                            },
                        )?;


                match compiled {

                    CompiledAccelerationScope::PeakShavingCausal(
                        kernel,
                    ) =>
                        kernel
                            .execute(
                                export_power,
                                demand_energy,
                                kernel_parameters,
                            )
                            .map_err(
                                native_error,
                            ),

                    _ =>
                        Err(
                            native_error(
                                "Acceleration cache type mismatch for rust_peakShavingCausal.",
                            ),
                        ),
                }
            },
        )?;


    Ok(
        PeakShavingCausalResultOutput {

            cut_level_power:
                result
                    .cut_level_power,

            normal_plan_survives:
                result
                    .normal_plan_survives,

            maximum_pv_plan_survives:
                result
                    .maximum_pv_plan_survives,

            survival_checks:
                result
                    .survival_checks,

            binary_search_iterations:
                result
                    .binary_search_iterations,
        },
    )
}


#[napi]
pub fn run_peak_shaving_causal_kernel(
    source:
        String,

    scope_id:
        String,

    export_power:
        Float64Array,

    demand_energy:
        Float64Array,

    parameters:
        PeakShavingCausalParametersInput,
) -> Result<
    PeakShavingCausalResultOutput
> {

    if scope_id !=
        "rust_peakShavingCausal"
    {

        return Err(
            invalid_argument(
                format!(
                    "runPeakShavingCausalKernel received wrong scope ID: {scope_id}"
                ),
            ),
        );
    }


    let cache_key =
        ensure_compiled_scope(
            &source,
            &scope_id,
        )?;


    execute_peak_shaving_causal_from_cache_key(
        cache_key,
        &export_power,
        &demand_energy,
        parameters,
    )
}


#[napi]
pub fn run_peak_shaving_causal_kernel_registered(
    handle:
        u32,

    export_power:
        Float64Array,

    demand_energy:
        Float64Array,

    parameters:
        PeakShavingCausalParametersInput,
) -> Result<
    PeakShavingCausalResultOutput
> {

    let cache_key =
        registered_cache_key(
            handle,
        )?;


    execute_peak_shaving_causal_from_cache_key(
        cache_key,
        &export_power,
        &demand_energy,
        parameters,
    )
}


/*
 * ================================================================
 * CACHE CONTROL
 * ================================================================
 */


#[napi]
pub fn clear_acceleration_scope_cache() {

    ACCELERATION_SCOPE_CACHE.with(
        |cache| {

            cache
                .borrow_mut()
                .clear();
        },
    );


    REGISTERED_SCOPE_HANDLES.with(
        |handles| {

            handles
                .borrow_mut()
                .clear();
        },
    );


    NEXT_REGISTERED_SCOPE_HANDLE.with(
        |next| {

            next.set(
                1,
            );
        },
    );
}


#[napi]
pub fn get_acceleration_scope_cache_size()
    -> u32 {

    ACCELERATION_SCOPE_CACHE.with(
        |cache| {

            cache
                .borrow()
                .len()
                as u32
        },
    )
}


/*
 * Backwards-compatible names used by the existing 7-day tests.
 */


#[napi]
pub fn clear_no_sunshine_kernel_cache() {

    clear_acceleration_scope_cache();
}


#[napi]
pub fn get_no_sunshine_kernel_cache_size()
    -> u32 {

    get_acceleration_scope_cache_size()
}

// SunOpt CPU/GPU worker controller
mod acceleration_controller;
mod cuda_context;

/*
 * B12 research B-side: fixed flat numeric ABI only.
 * File/domain-object mapping remains on the A side.
 */
mod b12_phase3c;

pub use acceleration_controller::{
    test_acceleration_config,
    AccelerationOptions,
    AccelerationStatus,
};

pub use cuda_context::{
    test_cuda_context,
    CudaContextStatus,
};
