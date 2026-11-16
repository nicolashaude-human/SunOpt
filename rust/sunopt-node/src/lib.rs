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

    /*
     * Approach A2:
     *
     * Static numerical inputs are copied once into native-owned memory.
     * Hot-path calls pass only a small data handle plus dynamic parameters.
     */
    static REGISTERED_DATA_PAIRS:
        RefCell<
            HashMap<
                u32,
                RegisteredDataPair
            >
        > =
        RefCell::new(
            HashMap::new(),
        );

    static NEXT_REGISTERED_DATA_HANDLE:
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


struct RegisteredDataPair {
    first:
        Vec<f64>,

    second:
        Vec<f64>,
}


fn next_registered_data_handle()
    -> u32 {

    NEXT_REGISTERED_DATA_HANDLE.with(
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
    )
}


#[napi]
pub fn register_acceleration_data_pair(
    first:
        Float64Array,

    second:
        Float64Array,
) -> Result<u32> {

    if first.is_empty() {
        return Err(
            invalid_argument(
                "Acceleration data pair must not be empty.",
            ),
        );
    }


    if first.len() !=
        second.len()
    {
        return Err(
            invalid_argument(
                "Acceleration data arrays must have equal length.",
            ),
        );
    }


    let handle =
        next_registered_data_handle();


    REGISTERED_DATA_PAIRS.with(
        |pairs| {

            pairs
                .borrow_mut()
                .insert(
                    handle,
                    RegisteredDataPair {
                        first:
                            first
                                .to_vec(),

                        second:
                            second
                                .to_vec(),
                    },
                );
        },
    );


    Ok(
        handle,
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
 * SHARED PARAMETER / RESULT CONVERSION
 * ================================================================
 *
 * The consolidated production path and optional packed utilities share the
 * same numerical conversion helpers.
 */


fn no_sunshine_parameters_from_input(
    parameters:
        &NoSunshineParametersInput,
) -> NoSunshineKernelParameters {

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
    }
}


fn no_sunshine_output(
    result:
        sunopt_core::NoSunshineKernelResult,
) -> NoSunshineResultOutput {

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
    }
}


fn causal_parameters_from_input(
    parameters:
        &PeakShavingCausalParametersInput,
) -> PeakShavingCausalKernelParameters {

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
    }
}


fn causal_output(
    result:
        sunopt_core::PeakShavingCausalKernelResult,
) -> PeakShavingCausalResultOutput {

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
    }
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
        &[f64],

    load_energy:
        &[f64],

    parameters:
        NoSunshineParametersInput,
) -> Result<
    NoSunshineResultOutput
> {

    let kernel_parameters =
        no_sunshine_parameters_from_input(
            &parameters,
        );


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
        no_sunshine_output(
            result,
        ),
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


#[napi]
pub fn run_no_sunshine_kernel_context(
    scope_handle:
        u32,

    data_handle:
        u32,

    parameters:
        NoSunshineParametersInput,
) -> Result<
    NoSunshineResultOutput
> {

    let cache_key =
        registered_cache_key(
            scope_handle,
        )?;


    REGISTERED_DATA_PAIRS.with(
        |pairs| {

            let pairs =
                pairs.borrow();


            let pair =
                pairs
                    .get(
                        &data_handle,
                    )
                    .ok_or_else(
                        || {
                            invalid_argument(
                                format!(
                                    "Unknown or expired acceleration data handle: {data_handle}"
                                ),
                            )
                        },
                    )?;


            execute_no_sunshine_from_cache_key(
                cache_key,
                pair.first.as_slice(),
                pair.second.as_slice(),
                parameters,
            )
        },
    )
}


fn execute_peak_shaving_causal_from_cache_key(
    cache_key:
        u64,

    export_power:
        &[f64],

    demand_energy:
        &[f64],

    parameters:
        PeakShavingCausalParametersInput,
) -> Result<
    PeakShavingCausalResultOutput
> {

    let kernel_parameters =
        causal_parameters_from_input(
            &parameters,
        );


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
        causal_output(
            result,
        ),
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


#[napi]
pub fn run_peak_shaving_causal_kernel_context(
    scope_handle:
        u32,

    data_handle:
        u32,

    parameters:
        PeakShavingCausalParametersInput,
) -> Result<
    PeakShavingCausalResultOutput
> {

    let cache_key =
        registered_cache_key(
            scope_handle,
        )?;


    REGISTERED_DATA_PAIRS.with(
        |pairs| {

            let pairs =
                pairs.borrow();


            let pair =
                pairs
                    .get(
                        &data_handle,
                    )
                    .ok_or_else(
                        || {
                            invalid_argument(
                                format!(
                                    "Unknown or expired acceleration data handle: {data_handle}"
                                ),
                            )
                        },
                    )?;


            execute_peak_shaving_causal_from_cache_key(
                cache_key,
                pair.first.as_slice(),
                pair.second.as_slice(),
                parameters,
            )
        },
    )
}


/*
 * ================================================================
 * OPTIONAL PACKED-BATCH UTILITY
 * ================================================================
 *
 * This API is retained only for callers that have independently proven that
 * jobs may be evaluated independently. Feedback-dependent sequential time
 * steps must not be grouped through this interface.
 */

const PACKED_NO_SUNSHINE_STRIDE: usize = 15;
const PACKED_CAUSAL_STRIDE: usize = 14;


fn packed_u32(
    value: f64,
    name: &str,
) -> Result<u32> {

    if !value.is_finite() ||
        value < 0.0 ||
        value > u32::MAX as f64 ||
        value.fract() != 0.0
    {
        return Err(
            invalid_argument(
                format!(
                    "Packed field {name} must be a non-negative integer, got {value}."
                ),
            ),
        );
    }

    Ok(value as u32)
}


#[napi]
pub fn run_no_sunshine_kernel_context_batch_packed(
    scope_handle: u32,
    data_handle: u32,
    packed: Float64Array,
) -> Result<Vec<NoSunshineResultOutput>> {

    if packed.len() % PACKED_NO_SUNSHINE_STRIDE != 0 {
        return Err(
            invalid_argument(
                format!(
                    "Packed noSunshine batch length {} is not divisible by {}.",
                    packed.len(),
                    PACKED_NO_SUNSHINE_STRIDE,
                ),
            ),
        );
    }

    let count =
        packed.len() /
        PACKED_NO_SUNSHINE_STRIDE;

    let cache_key =
        registered_cache_key(
            scope_handle,
        )?;

    REGISTERED_DATA_PAIRS.with(
        |pairs| {

            let pairs =
                pairs.borrow();

            let pair =
                pairs.get(
                    &data_handle,
                )
                .ok_or_else(
                    || invalid_argument(
                        format!(
                            "Unknown acceleration data handle: {data_handle}"
                        ),
                    ),
                )?;

            ACCELERATION_SCOPE_CACHE.with(
                |cache| {

                    let cache =
                        cache.borrow();

                    let compiled =
                        cache.get(
                            &cache_key,
                        )
                        .ok_or_else(
                            || native_error(
                                "Compiled noSunshine scope disappeared.",
                            ),
                        )?;

                    let kernel =
                        match compiled {
                            CompiledAccelerationScope::NoSunshine(kernel) =>
                                kernel,

                            _ =>
                                return Err(
                                    native_error(
                                        "Packed noSunshine batch cache type mismatch.",
                                    ),
                                ),
                        };

                    let mut outputs =
                        Vec::with_capacity(
                            count,
                        );

                    for job in 0..count {

                        let b =
                            job *
                            PACKED_NO_SUNSHINE_STRIDE;

                        let parameters =
                            NoSunshineKernelParameters {
                                current_index:
                                    packed_u32(
                                        packed[b],
                                        "currentIndex",
                                    )? as usize,

                                horizon_steps:
                                    packed_u32(
                                        packed[b + 1],
                                        "horizonSteps",
                                    )? as usize,

                                protected_steps:
                                    packed_u32(
                                        packed[b + 2],
                                        "protectedSteps",
                                    )? as usize,

                                time_step_hours:
                                    packed[b + 3],

                                household_reserve_power:
                                    packed[b + 4],

                                maximum_heat_pump_electrical_power:
                                    packed[b + 5],

                                cop:
                                    packed[b + 6],

                                initial_waterline_power:
                                    packed[b + 7],

                                target_storage_energy:
                                    packed[b + 8],

                                current_storage_energy:
                                    packed[b + 9],

                                storage_energy_per_kelvin:
                                    packed[b + 10],

                                storage_conductance_w_per_kelvin:
                                    packed[b + 11],

                                storage_min_temperature:
                                    packed[b + 12],

                                ambient_temperature:
                                    packed[b + 13],

                                daily_heat_loss:
                                    packed[b + 14],
                            };

                        let result =
                            kernel.execute(
                                pair.first.as_slice(),
                                pair.second.as_slice(),
                                parameters,
                            )
                            .map_err(
                                native_error,
                            )?;

                        outputs.push(
                            no_sunshine_output(
                                result,
                            ),
                        );
                    }

                    Ok(outputs)
                },
            )
        },
    )
}


#[napi]
pub fn run_peak_shaving_causal_kernel_context_batch_packed(
    scope_handle: u32,
    data_handle: u32,
    packed: Float64Array,
) -> Result<Vec<PeakShavingCausalResultOutput>> {

    if packed.len() % PACKED_CAUSAL_STRIDE != 0 {
        return Err(
            invalid_argument(
                format!(
                    "Packed causal batch length {} is not divisible by {}.",
                    packed.len(),
                    PACKED_CAUSAL_STRIDE,
                ),
            ),
        );
    }

    let count =
        packed.len() /
        PACKED_CAUSAL_STRIDE;

    let cache_key =
        registered_cache_key(
            scope_handle,
        )?;

    REGISTERED_DATA_PAIRS.with(
        |pairs| {

            let pairs =
                pairs.borrow();

            let pair =
                pairs.get(
                    &data_handle,
                )
                .ok_or_else(
                    || invalid_argument(
                        format!(
                            "Unknown acceleration data handle: {data_handle}"
                        ),
                    ),
                )?;

            ACCELERATION_SCOPE_CACHE.with(
                |cache| {

                    let cache =
                        cache.borrow();

                    let compiled =
                        cache.get(
                            &cache_key,
                        )
                        .ok_or_else(
                            || native_error(
                                "Compiled causal scope disappeared.",
                            ),
                        )?;

                    let kernel =
                        match compiled {
                            CompiledAccelerationScope::PeakShavingCausal(kernel) =>
                                kernel,

                            _ =>
                                return Err(
                                    native_error(
                                        "Packed causal batch cache type mismatch.",
                                    ),
                                ),
                        };

                    let mut outputs =
                        Vec::with_capacity(
                            count,
                        );

                    for job in 0..count {

                        let b =
                            job *
                            PACKED_CAUSAL_STRIDE;

                        let parameters =
                            PeakShavingCausalKernelParameters {
                                current_index:
                                    packed_u32(
                                        packed[b],
                                        "currentIndex",
                                    )? as usize,

                                horizon_steps:
                                    packed_u32(
                                        packed[b + 1],
                                        "horizonSteps",
                                    )? as usize,

                                current_storage_energy:
                                    packed[b + 2],

                                minimum_reserve_energy:
                                    packed[b + 3],

                                target_storage_energy:
                                    packed[b + 4],

                                maximum_heat_pump_electrical_power:
                                    packed[b + 5],

                                cop:
                                    packed[b + 6],

                                energy_cut_level:
                                    packed[b + 7],

                                time_step_hours:
                                    packed[b + 8],

                                storage_energy_per_kelvin:
                                    packed[b + 9],

                                storage_conductance_w_per_kelvin:
                                    packed[b + 10],

                                storage_min_temperature:
                                    packed[b + 11],

                                ambient_temperature:
                                    packed[b + 12],

                                daily_heat_loss:
                                    packed[b + 13],
                            };

                        let result =
                            kernel.execute(
                                pair.first.as_slice(),
                                pair.second.as_slice(),
                                parameters,
                            )
                            .map_err(
                                native_error,
                            )?;

                        outputs.push(
                            causal_output(
                                result,
                            ),
                        );
                    }

                    Ok(outputs)
                },
            )
        },
    )
}


/*
 * ================================================================
 * OPTIONAL PACKED-INTO UTILITY
 * ================================================================
 *
 * Retained from the successful A4/H5 experiments as a stable specialised
 * native API. It is intentionally NOT part of the default TypeScript runtime
 * path because A7 measured only ~0.5 % full-production improvement.
 */

const PACKED_NO_SUNSHINE_OUTPUT_LEN: usize = 11;
const PACKED_CAUSAL_OUTPUT_LEN: usize = 5;


fn no_sunshine_parameters_from_packed(
    packed: &[f64],
) -> Result<NoSunshineKernelParameters> {

    if packed.len() != PACKED_NO_SUNSHINE_STRIDE {
        return Err(
            invalid_argument(
                format!(
                    "Packed noSunshine input length must be {}, got {}.",
                    PACKED_NO_SUNSHINE_STRIDE,
                    packed.len(),
                ),
            ),
        );
    }

    Ok(
        NoSunshineKernelParameters {
            current_index:
                packed_u32(packed[0], "currentIndex")?
                    as usize,

            horizon_steps:
                packed_u32(packed[1], "horizonSteps")?
                    as usize,

            protected_steps:
                packed_u32(packed[2], "protectedSteps")?
                    as usize,

            time_step_hours: packed[3],
            household_reserve_power: packed[4],
            maximum_heat_pump_electrical_power: packed[5],
            cop: packed[6],
            initial_waterline_power: packed[7],
            target_storage_energy: packed[8],
            current_storage_energy: packed[9],
            storage_energy_per_kelvin: packed[10],
            storage_conductance_w_per_kelvin: packed[11],
            storage_min_temperature: packed[12],
            ambient_temperature: packed[13],
            daily_heat_loss: packed[14],
        },
    )
}


fn causal_parameters_from_packed(
    packed: &[f64],
) -> Result<PeakShavingCausalKernelParameters> {

    if packed.len() != PACKED_CAUSAL_STRIDE {
        return Err(
            invalid_argument(
                format!(
                    "Packed causal input length must be {}, got {}.",
                    PACKED_CAUSAL_STRIDE,
                    packed.len(),
                ),
            ),
        );
    }

    Ok(
        PeakShavingCausalKernelParameters {
            current_index:
                packed_u32(packed[0], "currentIndex")?
                    as usize,

            horizon_steps:
                packed_u32(packed[1], "horizonSteps")?
                    as usize,

            current_storage_energy: packed[2],
            minimum_reserve_energy: packed[3],
            target_storage_energy: packed[4],
            maximum_heat_pump_electrical_power: packed[5],
            cop: packed[6],
            energy_cut_level: packed[7],
            time_step_hours: packed[8],
            storage_energy_per_kelvin: packed[9],
            storage_conductance_w_per_kelvin: packed[10],
            storage_min_temperature: packed[11],
            ambient_temperature: packed[12],
            daily_heat_loss: packed[13],
        },
    )
}


fn execute_no_sunshine_context_packed(
    scope_handle: u32,
    data_handle: u32,
    parameters: NoSunshineKernelParameters,
) -> Result<sunopt_core::NoSunshineKernelResult> {

    let cache_key =
        registered_cache_key(
            scope_handle,
        )?;

    REGISTERED_DATA_PAIRS.with(
        |pairs| {
            let pairs =
                pairs.borrow();

            let pair =
                pairs
                    .get(&data_handle)
                    .ok_or_else(
                        || invalid_argument(
                            format!(
                                "Unknown or expired acceleration data handle: {data_handle}"
                            ),
                        ),
                    )?;

            ACCELERATION_SCOPE_CACHE.with(
                |cache| {
                    let cache =
                        cache.borrow();

                    let compiled =
                        cache
                            .get(&cache_key)
                            .ok_or_else(
                                || native_error(
                                    "Compiled rust_noSunshine scope disappeared from cache.",
                                ),
                            )?;

                    match compiled {
                        CompiledAccelerationScope::NoSunshine(kernel) =>
                            kernel
                                .execute(
                                    pair.first.as_slice(),
                                    pair.second.as_slice(),
                                    parameters,
                                )
                                .map_err(native_error),

                        _ =>
                            Err(
                                native_error(
                                    "Acceleration cache type mismatch for packed noSunshine utility.",
                                ),
                            ),
                    }
                },
            )
        },
    )
}


fn execute_causal_context_packed(
    scope_handle: u32,
    data_handle: u32,
    parameters: PeakShavingCausalKernelParameters,
) -> Result<sunopt_core::PeakShavingCausalKernelResult> {

    let cache_key =
        registered_cache_key(
            scope_handle,
        )?;

    REGISTERED_DATA_PAIRS.with(
        |pairs| {
            let pairs =
                pairs.borrow();

            let pair =
                pairs
                    .get(&data_handle)
                    .ok_or_else(
                        || invalid_argument(
                            format!(
                                "Unknown or expired acceleration data handle: {data_handle}"
                            ),
                        ),
                    )?;

            ACCELERATION_SCOPE_CACHE.with(
                |cache| {
                    let cache =
                        cache.borrow();

                    let compiled =
                        cache
                            .get(&cache_key)
                            .ok_or_else(
                                || native_error(
                                    "Compiled rust_peakShavingCausal scope disappeared from cache.",
                                ),
                            )?;

                    match compiled {
                        CompiledAccelerationScope::PeakShavingCausal(kernel) =>
                            kernel
                                .execute(
                                    pair.first.as_slice(),
                                    pair.second.as_slice(),
                                    parameters,
                                )
                                .map_err(native_error),

                        _ =>
                            Err(
                                native_error(
                                    "Acceleration cache type mismatch for packed causal utility.",
                                ),
                            ),
                    }
                },
            )
        },
    )
}


#[napi]
pub fn run_no_sunshine_kernel_context_packed_into(
    scope_handle: u32,
    data_handle: u32,
    packed: Float64Array,
    mut output: Float64Array,
) -> Result<()> {

    if output.len() != PACKED_NO_SUNSHINE_OUTPUT_LEN {
        return Err(
            invalid_argument(
                format!(
                    "Packed noSunshine output length must be {}, got {}.",
                    PACKED_NO_SUNSHINE_OUTPUT_LEN,
                    output.len(),
                ),
            ),
        );
    }

    let result =
        execute_no_sunshine_context_packed(
            scope_handle,
            data_handle,
            no_sunshine_parameters_from_packed(
                &packed,
            )?,
        )?;

    let target =
        unsafe {
            output.as_mut()
        };

    target[0] = result.kwh_no_sunshine;
    target[1] = result.bucket_demand_energy;
    target[2] = result.bucket_cooling_loss_energy;
    target[3] = result.pre_bucket_holding_loss_energy;
    target[4] = result.allocated_pv_thermal_energy;
    target[5] = result.unavoidable_thermal_deficit;
    target[6] = result.required_storage_energy_at_start;
    target[7] = result.additional_storage_energy_needed_at_start;
    target[8] = result.current_waterline_power;
    target[9] = result.current_scheduled_hp_electrical_power;
    target[10] = result.current_scheduled_hp_thermal_power;

    Ok(())
}


#[napi]
pub fn run_peak_shaving_causal_kernel_context_packed_into(
    scope_handle: u32,
    data_handle: u32,
    packed: Float64Array,
    mut output: Float64Array,
) -> Result<()> {

    if output.len() != PACKED_CAUSAL_OUTPUT_LEN {
        return Err(
            invalid_argument(
                format!(
                    "Packed causal output length must be {}, got {}.",
                    PACKED_CAUSAL_OUTPUT_LEN,
                    output.len(),
                ),
            ),
        );
    }

    let result =
        execute_causal_context_packed(
            scope_handle,
            data_handle,
            causal_parameters_from_packed(
                &packed,
            )?,
        )?;

    let target =
        unsafe {
            output.as_mut()
        };

    target[0] = result.cut_level_power;
    target[1] = if result.normal_plan_survives { 1.0 } else { 0.0 };
    target[2] = if result.maximum_pv_plan_survives { 1.0 } else { 0.0 };
    target[3] = result.survival_checks;
    target[4] = result.binary_search_iterations;

    Ok(())
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


    REGISTERED_DATA_PAIRS.with(
        |pairs| {

            pairs
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


    NEXT_REGISTERED_DATA_HANDLE.with(
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
