use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::{c_char, c_int, CStr};
use std::ptr::NonNull;

#[repr(C)]
struct SunoptCudaContextOpaque {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SunoptCausalParameters {
    data_length: usize,
    current_index: usize,
    horizon_steps: usize,

    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    target_storage_energy: f64,
    maximum_heat_pump_electrical_power: f64,
    cop: f64,
    requested_electrical_energy: f64,
    energy_cut_level: f64,
    time_step_hours: f64,
    storage_energy_per_kelvin: f64,
    storage_conductance_w_per_kelvin: f64,
    storage_min_temperature: f64,
    ambient_temperature: f64,
    daily_heat_loss: f64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SunoptCausalDynamic {
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    requested_electrical_energy: f64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct SunoptNoSunshineTankParameters {
    horizon_steps: usize,
    protected_steps: usize,

    time_step_hours: f64,

    maximum_heat_pump_electrical_power: f64,
    cop: f64,

    initial_waterline_power: f64,

    target_storage_energy: f64,

    storage_energy_per_kelvin: f64,
    storage_conductance_w_per_kelvin: f64,
    storage_min_temperature: f64,
    ambient_temperature: f64,
    daily_heat_loss: f64,
}


#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SunoptCausalResult {
    cut_level_power: f64,
    normal_plan_survives: c_int,
    maximum_pv_plan_survives: c_int,
    survival_checks: u32,
    binary_search_iterations: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SunoptCudaProfile {
    causal_calls: u64,
    causal_jobs: u64,

    parameter_h2d_ms: f64,
    kernel_ms: f64,
    result_d2h_ms: f64,
    gpu_total_ms: f64,
    host_wait_ms: f64,
    wall_ms: f64,

    profiled_search_jobs: u64,

    energy_phase_cycles: u64,
    boundary_phase_cycles: u64,
    radix_round_0_cycles: u64,
    radix_round_1_cycles: u64,
    radix_round_2_cycles: u64,
    radix_round_3_cycles: u64,

    boundary_storage_steps: u64,
    radix_round_0_storage_steps: u64,
    radix_round_1_storage_steps: u64,
    radix_round_2_storage_steps: u64,
    radix_round_3_storage_steps: u64,
}

unsafe extern "C" {
    fn sunopt_cuda_create(
        device_id: c_int,
        out_context: *mut *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_destroy(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_device_id(
        context: *const SunoptCudaContextOpaque,
        out_device_id: *mut c_int,
    ) -> c_int;

    fn sunopt_cuda_stream_synchronize(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_reserve(
        context: *mut SunoptCudaContextOpaque,
        input_bytes: usize,
        workspace_bytes: usize,
        result_bytes: usize,
    ) -> c_int;

    fn sunopt_cuda_capacities(
        context: *const SunoptCudaContextOpaque,
        out_input_bytes: *mut usize,
        out_workspace_bytes: *mut usize,
        out_result_bytes: *mut usize,
    ) -> c_int;

    fn sunopt_cuda_upload_causal_arrays(
        context: *mut SunoptCudaContextOpaque,
        export_power: *const f64,
        demand_energy: *const f64,
        length: usize,
    ) -> c_int;

    fn sunopt_cuda_run_causal(
        context: *mut SunoptCudaContextOpaque,
        parameters: *const SunoptCausalParameters,
        host_result: *mut SunoptCausalResult,
    ) -> c_int;

    fn sunopt_cuda_run_causal_batch(
        context: *mut SunoptCudaContextOpaque,
        host_parameters: *const SunoptCausalParameters,
        job_count: usize,
        host_results: *mut SunoptCausalResult,
    ) -> c_int;


    fn sunopt_cuda_configure_causal_static_parameters(
        context: *mut SunoptCudaContextOpaque,
        host_parameters: *const SunoptCausalParameters,
        job_count: usize,
    ) -> c_int;


    fn sunopt_cuda_run_causal_dynamic_batch(
        context: *mut SunoptCudaContextOpaque,
        current_index: usize,
        horizon_steps: usize,
        host_dynamic_parameters: *const SunoptCausalDynamic,
        job_count: usize,
        host_results: *mut SunoptCausalResult,
    ) -> c_int;


    fn sunopt_cuda_precompute_no_sunshine_waterlines(
        context: *mut SunoptCudaContextOpaque,
        host_parameters: *const SunoptNoSunshineTankParameters,
        tank_count: usize,
        host_waterlines: *mut f64,
    ) -> c_int;

    fn sunopt_cuda_get_profile(
        context: *const SunoptCudaContextOpaque,
        out_profile: *mut SunoptCudaProfile,
    ) -> c_int;

    fn sunopt_cuda_reset_profile(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_error_string(
        error_code: c_int,
    ) -> *const c_char;
}

fn cuda_error(
    operation: &str,
    code: c_int,
) -> Error {
    let message = unsafe {
        let ptr =
            sunopt_cuda_error_string(
                code,
            );

        if ptr.is_null() {
            format!(
                "CUDA error {code}"
            )
        } else {
            CStr::from_ptr(ptr)
                .to_string_lossy()
                .into_owned()
        }
    };

    Error::from_reason(
        format!(
            "{operation} failed: {message} \
             (CUDA error {code})"
        ),
    )
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
)]
pub struct CudaBufferCapacities {
    pub input_bytes: usize,
    pub workspace_bytes: usize,
    pub result_bytes: usize,
}

#[derive(Debug)]
pub struct CudaSimulationContext {
    raw:
        NonNull<
            SunoptCudaContextOpaque
        >,
}

unsafe impl Send
    for CudaSimulationContext
{
}

impl CudaSimulationContext {
    pub fn new(
        device_id: i32,
    ) -> Result<Self> {
        let mut raw =
            std::ptr::null_mut();

        let code = unsafe {
            sunopt_cuda_create(
                device_id as c_int,
                &mut raw,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_create",
                    code,
                ),
            );
        }

        let raw =
            NonNull::new(raw)
                .ok_or_else(|| {
                    Error::from_reason(
                        "sunopt_cuda_create returned \
                         success with a null context",
                    )
                })?;

        Ok(Self { raw })
    }

    pub fn device_id(
        &self,
    ) -> Result<i32> {
        let mut device_id:
            c_int = -1;

        let code = unsafe {
            sunopt_cuda_device_id(
                self.raw.as_ptr(),
                &mut device_id,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_device_id",
                    code,
                ),
            );
        }

        Ok(
            device_id as i32,
        )
    }

    pub fn synchronize(
        &mut self,
    ) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_stream_synchronize(
                self.raw.as_ptr(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_stream_synchronize",
                    code,
                ),
            );
        }

        Ok(())
    }

    pub fn reserve(
        &mut self,
        input_bytes: usize,
        workspace_bytes: usize,
        result_bytes: usize,
    ) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_reserve(
                self.raw.as_ptr(),
                input_bytes,
                workspace_bytes,
                result_bytes,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_reserve",
                    code,
                ),
            );
        }

        Ok(())
    }

    pub fn capacities(
        &self,
    ) -> Result<
        CudaBufferCapacities
    > {
        let mut input_bytes =
            0usize;

        let mut workspace_bytes =
            0usize;

        let mut result_bytes =
            0usize;

        let code = unsafe {
            sunopt_cuda_capacities(
                self.raw.as_ptr(),
                &mut input_bytes,
                &mut workspace_bytes,
                &mut result_bytes,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_capacities",
                    code,
                ),
            );
        }

        Ok(
            CudaBufferCapacities {
                input_bytes,
                workspace_bytes,
                result_bytes,
            },
        )
    }

    pub fn upload_causal_arrays(
        &mut self,
        export_power: &[f64],
        demand_energy: &[f64],
    ) -> Result<()> {
        if export_power.is_empty() {
            return Err(
                Error::from_reason(
                    "exportPower must not be empty.",
                ),
            );
        }

        if (
            export_power.len() !=
            demand_energy.len()
        ) {
            return Err(
                Error::from_reason(
                    "exportPower and demandEnergy \
                     must have equal length.",
                ),
            );
        }

        let code = unsafe {
            sunopt_cuda_upload_causal_arrays(
                self.raw.as_ptr(),
                export_power.as_ptr(),
                demand_energy.as_ptr(),
                export_power.len(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_upload_causal_arrays",
                    code,
                ),
            );
        }

        Ok(())
    }

    fn run_causal(
        &mut self,
        parameters:
            SunoptCausalParameters,
    ) -> Result<
        SunoptCausalResult
    > {
        let mut result =
            SunoptCausalResult::default();

        let code = unsafe {
            sunopt_cuda_run_causal(
                self.raw.as_ptr(),
                &parameters,
                &mut result,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_causal",
                    code,
                ),
            );
        }

        Ok(result)
    }

    fn run_causal_batch(
        &mut self,
        parameters:
            &[SunoptCausalParameters],
    ) -> Result<
        Vec<SunoptCausalResult>
    > {
        if parameters.is_empty() {
            return Ok(Vec::new());
        }

        let mut results =
            vec![
                SunoptCausalResult::default();
                parameters.len()
            ];

        let code = unsafe {
            sunopt_cuda_run_causal_batch(
                self.raw.as_ptr(),
                parameters.as_ptr(),
                parameters.len(),
                results.as_mut_ptr(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_causal_batch",
                    code,
                ),
            );
        }

        Ok(results)
    }

    fn configure_causal_static_parameters(
        &mut self,
        parameters:
            &[SunoptCausalParameters],
    ) -> Result<()> {
        if parameters.is_empty() {
            return Err(
                Error::from_reason(
                    "Static causal parameter configuration cannot be empty.",
                ),
            );
        }


        let code = unsafe {
            sunopt_cuda_configure_causal_static_parameters(
                self.raw.as_ptr(),
                parameters.as_ptr(),
                parameters.len(),
            )
        };


        if (
            code !=
            0
        ) {
            return Err(
                cuda_error(
                    "sunopt_cuda_configure_causal_static_parameters",
                    code,
                ),
            );
        }


        Ok(())
    }


    fn run_causal_dynamic_batch(
        &mut self,
        current_index:
            usize,

        horizon_steps:
            usize,

        dynamic:
            &[SunoptCausalDynamic],
    ) -> Result<
        Vec<SunoptCausalResult>
    > {
        if dynamic.is_empty() {
            return Ok(
                Vec::new(),
            );
        }


        let mut results =
            vec![
                SunoptCausalResult::default();
                dynamic.len()
            ];


        let code = unsafe {
            sunopt_cuda_run_causal_dynamic_batch(
                self.raw.as_ptr(),
                current_index,
                horizon_steps,
                dynamic.as_ptr(),
                dynamic.len(),
                results.as_mut_ptr(),
            )
        };


        if (
            code !=
            0
        ) {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_causal_dynamic_batch",
                    code,
                ),
            );
        }


        Ok(
            results,
        )
    }


    fn precompute_no_sunshine_waterlines(
        &mut self,
        parameters:
            &[SunoptNoSunshineTankParameters],

        data_length:
            usize,
    ) -> Result<Vec<f64>> {
        if parameters.is_empty() {
            return Ok(
                Vec::new(),
            );
        }

        let total_jobs =
            parameters
                .len()
                .checked_mul(
                    data_length,
                )
                .ok_or_else(|| {
                    Error::from_reason(
                        "No-sunshine CUDA result size overflow.",
                    )
                })?;

        let mut waterlines =
            vec![
                0.0f64;
                total_jobs
            ];

        let code = unsafe {
            sunopt_cuda_precompute_no_sunshine_waterlines(
                self.raw.as_ptr(),
                parameters.as_ptr(),
                parameters.len(),
                waterlines.as_mut_ptr(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_precompute_no_sunshine_waterlines",
                    code,
                ),
            );
        }

        Ok(
            waterlines,
        )
    }


    fn causal_profile(
        &self,
    ) -> Result<SunoptCudaProfile> {
        let mut profile =
            SunoptCudaProfile::default();

        let code = unsafe {
            sunopt_cuda_get_profile(
                self.raw.as_ptr(),
                &mut profile,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_get_profile",
                    code,
                ),
            );
        }

        Ok(profile)
    }

    fn reset_causal_profile(
        &mut self,
    ) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_reset_profile(
                self.raw.as_ptr(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_reset_profile",
                    code,
                ),
            );
        }

        Ok(())
    }
}

impl Drop
    for CudaSimulationContext
{
    fn drop(&mut self) {
        let _ = unsafe {
            sunopt_cuda_destroy(
                self.raw.as_ptr(),
            )
        };
    }
}

#[napi(object)]
pub struct CudaContextStatus {
    pub device_id: i32,
    pub stream_ready: bool,
}

#[napi(object)]
pub struct CudaBufferStatus {
    pub device_id: i32,
    pub input_bytes: u32,
    pub workspace_bytes: u32,
    pub result_bytes: u32,
}

#[napi(object)]
pub struct CudaCausalJob {
    pub current_index: u32,
    pub horizon_steps: u32,

    pub current_storage_energy: f64,
    pub minimum_reserve_energy: f64,
    pub target_storage_energy: f64,
    pub maximum_heat_pump_electrical_power: f64,
    pub cop: f64,

    /*
     * Optional for backwards compatibility with existing direct CUDA
     * tests. When absent, the native kernel keeps using energyCutLevel.
     */
    pub requested_electrical_energy: Option<f64>,

    pub energy_cut_level: f64,
    pub time_step_hours: f64,
    pub storage_energy_per_kelvin: f64,
    pub storage_conductance_w_per_kelvin: f64,
    pub storage_min_temperature: f64,
    pub ambient_temperature: f64,
    pub daily_heat_loss: f64,
}

#[napi(object)]
pub struct CudaNoSunshineTankJob {
    pub horizon_steps: u32,
    pub protected_steps: u32,

    pub time_step_hours: f64,

    pub maximum_heat_pump_electrical_power: f64,
    pub cop: f64,

    pub initial_waterline_power: f64,

    pub target_storage_energy: f64,

    pub storage_energy_per_kelvin: f64,
    pub storage_conductance_w_per_kelvin: f64,
    pub storage_min_temperature: f64,
    pub ambient_temperature: f64,
    pub daily_heat_loss: f64,
}


#[napi(object)]
pub struct CudaCausalResult {
    pub cut_level_power: f64,
    pub normal_plan_survives: bool,
    pub maximum_pv_plan_survives: bool,
    pub survival_checks: u32,
    pub binary_search_iterations: u32,
}

#[napi(object)]
pub struct CudaCausalProfile {
    pub causal_calls: u32,
    pub causal_jobs: u32,

    pub parameter_h2d_ms: f64,
    pub kernel_ms: f64,
    pub result_d2h_ms: f64,
    pub gpu_total_ms: f64,
    pub host_wait_ms: f64,
    pub wall_ms: f64,

    pub average_kernel_us_per_job: f64,
    pub average_gpu_us_per_job: f64,
}


#[napi]
pub struct CudaCausalWorker {
    context:
        CudaSimulationContext,

    data_length:
        usize,

    /*
     * Production lockstep cache.
     *
     * The 12-value per-tank layout contains only three values that
     * actually change each annual timestep:
     *
     *   currentStorageEnergy
     *   minimumReserveEnergy
     *   requestedElectricalEnergy
     *
     * Cache the complete native structs once and update only those
     * dynamic fields plus the common currentIndex/horizonSteps.
     */
    lockstep_parameters:
        Vec<SunoptCausalParameters>,

    lockstep_dynamic:
        Vec<SunoptCausalDynamic>,
}

impl CudaCausalWorker {
    fn validate_job(
        &self,
        job: &CudaCausalJob,
    ) -> Result<()> {
        if (
            job.current_index as usize >=
            self.data_length
        ) {
            return Err(
                Error::from_reason(
                    "currentIndex is outside annual arrays.",
                ),
            );
        }

        if job.horizon_steps == 0 {
            return Err(
                Error::from_reason(
                    "horizonSteps must be greater than zero.",
                ),
            );
        }

        if (
            job.horizon_steps as usize >
            self.data_length
        ) {
            return Err(
                Error::from_reason(
                    "horizonSteps cannot exceed annual array length.",
                ),
            );
        }

        if job.time_step_hours <= 0.0 {
            return Err(
                Error::from_reason(
                    "timeStepHours must be greater than zero.",
                ),
            );
        }

        if (
            job.storage_energy_per_kelvin <=
            0.0
        ) {
            return Err(
                Error::from_reason(
                    "storageEnergyPerKelvin must be greater than zero.",
                ),
            );
        }

        if (
            job.maximum_heat_pump_electrical_power <
            0.0
        ) {
            return Err(
                Error::from_reason(
                    "maximumHeatPumpElectricalPower cannot be negative.",
                ),
            );
        }

        if job.cop <= 0.0 {
            return Err(
                Error::from_reason(
                    "COP must be greater than zero.",
                ),
            );
        }

        if job.target_storage_energy < 0.0 {
            return Err(
                Error::from_reason(
                    "targetStorageEnergy cannot be negative.",
                ),
            );
        }

        Ok(())
    }

    fn validate_no_sunshine_tank(
        &self,
        job:
            &CudaNoSunshineTankJob,
    ) -> Result<()> {
        if job.horizon_steps == 0 {
            return Err(
                Error::from_reason(
                    "No-sunshine horizonSteps must be greater than zero.",
                ),
            );
        }

        if (
            job.horizon_steps as usize >
            self.data_length
        ) {
            return Err(
                Error::from_reason(
                    "No-sunshine horizonSteps cannot exceed annual array length.",
                ),
            );
        }

        if (
            job.protected_steps >
            job.horizon_steps
        ) {
            return Err(
                Error::from_reason(
                    "No-sunshine protectedSteps cannot exceed horizonSteps.",
                ),
            );
        }

        if job.time_step_hours <= 0.0 {
            return Err(
                Error::from_reason(
                    "No-sunshine timeStepHours must be greater than zero.",
                ),
            );
        }

        if job.cop <= 0.0 {
            return Err(
                Error::from_reason(
                    "No-sunshine COP must be greater than zero.",
                ),
            );
        }

        if (
            job.maximum_heat_pump_electrical_power <
            0.0
        ) {
            return Err(
                Error::from_reason(
                    "No-sunshine maximumHeatPumpElectricalPower cannot be negative.",
                ),
            );
        }

        if (
            job.target_storage_energy <=
            0.0
        ) {
            return Err(
                Error::from_reason(
                    "No-sunshine targetStorageEnergy must be greater than zero.",
                ),
            );
        }

        if (
            job.storage_energy_per_kelvin <=
            0.0
        ) {
            return Err(
                Error::from_reason(
                    "No-sunshine storageEnergyPerKelvin must be greater than zero.",
                ),
            );
        }

        Ok(())
    }


    fn native_no_sunshine_parameters(
        job:
            &CudaNoSunshineTankJob,
    ) -> SunoptNoSunshineTankParameters {
        SunoptNoSunshineTankParameters {
            horizon_steps:
                job.horizon_steps as usize,

            protected_steps:
                job.protected_steps as usize,

            time_step_hours:
                job.time_step_hours,

            maximum_heat_pump_electrical_power:
                job.maximum_heat_pump_electrical_power,

            cop:
                job.cop,

            initial_waterline_power:
                job.initial_waterline_power,

            target_storage_energy:
                job.target_storage_energy,

            storage_energy_per_kelvin:
                job.storage_energy_per_kelvin,

            storage_conductance_w_per_kelvin:
                job.storage_conductance_w_per_kelvin,

            storage_min_temperature:
                job.storage_min_temperature,

            ambient_temperature:
                job.ambient_temperature,

            daily_heat_loss:
                job.daily_heat_loss,
        }
    }


    fn native_parameters(
        &self,
        job: &CudaCausalJob,
    ) -> SunoptCausalParameters {
        SunoptCausalParameters {
            data_length:
                self.data_length,

            current_index:
                job.current_index as usize,

            horizon_steps:
                job.horizon_steps as usize,

            current_storage_energy:
                job.current_storage_energy,

            minimum_reserve_energy:
                job.minimum_reserve_energy,

            target_storage_energy:
                job.target_storage_energy,

            maximum_heat_pump_electrical_power:
                job.maximum_heat_pump_electrical_power,

            cop:
                job.cop,

            requested_electrical_energy:
                job
                    .requested_electrical_energy
                    .unwrap_or(-1.0),

            energy_cut_level:
                job.energy_cut_level,

            time_step_hours:
                job.time_step_hours,

            storage_energy_per_kelvin:
                job.storage_energy_per_kelvin,

            storage_conductance_w_per_kelvin:
                job.storage_conductance_w_per_kelvin,

            storage_min_temperature:
                job.storage_min_temperature,

            ambient_temperature:
                job.ambient_temperature,

            daily_heat_loss:
                job.daily_heat_loss,
        }
    }

    fn public_result(
        native:
            SunoptCausalResult,
    ) -> CudaCausalResult {
        CudaCausalResult {
            cut_level_power:
                native.cut_level_power,

            normal_plan_survives:
                native.normal_plan_survives != 0,

            maximum_pv_plan_survives:
                native.maximum_pv_plan_survives != 0,

            survival_checks:
                native.survival_checks,

            binary_search_iterations:
                native.binary_search_iterations,
        }
    }
}

#[napi]
impl CudaCausalWorker {
    #[napi(constructor)]
    pub fn new(
        export_power:
            Float64Array,

        demand_energy:
            Float64Array,

        device_id:
            Option<i32>,
    ) -> Result<Self> {
        if export_power.is_empty() {
            return Err(
                Error::from_reason(
                    "exportPower must not be empty.",
                ),
            );
        }

        if (
            export_power.len() !=
            demand_energy.len()
        ) {
            return Err(
                Error::from_reason(
                    "exportPower and demandEnergy must have equal length.",
                ),
            );
        }


        if (
            export_power
                .iter()
                .any(
                    |value| {
                        !value.is_finite()
                    },
                ) ||
            demand_energy
                .iter()
                .any(
                    |value| {
                        !value.is_finite()
                    },
                )
        ) {
            return Err(
                Error::from_reason(
                    "exportPower and demandEnergy must contain only finite values.",
                ),
            );
        }


        let data_length =
            export_power.len();


        /*
         * CUDA causal/no-sunshine code has historically applied:
         *
         *   if (value < 0.0) value = 0.0;
         *
         * every time an immutable annual export/demand value is read.
         *
         * Do that normalization once here instead. The constructor
         * validates finite inputs above, so resident arrays are finite
         * and nonnegative for the CUDA hot path.
         */
        let export_power_nonnegative:
            Vec<f64> =
            export_power
                .iter()
                .map(
                    |value| {
                        if *value < 0.0 {
                            0.0
                        } else {
                            *value
                        }
                    },
                )
                .collect();


        let demand_energy_nonnegative:
            Vec<f64> =
            demand_energy
                .iter()
                .map(
                    |value| {
                        if *value < 0.0 {
                            0.0
                        } else {
                            *value
                        }
                    },
                )
                .collect();


        let mut context =
            CudaSimulationContext::new(
                device_id.unwrap_or(0),
            )?;

        context
            .upload_causal_arrays(
                &export_power_nonnegative,
                &demand_energy_nonnegative,
            )?;

        Ok(Self {
            context,
            data_length,

            lockstep_parameters:
                Vec::new(),

            lockstep_dynamic:
                Vec::new(),
        })
    }

    #[napi]
    pub fn run(
        &mut self,
        job:
            CudaCausalJob,
    ) -> Result<
        CudaCausalResult
    > {
        self.validate_job(&job)?;

        let native =
            self.context
                .run_causal(
                    self.native_parameters(
                        &job,
                    ),
                )?;

        Ok(
            Self::public_result(
                native,
            ),
        )
    }

    /// Parallel GPU batch:
    /// one CUDA thread executes one complete causal job.
    #[napi]
    pub fn run_batch(
        &mut self,
        jobs:
            Vec<CudaCausalJob>,
    ) -> Result<
        Vec<CudaCausalResult>
    > {
        if jobs.is_empty() {
            return Ok(
                Vec::new(),
            );
        }

        for job in &jobs {
            self.validate_job(job)?;
        }

        let parameters:
            Vec<
                SunoptCausalParameters
            > =
            jobs
                .iter()
                .map(
                    |job| {
                        self.native_parameters(
                            job,
                        )
                    },
                )
                .collect();

        let native_results =
            self.context
                .run_causal_batch(
                    &parameters,
                )?;

        Ok(
            native_results
                .into_iter()
                .map(
                    Self::public_result,
                )
                .collect(),
        )
    }

    /// Configure the production lockstep native-parameter cache.
    ///
    /// Input uses the existing stride-12 layout so the TypeScript
    /// controller can configure this directly from its already
    /// prepared constant array. The three dynamic fields are copied
    /// too, but will be overwritten by run_lockstep_dynamic().
    #[napi]
    pub fn configure_lockstep_packed(
        &mut self,
        packed:
            Float64Array,
    ) -> Result<()> {
        const STRIDE: usize =
            12;

        if (
            packed.is_empty() ||
            packed.len() %
                STRIDE !=
                0
        ) {
            return Err(
                Error::from_reason(
                    "Packed lockstep configuration length must be a non-zero multiple of 12.",
                ),
            );
        }


        const CONFIG_CURRENT_INDEX:
            usize =
            0;

        const CONFIG_HORIZON_STEPS:
            usize =
            1;


        let tank_count =
            packed.len() /
            STRIDE;


        self.lockstep_parameters
            .clear();

        self.lockstep_parameters
            .reserve(
                tank_count,
            );


        self.lockstep_dynamic
            .clear();

        self.lockstep_dynamic
            .reserve(
                tank_count,
            );


        for tank_index in 0..tank_count {
            let base =
                tank_index *
                STRIDE;


            let current_storage_energy =
                packed[
                    base +
                    0
                ];

            let minimum_reserve_energy =
                packed[
                    base +
                    1
                ];

            let target_storage_energy =
                packed[
                    base +
                    2
                ];

            let maximum_heat_pump_electrical_power =
                packed[
                    base +
                    3
                ];

            let cop =
                packed[
                    base +
                    4
                ];

            let requested_electrical_energy =
                packed[
                    base +
                    5
                ];

            let time_step_hours =
                packed[
                    base +
                    6
                ];

            let storage_energy_per_kelvin =
                packed[
                    base +
                    7
                ];

            let storage_conductance_w_per_kelvin =
                packed[
                    base +
                    8
                ];

            let storage_min_temperature =
                packed[
                    base +
                    9
                ];

            let ambient_temperature =
                packed[
                    base +
                    10
                ];

            let daily_heat_loss =
                packed[
                    base +
                    11
                ];


            if (
                time_step_hours <=
                    0.0 ||
                storage_energy_per_kelvin <=
                    0.0 ||
                maximum_heat_pump_electrical_power <
                    0.0 ||
                cop <=
                    0.0 ||
                target_storage_energy <
                    0.0
            ) {
                self.lockstep_parameters
                    .clear();

                return Err(
                    Error::from_reason(
                        format!(
                            "Invalid packed lockstep configuration for tank index {tank_index}.",
                        ),
                    ),
                );
            }


            self.lockstep_parameters
                .push(
                    SunoptCausalParameters {
                        data_length:
                            self.data_length,

                        current_index:
                            CONFIG_CURRENT_INDEX,

                        horizon_steps:
                            CONFIG_HORIZON_STEPS,

                        current_storage_energy,

                        minimum_reserve_energy,

                        target_storage_energy,

                        maximum_heat_pump_electrical_power,

                        cop,

                        requested_electrical_energy,

                        energy_cut_level:
                            0.0,

                        time_step_hours,

                        storage_energy_per_kelvin,

                        storage_conductance_w_per_kelvin,

                        storage_min_temperature,

                        ambient_temperature,

                        daily_heat_loss,
                    },
                );


            self.lockstep_dynamic
                .push(
                    SunoptCausalDynamic {
                        current_storage_energy,

                        minimum_reserve_energy,

                        requested_electrical_energy,
                    },
                );
        }


        self.context
            .configure_causal_static_parameters(
                &self.lockstep_parameters,
            )?;


        Ok(())
    }


    /// Production lockstep hot interface.
    ///
    /// Dynamic layout per tank, stride = 3:
    ///
    ///   0 currentStorageEnergy
    ///   1 minimumReserveEnergy
    ///   2 requestedElectricalEnergy
    ///
    /// All other native fields are reused from the one-time
    /// configure_lockstep_packed() cache.
    #[napi]
    pub fn run_lockstep_dynamic(
        &mut self,
        current_index:
            u32,

        horizon_steps:
            u32,

        dynamic:
            Float64Array,
    ) -> Result<Vec<f64>> {
        const DYNAMIC_STRIDE:
            usize =
            3;


        if (
            current_index as usize >=
            self.data_length
        ) {
            return Err(
                Error::from_reason(
                    "currentIndex is outside annual arrays.",
                ),
            );
        }


        if (
            horizon_steps ==
                0 ||
            horizon_steps as usize >
                self.data_length
        ) {
            return Err(
                Error::from_reason(
                    "horizonSteps must be within annual array length.",
                ),
            );
        }


        if (
            self.lockstep_parameters
                .is_empty() ||
            self.lockstep_dynamic
                .len() !=
                self.lockstep_parameters
                    .len()
        ) {
            return Err(
                Error::from_reason(
                    "Lockstep parameters are not configured. Call configureLockstepPacked first.",
                ),
            );
        }


        let expected_dynamic_length =
            self.lockstep_dynamic
                .len() *
            DYNAMIC_STRIDE;


        if (
            dynamic.len() !=
            expected_dynamic_length
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Dynamic lockstep array length must be {expected_dynamic_length}.",
                    ),
                ),
            );
        }


        for (
            tank_index,
            native_dynamic
        ) in self
            .lockstep_dynamic
            .iter_mut()
            .enumerate()
        {
            let base =
                tank_index *
                DYNAMIC_STRIDE;


            let requested_electrical_energy =
                dynamic[
                    base +
                    2
                ];


            if (
                requested_electrical_energy <
                0.0
            ) {
                return Err(
                    Error::from_reason(
                        format!(
                            "Invalid requestedElectricalEnergy for tank index {tank_index}.",
                        ),
                    ),
                );
            }


            native_dynamic
                .current_storage_energy =
                    dynamic[
                        base +
                        0
                    ];


            native_dynamic
                .minimum_reserve_energy =
                    dynamic[
                        base +
                        1
                    ];


            native_dynamic
                .requested_electrical_energy =
                    requested_electrical_energy;
        }


        let native_results =
            self.context
                .run_causal_dynamic_batch(
                    current_index as usize,
                    horizon_steps as usize,
                    &self.lockstep_dynamic,
                )?;


        Ok(
            native_results
                .into_iter()
                .map(
                    |result| {
                        result
                            .cut_level_power
                    },
                )
                .collect(),
        )
    }


    /// Low-overhead lockstep batch interface.
    ///
    /// Packed layout per tank, stride = 12:
    ///
    /// 0 currentStorageEnergy
    /// 1 minimumReserveEnergy
    /// 2 targetStorageEnergy
    /// 3 maximumHeatPumpElectricalPower
    /// 4 cop
    /// 5 requestedElectricalEnergy
    /// 6 timeStepHours
    /// 7 storageEnergyPerKelvin
    /// 8 storageConductanceWPerKelvin
    /// 9 storageMinTemperature
    /// 10 ambientTemperature
    /// 11 dailyHeatLoss
    ///
    /// currentIndex and horizonSteps are common to the whole batch.
    /// Only cutLevelPower values are returned because that is the only
    /// causal result consumed by the lockstep controller.
    #[napi]
    pub fn run_lockstep_packed(
        &mut self,
        current_index: u32,
        horizon_steps: u32,
        packed: Float64Array,
    ) -> Result<Vec<f64>> {
        const STRIDE: usize = 12;

        if current_index as usize >= self.data_length {
            return Err(
                Error::from_reason(
                    "currentIndex is outside annual arrays.",
                ),
            );
        }

        if (
            horizon_steps == 0 ||
            horizon_steps as usize > self.data_length
        ) {
            return Err(
                Error::from_reason(
                    "horizonSteps must be within annual array length.",
                ),
            );
        }

        if (
            packed.is_empty() ||
            packed.len() % STRIDE != 0
        ) {
            return Err(
                Error::from_reason(
                    "Packed lockstep array length must be a non-zero multiple of 12.",
                ),
            );
        }

        let tank_count =
            packed.len() /
            STRIDE;

        let mut parameters:
            Vec<SunoptCausalParameters> =
            Vec::with_capacity(
                tank_count,
            );

        for tank_index in 0..tank_count {
            let base =
                tank_index *
                STRIDE;

            let current_storage_energy =
                packed[base + 0];

            let minimum_reserve_energy =
                packed[base + 1];

            let target_storage_energy =
                packed[base + 2];

            let maximum_heat_pump_electrical_power =
                packed[base + 3];

            let cop =
                packed[base + 4];

            let requested_electrical_energy =
                packed[base + 5];

            let time_step_hours =
                packed[base + 6];

            let storage_energy_per_kelvin =
                packed[base + 7];

            let storage_conductance_w_per_kelvin =
                packed[base + 8];

            let storage_min_temperature =
                packed[base + 9];

            let ambient_temperature =
                packed[base + 10];

            let daily_heat_loss =
                packed[base + 11];

            if (
                time_step_hours <= 0.0 ||
                storage_energy_per_kelvin <= 0.0 ||
                maximum_heat_pump_electrical_power < 0.0 ||
                cop <= 0.0 ||
                target_storage_energy < 0.0 ||
                requested_electrical_energy < 0.0
            ) {
                return Err(
                    Error::from_reason(
                        format!(
                            "Invalid packed lockstep parameters for tank index {tank_index}.",
                        ),
                    ),
                );
            }

            parameters.push(
                SunoptCausalParameters {
                    data_length:
                        self.data_length,

                    current_index:
                        current_index as usize,

                    horizon_steps:
                        horizon_steps as usize,

                    current_storage_energy,
                    minimum_reserve_energy,
                    target_storage_energy,
                    maximum_heat_pump_electrical_power,
                    cop,
                    requested_electrical_energy,

                    /*
                     * The fused energy search replaces this before the
                     * causal survival checks.
                     */
                    energy_cut_level:
                        0.0,

                    time_step_hours,
                    storage_energy_per_kelvin,
                    storage_conductance_w_per_kelvin,
                    storage_min_temperature,
                    ambient_temperature,
                    daily_heat_loss,
                },
            );
        }

        let native_results =
            self.context
                .run_causal_batch(
                    &parameters,
                )?;

        Ok(
            native_results
                .into_iter()
                .map(
                    |result| {
                        result
                            .cut_level_power
                    },
                )
                .collect(),
        )
    }


    /// Precompute the complete annual no-sunshine current-waterline
    /// matrix in one CUDA launch.
    ///
    /// Output layout is tank-major:
    ///
    ///   result[tankIndex * dataLength + currentIndex]
    #[napi]
    pub fn precompute_no_sunshine_waterlines(
        &mut self,

        tanks:
            Vec<CudaNoSunshineTankJob>,
    ) -> Result<Vec<f64>> {
        if tanks.is_empty() {
            return Ok(
                Vec::new(),
            );
        }

        for tank in &tanks {
            self.validate_no_sunshine_tank(
                tank,
            )?;
        }

        let parameters:
            Vec<
                SunoptNoSunshineTankParameters
            > =
            tanks
                .iter()
                .map(
                    Self::native_no_sunshine_parameters,
                )
                .collect();

        self.context
            .precompute_no_sunshine_waterlines(
                &parameters,
                self.data_length,
            )
    }


    #[napi]
    pub fn profile(
        &self,
    ) -> Result<CudaCausalProfile> {
        let profile =
            self.context
                .causal_profile()?;

        let job_count =
            profile.causal_jobs
                as f64;

        let average_kernel_us_per_job =
            if job_count > 0.0 {
                profile.kernel_ms *
                1000.0 /
                job_count
            } else {
                0.0
            };

        let average_gpu_us_per_job =
            if job_count > 0.0 {
                profile.gpu_total_ms *
                1000.0 /
                job_count
            } else {
                0.0
            };

        Ok(
            CudaCausalProfile {
                causal_calls:
                    profile.causal_calls
                        .min(u32::MAX as u64)
                        as u32,

                causal_jobs:
                    profile.causal_jobs
                        .min(u32::MAX as u64)
                        as u32,

                parameter_h2d_ms:
                    profile.parameter_h2d_ms,

                kernel_ms:
                    profile.kernel_ms,

                result_d2h_ms:
                    profile.result_d2h_ms,

                gpu_total_ms:
                    profile.gpu_total_ms,

                host_wait_ms:
                    profile.host_wait_ms,

                wall_ms:
                    profile.wall_ms,

                average_kernel_us_per_job,

                average_gpu_us_per_job,
            },
        )
    }

    #[napi]
    pub fn reset_profile(
        &mut self,
    ) -> Result<()> {
        self.context
            .reset_causal_profile()
    }

    #[napi(getter)]
    pub fn data_length(
        &self,
    ) -> u32 {
        self.data_length
            as u32
    }

    #[napi(getter)]
    pub fn device_id(
        &self,
    ) -> Result<i32> {
        self.context
            .device_id()
    }
}

#[napi]
pub fn test_cuda_context(
    device_id:
        Option<i32>,
) -> Result<
    CudaContextStatus
> {
    let mut context =
        CudaSimulationContext::new(
            device_id.unwrap_or(0),
        )?;

    context.synchronize()?;

    Ok(
        CudaContextStatus {
            device_id:
                context.device_id()?,

            stream_ready:
                true,
        },
    )
}

#[napi]
pub fn test_cuda_buffers(
    input_bytes:
        u32,

    workspace_bytes:
        u32,

    result_bytes:
        u32,

    device_id:
        Option<i32>,
) -> Result<
    CudaBufferStatus
> {
    let mut context =
        CudaSimulationContext::new(
            device_id.unwrap_or(0),
        )?;

    context.reserve(
        input_bytes as usize,
        workspace_bytes as usize,
        result_bytes as usize,
    )?;

    context.synchronize()?;

    let capacities =
        context.capacities()?;

    Ok(
        CudaBufferStatus {
            device_id:
                context.device_id()?,

            input_bytes:
                capacities.input_bytes
                    as u32,

            workspace_bytes:
                capacities.workspace_bytes
                    as u32,

            result_bytes:
                capacities.result_bytes
                    as u32,
        },
    )
}
