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
    pub energy_cut_level: f64,
    pub time_step_hours: f64,
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

        let data_length =
            export_power.len();

        let mut context =
            CudaSimulationContext::new(
                device_id.unwrap_or(0),
            )?;

        context
            .upload_causal_arrays(
                &export_power,
                &demand_energy,
            )?;

        Ok(Self {
            context,
            data_length,
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
