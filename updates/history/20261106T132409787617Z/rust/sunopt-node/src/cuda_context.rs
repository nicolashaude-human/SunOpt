use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::{c_char, c_int, CStr};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Instant;

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
        host_cut_level_results: *mut f64,
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


    fn run_causal_dynamic_batch_into(
        &mut self,
        current_index:
            usize,

        horizon_steps:
            usize,

        dynamic:
            &[SunoptCausalDynamic],

        output:
            &mut [f64],
    ) -> Result<()> {
        if dynamic.is_empty() {
            if (
                !output.is_empty()
            ) {
                return Err(
                    Error::from_reason(
                        "Empty dynamic batch requires empty output.",
                    ),
                );
            }

            return Ok(());
        }


        if (
            dynamic.len() >
            32
        ) {
            return Err(
                Error::from_reason(
                    "Kernel-argument dynamic batch supports at most 32 tanks.",
                ),
            );
        }


        if (
            output.len() !=
            dynamic.len()
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Dynamic result buffer length must be {}; received {}.",
                        dynamic.len(),
                        output.len(),
                    ),
                ),
            );
        }


        let code = unsafe {
            sunopt_cuda_run_causal_dynamic_batch(
                self.raw.as_ptr(),
                current_index,
                horizon_steps,
                dynamic.as_ptr(),
                dynamic.len(),
                output.as_mut_ptr(),
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
    ) -> Result<Vec<f64>> {
        let mut cut_levels =
            vec![
                0.0f64;
                dynamic.len()
            ];


        self.run_causal_dynamic_batch_into(
            current_index,
            horizon_steps,
            dynamic,
            &mut cut_levels,
        )?;


        Ok(
            cut_levels,
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

const BLACKWELL_SNAPSHOT_MAGIC: &[u8; 16] = b"SUNOPT-BW-BENCH\0";
const BLACKWELL_SNAPSHOT_VERSION: u32 = 1;
const BLACKWELL_CONFIG_STRIDE: usize = 12;
const BLACKWELL_DYNAMIC_STRIDE: usize = 3;

#[derive(Clone)]
struct BlackwellReplayStep {
    current_index: u32,
    horizon_steps: u32,
    dynamic: Vec<f64>,
}

struct BlackwellReplaySnapshot {
    data_length: usize,
    tank_count: usize,
    export_power: Vec<f64>,
    demand_energy: Vec<f64>,
    packed_config: Vec<f64>,
    steps: Vec<BlackwellReplayStep>,
}

struct BlackwellCapture {
    path: PathBuf,
    data_length: usize,
    export_power: Vec<f64>,
    demand_energy: Vec<f64>,
    packed_config: Vec<f64>,
    steps: Vec<BlackwellReplayStep>,
}

fn blackwell_default_snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("blackwell-benchmark.bin")
}

fn blackwell_capture_requested(path: &Path) -> bool {
    match std::env::var("SUNOPT_BLACKWELL_CAPTURE") {
        Ok(value) => {
            let value = value.trim().to_ascii_lowercase();
            if value == "0" || value == "false" || value == "off" {
                false
            } else if value == "1" || value == "true" || value == "on" || value == "force" {
                true
            } else {
                !path.exists()
            }
        }
        Err(_) => !path.exists(),
    }
}

fn write_u32(writer: &mut File, value: u32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_f64_slice(writer: &mut File, values: &[f64]) -> std::io::Result<()> {
    for value in values {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    Ok(())
}

fn read_u32(reader: &mut File) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f64_vec(reader: &mut File, count: usize) -> std::io::Result<Vec<f64>> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        values.push(f64::from_bits(u64::from_le_bytes(bytes)));
    }
    Ok(values)
}

impl BlackwellCapture {
    fn new(path: PathBuf, export_power: &[f64], demand_energy: &[f64]) -> Self {
        Self {
            path,
            data_length: export_power.len(),
            export_power: export_power.to_vec(),
            demand_energy: demand_energy.to_vec(),
            packed_config: Vec::new(),
            steps: Vec::with_capacity(export_power.len()),
        }
    }

    fn configure(&mut self, packed: &[f64]) {
        self.packed_config.clear();
        self.packed_config.extend_from_slice(packed);
        self.steps.clear();
    }

    fn push_step(&mut self, current_index: u32, horizon_steps: u32, dynamic: &[f64]) {
        self.steps.push(BlackwellReplayStep {
            current_index,
            horizon_steps,
            dynamic: dynamic.to_vec(),
        });
    }

    fn complete(&self) -> bool {
        !self.packed_config.is_empty() && self.steps.len() == self.data_length
    }

    fn write_snapshot(&self) -> std::io::Result<()> {
        let tank_count = self.packed_config.len() / BLACKWELL_CONFIG_STRIDE;
        if tank_count == 0 || self.packed_config.len() != tank_count * BLACKWELL_CONFIG_STRIDE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid Blackwell packed configuration.",
            ));
        }
        let expected_dynamic = tank_count * BLACKWELL_DYNAMIC_STRIDE;
        if self.steps.iter().any(|step| step.dynamic.len() != expected_dynamic) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid Blackwell dynamic replay stride.",
            ));
        }

        let temporary_path = self.path.with_extension("bin.tmp");
        let mut writer = File::create(&temporary_path)?;
        writer.write_all(BLACKWELL_SNAPSHOT_MAGIC)?;
        write_u32(&mut writer, BLACKWELL_SNAPSHOT_VERSION)?;
        write_u32(&mut writer, self.data_length as u32)?;
        write_u32(&mut writer, tank_count as u32)?;
        write_u32(&mut writer, self.steps.len() as u32)?;
        write_u32(&mut writer, BLACKWELL_CONFIG_STRIDE as u32)?;
        write_u32(&mut writer, BLACKWELL_DYNAMIC_STRIDE as u32)?;
        write_f64_slice(&mut writer, &self.export_power)?;
        write_f64_slice(&mut writer, &self.demand_energy)?;
        write_f64_slice(&mut writer, &self.packed_config)?;
        for step in &self.steps {
            write_u32(&mut writer, step.current_index)?;
            write_u32(&mut writer, step.horizon_steps)?;
            write_f64_slice(&mut writer, &step.dynamic)?;
        }
        writer.flush()?;
        drop(writer);
        std::fs::rename(temporary_path, &self.path)?;
        Ok(())
    }
}

fn read_blackwell_snapshot(path: &Path) -> std::io::Result<BlackwellReplaySnapshot> {
    let mut reader = File::open(path)?;
    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic)?;
    if &magic != BLACKWELL_SNAPSHOT_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Blackwell snapshot magic does not match.",
        ));
    }

    let version = read_u32(&mut reader)?;
    if version != BLACKWELL_SNAPSHOT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported Blackwell snapshot version {version}."),
        ));
    }

    let data_length = read_u32(&mut reader)? as usize;
    let tank_count = read_u32(&mut reader)? as usize;
    let step_count = read_u32(&mut reader)? as usize;
    let config_stride = read_u32(&mut reader)? as usize;
    let dynamic_stride = read_u32(&mut reader)? as usize;
    if data_length == 0 || tank_count == 0 || step_count == 0 ||
        config_stride != BLACKWELL_CONFIG_STRIDE || dynamic_stride != BLACKWELL_DYNAMIC_STRIDE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Blackwell snapshot dimensions are invalid.",
        ));
    }

    let export_power = read_f64_vec(&mut reader, data_length)?;
    let demand_energy = read_f64_vec(&mut reader, data_length)?;
    let packed_config = read_f64_vec(&mut reader, tank_count * config_stride)?;
    let mut steps = Vec::with_capacity(step_count);
    for _ in 0..step_count {
        let current_index = read_u32(&mut reader)?;
        let horizon_steps = read_u32(&mut reader)?;
        let dynamic = read_f64_vec(&mut reader, tank_count * dynamic_stride)?;
        steps.push(BlackwellReplayStep { current_index, horizon_steps, dynamic });
    }

    Ok(BlackwellReplaySnapshot {
        data_length,
        tank_count,
        export_power,
        demand_energy,
        packed_config,
        steps,
    })
}

fn blackwell_fnv1a_update(hash: &mut u64, value: f64) {
    for byte in value.to_bits().to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(1099511628211);
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


#[napi(object)]
pub struct BlackwellReplayResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub data_length: u32,
    pub tank_count: u32,
    pub step_count: u32,
    pub repetitions: u32,
    pub wall_ms: Vec<f64>,
    pub kernel_ms: Vec<f64>,
    pub gpu_total_ms: Vec<f64>,
    pub host_wait_ms: Vec<f64>,
    pub native_profile_wall_ms: Vec<f64>,
    pub median_wall_ms: f64,
    pub min_wall_ms: f64,
    pub max_wall_ms: f64,
    pub checksum_hex: String,
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

    blackwell_capture:
        Option<BlackwellCapture>,
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

        let snapshot_path =
            blackwell_default_snapshot_path();

        let blackwell_capture =
            if blackwell_capture_requested(
                &snapshot_path,
            ) {
                Some(
                    BlackwellCapture::new(
                        snapshot_path,
                        &export_power_nonnegative,
                        &demand_energy_nonnegative,
                    ),
                )
            } else {
                None
            };

        Ok(Self {
            context,
            data_length,

            lockstep_parameters:
                Vec::new(),

            lockstep_dynamic:
                Vec::new(),

            blackwell_capture,
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


        if let Some(capture) = self.blackwell_capture.as_mut() {
            let packed_capture: Vec<f64> =
                packed.iter().copied().collect();
            capture.configure(&packed_capture);
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


        self.context
            .run_causal_dynamic_batch(
                current_index as usize,
                horizon_steps as usize,
                &self.lockstep_dynamic,
            )
    }


    /// Lowest-overhead production lockstep interface.
    ///
    /// Both TypedArrays are borrowed synchronously from JavaScript:
    ///
    ///   dynamic:
    ///     3 doubles per tank
    ///
    ///   output:
    ///     1 cutLevelPower double per tank
    ///
    /// No Rust Vec result allocation and no JS Array conversion are
    /// performed. The caller owns and reuses the output Float64Array.
    #[napi]
    pub fn run_lockstep_dynamic_into(
        &mut self,
        current_index:
            u32,

        horizon_steps:
            u32,

        dynamic:
            Float64ArraySlice<'_>,

        mut output:
            Float64ArraySlice<'_>,
    ) -> Result<()> {
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


        if (
            output.len() !=
            self.lockstep_dynamic
                .len()
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Output lockstep array length must be {}; received {}.",
                        self.lockstep_dynamic.len(),
                        output.len(),
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


        /*
         * SAFETY:
         *
         * This N-API method is synchronous. JavaScript cannot execute
         * or access the output TypedArray until this call returns.
         * Therefore the native call has exclusive access to the
         * backing bytes for the duration of this mutable borrow.
         */
        let output_slice =
            unsafe {
                output.as_mut()
            };


        self.context
            .run_causal_dynamic_batch_into(
                current_index as usize,
                horizon_steps as usize,
                &self.lockstep_dynamic,
                output_slice,
            )?;

        let mut completed_capture = false;
        if let Some(capture) = self.blackwell_capture.as_mut() {
            let dynamic_capture: Vec<f64> =
                dynamic.iter().copied().collect();
            capture.push_step(
                current_index,
                horizon_steps,
                &dynamic_capture,
            );
            completed_capture = capture.complete();
            if completed_capture {
                match capture.write_snapshot() {
                    Ok(()) => println!(
                        "[SunOpt] Blackwell benchmark snapshot written: {}",
                        capture.path.display(),
                    ),
                    Err(error) => eprintln!(
                        "[SunOpt] Blackwell benchmark snapshot write failed: {error}",
                    ),
                }
            }
        }
        if completed_capture {
            self.blackwell_capture = None;
        }

        Ok(())
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


fn blackwell_parameters_from_packed(
    packed: &[f64],
    data_length: usize,
) -> Result<Vec<SunoptCausalParameters>> {
    if packed.is_empty() || packed.len() % BLACKWELL_CONFIG_STRIDE != 0 {
        return Err(Error::from_reason(
            "Blackwell replay packed configuration is invalid.",
        ));
    }

    let tank_count = packed.len() / BLACKWELL_CONFIG_STRIDE;
    let mut parameters = Vec::with_capacity(tank_count);
    for tank_index in 0..tank_count {
        let base = tank_index * BLACKWELL_CONFIG_STRIDE;
        parameters.push(SunoptCausalParameters {
            data_length,
            current_index: 0,
            horizon_steps: 1,
            current_storage_energy: packed[base + 0],
            minimum_reserve_energy: packed[base + 1],
            target_storage_energy: packed[base + 2],
            maximum_heat_pump_electrical_power: packed[base + 3],
            cop: packed[base + 4],
            requested_electrical_energy: packed[base + 5],
            energy_cut_level: 0.0,
            time_step_hours: packed[base + 6],
            storage_energy_per_kelvin: packed[base + 7],
            storage_conductance_w_per_kelvin: packed[base + 8],
            storage_min_temperature: packed[base + 9],
            ambient_temperature: packed[base + 10],
            daily_heat_loss: packed[base + 11],
        });
    }
    Ok(parameters)
}

fn blackwell_dynamic_from_values(
    values: &[f64],
    tank_count: usize,
    dynamic: &mut [SunoptCausalDynamic],
) -> Result<()> {
    if values.len() != tank_count * BLACKWELL_DYNAMIC_STRIDE || dynamic.len() != tank_count {
        return Err(Error::from_reason(
            "Blackwell replay dynamic dimensions are invalid.",
        ));
    }
    for tank_index in 0..tank_count {
        let base = tank_index * BLACKWELL_DYNAMIC_STRIDE;
        dynamic[tank_index] = SunoptCausalDynamic {
            current_storage_energy: values[base + 0],
            minimum_reserve_energy: values[base + 1],
            requested_electrical_energy: values[base + 2],
        };
    }
    Ok(())
}

fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = sorted.len()/2;
    if sorted.len()%2==0 { (sorted[m-1]+sorted[m])/2.0 } else { sorted[m] }
}

#[napi]
pub fn run_blackwell_replay(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellReplayResult> {
    let path = snapshot_path.map(PathBuf::from).unwrap_or_else(blackwell_default_snapshot_path);
    let snapshot = read_blackwell_snapshot(&path).map_err(|error| {
        Error::from_reason(format!(
            "Unable to read Blackwell benchmark snapshot {}: {error}",
            path.display(),
        ))
    })?;

    if snapshot.steps.len() != snapshot.data_length {
        return Err(Error::from_reason(format!(
            "Blackwell snapshot contains {} steps but annual data length is {}.",
            snapshot.steps.len(), snapshot.data_length,
        )));
    }

    let parameters = blackwell_parameters_from_packed(&snapshot.packed_config, snapshot.data_length)?;
    if parameters.len() != snapshot.tank_count {
        return Err(Error::from_reason(
            "Blackwell snapshot tank count does not match packed configuration.",
        ));
    }

    let selected_device = device_id.unwrap_or(0);
    let mut context = CudaSimulationContext::new(selected_device)?;
    context.upload_causal_arrays(&snapshot.export_power, &snapshot.demand_energy)?;
    context.configure_causal_static_parameters(&parameters)?;

    let mut dynamic = vec![SunoptCausalDynamic::default(); snapshot.tank_count];
    let mut output = vec![0.0f64; snapshot.tank_count];
    let repetition_count = repetitions.unwrap_or(3).clamp(1,20);

    let mut wall_ms = Vec::with_capacity(repetition_count as usize);
    let mut kernel_ms = Vec::with_capacity(repetition_count as usize);
    let mut gpu_total_ms = Vec::with_capacity(repetition_count as usize);
    let mut host_wait_ms = Vec::with_capacity(repetition_count as usize);
    let mut native_profile_wall_ms = Vec::with_capacity(repetition_count as usize);

    for _ in 0..repetition_count {
        context.reset_causal_profile()?;
        let started = Instant::now();
        for step in &snapshot.steps {
            blackwell_dynamic_from_values(&step.dynamic, snapshot.tank_count, &mut dynamic)?;
            context.run_causal_dynamic_batch_into(
                step.current_index as usize,
                step.horizon_steps as usize,
                &dynamic,
                &mut output,
            )?;
        }
        wall_ms.push(started.elapsed().as_secs_f64()*1000.0);
        let profile = context.causal_profile()?;
        kernel_ms.push(profile.kernel_ms);
        gpu_total_ms.push(profile.gpu_total_ms);
        host_wait_ms.push(profile.host_wait_ms);
        native_profile_wall_ms.push(profile.wall_ms);
    }

    let mut checksum = 1469598103934665603u64;
    for step in &snapshot.steps {
        blackwell_dynamic_from_values(&step.dynamic, snapshot.tank_count, &mut dynamic)?;
        context.run_causal_dynamic_batch_into(
            step.current_index as usize,
            step.horizon_steps as usize,
            &dynamic,
            &mut output,
        )?;
        for value in &output { blackwell_fnv1a_update(&mut checksum, *value); }
    }

    let median_wall_ms = median_f64(&wall_ms);
    let min_wall_ms = wall_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_wall_ms = wall_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    Ok(BlackwellReplayResult {
        snapshot_path: path.display().to_string(),
        device_id: selected_device,
        data_length: snapshot.data_length as u32,
        tank_count: snapshot.tank_count as u32,
        step_count: snapshot.steps.len() as u32,
        repetitions: repetition_count,
        wall_ms,
        kernel_ms,
        gpu_total_ms,
        host_wait_ms,
        native_profile_wall_ms,
        median_wall_ms,
        min_wall_ms,
        max_wall_ms,
        checksum_hex: format!("{checksum:016x}"),
    })
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
