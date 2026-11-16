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

    // SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FFI_BEGIN
    fn sunopt_cuda_run_causal_graph_replay(
        context:
            *mut SunoptCudaContextOpaque,

        host_current_indices:
            *const u32,

        host_horizon_steps:
            *const u32,

        host_dynamic_parameters:
            *const SunoptCausalDynamic,

        step_count:
            usize,

        tank_count:
            usize,

        repetitions:
            u32,

        host_results:
            *mut f64,

        host_wall_ms:
            *mut f64,

        host_gpu_ms:
            *mut f64,

        out_graph_build_ms:
            *mut f64,

        out_result_d2h_ms:
            *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FFI_END


// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_BEGIN
    fn sunopt_cuda_run_max_affine_tree_benchmark(
        device_id:
            c_int,

        host_export_power:
            *const f64,

        host_demand_energy:
            *const f64,

        data_length:
            usize,

        host_tank_parameters:
            *const f64,

        tank_count:
            usize,

        host_current_indices:
            *const u32,

        host_horizon_steps:
            *const u32,

        host_dynamic_values:
            *const f64,

        host_cut_levels:
            *const f64,

        step_count:
            usize,

        repetitions:
            u32,

        host_survival:
            *mut u32,

        out_h2d_ms:
            *mut f64,

        out_kernel_total_ms:
            *mut f64,

        out_d2h_ms:
            *mut f64,

        out_wall_ms:
            *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FFI_END


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


// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellGraphReplayResult {
    pub snapshot_path:
        String,

    pub device_id:
        i32,

    pub data_length:
        u32,

    pub tank_count:
        u32,

    pub step_count:
        u32,

    pub causal_jobs:
        u32,

    pub repetitions:
        u32,

    pub graph_build_ms:
        f64,

    pub result_d2h_ms:
        f64,

    pub wall_ms:
        Vec<f64>,

    pub gpu_ms:
        Vec<f64>,

    pub median_wall_ms:
        f64,

    pub median_gpu_ms:
        f64,

    pub min_wall_ms:
        f64,

    pub max_wall_ms:
        f64,

    pub checksum_hex:
        String,
}
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_RESULT_END


// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellMaxAffineCudaTreeResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,

    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub survival_check_count: u32,

    pub supported_jobs: u32,
    pub unsupported_jobs: u32,

    pub reference_full_causal_gpu_ms: f64,

    pub candidate_h2d_ms: f64,
    pub candidate_kernel_total_ms: f64,
    pub candidate_kernel_ms: f64,
    pub candidate_d2h_ms: f64,
    pub candidate_wall_ms: f64,

    pub candidate_ns_per_survival_check: f64,

    pub normal_survival_mismatches: u32,
    pub maxpv_survival_mismatches: u32,
    pub unexpected_gpu_unsupported_checks: u32,
}
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_END


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

// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FN_BEGIN
#[napi]
pub fn run_blackwell_graph_replay(
    snapshot_path:
        Option<String>,

    repetitions:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<BlackwellGraphReplayResult> {
    let path =
        snapshot_path
            .map(
                PathBuf::from,
            )
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );


    let snapshot =
        read_blackwell_snapshot(
            &path,
        )
        .map_err(
            |error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell benchmark snapshot {}: {error}",
                        path.display(),
                    ),
                )
            },
        )?;


    if (
        snapshot.steps.len() !=
        snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "Blackwell snapshot contains {} steps but annual data length is {}.",
                    snapshot.steps.len(),
                    snapshot.data_length,
                ),
            ),
        );
    }


    let parameters =
        blackwell_parameters_from_packed(
            &snapshot.packed_config,
            snapshot.data_length,
        )?;


    if (
        parameters.len() !=
        snapshot.tank_count
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot tank count does not match packed configuration.",
            ),
        );
    }


    let selected_device =
        device_id
            .unwrap_or(0);


    let mut context =
        CudaSimulationContext::new(
            selected_device,
        )?;


    context
        .upload_causal_arrays(
            &snapshot.export_power,
            &snapshot.demand_energy,
        )?;


    context
        .configure_causal_static_parameters(
            &parameters,
        )?;


    let repetition_count =
        repetitions
            .unwrap_or(3)
            .clamp(
                1,
                20,
            );


    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;


    let total_jobs =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Blackwell graph replay job count overflow.",
                    )
                },
            )?;


    if (
        total_jobs >
        u32::MAX as usize
    ) {
        return Err(
            Error::from_reason(
                "Blackwell graph replay job count exceeds u32 result capacity.",
            ),
        );
    }


    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<SunoptCausalDynamic>::with_capacity(
            total_jobs,
        );

    let mut dynamic_step =
        vec![
            SunoptCausalDynamic::default();
            tank_count
        ];


    for step in
        &snapshot.steps
    {
        blackwell_dynamic_from_values(
            &step.dynamic,
            tank_count,
            &mut dynamic_step,
        )?;


        current_indices.push(
            step.current_index,
        );

        horizon_steps.push(
            step.horizon_steps,
        );

        dynamic_flat.extend_from_slice(
            &dynamic_step,
        );
    }


    let mut results =
        vec![
            0.0_f64;
            total_jobs
        ];

    let mut wall_ms =
        vec![
            0.0_f64;
            repetition_count as usize
        ];

    let mut gpu_ms =
        vec![
            0.0_f64;
            repetition_count as usize
        ];

    let mut graph_build_ms =
        0.0_f64;

    let mut result_d2h_ms =
        0.0_f64;


    let code =
        unsafe {
            sunopt_cuda_run_causal_graph_replay(
                context.raw.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                tank_count,
                repetition_count,
                results.as_mut_ptr(),
                wall_ms.as_mut_ptr(),
                gpu_ms.as_mut_ptr(),
                &mut graph_build_ms,
                &mut result_d2h_ms,
            )
        };


    if (
        code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_causal_graph_replay",
                code,
            ),
        );
    }


    let mut checksum =
        1469598103934665603u64;


    for value in
        &results
    {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }


    let median_wall_ms =
        median_f64(
            &wall_ms,
        );

    let median_gpu_ms =
        median_f64(
            &gpu_ms,
        );

    let min_wall_ms =
        wall_ms
            .iter()
            .copied()
            .fold(
                f64::INFINITY,
                f64::min,
            );

    let max_wall_ms =
        wall_ms
            .iter()
            .copied()
            .fold(
                f64::NEG_INFINITY,
                f64::max,
            );


    Ok(
        BlackwellGraphReplayResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            data_length:
                snapshot.data_length
                    as u32,

            tank_count:
                tank_count
                    as u32,

            step_count:
                step_count
                    as u32,

            causal_jobs:
                total_jobs
                    as u32,

            repetitions:
                repetition_count,

            graph_build_ms,

            result_d2h_ms,

            wall_ms,
            gpu_ms,

            median_wall_ms,
            median_gpu_ms,
            min_wall_ms,
            max_wall_ms,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FN_END


// SUNOPT_BLACKWELL_PHYSICS_COMPRESSION_RESEARCH_BEGIN
/*
 * ============================================================================
 * BLACKWELL PHYSICS COMPRESSION RESEARCH
 * ============================================================================
 *
 * Benchmark/research only. Production lockstep is untouched.
 *
 * Two ideas are tested against the exact recorded Blackwell snapshot:
 *
 * 1. Exact max-affine causal compression
 *
 *    For one future bucket, with the same affine heat-loss model used by the
 *    CUDA causal kernel, the minimum energy required BEFORE that bucket can be
 *    written as
 *
 *        R_i = max(A_i, a * R_{i+1} + B_i)
 *
 *    where R_{i+1} is the energy required after this bucket's forecast charge.
 *    A function F(x)=max(c, a*x+b) is closed under composition, so an entire
 *    48 h horizon can be represented by three max-affine coefficients plus
 *    one feasibility bound. Composition is associative and can therefore be
 *    tree-reduced on a GPU in O(log N)
 *    dependency depth (8 levels for <=256 buckets) instead of walking ~192
 *    storage states serially.
 *
 * 2. Truncated Fourier reconstruction
 *
 *    The 48 h export and demand windows are reconstructed from a configurable
 *    number of real Fourier harmonics, then run through the same causal logic.
 *    This is deliberately approximate and is evaluated for signal error,
 *    Waterline error and survival-decision mismatches.
 */

#[derive(Clone, Copy)]
struct ResearchTankParameters {
    target_storage_energy: f64,
    maximum_heat_pump_electrical_power: f64,
    cop: f64,
    time_step_hours: f64,
    heat_loss_retention: f64,
    heat_loss_intercept: f64,
    affine_supported: bool,
}

#[derive(Clone, Copy, Default)]
struct ResearchCausalResult {
    cut_level_power: f64,
    normal_plan_survives: bool,
    maximum_pv_plan_survives: bool,
}

#[derive(Clone, Copy)]
struct ResearchMaxAffine {
    /*
     * F(x) = max(c, a*x+b), valid only for x <= max_input.
     *
     * The domain limit is required because every intermediate storage
     * state is capped by targetStorageEnergy.  Without carrying this
     * feasibility domain a later impossible requirement (> target)
     * could be hidden by a large earlier forecast charge.
     */
    a: f64,
    b: f64,
    c: f64,
    max_input: f64,
}

impl ResearchMaxAffine {
    fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: f64::NEG_INFINITY,
            max_input: f64::INFINITY,
        }
    }

    fn eval(self, x: f64) -> Option<f64> {
        if x > self.max_input + 1.0e-12 {
            return None;
        }

        Some(self.c.max(self.a * x + self.b))
    }
}

fn research_compose_max_affine(
    left: ResearchMaxAffine,
    right: ResearchMaxAffine,
) -> ResearchMaxAffine {
    /*
     * left(right(x)).  Besides the max-affine coefficients we propagate
     * the exact domain for which right(x) is a feasible input to left.
     */
    let mut max_input = right.max_input;

    if right.c > left.max_input + 1.0e-12 {
        max_input = f64::NEG_INFINITY;
    } else if right.a > 0.0 {
        let bound =
            (left.max_input - right.b) /
            right.a;

        max_input = max_input.min(bound);
    } else if right.b > left.max_input + 1.0e-12 {
        max_input = f64::NEG_INFINITY;
    }

    ResearchMaxAffine {
        a: left.a * right.a,
        b: left.a * right.b + left.b,
        c: left.c.max(left.a * right.c + left.b),
        max_input,
    }
}

fn research_tank_from_packed(
    packed: &[f64],
    tank_index: usize,
) -> ResearchTankParameters {
    let base = tank_index * BLACKWELL_CONFIG_STRIDE;

    let target_storage_energy = packed[base + 2];
    let maximum_heat_pump_electrical_power = packed[base + 3];
    let cop = packed[base + 4];
    let time_step_hours = packed[base + 6];
    let storage_energy_per_kelvin = packed[base + 7];
    let storage_conductance_w_per_kelvin = packed[base + 8];
    let storage_min_temperature = packed[base + 9];
    let ambient_temperature = packed[base + 10];
    let daily_heat_loss = packed[base + 11];

    let (heat_loss_retention, heat_loss_intercept, affine_supported) =
        if daily_heat_loss >= 0.0 {
            (
                1.0,
                daily_heat_loss * time_step_hours / 24.0,
                true,
            )
        } else if storage_energy_per_kelvin > 0.0 {
            let conductance_energy =
                storage_conductance_w_per_kelvin * time_step_hours / 1000.0;

            let slope =
                conductance_energy / storage_energy_per_kelvin;

            let intercept =
                conductance_energy *
                (storage_min_temperature - ambient_temperature);

            (
                1.0 - slope,
                intercept,
                slope >= 0.0 &&
                    slope < 1.0 &&
                    intercept >= 0.0,
            )
        } else {
            (0.0, 0.0, false)
        };

    ResearchTankParameters {
        target_storage_energy,
        maximum_heat_pump_electrical_power,
        cop,
        time_step_hours,
        heat_loss_retention,
        heat_loss_intercept,
        affine_supported,
    }
}

fn research_hp_thermal_energy(
    export_power: f64,
    cut_level_power: f64,
    tank: ResearchTankParameters,
) -> f64 {
    let hp_power =
        (export_power - cut_level_power)
            .max(0.0)
            .min(tank.maximum_heat_pump_electrical_power);

    hp_power * tank.cop * tank.time_step_hours
}

fn research_electrical_energy_at_cut(
    export_window: &[f64],
    cut_level_power: f64,
    tank: ResearchTankParameters,
) -> f64 {
    let mut result = 0.0;

    for &export_power in export_window {
        result +=
            (export_power - cut_level_power)
                .max(0.0)
                .min(tank.maximum_heat_pump_electrical_power) *
            tank.time_step_hours;
    }

    result
}

fn research_energy_cut_level(
    export_window: &[f64],
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> f64 {
    let mut maximum_export_power = 0.0_f64;

    for &value in export_window {
        maximum_export_power = maximum_export_power.max(value);
    }

    let maximum_schedulable =
        research_electrical_energy_at_cut(
            export_window,
            0.0,
            tank,
        );

    let target =
        requested_electrical_energy
            .max(0.0)
            .min(maximum_schedulable);

    if target <= 0.0 || maximum_export_power <= 0.0 {
        return maximum_export_power;
    }

    if target >= maximum_schedulable {
        return 0.0;
    }

    let mut lower = 0.0;
    let mut upper = maximum_export_power;

    for _ in 0..32 {
        let middle = (lower + upper) * 0.5;
        let energy =
            research_electrical_energy_at_cut(
                export_window,
                middle,
                tank,
            );

        if energy > target {
            lower = middle;
        } else {
            upper = middle;
        }
    }

    (lower + upper) * 0.5
}

fn research_scalar_survives(
    export_window: &[f64],
    demand_window: &[f64],
    cut_level_power: f64,
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    tank: ResearchTankParameters,
) -> bool {
    const EPSILON: f64 = 1.0e-6;

    let reserve =
        minimum_reserve_energy
            .max(0.0)
            .min(tank.target_storage_energy);

    let mut stored =
        current_storage_energy
            .max(0.0)
            .min(tank.target_storage_energy);

    for offset in 0..export_window.len() {
        let charge =
            research_hp_thermal_energy(
                export_window[offset],
                cut_level_power,
                tank,
            );

        if offset == 0 {
            stored =
                (stored + charge)
                    .min(tank.target_storage_energy);
            continue;
        }

        let requested_loss =
            (
                (1.0 - tank.heat_loss_retention) * stored +
                tank.heat_loss_intercept
            )
            .max(0.0);

        let heat_loss = requested_loss.min(stored);
        let after_loss = (stored - heat_loss).max(0.0);
        let demand = demand_window[offset].max(0.0);
        let discharged = demand.min(after_loss);
        let before_charge = after_loss - discharged;
        let unmet = (demand - discharged).max(0.0);

        if unmet > EPSILON {
            return false;
        }

        if reserve - before_charge > EPSILON {
            return false;
        }

        stored =
            (before_charge + charge)
                .min(tank.target_storage_energy);
    }

    true
}

fn research_max_affine_survives(
    export_window: &[f64],
    demand_window: &[f64],
    cut_level_power: f64,
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    tank: ResearchTankParameters,
) -> Option<bool> {
    const EPSILON: f64 = 1.0e-6;

    if !tank.affine_supported ||
        tank.heat_loss_retention <= 0.0 ||
        export_window.len() != demand_window.len() ||
        export_window.is_empty()
    {
        return None;
    }

    let reserve =
        minimum_reserve_energy
            .max(0.0)
            .min(tank.target_storage_energy);

    /*
     * With a strictly positive reserve the canonical tolerance can be
     * folded exactly into the required-energy threshold.  At a zero
     * reserve, heat-loss clipping introduces a discontinuous zero-energy
     * branch; that needs a richer piecewise summary than one max-affine
     * segment.  The first experiment therefore reports those jobs as
     * unsupported instead of pretending exact parity.
     */
    if reserve <= EPSILON {
        return None;
    }

    let current =
        current_storage_energy
            .max(0.0)
            .min(tank.target_storage_energy);

    if export_window.len() == 1 {
        return Some(true);
    }

    let inverse_retention =
        1.0 / tank.heat_loss_retention;

    let mut transforms =
        Vec::<ResearchMaxAffine>::with_capacity(
            export_window.len() - 1,
        );

    for offset in 1..export_window.len() {
        let demand = demand_window[offset].max(0.0);
        let charge =
            research_hp_thermal_energy(
                export_window[offset],
                cut_level_power,
                tank,
            );

        /*
         * Required-energy recurrence.  Because reserve > EPSILON for this
         * exact branch, the scalar 1e-6 survival tolerance folds directly
         * into the reserve threshold:
         *
         * R_i = max(
         *   (lossIntercept + demand + reserve - EPSILON) / retention,
         *   R_{i+1}/retention +
         *     (lossIntercept + demand - charge)/retention
         * )
         */
        transforms.push(
            ResearchMaxAffine {
                a: inverse_retention,
                b:
                    (
                        tank.heat_loss_intercept +
                        demand -
                        charge
                    ) *
                    inverse_retention,
                c:
                    (
                        tank.heat_loss_intercept +
                        demand +
                        reserve -
                        EPSILON
                    ) *
                    inverse_retention,
                max_input:
                    tank.target_storage_energy,
            },
        );
    }

    /*
     * Ordered associative tree reduction.
     * Adjacent transforms are combined as left(right(x)).
     */
    while transforms.len() > 1 {
        let old_len = transforms.len();
        let mut write = 0usize;
        let mut read = 0usize;

        while read + 1 < old_len {
            transforms[write] =
                research_compose_max_affine(
                    transforms[read],
                    transforms[read + 1],
                );
            write += 1;
            read += 2;
        }

        if read < old_len {
            transforms[write] = transforms[read];
            write += 1;
        }

        transforms.truncate(write);
    }

    let summary =
        transforms
            .first()
            .copied()
            .unwrap_or_else(ResearchMaxAffine::identity);

    let required_after_current_bucket =
        match summary.eval(0.0) {
            Some(value) => value,
            None => return Some(false),
        };

    if required_after_current_bucket >
        tank.target_storage_energy + 1.0e-9
    {
        return Some(false);
    }

    let current_charge =
        research_hp_thermal_energy(
            export_window[0],
            cut_level_power,
            tank,
        );

    let after_current_charge =
        (current + current_charge)
            .min(tank.target_storage_energy);

    Some(
        after_current_charge + 1.0e-12 >=
        required_after_current_bucket,
    )
}

fn research_run_scalar(
    export_window: &[f64],
    demand_window: &[f64],
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> ResearchCausalResult {
    let energy_cut =
        research_energy_cut_level(
            export_window,
            requested_electrical_energy,
            tank,
        );

    let normal =
        research_scalar_survives(
            export_window,
            demand_window,
            energy_cut,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        );

    if normal {
        return ResearchCausalResult {
            cut_level_power: energy_cut,
            normal_plan_survives: true,
            maximum_pv_plan_survives: true,
        };
    }

    let maximum_pv =
        research_scalar_survives(
            export_window,
            demand_window,
            0.0,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        );

    if !maximum_pv {
        return ResearchCausalResult {
            cut_level_power: 0.0,
            normal_plan_survives: false,
            maximum_pv_plan_survives: false,
        };
    }

    let mut lower = 0.0;
    let mut upper = energy_cut;

    for _ in 0..20 {
        let middle = (lower + upper) * 0.5;

        if research_scalar_survives(
            export_window,
            demand_window,
            middle,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        ) {
            lower = middle;
        } else {
            upper = middle;
        }
    }

    ResearchCausalResult {
        cut_level_power: lower,
        normal_plan_survives: false,
        maximum_pv_plan_survives: true,
    }
}

fn research_run_max_affine(
    export_window: &[f64],
    demand_window: &[f64],
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> Option<ResearchCausalResult> {
    let energy_cut =
        research_energy_cut_level(
            export_window,
            requested_electrical_energy,
            tank,
        );

    let normal =
        research_max_affine_survives(
            export_window,
            demand_window,
            energy_cut,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        )?;

    if normal {
        return Some(
            ResearchCausalResult {
                cut_level_power: energy_cut,
                normal_plan_survives: true,
                maximum_pv_plan_survives: true,
            },
        );
    }

    let maximum_pv =
        research_max_affine_survives(
            export_window,
            demand_window,
            0.0,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        )?;

    if !maximum_pv {
        return Some(
            ResearchCausalResult {
                cut_level_power: 0.0,
                normal_plan_survives: false,
                maximum_pv_plan_survives: false,
            },
        );
    }

    let mut lower = 0.0;
    let mut upper = energy_cut;

    for _ in 0..20 {
        let middle = (lower + upper) * 0.5;

        if research_max_affine_survives(
            export_window,
            demand_window,
            middle,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        )? {
            lower = middle;
        } else {
            upper = middle;
        }
    }

    Some(
        ResearchCausalResult {
            cut_level_power: lower,
            normal_plan_survives: false,
            maximum_pv_plan_survives: true,
        },
    )
}

fn research_window(
    source: &[f64],
    current_index: usize,
    horizon_steps: usize,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(horizon_steps);

    for offset in 0..horizon_steps {
        result.push(
            source[
                (current_index + offset) % source.len()
            ],
        );
    }

    result
}

fn research_fourier_reconstruct(
    source: &[f64],
    harmonics: usize,
) -> Vec<f64> {
    let n = source.len();

    if n == 0 {
        return Vec::new();
    }

    let maximum_harmonics =
        harmonics.min((n.saturating_sub(1)) / 2);

    let inv_n = 1.0 / n as f64;
    let two_over_n = 2.0 * inv_n;
    let tau = std::f64::consts::PI * 2.0;

    let dc = source.iter().copied().sum::<f64>() * inv_n;

    let mut cosine = vec![0.0_f64; maximum_harmonics + 1];
    let mut sine = vec![0.0_f64; maximum_harmonics + 1];

    for k in 1..=maximum_harmonics {
        let mut c = 0.0;
        let mut s = 0.0;

        for (index, &value) in source.iter().enumerate() {
            let angle =
                tau *
                k as f64 *
                index as f64 *
                inv_n;

            c += value * angle.cos();
            s += value * angle.sin();
        }

        cosine[k] = c * two_over_n;
        sine[k] = s * two_over_n;
    }

    let mut result = vec![dc; n];

    for index in 0..n {
        let mut value = dc;

        for k in 1..=maximum_harmonics {
            let angle =
                tau *
                k as f64 *
                index as f64 *
                inv_n;

            value +=
                cosine[k] * angle.cos() +
                sine[k] * angle.sin();
        }

        /* Physical arrays are non-negative. */
        result[index] = value.max(0.0);
    }

    result
}

fn research_signal_error(
    reference: &[f64],
    approximate: &[f64],
) -> (f64, f64, f64) {
    let mut sum_abs = 0.0;
    let mut sum_sq = 0.0;
    let mut max_abs = 0.0_f64;

    for (&reference_value, &approximate_value) in
        reference.iter().zip(approximate.iter())
    {
        let error = (reference_value - approximate_value).abs();
        sum_abs += error;
        sum_sq += error * error;
        max_abs = max_abs.max(error);
    }

    let count = reference.len().max(1) as f64;

    (
        sum_abs / count,
        (sum_sq / count).sqrt(),
        max_abs,
    )
}

#[napi(object)]
pub struct BlackwellPhysicsCompressionResearchResult {
    pub snapshot_path: String,
    pub data_length: u32,
    pub tank_count: u32,
    pub sample_step_count: u32,
    pub sample_job_count: u32,

    pub gpu_reference_ms: f64,
    pub scalar_reference_ms: f64,
    pub max_affine_ms: f64,
    pub max_affine_tree_levels: u32,
    pub max_affine_supported_jobs: u32,

    pub scalar_gpu_max_abs_cut_error: f64,
    pub scalar_gpu_cut_error_over_1e6: u32,

    pub max_affine_scalar_max_abs_cut_error: f64,
    pub max_affine_scalar_cut_error_over_1e6: u32,
    pub max_affine_normal_survival_mismatches: u32,
    pub max_affine_maxpv_survival_mismatches: u32,

    pub fourier_harmonics: Vec<u32>,
    pub fourier_coefficients: Vec<u32>,
    pub fourier_signal_fraction: Vec<f64>,
    pub fourier_ms: Vec<f64>,

    pub fourier_export_mae: Vec<f64>,
    pub fourier_export_rmse: Vec<f64>,
    pub fourier_export_max_abs_error: Vec<f64>,
    pub fourier_demand_mae: Vec<f64>,
    pub fourier_demand_rmse: Vec<f64>,
    pub fourier_demand_max_abs_error: Vec<f64>,

    pub fourier_cut_mae: Vec<f64>,
    pub fourier_cut_rmse: Vec<f64>,
    pub fourier_cut_max_abs_error: Vec<f64>,
    pub fourier_cut_error_over_001: Vec<u32>,
    pub fourier_cut_error_over_01: Vec<u32>,
    pub fourier_normal_survival_mismatches: Vec<u32>,
    pub fourier_maxpv_survival_mismatches: Vec<u32>,
}

#[napi]
pub fn run_blackwell_physics_compression_research(
    snapshot_path: Option<String>,
    sample_steps: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellPhysicsCompressionResearchResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(blackwell_default_snapshot_path);

    let snapshot =
        read_blackwell_snapshot(&path)
            .map_err(|error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell benchmark snapshot {}: {error}",
                        path.display(),
                    ),
                )
            })?;

    if snapshot.steps.is_empty() || snapshot.tank_count == 0 {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty.",
            ),
        );
    }

    let requested_samples =
        sample_steps
            .unwrap_or(192)
            .clamp(2, snapshot.steps.len() as u32)
            as usize;

    let mut sample_positions = Vec::<usize>::with_capacity(requested_samples);

    for sample in 0..requested_samples {
        let position =
            sample *
            (snapshot.steps.len() - 1) /
            (requested_samples - 1);

        if sample_positions.last().copied() != Some(position) {
            sample_positions.push(position);
        }
    }

    let tank_parameters =
        (0..snapshot.tank_count)
            .map(|tank| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank,
                )
            })
            .collect::<Vec<_>>();

    let selected_device = device_id.unwrap_or(0);
    let cuda_parameters =
        blackwell_parameters_from_packed(
            &snapshot.packed_config,
            snapshot.data_length,
        )?;

    let mut context = CudaSimulationContext::new(selected_device)?;
    context.upload_causal_arrays(
        &snapshot.export_power,
        &snapshot.demand_energy,
    )?;
    context.configure_causal_static_parameters(&cuda_parameters)?;

    let sample_job_count =
        sample_positions.len() * snapshot.tank_count;

    let mut gpu_cut_levels =
        Vec::<f64>::with_capacity(sample_job_count);

    let mut dynamic =
        vec![
            SunoptCausalDynamic::default();
            snapshot.tank_count
        ];

    let mut gpu_output =
        vec![0.0_f64; snapshot.tank_count];

    let gpu_started = Instant::now();

    for &position in &sample_positions {
        let step = &snapshot.steps[position];

        blackwell_dynamic_from_values(
            &step.dynamic,
            snapshot.tank_count,
            &mut dynamic,
        )?;

        context.run_causal_dynamic_batch_into(
            step.current_index as usize,
            step.horizon_steps as usize,
            &dynamic,
            &mut gpu_output,
        )?;

        gpu_cut_levels.extend_from_slice(&gpu_output);
    }

    let gpu_reference_ms =
        gpu_started.elapsed().as_secs_f64() * 1000.0;

    let mut scalar_results =
        Vec::<ResearchCausalResult>::with_capacity(sample_job_count);

    let scalar_started = Instant::now();

    for &position in &sample_positions {
        let step = &snapshot.steps[position];
        let horizon = step.horizon_steps as usize;
        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                horizon,
            );
        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index as usize,
                horizon,
            );

        for tank_index in 0..snapshot.tank_count {
            let base = tank_index * BLACKWELL_DYNAMIC_STRIDE;

            scalar_results.push(
                research_run_scalar(
                    &export_window,
                    &demand_window,
                    step.dynamic[base + 0],
                    step.dynamic[base + 1],
                    step.dynamic[base + 2],
                    tank_parameters[tank_index],
                ),
            );
        }
    }

    let scalar_reference_ms =
        scalar_started.elapsed().as_secs_f64() * 1000.0;

    let mut scalar_gpu_max_abs_cut_error = 0.0_f64;
    let mut scalar_gpu_cut_error_over_1e6 = 0_u32;

    for (scalar, &gpu) in
        scalar_results.iter().zip(gpu_cut_levels.iter())
    {
        let error = (scalar.cut_level_power - gpu).abs();
        scalar_gpu_max_abs_cut_error =
            scalar_gpu_max_abs_cut_error.max(error);

        if error > 1.0e-6 {
            scalar_gpu_cut_error_over_1e6 += 1;
        }
    }

    let mut max_affine_results =
        Vec::<Option<ResearchCausalResult>>::with_capacity(sample_job_count);

    let max_affine_started = Instant::now();

    for &position in &sample_positions {
        let step = &snapshot.steps[position];
        let horizon = step.horizon_steps as usize;
        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                horizon,
            );
        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index as usize,
                horizon,
            );

        for tank_index in 0..snapshot.tank_count {
            let base = tank_index * BLACKWELL_DYNAMIC_STRIDE;

            max_affine_results.push(
                research_run_max_affine(
                    &export_window,
                    &demand_window,
                    step.dynamic[base + 0],
                    step.dynamic[base + 1],
                    step.dynamic[base + 2],
                    tank_parameters[tank_index],
                ),
            );
        }
    }

    let max_affine_ms =
        max_affine_started.elapsed().as_secs_f64() * 1000.0;

    let maximum_horizon =
        sample_positions
            .iter()
            .map(|&position| snapshot.steps[position].horizon_steps as usize)
            .max()
            .unwrap_or(1);

    let max_affine_tree_levels =
        if maximum_horizon <= 2 {
            0
        } else {
            (usize::BITS -
                (maximum_horizon - 2).leading_zeros())
                as u32
        };

    let mut max_affine_supported_jobs = 0_u32;
    let mut max_affine_scalar_max_abs_cut_error = 0.0_f64;
    let mut max_affine_scalar_cut_error_over_1e6 = 0_u32;
    let mut max_affine_normal_survival_mismatches = 0_u32;
    let mut max_affine_maxpv_survival_mismatches = 0_u32;

    for (reference, candidate) in
        scalar_results.iter().zip(max_affine_results.iter())
    {
        let Some(candidate) = candidate else {
            continue;
        };

        max_affine_supported_jobs += 1;

        let error =
            (reference.cut_level_power - candidate.cut_level_power).abs();

        max_affine_scalar_max_abs_cut_error =
            max_affine_scalar_max_abs_cut_error.max(error);

        if error > 1.0e-6 {
            max_affine_scalar_cut_error_over_1e6 += 1;
        }

        if reference.normal_plan_survives !=
            candidate.normal_plan_survives
        {
            max_affine_normal_survival_mismatches += 1;
        }

        if reference.maximum_pv_plan_survives !=
            candidate.maximum_pv_plan_survives
        {
            max_affine_maxpv_survival_mismatches += 1;
        }
    }

    let harmonics_list =
        [4usize, 8, 16, 24, 32, 48, 64, 80];

    let mut fourier_harmonics = Vec::<u32>::new();
    let mut fourier_coefficients = Vec::<u32>::new();
    let mut fourier_signal_fraction = Vec::<f64>::new();
    let mut fourier_ms = Vec::<f64>::new();

    let mut fourier_export_mae = Vec::<f64>::new();
    let mut fourier_export_rmse = Vec::<f64>::new();
    let mut fourier_export_max_abs_error = Vec::<f64>::new();
    let mut fourier_demand_mae = Vec::<f64>::new();
    let mut fourier_demand_rmse = Vec::<f64>::new();
    let mut fourier_demand_max_abs_error = Vec::<f64>::new();

    let mut fourier_cut_mae = Vec::<f64>::new();
    let mut fourier_cut_rmse = Vec::<f64>::new();
    let mut fourier_cut_max_abs_error = Vec::<f64>::new();
    let mut fourier_cut_error_over_001 = Vec::<u32>::new();
    let mut fourier_cut_error_over_01 = Vec::<u32>::new();
    let mut fourier_normal_survival_mismatches = Vec::<u32>::new();
    let mut fourier_maxpv_survival_mismatches = Vec::<u32>::new();

    for &harmonics in &harmonics_list {
        if harmonics >= maximum_horizon / 2 {
            continue;
        }

        let started = Instant::now();

        let mut export_abs_sum = 0.0;
        let mut export_sq_sum = 0.0;
        let mut export_max = 0.0_f64;
        let mut demand_abs_sum = 0.0;
        let mut demand_sq_sum = 0.0;
        let mut demand_max = 0.0_f64;
        let mut signal_count = 0usize;

        let mut cut_abs_sum = 0.0;
        let mut cut_sq_sum = 0.0;
        let mut cut_max = 0.0_f64;
        let mut cut_count = 0usize;
        let mut over_001 = 0_u32;
        let mut over_01 = 0_u32;
        let mut normal_mismatches = 0_u32;
        let mut maxpv_mismatches = 0_u32;

        let mut reference_offset = 0usize;

        for &position in &sample_positions {
            let step = &snapshot.steps[position];
            let horizon = step.horizon_steps as usize;

            let export_window =
                research_window(
                    &snapshot.export_power,
                    step.current_index as usize,
                    horizon,
                );
            let demand_window =
                research_window(
                    &snapshot.demand_energy,
                    step.current_index as usize,
                    horizon,
                );

            let export_fourier =
                research_fourier_reconstruct(
                    &export_window,
                    harmonics,
                );
            let demand_fourier =
                research_fourier_reconstruct(
                    &demand_window,
                    harmonics,
                );

            let (export_mae, export_rmse, export_max_error) =
                research_signal_error(
                    &export_window,
                    &export_fourier,
                );
            let (demand_mae, demand_rmse, demand_max_error) =
                research_signal_error(
                    &demand_window,
                    &demand_fourier,
                );

            let n = horizon as f64;
            export_abs_sum += export_mae * n;
            export_sq_sum += export_rmse * export_rmse * n;
            export_max = export_max.max(export_max_error);
            demand_abs_sum += demand_mae * n;
            demand_sq_sum += demand_rmse * demand_rmse * n;
            demand_max = demand_max.max(demand_max_error);
            signal_count += horizon;

            for tank_index in 0..snapshot.tank_count {
                let base = tank_index * BLACKWELL_DYNAMIC_STRIDE;
                let candidate =
                    research_run_scalar(
                        &export_fourier,
                        &demand_fourier,
                        step.dynamic[base + 0],
                        step.dynamic[base + 1],
                        step.dynamic[base + 2],
                        tank_parameters[tank_index],
                    );

                let reference =
                    scalar_results[reference_offset + tank_index];

                let cut_error =
                    (candidate.cut_level_power -
                        reference.cut_level_power)
                        .abs();

                cut_abs_sum += cut_error;
                cut_sq_sum += cut_error * cut_error;
                cut_max = cut_max.max(cut_error);
                cut_count += 1;

                if cut_error > 0.01 {
                    over_001 += 1;
                }

                if cut_error > 0.1 {
                    over_01 += 1;
                }

                if candidate.normal_plan_survives !=
                    reference.normal_plan_survives
                {
                    normal_mismatches += 1;
                }

                if candidate.maximum_pv_plan_survives !=
                    reference.maximum_pv_plan_survives
                {
                    maxpv_mismatches += 1;
                }
            }

            reference_offset += snapshot.tank_count;
        }

        let elapsed_ms =
            started.elapsed().as_secs_f64() * 1000.0;

        let signal_denominator = signal_count.max(1) as f64;
        let cut_denominator = cut_count.max(1) as f64;

        fourier_harmonics.push(harmonics as u32);
        fourier_coefficients.push((1 + 2 * harmonics) as u32);
        fourier_signal_fraction.push(
            (1 + 2 * harmonics) as f64 /
            maximum_horizon as f64,
        );
        fourier_ms.push(elapsed_ms);

        fourier_export_mae.push(export_abs_sum / signal_denominator);
        fourier_export_rmse.push((export_sq_sum / signal_denominator).sqrt());
        fourier_export_max_abs_error.push(export_max);
        fourier_demand_mae.push(demand_abs_sum / signal_denominator);
        fourier_demand_rmse.push((demand_sq_sum / signal_denominator).sqrt());
        fourier_demand_max_abs_error.push(demand_max);

        fourier_cut_mae.push(cut_abs_sum / cut_denominator);
        fourier_cut_rmse.push((cut_sq_sum / cut_denominator).sqrt());
        fourier_cut_max_abs_error.push(cut_max);
        fourier_cut_error_over_001.push(over_001);
        fourier_cut_error_over_01.push(over_01);
        fourier_normal_survival_mismatches.push(normal_mismatches);
        fourier_maxpv_survival_mismatches.push(maxpv_mismatches);
    }

    Ok(
        BlackwellPhysicsCompressionResearchResult {
            snapshot_path: path.display().to_string(),
            data_length: snapshot.data_length as u32,
            tank_count: snapshot.tank_count as u32,
            sample_step_count: sample_positions.len() as u32,
            sample_job_count: sample_job_count as u32,

            gpu_reference_ms,
            scalar_reference_ms,
            max_affine_ms,
            max_affine_tree_levels,
            max_affine_supported_jobs,

            scalar_gpu_max_abs_cut_error,
            scalar_gpu_cut_error_over_1e6,

            max_affine_scalar_max_abs_cut_error,
            max_affine_scalar_cut_error_over_1e6,
            max_affine_normal_survival_mismatches,
            max_affine_maxpv_survival_mismatches,

            fourier_harmonics,
            fourier_coefficients,
            fourier_signal_fraction,
            fourier_ms,

            fourier_export_mae,
            fourier_export_rmse,
            fourier_export_max_abs_error,
            fourier_demand_mae,
            fourier_demand_rmse,
            fourier_demand_max_abs_error,

            fourier_cut_mae,
            fourier_cut_rmse,
            fourier_cut_max_abs_error,
            fourier_cut_error_over_001,
            fourier_cut_error_over_01,
            fourier_normal_survival_mismatches,
            fourier_maxpv_survival_mismatches,
        },
    )
}
// SUNOPT_BLACKWELL_PHYSICS_COMPRESSION_RESEARCH_END


// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_BEGIN
#[napi]
pub fn run_blackwell_max_affine_cuda_tree(
    snapshot_path:
        Option<String>,

    sample_steps:
        Option<u32>,

    repetitions:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<BlackwellMaxAffineCudaTreeResult> {
    let path =
        snapshot_path
            .map(
                PathBuf::from,
            )
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(
            &path,
        )
        .map_err(
            |error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell benchmark snapshot {}: {error}",
                        path.display(),
                    ),
                )
            },
        )?;

    if (
        snapshot.steps.is_empty() ||
        snapshot.tank_count == 0 ||
        snapshot.export_power.is_empty() ||
        snapshot.demand_energy.len() !=
            snapshot.export_power.len()
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or inconsistent.",
            ),
        );
    }

    let requested_samples =
        sample_steps
            .unwrap_or(
                192,
            )
            .clamp(
                2,
                snapshot.steps.len()
                    as u32,
            )
            as usize;

    let mut sample_positions =
        Vec::<usize>::with_capacity(
            requested_samples,
        );

    for sample in
        0..
        requested_samples
    {
        let position =
            sample *
            (
                snapshot.steps.len() -
                1
            ) /
            (
                requested_samples -
                1
            );

        if (
            sample_positions
                .last()
                .copied() !=
            Some(
                position,
            )
        ) {
            sample_positions.push(
                position,
            );
        }
    }

    let selected_device =
        device_id
            .unwrap_or(
                0,
            );

    let repetitions =
        repetitions
            .unwrap_or(
                20,
            )
            .clamp(
                1,
                10_000,
            );

    let tank_parameters =
        (
            0..
            snapshot.tank_count
        )
        .map(
            |tank_index| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank_index,
                )
            },
        )
        .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count *
            6,
        );

    for tank in
        &tank_parameters
    {
        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            tank.maximum_heat_pump_electrical_power,
        );

        packed_tanks.push(
            tank.cop *
            tank.time_step_hours,
        );

        packed_tanks.push(
            tank.heat_loss_retention,
        );

        packed_tanks.push(
            tank.heat_loss_intercept,
        );

        packed_tanks.push(
            if tank.affine_supported {
                1.0
            } else {
                0.0
            },
        );
    }

    let step_count =
        sample_positions.len();

    let job_count =
        step_count *
        snapshot.tank_count;

    let check_count =
        job_count *
        2;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_values =
        Vec::<f64>::with_capacity(
            job_count *
            2,
        );

    let mut cut_levels =
        Vec::<f64>::with_capacity(
            check_count,
        );

    let mut expected_survival =
        Vec::<u32>::with_capacity(
            check_count,
        );

    let mut supported_jobs =
        0_u32;

    for &position in
        &sample_positions
    {
        let step =
            &snapshot.steps[
                position
            ];

        let horizon =
            step.horizon_steps
                as usize;

        if (
            horizon == 0 ||
            horizon > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Max-affine CUDA tree benchmark supports horizon 1..=256; snapshot contains horizon {horizon}.",
                    ),
                ),
            );
        }

        current_indices.push(
            step.current_index,
        );

        horizon_steps.push(
            step.horizon_steps,
        );

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index
                    as usize,
                horizon,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index
                    as usize,
                horizon,
            );

        for tank_index in
            0..
            snapshot.tank_count
        {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let current_storage_energy =
                step.dynamic[
                    base +
                    0
                ];

            let minimum_reserve_energy =
                step.dynamic[
                    base +
                    1
                ];

            let requested_electrical_energy =
                step.dynamic[
                    base +
                    2
                ];

            let tank =
                tank_parameters[
                    tank_index
                ];

            dynamic_values.push(
                current_storage_energy,
            );

            dynamic_values.push(
                minimum_reserve_energy,
            );

            let energy_cut =
                research_energy_cut_level(
                    &export_window,
                    requested_electrical_energy,
                    tank,
                );

            cut_levels.push(
                energy_cut,
            );

            cut_levels.push(
                0.0,
            );

            let reserve =
                minimum_reserve_energy
                    .max(
                        0.0,
                    )
                    .min(
                        tank.target_storage_energy,
                    );

            let exact_supported =
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                reserve >
                    1.0e-6;

            if exact_supported {
                supported_jobs +=
                    1;

                expected_survival.push(
                    if research_scalar_survives(
                        &export_window,
                        &demand_window,
                        energy_cut,
                        current_storage_energy,
                        minimum_reserve_energy,
                        tank,
                    ) {
                        1
                    } else {
                        0
                    },
                );

                expected_survival.push(
                    if research_scalar_survives(
                        &export_window,
                        &demand_window,
                        0.0,
                        current_storage_energy,
                        minimum_reserve_energy,
                        tank,
                    ) {
                        1
                    } else {
                        0
                    },
                );
            } else {
                expected_survival.push(
                    2,
                );

                expected_survival.push(
                    2,
                );
            }
        }
    }

    /*
     * Existing complete causal GPU reference on the same sampled steps.
     * This is context only; it performs the whole causal cut search.
     */
    let cuda_parameters =
        blackwell_parameters_from_packed(
            &snapshot.packed_config,
            snapshot.data_length,
        )?;

    let mut reference_context =
        CudaSimulationContext::new(
            selected_device,
        )?;

    reference_context
        .upload_causal_arrays(
            &snapshot.export_power,
            &snapshot.demand_energy,
        )?;

    reference_context
        .configure_causal_static_parameters(
            &cuda_parameters,
        )?;

    let mut reference_dynamic =
        vec![
            SunoptCausalDynamic::default();
            snapshot.tank_count
        ];

    let mut reference_output =
        vec![
            0.0_f64;
            snapshot.tank_count
        ];

    let reference_started =
        Instant::now();

    for &position in
        &sample_positions
    {
        let step =
            &snapshot.steps[
                position
            ];

        blackwell_dynamic_from_values(
            &step.dynamic,
            snapshot.tank_count,
            &mut reference_dynamic,
        )?;

        reference_context
            .run_causal_dynamic_batch_into(
                step.current_index
                    as usize,
                step.horizon_steps
                    as usize,
                &reference_dynamic,
                &mut reference_output,
            )?;
    }

    let reference_full_causal_gpu_ms =
        reference_started
            .elapsed()
            .as_secs_f64() *
        1000.0;

    let mut gpu_survival =
        vec![
            0_u32;
            check_count
        ];

    let mut h2d_ms =
        0.0_f64;

    let mut kernel_total_ms =
        0.0_f64;

    let mut d2h_ms =
        0.0_f64;

    let mut wall_ms =
        0.0_f64;

    let error =
        unsafe {
            sunopt_cuda_run_max_affine_tree_benchmark(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                snapshot.tank_count,
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_values
                    .as_ptr(),
                cut_levels
                    .as_ptr(),
                step_count,
                repetitions,
                gpu_survival
                    .as_mut_ptr(),
                &mut h2d_ms,
                &mut kernel_total_ms,
                &mut d2h_ms,
                &mut wall_ms,
            )
        };

    if (
        error !=
        0
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "CUDA max-affine tree benchmark failed with CUDA error code {error}.",
                ),
            ),
        );
    }

    let mut normal_survival_mismatches =
        0_u32;

    let mut maxpv_survival_mismatches =
        0_u32;

    let mut unexpected_gpu_unsupported_checks =
        0_u32;

    for job in
        0..
        job_count
    {
        let normal_index =
            job *
            2;

        let maxpv_index =
            normal_index +
            1;

        let expected_normal =
            expected_survival[
                normal_index
            ];

        let expected_maxpv =
            expected_survival[
                maxpv_index
            ];

        let gpu_normal =
            gpu_survival[
                normal_index
            ];

        let gpu_maxpv =
            gpu_survival[
                maxpv_index
            ];

        if (
            expected_normal != 2 &&
            gpu_normal == 2
        ) {
            unexpected_gpu_unsupported_checks +=
                1;
        }

        if (
            expected_maxpv != 2 &&
            gpu_maxpv == 2
        ) {
            unexpected_gpu_unsupported_checks +=
                1;
        }

        if (
            expected_normal != 2 &&
            gpu_normal !=
                expected_normal
        ) {
            normal_survival_mismatches +=
                1;
        }

        if (
            expected_maxpv != 2 &&
            gpu_maxpv !=
                expected_maxpv
        ) {
            maxpv_survival_mismatches +=
                1;
        }
    }

    let candidate_kernel_ms =
        kernel_total_ms /
        repetitions
            as f64;

    let supported_check_count =
        supported_jobs
            as f64 *
        2.0;

    let candidate_ns_per_survival_check =
        if (
            supported_check_count >
            0.0
        ) {
            candidate_kernel_ms *
            1_000_000.0 /
            supported_check_count
        } else {
            0.0
        };

    Ok(
        BlackwellMaxAffineCudaTreeResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions,

            sample_step_count:
                step_count
                    as u32,

            sample_job_count:
                job_count
                    as u32,

            survival_check_count:
                check_count
                    as u32,

            supported_jobs,

            unsupported_jobs:
                job_count
                    as u32 -
                supported_jobs,

            reference_full_causal_gpu_ms,

            candidate_h2d_ms:
                h2d_ms,

            candidate_kernel_total_ms:
                kernel_total_ms,

            candidate_kernel_ms,

            candidate_d2h_ms:
                d2h_ms,

            candidate_wall_ms:
                wall_ms,

            candidate_ns_per_survival_check,

            normal_survival_mismatches,

            maxpv_survival_mismatches,

            unexpected_gpu_unsupported_checks,
        },
    )
}
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_END


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
