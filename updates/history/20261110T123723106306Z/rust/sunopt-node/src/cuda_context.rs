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


// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FFI_BEGIN
    fn sunopt_cuda_run_piecewise_affine_b3_benchmark(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        host_energy_cut_levels: *const f64,
        step_count: usize,
        max_correction_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_iterations: *mut u32,
        host_evaluations: *mut u32,
        host_bracket_width: *mut f64,
        out_upload_ms: *mut f64,
        out_kernel_total_ms: *mut f64,
        out_download_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_BEGIN
    fn sunopt_cuda_run_digital_hp_breakpoints(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_kernel_total_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FFI_BEGIN
    fn sunopt_cuda_run_digital_lockstep_candidate(
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

        step_count:
            usize,

        search_iterations:
            u32,

        repetitions:
            u32,

        host_cut_results:
            *mut f64,

        host_status:
            *mut u32,

        host_survival_evaluations:
            *mut u32,

        out_static_h2d_ms:
            *mut f64,

        out_gpu_total_ms:
            *mut f64,

        out_host_wait_ms:
            *mut f64,

        out_wall_ms:
            *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FFI_END


    // SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FFI_BEGIN
    fn sunopt_cuda_run_digital_lockstep_graph_candidate(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_graph_build_ms: *mut f64,
        out_gpu_total_ms: *mut f64,
        out_host_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FFI_BEGIN
    fn sunopt_cuda_run_digital_lockstep_presorted_candidate(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_presorted_h2d_ms: *mut f64,
        out_gpu_total_ms: *mut f64,
        out_host_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_FFI_BEGIN
    fn sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_presorted_h2d_ms: *mut f64,
        out_gpu_total_ms: *mut f64,
        out_host_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FFI_BEGIN
    fn sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_static_h2d_ms: *mut f64,
        out_presorted_h2d_ms: *mut f64,
        out_gpu_total_ms: *mut f64,
        out_host_wait_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FFI_BEGIN
    fn sunopt_cuda_run_digital_resident_chunk_replay(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        chunk_steps: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_input_h2d_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_d2h_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FFI_BEGIN
    fn sunopt_cuda_run_digital_resident_job_batch(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        search_iterations: u32,
        jobs_per_launch: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_input_h2d_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_d2h_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_FFI_BEGIN
    fn sunopt_cuda_run_digital_multilane_lockstep(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        lane_count: u32,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_input_h2d_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_d2h_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_FFI_END


// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FFI_BEGIN
    fn sunopt_cuda_run_digital_cluster4_lockstep(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_presorted_export_windows: *const f64,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        step_count: usize,
        repetitions: u32,
        host_cut_results: *mut f64,
        host_status: *mut u32,
        host_survival_evaluations: *mut u32,
        out_input_h2d_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_d2h_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FFI_END


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


// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellPiecewiseAffineB3Result {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub annual_step_count: u32,
    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub supported_jobs: u32,
    pub unsupported_jobs: u32,
    pub correction_search_jobs: u32,
    pub reference_full_causal_gpu_ms: f64,

    pub iteration_limits: Vec<u32>,
    pub upload_ms: Vec<f64>,
    pub kernel_ms: Vec<f64>,
    pub download_ms: Vec<f64>,
    pub wall_ms: Vec<f64>,
    pub annual_projected_kernel_ms: Vec<f64>,

    pub status_mismatches: Vec<u32>,
    pub max_abs_cut_error: Vec<f64>,
    pub mean_abs_cut_error: Vec<f64>,
    pub cut_error_over_1e6: Vec<u32>,
    pub cut_error_over_1e5: Vec<u32>,
    pub cut_error_over_1e4: Vec<u32>,

    pub average_correction_iterations: Vec<f64>,
    pub average_evaluations_supported: Vec<f64>,
    pub maximum_bracket_width: Vec<f64>,
}
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_RESULT_END



// SUNOPT_BLACKWELL_B14_EXACT_ENVELOPE_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellB14ExactEnvelopeResult {
    pub snapshot_path: String,
    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub supported_jobs: u32,
    pub unsupported_jobs: u32,
    pub correction_search_jobs: u32,
    pub elapsed_ms: f64,
    pub status_mismatches: u32,
    pub cut_mismatches_1e12: u32,
    pub cut_mismatches_1e9: u32,
    pub cut_mismatches_1e6: u32,
    pub max_abs_cut_error: f64,
    pub mean_abs_cut_error: f64,
    pub min_segments: u32,
    pub median_segments: f64,
    pub p95_segments: u32,
    pub max_segments: u32,
    pub mean_segments: f64,
    pub min_hp_breakpoints: u32,
    pub median_hp_breakpoints: f64,
    pub p95_hp_breakpoints: u32,
    pub max_hp_breakpoints: u32,
    pub mean_hp_breakpoints: f64,
    pub extra_segments_total: u32,
}
// SUNOPT_BLACKWELL_B14_EXACT_ENVELOPE_RESULT_END

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

    pub normal_survives_jobs: u32,
    pub normal_fails_jobs: u32,
    pub maxpv_fails_jobs: u32,
    pub correction_search_jobs: u32,

    pub projected_survival_check_count: u32,
    pub projected_tree_kernel_ms: f64,

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


// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_BEGIN
#[derive(Clone, Copy, Default)]
struct DigitalResearchResult {
    cut_level_power: f64,
    normal_plan_survives: bool,
    maximum_pv_plan_survives: bool,
}

fn digital_hp_charge(
    export_power: f64,
    cut_level_power: f64,
    tank: ResearchTankParameters,
) -> f64 {
    if tank.maximum_heat_pump_electrical_power > 0.0 &&
        export_power - cut_level_power >=
            tank.maximum_heat_pump_electrical_power
    {
        tank.maximum_heat_pump_electrical_power *
        tank.cop *
        tank.time_step_hours
    } else {
        0.0
    }
}

fn digital_survives_scalar(
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
            digital_hp_charge(
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
                (1.0 - tank.heat_loss_retention) *
                stored +
                tank.heat_loss_intercept
            )
            .max(0.0);

        let heat_loss =
            requested_loss.min(stored);

        let after_loss =
            (stored - heat_loss).max(0.0);

        let demand =
            demand_window[offset].max(0.0);

        let discharged =
            demand.min(after_loss);

        let before_charge =
            after_loss -
            discharged;

        let unmet =
            (demand - discharged).max(0.0);

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

fn digital_normal_cut(
    export_window: &[f64],
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> f64 {
    let pmax =
        tank.maximum_heat_pump_electrical_power;

    if pmax <= 0.0 ||
        tank.time_step_hours <= 0.0
    {
        return 0.0;
    }

    let mut breakpoints =
        export_window
            .iter()
            .copied()
            .filter(|&export| export >= pmax)
            .map(|export| export - pmax)
            .collect::<Vec<_>>();

    breakpoints.sort_by(
        |a, b| {
            a.partial_cmp(b)
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
        },
    );

    let count =
        breakpoints.len();

    if count == 0 {
        return pmax;
    }

    let per_bucket =
        pmax *
        tank.time_step_hours;

    let maximum_schedulable =
        count as f64 *
        per_bucket;

    let target =
        requested_electrical_energy
            .max(0.0)
            .min(maximum_schedulable);

    if target <= 0.0 {
        return breakpoints[count - 1] + pmax;
    }

    if target >= maximum_schedulable - 1.0e-12 {
        return 0.0;
    }

    let required_on =
        ((target / per_bucket - 1.0e-12).ceil() as usize)
            .max(1)
            .min(count);

    breakpoints[
        count -
        required_on
    ]
}

fn digital_run_scalar(
    export_window: &[f64],
    demand_window: &[f64],
    current_storage_energy: f64,
    minimum_reserve_energy: f64,
    requested_electrical_energy: f64,
    tank: ResearchTankParameters,
) -> DigitalResearchResult {
    let normal_cut =
        digital_normal_cut(
            export_window,
            requested_electrical_energy,
            tank,
        );

    if digital_survives_scalar(
        export_window,
        demand_window,
        normal_cut,
        current_storage_energy,
        minimum_reserve_energy,
        tank,
    ) {
        return DigitalResearchResult {
            cut_level_power:
                normal_cut,
            normal_plan_survives:
                true,
            maximum_pv_plan_survives:
                true,
        };
    }

    if !digital_survives_scalar(
        export_window,
        demand_window,
        0.0,
        current_storage_energy,
        minimum_reserve_energy,
        tank,
    ) {
        return DigitalResearchResult {
            cut_level_power:
                0.0,
            normal_plan_survives:
                false,
            maximum_pv_plan_survives:
                false,
        };
    }

    let pmax =
        tank.maximum_heat_pump_electrical_power;

    let mut candidates =
        export_window
            .iter()
            .copied()
            .filter(|&export| export >= pmax)
            .map(|export| export - pmax)
            .collect::<Vec<_>>();

    candidates.sort_by(
        |a, b| {
            a.partial_cmp(b)
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
        },
    );

    let mut left =
        0usize;

    let mut right =
        candidates.len();

    let mut best =
        0.0_f64;

    while left < right {
        let middle =
            left +
            (right - left) /
            2;

        let cut =
            candidates[middle].max(0.0);

        if digital_survives_scalar(
            export_window,
            demand_window,
            cut,
            current_storage_energy,
            minimum_reserve_energy,
            tank,
        ) {
            best =
                cut;
            left =
                middle +
                1;
        } else {
            right =
                middle;
        }
    }

    DigitalResearchResult {
        cut_level_power:
            best,
        normal_plan_survives:
            false,
        maximum_pv_plan_survives:
            true,
    }
}

#[napi(object)]
pub struct BlackwellDigitalHpBreakpointVariant {
    pub search_iterations: u32,
    pub kernel_ms: f64,
    pub wall_ms: f64,
    pub annual_projected_kernel_ms: f64,

    pub supported_jobs: u32,
    pub unsupported_jobs: u32,

    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,

    pub average_survival_evaluations: f64,
    pub maximum_survival_evaluations: u32,
}

#[napi(object)]
pub struct BlackwellDigitalHpBreakpointResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub annual_step_count: u32,
    pub sample_step_count: u32,
    pub sample_job_count: u32,
    pub supported_jobs: u32,
    pub unsupported_jobs: u32,
    pub scalar_reference_ms: f64,
    pub variants: Vec<BlackwellDigitalHpBreakpointVariant>,
}
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalLockstepCandidateResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub gpu_compressed_jobs: u32,
    pub gpu_serial_fallback_jobs: u32,
    pub unexpected_unsupported_jobs: u32,
    pub static_h2d_ms: f64,
    pub direct_gpu_total_ms: f64,
    pub direct_host_wait_ms: f64,
    pub direct_wall_ms: f64,
    pub candidate_causal_ms: f64,
    pub candidate_average_us_per_job: f64,
    pub parity_sample_steps: u32,
    pub parity_status_mismatches: u32,
    pub parity_cut_mismatches: u32,
    pub parity_max_cut_error: f64,
    pub average_survival_evaluations: f64,
    pub maximum_survival_evaluations: u32,
    pub checksum_hex: String,
}
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalGraphAbResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub gpu_compressed_jobs: u32,
    pub gpu_serial_fallback_jobs: u32,

    pub direct_static_h2d_ms: f64,
    pub direct_gpu_total_ms: f64,
    pub direct_host_wait_ms: f64,
    pub direct_wall_ms: f64,

    pub graph_static_h2d_ms: f64,
    pub graph_build_ms: f64,
    pub graph_gpu_total_ms: f64,
    pub graph_host_wait_ms: f64,
    pub graph_wall_ms: f64,

    pub wall_saving_ms: f64,
    pub wall_speedup: f64,
    pub wall_reduction_percent: f64,

    pub direct_graph_status_mismatches: u32,
    pub direct_graph_cut_mismatches: u32,
    pub direct_graph_max_cut_error: f64,

    pub parity_sample_steps: u32,
    pub parity_status_mismatches: u32,
    pub parity_cut_mismatches: u32,
    pub parity_max_cut_error: f64,

    pub checksum_hex: String,
}
// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalPresortAbResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub presorted_values: u32,
    pub presorted_mib: f64,
    pub presort_cpu_ms: f64,
    pub direct_gpu_total_ms: f64,
    pub direct_wall_ms: f64,
    pub presorted_window_h2d_ms: f64,
    pub presorted_gpu_total_ms: f64,
    pub presorted_wall_ms: f64,
    pub wall_saving_ms: f64,
    pub wall_speedup: f64,
    pub wall_reduction_percent: f64,
    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,
    pub checksum_hex: String,
}
// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalWarpTreeAbResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub presorted_mib: f64,
    pub presort_cpu_ms: f64,
    pub block_tree_gpu_total_ms: f64,
    pub block_tree_wall_ms: f64,
    pub warp_tree_gpu_total_ms: f64,
    pub warp_tree_wall_ms: f64,
    pub wall_saving_ms: f64,
    pub wall_speedup: f64,
    pub wall_reduction_percent: f64,
    pub block_warp_status_mismatches: u32,
    pub block_warp_cut_mismatches: u32,
    pub block_warp_max_cut_error: f64,
    pub evaluation_mismatches: u32,
    pub parity_sample_steps: u32,
    pub parity_status_mismatches: u32,
    pub parity_cut_mismatches: u32,
    pub parity_max_cut_error: f64,
    pub checksum_hex: String,
}
// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalPiecewiseZeroAbResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,

    pub legacy_compressed_jobs: u32,
    pub legacy_serial_fallback_jobs: u32,
    pub piecewise_compressed_jobs: u32,
    pub piecewise_serial_fallback_jobs: u32,

    pub legacy_gpu_total_ms: f64,
    pub legacy_wall_ms: f64,

    pub piecewise_gpu_total_ms: f64,
    pub piecewise_wall_ms: f64,

    pub wall_saving_ms: f64,
    pub wall_speedup: f64,
    pub wall_reduction_percent: f64,

    pub legacy_piecewise_status_mismatches: u32,
    pub legacy_piecewise_cut_mismatches: u32,
    pub legacy_piecewise_max_cut_error: f64,
    pub evaluation_mismatches: u32,

    pub parity_sample_steps: u32,
    pub parity_status_mismatches: u32,
    pub parity_cut_mismatches: u32,
    pub parity_max_cut_error: f64,

    pub checksum_hex: String,
}
// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalResidentChunkVariant {
    pub chunk_steps: u32,
    pub launches_per_year: u32,
    pub input_h2d_ms: f64,
    pub kernel_ms: f64,
    pub result_d2h_ms: f64,
    pub wall_ms: f64,
    pub speedup_vs_lockstep: f64,
    pub reduction_vs_lockstep_percent: f64,
    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,
    pub evaluation_mismatches: u32,
    pub checksum_hex: String,
}

#[napi(object)]
pub struct BlackwellDigitalResidentChunkResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub presorted_mib: f64,
    pub lockstep_gpu_total_ms: f64,
    pub lockstep_wall_ms: f64,
    pub lockstep_checksum_hex: String,
    pub variants: Vec<BlackwellDigitalResidentChunkVariant>,
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalResidentJobBatchVariant {
    pub jobs_per_launch: u32,
    pub launches_per_year: u32,
    pub input_h2d_ms: f64,
    pub kernel_ms: f64,
    pub result_d2h_ms: f64,
    pub wall_ms: f64,
    pub speedup_vs_chunk_28: f64,
    pub speedup_vs_lockstep: f64,
    pub reduction_vs_lockstep_percent: f64,
    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,
    pub evaluation_mismatches: u32,
    pub checksum_hex: String,
}


#[napi(object)]
pub struct BlackwellDigitalResidentJobBatchResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub search_iterations: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub presorted_mib: f64,
    pub lockstep_wall_ms: f64,
    pub chunk_28_wall_ms: f64,
    pub reference_checksum_hex: String,
    pub variants: Vec<BlackwellDigitalResidentJobBatchVariant>,
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalMultilaneVariant {
    pub lane_count: u32,
    pub threads_per_block: u32,
    pub input_h2d_ms: f64,
    pub kernel_ms: f64,
    pub result_d2h_ms: f64,
    pub wall_ms: f64,
    pub speedup_vs_binary: f64,
    pub reduction_vs_binary_percent: f64,
    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,
    pub average_survival_evaluations: f64,
    pub maximum_survival_evaluations: u32,
    pub checksum_hex: String,
}


#[napi(object)]
pub struct BlackwellDigitalMultilaneResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub presorted_mib: f64,
    pub binary_kernel_ms: f64,
    pub binary_wall_ms: f64,
    pub binary_checksum_hex: String,
    pub variants: Vec<BlackwellDigitalMultilaneVariant>,
}
// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_RESULT_END


// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellDigitalCluster4Result {
    pub snapshot_path: String,
    pub device_id: i32,
    pub repetitions: u32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,

    pub binary_kernel_ms: f64,
    pub binary_wall_ms: f64,

    pub cluster_blocks_per_tank: u32,
    pub cluster_threads_per_block: u32,
    pub cluster_total_blocks_per_step: u32,

    pub cluster_input_h2d_ms: f64,
    pub cluster_kernel_ms: f64,
    pub cluster_result_d2h_ms: f64,
    pub cluster_wall_ms: f64,

    pub speedup_vs_binary: f64,
    pub reduction_vs_binary_percent: f64,

    pub status_mismatches: u32,
    pub cut_mismatches: u32,
    pub maximum_cut_error: f64,
    pub evaluation_mismatches: u32,

    pub binary_checksum_hex: String,
    pub cluster_checksum_hex: String,
}
// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_RESULT_END


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


// SUNOPT_BLACKWELL_B14_EXACT_ENVELOPE_HELPERS_BEGIN
#[derive(Clone, Copy, Debug)]
struct B14Segment { lo: f64, hi: f64, m: f64, b: f64 }

fn b14_eval(segments: &[B14Segment], x: f64) -> f64 {
    let mut s = segments.last().copied().unwrap();
    for &candidate in segments {
        if x <= candidate.hi + 1.0e-12 { s = candidate; break; }
    }
    s.m * x + s.b
}

fn b14_merge(mut v: Vec<B14Segment>) -> Vec<B14Segment> {
    v.retain(|s| s.hi > s.lo + 1.0e-13);
    let mut out: Vec<B14Segment> = Vec::with_capacity(v.len());
    for s in v {
        if let Some(last) = out.last_mut() {
            let scale = 1.0 + last.m.abs().max(s.m.abs()).max(last.b.abs()).max(s.b.abs());
            if (last.hi-s.lo).abs() <= 1.0e-11 &&
               (last.m-s.m).abs() <= 1.0e-11*scale &&
               (last.b-s.b).abs() <= 1.0e-10*scale {
                last.hi=s.hi; continue;
            }
        }
        out.push(s);
    }
    out
}

fn b14_clip_constant(input: &[B14Segment], c: f64, take_min: bool) -> Vec<B14Segment> {
    let mut out=Vec::new();
    for &s in input {
        let y0=s.m*s.lo+s.b; let y1=s.m*s.hi+s.b;
        let d0=y0-c; let d1=y1-c;
        if s.m.abs() < 1.0e-15 || d0*d1 >= 0.0 {
            let affine = if take_min { y0 <= c && y1 <= c } else { y0 >= c && y1 >= c };
            out.push(if affine {s} else {B14Segment{lo:s.lo,hi:s.hi,m:0.0,b:c}});
        } else {
            let x=((c-s.b)/s.m).max(s.lo).min(s.hi);
            let left_affine = if take_min { y0 <= c } else { y0 >= c };
            if x>s.lo+1e-13 { out.push(if left_affine {B14Segment{lo:s.lo,hi:x,..s}} else {B14Segment{lo:s.lo,hi:x,m:0.0,b:c}}); }
            if s.hi>x+1e-13 { out.push(if left_affine {B14Segment{lo:x,hi:s.hi,m:0.0,b:c}} else {B14Segment{lo:x,hi:s.hi,..s}}); }
        }
    }
    b14_merge(out)
}

fn b14_add(a:&[B14Segment], b:&[B14Segment], hi:f64) -> Vec<B14Segment> {
    let mut xs=vec![0.0,hi];
    for s in a { if s.lo>0.0 && s.lo<hi {xs.push(s.lo)} if s.hi>0.0&&s.hi<hi{xs.push(s.hi)} }
    for s in b { if s.lo>0.0 && s.lo<hi {xs.push(s.lo)} if s.hi>0.0&&s.hi<hi{xs.push(s.hi)} }
    xs.sort_by(|x,y|x.partial_cmp(y).unwrap()); xs.dedup_by(|x,y|(*x-*y).abs()<1e-12);
    let mut out=Vec::new();
    for w in xs.windows(2) { let lo=w[0];let h=w[1]; if h<=lo+1e-13{continue} let mid=(lo+h)*0.5;
        let sa=a.iter().find(|s|mid>=s.lo-1e-12&&mid<=s.hi+1e-12).unwrap();
        let sb=b.iter().find(|s|mid>=s.lo-1e-12&&mid<=s.hi+1e-12).unwrap();
        out.push(B14Segment{lo,hi:h,m:sa.m+sb.m,b:sa.b+sb.b});
    }
    b14_merge(out)
}

fn b14_charge(export:f64, hi:f64, tank:ResearchTankParameters)->Vec<B14Segment>{
    let factor=tank.cop*tank.time_step_hours; let p=tank.maximum_heat_pump_electrical_power;
    let mut xs=vec![0.0,hi]; for x in [export-p,export] { if x>0.0&&x<hi{xs.push(x)} }
    xs.sort_by(|a,b|a.partial_cmp(b).unwrap()); xs.dedup_by(|a,b|(*a-*b).abs()<1e-12);
    let mut out=Vec::new(); for w in xs.windows(2){let lo=w[0];let h=w[1];let mid=(lo+h)*0.5;let raw=export-mid;
      if raw<=0.0 {out.push(B14Segment{lo,hi:h,m:0.0,b:0.0})}
      else if raw>=p {out.push(B14Segment{lo,hi:h,m:0.0,b:p*factor})}
      else {out.push(B14Segment{lo,hi:h,m:-factor,b:export*factor})}}
    b14_merge(out)
}

fn b14_truncate_feasible(input:&[B14Segment], threshold:f64, current_hi:f64)->f64{
    let mut hi=current_hi;
    for &s in input {
        if s.lo>=current_hi {break} let sh=s.hi.min(current_hi); let ylo=s.m*s.lo+s.b; let yhi=s.m*sh+s.b;
        if ylo+1e-12 < threshold { return s.lo.min(hi); }
        if yhi+1e-12 < threshold && s.m < 0.0 { hi=hi.min(((threshold-s.b)/s.m).max(s.lo).min(sh)); return hi; }
    }
    hi
}

fn b14_exact_cut(
 export:&[f64], demand:&[f64], current:f64, reserve_input:f64, requested:f64, tank:ResearchTankParameters
)->Option<(ResearchCausalResult,usize,usize)> {
    const EPS:f64=1e-6;
    if !tank.affine_supported || tank.heat_loss_retention<=0.0 || export.is_empty() || export.len()!=demand.len(){return None}
    let reserve=reserve_input.max(0.0).min(tank.target_storage_energy); if reserve<=EPS{return None}
    let energy_cut=research_energy_cut_level(export,requested,tank);
    let hp_bp=export.iter().flat_map(|&e|[e-tank.maximum_heat_pump_electrical_power,e]).filter(|&x|x>0.0&&x<energy_cut).count();
    if research_scalar_survives(export,demand,energy_cut,current,reserve_input,tank){return Some((ResearchCausalResult{cut_level_power:energy_cut,normal_plan_survives:true,maximum_pv_plan_survives:true},1,hp_bp))}
    if !research_scalar_survives(export,demand,0.0,current,reserve_input,tank){return Some((ResearchCausalResult{cut_level_power:0.0,normal_plan_survives:false,maximum_pv_plan_survives:false},1,hp_bp))}
    let mut feasible_hi=energy_cut;
    let q0=b14_charge(export[0],energy_cut,tank);
    let base=vec![B14Segment{lo:0.0,hi:energy_cut,m:0.0,b:current.max(0.0).min(tank.target_storage_energy)}];
    let mut stored=b14_clip_constant(&b14_add(&base,&q0,energy_cut),tank.target_storage_energy,true);
    let mut max_segments=stored.len();
    for i in 1..export.len(){
        // On the feasible branch reserve>EPS guarantees positive post-demand energy.
        // Thus canonical loss->demand reduces exactly to retention*stored-intercept-demand.
        let mut before:Vec<B14Segment>=stored.iter().map(|s|B14Segment{lo:s.lo,hi:s.hi,m:tank.heat_loss_retention*s.m,b:tank.heat_loss_retention*s.b-tank.heat_loss_intercept-demand[i].max(0.0)}).collect();
        before=b14_merge(before);
        feasible_hi=b14_truncate_feasible(&before,reserve-EPS,feasible_hi);
        if feasible_hi<=0.0 {break}
        let q=b14_charge(export[i],energy_cut,tank);
        stored=b14_clip_constant(&b14_add(&before,&q,energy_cut),tank.target_storage_energy,true);
        max_segments=max_segments.max(stored.len()).max(before.len());
    }
    let mathematical=feasible_hi.max(0.0).min(energy_cut);
    // Production intentionally returns the lower endpoint after exactly 20 bisections.
    let mut lo=0.0; let mut hi=energy_cut;
    for _ in 0..20 { let mid=(lo+hi)*0.5; if mid<=mathematical+1e-12 {lo=mid}else{hi=mid} }
    Some((ResearchCausalResult{cut_level_power:lo,normal_plan_survives:false,maximum_pv_plan_survives:true},max_segments,hp_bp))
}
// SUNOPT_BLACKWELL_B14_EXACT_ENVELOPE_HELPERS_END

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


// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FN_BEGIN
#[napi]
pub fn run_blackwell_piecewise_affine_b3(
    snapshot_path: Option<String>,
    sample_steps: Option<u32>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellPiecewiseAffineB3Result> {
    let path = snapshot_path
        .map(PathBuf::from)
        .unwrap_or_else(blackwell_default_snapshot_path);

    let snapshot = read_blackwell_snapshot(&path).map_err(|error| {
        Error::from_reason(format!(
            "Unable to read Blackwell benchmark snapshot {}: {error}",
            path.display(),
        ))
    })?;

    if snapshot.steps.is_empty() || snapshot.tank_count == 0 ||
        snapshot.export_power.is_empty() ||
        snapshot.demand_energy.len() != snapshot.export_power.len()
    {
        return Err(Error::from_reason("Blackwell snapshot is empty or inconsistent."));
    }

    let requested_samples = sample_steps
        .unwrap_or(192)
        .clamp(2, snapshot.steps.len() as u32) as usize;

    let mut sample_positions = Vec::<usize>::with_capacity(requested_samples);
    for sample in 0..requested_samples {
        let position = sample * (snapshot.steps.len() - 1) / (requested_samples - 1);
        if sample_positions.last().copied() != Some(position) {
            sample_positions.push(position);
        }
    }

    let selected_device = device_id.unwrap_or(0);
    let repetitions = repetitions.unwrap_or(20).clamp(1, 10_000);

    let tank_parameters = (0..snapshot.tank_count)
        .map(|tank| research_tank_from_packed(&snapshot.packed_config, tank))
        .collect::<Vec<_>>();

    let mut packed_tanks = Vec::<f64>::with_capacity(snapshot.tank_count * 6);
    for tank in &tank_parameters {
        packed_tanks.push(tank.target_storage_energy);
        packed_tanks.push(tank.maximum_heat_pump_electrical_power);
        packed_tanks.push(tank.cop * tank.time_step_hours);
        packed_tanks.push(tank.heat_loss_retention);
        packed_tanks.push(tank.heat_loss_intercept);
        packed_tanks.push(if tank.affine_supported { 1.0 } else { 0.0 });
    }

    let step_count = sample_positions.len();
    let job_count = step_count * snapshot.tank_count;
    let mut current_indices = Vec::<u32>::with_capacity(step_count);
    let mut horizon_steps = Vec::<u32>::with_capacity(step_count);
    let mut dynamic_values = Vec::<f64>::with_capacity(job_count * 2);
    let mut energy_cut_levels = Vec::<f64>::with_capacity(job_count);
    let mut reference_results = Vec::<ResearchCausalResult>::with_capacity(job_count);
    let mut supported_mask = Vec::<bool>::with_capacity(job_count);

    let mut supported_jobs = 0_u32;
    let mut correction_search_jobs = 0_u32;

    for &position in &sample_positions {
        let step = &snapshot.steps[position];
        let horizon = step.horizon_steps as usize;
        if horizon == 0 || horizon > 256 {
            return Err(Error::from_reason(format!(
                "B3 benchmark supports horizon 1..=256; snapshot contains {horizon}."
            )));
        }

        current_indices.push(step.current_index);
        horizon_steps.push(step.horizon_steps);

        let export_window = research_window(
            &snapshot.export_power,
            step.current_index as usize,
            horizon,
        );
        let demand_window = research_window(
            &snapshot.demand_energy,
            step.current_index as usize,
            horizon,
        );

        for tank_index in 0..snapshot.tank_count {
            let base = tank_index * BLACKWELL_DYNAMIC_STRIDE;
            let current = step.dynamic[base + 0];
            let reserve_input = step.dynamic[base + 1];
            let requested = step.dynamic[base + 2];
            let tank = tank_parameters[tank_index];

            dynamic_values.push(current);
            dynamic_values.push(reserve_input);

            let energy_cut = research_energy_cut_level(&export_window, requested, tank);
            energy_cut_levels.push(energy_cut);

            let reference = research_run_scalar(
                &export_window,
                &demand_window,
                current,
                reserve_input,
                requested,
                tank,
            );

            let reserve = reserve_input.max(0.0).min(tank.target_storage_energy);
            let supported = tank.affine_supported &&
                tank.heat_loss_retention > 0.0 &&
                reserve > 1.0e-6;

            if supported {
                supported_jobs += 1;
                if !reference.normal_plan_survives && reference.maximum_pv_plan_survives {
                    correction_search_jobs += 1;
                }
            }

            supported_mask.push(supported);
            reference_results.push(reference);
        }
    }

    /* Existing complete causal GPU reference on the same sampled steps. */
    let cuda_parameters = blackwell_parameters_from_packed(
        &snapshot.packed_config,
        snapshot.data_length,
    )?;
    let mut reference_context = CudaSimulationContext::new(selected_device)?;
    reference_context.upload_causal_arrays(&snapshot.export_power, &snapshot.demand_energy)?;
    reference_context.configure_causal_static_parameters(&cuda_parameters)?;
    let mut reference_dynamic = vec![SunoptCausalDynamic::default(); snapshot.tank_count];
    let mut reference_output = vec![0.0_f64; snapshot.tank_count];

    let reference_started = Instant::now();
    for &position in &sample_positions {
        let step = &snapshot.steps[position];
        blackwell_dynamic_from_values(
            &step.dynamic,
            snapshot.tank_count,
            &mut reference_dynamic,
        )?;
        reference_context.run_causal_dynamic_batch_into(
            step.current_index as usize,
            step.horizon_steps as usize,
            &reference_dynamic,
            &mut reference_output,
        )?;
    }
    let reference_full_causal_gpu_ms = reference_started.elapsed().as_secs_f64() * 1000.0;

    let iteration_limits = vec![1_u32, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32];
    let mut upload_ms = Vec::<f64>::new();
    let mut kernel_ms = Vec::<f64>::new();
    let mut download_ms = Vec::<f64>::new();
    let mut wall_ms = Vec::<f64>::new();
    let mut annual_projected_kernel_ms = Vec::<f64>::new();
    let mut status_mismatches = Vec::<u32>::new();
    let mut max_abs_cut_error = Vec::<f64>::new();
    let mut mean_abs_cut_error = Vec::<f64>::new();
    let mut cut_error_over_1e6 = Vec::<u32>::new();
    let mut cut_error_over_1e5 = Vec::<u32>::new();
    let mut cut_error_over_1e4 = Vec::<u32>::new();
    let mut average_correction_iterations = Vec::<f64>::new();
    let mut average_evaluations_supported = Vec::<f64>::new();
    let mut maximum_bracket_width = Vec::<f64>::new();

    for &limit in &iteration_limits {
        let mut cuts = vec![0.0_f64; job_count];
        let mut status = vec![0_u32; job_count];
        let mut iterations = vec![0_u32; job_count];
        let mut evaluations = vec![0_u32; job_count];
        let mut widths = vec![0.0_f64; job_count];
        let mut this_upload = 0.0_f64;
        let mut kernel_total = 0.0_f64;
        let mut this_download = 0.0_f64;
        let mut this_wall = 0.0_f64;

        let error = unsafe {
            sunopt_cuda_run_piecewise_affine_b3_benchmark(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                snapshot.tank_count,
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_values.as_ptr(),
                energy_cut_levels.as_ptr(),
                step_count,
                limit,
                repetitions,
                cuts.as_mut_ptr(),
                status.as_mut_ptr(),
                iterations.as_mut_ptr(),
                evaluations.as_mut_ptr(),
                widths.as_mut_ptr(),
                &mut this_upload,
                &mut kernel_total,
                &mut this_download,
                &mut this_wall,
            )
        };

        if error != 0 {
            return Err(Error::from_reason(format!(
                "CUDA B3 piecewise-affine benchmark failed with CUDA error code {error}."
            )));
        }

        let this_kernel = kernel_total / repetitions as f64;
        let mut mismatch_count = 0_u32;
        let mut max_error = 0.0_f64;
        let mut sum_error = 0.0_f64;
        let mut count_error = 0_u32;
        let mut over_1e6 = 0_u32;
        let mut over_1e5 = 0_u32;
        let mut over_1e4 = 0_u32;
        let mut correction_iteration_sum = 0_u64;
        let mut correction_count = 0_u32;
        let mut evaluation_sum = 0_u64;
        let mut max_width = 0.0_f64;

        for job in 0..job_count {
            if !supported_mask[job] {
                if status[job] != 3 { mismatch_count += 1; }
                continue;
            }

            let reference = reference_results[job];
            let expected_status = if reference.normal_plan_survives {
                0_u32
            } else if !reference.maximum_pv_plan_survives {
                1_u32
            } else {
                2_u32
            };

            if status[job] != expected_status { mismatch_count += 1; }

            let cut_error = (cuts[job] - reference.cut_level_power).abs();
            max_error = max_error.max(cut_error);
            sum_error += cut_error;
            count_error += 1;
            if cut_error > 1.0e-6 { over_1e6 += 1; }
            if cut_error > 1.0e-5 { over_1e5 += 1; }
            if cut_error > 1.0e-4 { over_1e4 += 1; }

            evaluation_sum += evaluations[job] as u64;
            max_width = max_width.max(widths[job]);

            if expected_status == 2 {
                correction_iteration_sum += iterations[job] as u64;
                correction_count += 1;
            }
        }

        upload_ms.push(this_upload);
        kernel_ms.push(this_kernel);
        download_ms.push(this_download);
        wall_ms.push(this_wall);
        annual_projected_kernel_ms.push(
            this_kernel * snapshot.steps.len() as f64 / step_count as f64
        );
        status_mismatches.push(mismatch_count);
        max_abs_cut_error.push(max_error);
        mean_abs_cut_error.push(if count_error > 0 { sum_error / count_error as f64 } else { 0.0 });
        cut_error_over_1e6.push(over_1e6);
        cut_error_over_1e5.push(over_1e5);
        cut_error_over_1e4.push(over_1e4);
        average_correction_iterations.push(
            if correction_count > 0 {
                correction_iteration_sum as f64 / correction_count as f64
            } else { 0.0 }
        );
        average_evaluations_supported.push(
            if supported_jobs > 0 {
                evaluation_sum as f64 / supported_jobs as f64
            } else { 0.0 }
        );
        maximum_bracket_width.push(max_width);
    }

    Ok(BlackwellPiecewiseAffineB3Result {
        snapshot_path: path.display().to_string(),
        device_id: selected_device,
        repetitions,
        annual_step_count: snapshot.steps.len() as u32,
        sample_step_count: step_count as u32,
        sample_job_count: job_count as u32,
        supported_jobs,
        unsupported_jobs: job_count as u32 - supported_jobs,
        correction_search_jobs,
        reference_full_causal_gpu_ms,
        iteration_limits,
        upload_ms,
        kernel_ms,
        download_ms,
        wall_ms,
        annual_projected_kernel_ms,
        status_mismatches,
        max_abs_cut_error,
        mean_abs_cut_error,
        cut_error_over_1e6,
        cut_error_over_1e5,
        cut_error_over_1e4,
        average_correction_iterations,
        average_evaluations_supported,
        maximum_bracket_width,
    })
}
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_FN_END


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

    let mut normal_survives_jobs =
        0_u32;

    let mut normal_fails_jobs =
        0_u32;

    let mut maxpv_fails_jobs =
        0_u32;

    let mut correction_search_jobs =
        0_u32;

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

        if expected_normal != 2 {
            if expected_normal == 1 {
                normal_survives_jobs +=
                    1;
            } else {
                normal_fails_jobs +=
                    1;

                if expected_maxpv == 0 {
                    maxpv_fails_jobs +=
                        1;
                } else {
                    correction_search_jobs +=
                        1;
                }
            }
        }

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

    let projected_survival_check_count =
        supported_jobs +
        normal_fails_jobs +
        correction_search_jobs *
            20;

    let projected_tree_kernel_ms =
        if supported_jobs > 0 {
            candidate_kernel_ms *
            projected_survival_check_count
                as f64 /
            (
                supported_jobs
                    as f64 *
                2.0
            )
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

            normal_survives_jobs,

            normal_fails_jobs,

            maxpv_fails_jobs,

            correction_search_jobs,

            projected_survival_check_count,

            projected_tree_kernel_ms,

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


// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_hp_breakpoints(
    snapshot_path: Option<String>,
    sample_steps: Option<u32>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalHpBreakpointResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(&path)
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

    if snapshot.steps.is_empty() ||
        snapshot.tank_count == 0
    {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty.",
            ),
        );
    }

    let requested_samples =
        sample_steps
            .unwrap_or(192)
            .clamp(
                2,
                snapshot.steps.len() as u32,
            )
            as usize;

    let repetitions =
        repetitions
            .unwrap_or(20)
            .clamp(
                1,
                10_000,
            );

    let selected_device =
        device_id.unwrap_or(0);

    let mut sample_positions =
        Vec::<usize>::with_capacity(
            requested_samples,
        );

    for sample in 0..requested_samples {
        let position =
            sample *
            (snapshot.steps.len() - 1) /
            (requested_samples - 1);

        if sample_positions
            .last()
            .copied() !=
            Some(position)
        {
            sample_positions.push(position);
        }
    }

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count *
            7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            pmax,
        );

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        sample_positions.len();

    let job_count =
        step_count *
        snapshot.tank_count;

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
            3,
        );

    let mut scalar =
        Vec::<DigitalResearchResult>::with_capacity(
            job_count,
        );

    let mut exact_supported =
        Vec::<bool>::with_capacity(
            job_count,
        );

    let scalar_started =
        Instant::now();

    for &position in &sample_positions {
        let step =
            &snapshot.steps[position];

        let horizon =
            step.horizon_steps as usize;

        if horizon == 0 ||
            horizon > 256
        {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital breakpoint benchmark supports horizon 1..=256; got {horizon}.",
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
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let current_storage =
                step.dynamic[base + 0];

            let reserve =
                step.dynamic[base + 1];

            let requested =
                step.dynamic[base + 2];

            let tank =
                tanks[tank_index];

            dynamic_values.push(
                current_storage,
            );

            dynamic_values.push(
                reserve,
            );

            dynamic_values.push(
                requested,
            );

            scalar.push(
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    current_storage,
                    reserve,
                    requested,
                    tank,
                ),
            );

            let clamped_reserve =
                reserve
                    .max(0.0)
                    .min(
                        tank.target_storage_energy,
                    );

            exact_supported.push(
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                clamped_reserve >
                    1.0e-6,
            );
        }
    }

    let scalar_reference_ms =
        scalar_started
            .elapsed()
            .as_secs_f64() *
        1000.0;

    let supported_jobs =
        exact_supported
            .iter()
            .filter(|&&x| x)
            .count()
            as u32;

    let unsupported_jobs =
        job_count as u32 -
        supported_jobs;

    let iteration_limits =
        [
            1_u32,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ];

    let mut variants =
        Vec::<BlackwellDigitalHpBreakpointVariant>::new();

    for search_iterations in iteration_limits {
        let mut cuts =
            vec![
                0.0_f64;
                job_count
            ];

        let mut statuses =
            vec![
                0_u32;
                job_count
            ];

        let mut evaluations =
            vec![
                0_u32;
                job_count
            ];

        let mut kernel_total_ms =
            0.0_f64;

        let mut wall_ms =
            0.0_f64;

        let error =
            unsafe {
                sunopt_cuda_run_digital_hp_breakpoints(
                    selected_device,
                    snapshot.export_power.as_ptr(),
                    snapshot.demand_energy.as_ptr(),
                    snapshot.export_power.len(),
                    packed_tanks.as_ptr(),
                    snapshot.tank_count,
                    current_indices.as_ptr(),
                    horizon_steps.as_ptr(),
                    dynamic_values.as_ptr(),
                    step_count,
                    search_iterations,
                    repetitions,
                    cuts.as_mut_ptr(),
                    statuses.as_mut_ptr(),
                    evaluations.as_mut_ptr(),
                    &mut kernel_total_ms,
                    &mut wall_ms,
                )
            };

        if error != 0 {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital HP CUDA benchmark failed with CUDA error {error} at search_iterations={search_iterations}.",
                    ),
                ),
            );
        }

        let kernel_ms =
            kernel_total_ms /
            repetitions as f64;

        let mut status_mismatches =
            0_u32;

        let mut cut_mismatches =
            0_u32;

        let mut maximum_cut_error =
            0.0_f64;

        let mut evaluation_sum =
            0_u64;

        let mut maximum_survival_evaluations =
            0_u32;

        for job in 0..job_count {
            if !exact_supported[job] {
                continue;
            }

            let reference =
                scalar[job];

            let expected_status =
                if reference.normal_plan_survives {
                    0_u32
                } else if !reference.maximum_pv_plan_survives {
                    1_u32
                } else {
                    2_u32
                };

            if statuses[job] !=
                expected_status
            {
                status_mismatches +=
                    1;
            }

            let cut_error =
                (
                    cuts[job] -
                    reference.cut_level_power
                )
                .abs();

            maximum_cut_error =
                maximum_cut_error.max(
                    cut_error,
                );

            if cut_error >
                1.0e-9
            {
                cut_mismatches +=
                    1;
            }

            evaluation_sum +=
                evaluations[job]
                    as u64;

            maximum_survival_evaluations =
                maximum_survival_evaluations.max(
                    evaluations[job],
                );
        }

        let average_survival_evaluations =
            if supported_jobs >
                0
            {
                evaluation_sum as f64 /
                supported_jobs as f64
            } else {
                0.0
            };

        let annual_projected_kernel_ms =
            kernel_ms *
            snapshot.steps.len() as f64 /
            step_count as f64;

        variants.push(
            BlackwellDigitalHpBreakpointVariant {
                search_iterations,
                kernel_ms,
                wall_ms,
                annual_projected_kernel_ms,
                supported_jobs,
                unsupported_jobs,
                status_mismatches,
                cut_mismatches,
                maximum_cut_error,
                average_survival_evaluations,
                maximum_survival_evaluations,
            },
        );
    }

    Ok(
        BlackwellDigitalHpBreakpointResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions,

            annual_step_count:
                snapshot.steps.len()
                    as u32,

            sample_step_count:
                step_count
                    as u32,

            sample_job_count:
                job_count
                    as u32,

            supported_jobs,

            unsupported_jobs,

            scalar_reference_ms,

            variants,
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FN_END


// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_lockstep_candidate(
    snapshot_path:
        Option<String>,

    repetitions:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<BlackwellDigitalLockstepCandidateResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
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

    if (
        snapshot.steps.len() !=
        snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "Expected full annual lockstep snapshot: {} steps vs data length {}.",
                    snapshot.steps.len(),
                    snapshot.data_length,
                ),
            ),
        );
    }

    let selected_device =
        device_id
            .unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(
                1,
                5,
            );

    let search_iterations =
        8_u32;

    let tanks =
        (
            0..
            snapshot.tank_count
        )
        .map(
            |tank| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank,
                )
            },
        )
        .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count *
            7,
        );

    for tank in
        &tanks
    {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            pmax,
        );

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Digital lockstep candidate job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count *
            3,
        );

    for step in
        &snapshot.steps
    {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital lockstep candidate supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank in
            0..
            tank_count
        {
            let base =
                tank *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(
                step.dynamic[
                    base + 0
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 1
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 2
                ],
            );
        }
    }

    let mut cuts =
        vec![
            0.0_f64;
            job_count
        ];

    let mut statuses =
        vec![
            0_u32;
            job_count
        ];

    let mut evaluations =
        vec![
            0_u32;
            job_count
        ];

    let mut static_h2d_ms =
        0.0_f64;

    let mut gpu_total_ms =
        0.0_f64;

    let mut host_wait_ms =
        0.0_f64;

    let mut wall_ms =
        0.0_f64;

    let code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_candidate(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                tank_count,
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_flat
                    .as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                cuts
                    .as_mut_ptr(),
                statuses
                    .as_mut_ptr(),
                evaluations
                    .as_mut_ptr(),
                &mut static_h2d_ms,
                &mut gpu_total_ms,
                &mut host_wait_ms,
                &mut wall_ms,
            )
        };

    if (
        code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_candidate",
                code,
            ),
        );
    }

    let direct_gpu_total_ms =
        gpu_total_ms /
        repetition_count
            as f64;

    let direct_host_wait_ms =
        host_wait_ms /
        repetition_count
            as f64;

    let direct_wall_ms =
        wall_ms /
        repetition_count
            as f64;

    /*
     * Everything now stays on the GPU. Count which jobs use the compressed
     * max-affine branch and which use the exact serial GPU branch.
     */
    let gpu_statuses =
        statuses.clone();

    let gpu_cuts =
        cuts.clone();

    let mut gpu_serial_fallback_jobs =
        0_u32;

    for step in
        &snapshot.steps
    {
        for tank_index in
            0..
            tank_count
        {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reserve =
                step.dynamic[
                    base + 1
                ]
                .max(
                    0.0,
                )
                .min(
                    tanks[
                        tank_index
                    ]
                    .target_storage_energy,
                );

            let tank =
                tanks[
                    tank_index
                ];

            let max_affine_supported =
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                reserve >
                    1.0e-6;

            if (
                !max_affine_supported
            ) {
                gpu_serial_fallback_jobs +=
                    1;
            }
        }
    }

    let gpu_compressed_jobs =
        job_count
            as u32 -
        gpu_serial_fallback_jobs;

    let unexpected_unsupported_jobs =
        statuses
            .iter()
            .filter(
                |&&value| {
                    value ==
                    3
                },
            )
            .count()
            as u32;

    if (
        unexpected_unsupported_jobs >
        0
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "Digital all-GPU candidate leaked {unexpected_unsupported_jobs} status=3 jobs.",
                ),
            ),
        );
    }

    /*
     * Production-shaped parity sample: verify the direct 28-block lockstep
     * outputs against the scalar ideal-digital reference at 192 evenly spaced
     * annual steps. This includes the serial GPU fallback jobs.
     */
    let parity_sample_steps =
        192_usize.min(
            step_count,
        );

    let mut parity_status_mismatches =
        0_u32;

    let mut parity_cut_mismatches =
        0_u32;

    let mut parity_max_cut_error =
        0.0_f64;

    for sample in
        0..
        parity_sample_steps
    {
        let step_index =
            if (
                parity_sample_steps <=
                1
            ) {
                0
            } else {
                sample *
                (
                    step_count -
                    1
                ) /
                (
                    parity_sample_steps -
                    1
                )
            };

        let step =
            &snapshot.steps[
                step_index
            ];

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index
                    as usize,
                step.horizon_steps
                    as usize,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index
                    as usize,
                step.horizon_steps
                    as usize,
            );

        for tank_index in
            0..
            tank_count
        {
            let job =
                step_index *
                tank_count +
                tank_index;

            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reference =
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    step.dynamic[
                        base + 0
                    ],
                    step.dynamic[
                        base + 1
                    ],
                    step.dynamic[
                        base + 2
                    ],
                    tanks[
                        tank_index
                    ],
                );

            let expected_status =
                if reference
                    .normal_plan_survives
                {
                    0_u32
                } else if !reference
                    .maximum_pv_plan_survives
                {
                    1_u32
                } else {
                    2_u32
                };

            if (
                gpu_statuses[
                    job
                ] !=
                expected_status
            ) {
                parity_status_mismatches +=
                    1;
            }

            let cut_error =
                (
                    gpu_cuts[
                        job
                    ] -
                    reference
                        .cut_level_power
                )
                .abs();

            parity_max_cut_error =
                parity_max_cut_error.max(
                    cut_error,
                );

            if (
                cut_error >
                1.0e-9
            ) {
                parity_cut_mismatches +=
                    1;
            }
        }
    }

    let mut evaluation_sum =
        0_u64;

    let mut maximum_survival_evaluations =
        0_u32;

    for (
        job,
        &value,
    ) in
        evaluations
            .iter()
            .enumerate()
    {
        evaluation_sum +=
            value
                as u64;

        maximum_survival_evaluations =
            maximum_survival_evaluations.max(
                value,
            );
    }

    let average_survival_evaluations =
        if (
            job_count >
            0
        ) {
            evaluation_sum
                as f64 /
            job_count
                as f64
        } else {
            0.0
        };

    /* No CPU causal fallback remains. */
    let candidate_causal_ms =
        direct_wall_ms;

    let candidate_average_us_per_job =
        candidate_causal_ms *
        1000.0 /
        job_count
            as f64;

    let mut checksum =
        1469598103934665603u64;

    for value in
        &cuts
    {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    Ok(
        BlackwellDigitalLockstepCandidateResult {
            snapshot_path:
                path
                    .display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            search_iterations,

            annual_step_count:
                step_count
                    as u32,

            tank_count:
                tank_count
                    as u32,

            causal_jobs:
                job_count
                    as u32,

            gpu_compressed_jobs,

            gpu_serial_fallback_jobs,

            unexpected_unsupported_jobs,

            static_h2d_ms,

            direct_gpu_total_ms,

            direct_host_wait_ms,

            direct_wall_ms,

            candidate_causal_ms,

            candidate_average_us_per_job,

            parity_sample_steps:
                parity_sample_steps
                    as u32,

            parity_status_mismatches,

            parity_cut_mismatches,

            parity_max_cut_error,

            average_survival_evaluations,

            maximum_survival_evaluations,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_END


// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_graph_ab(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalGraphAbResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
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
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a complete annual lockstep snapshot.",
            ),
        );
    }

    let selected_device =
        device_id.unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(1, 3);

    let search_iterations =
        8_u32;

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(pmax);

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(tank_count)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Digital graph A/B job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count * 3,
        );

    let mut gpu_serial_fallback_jobs =
        0_u32;

    for step in &snapshot.steps {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital graph A/B supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank_index in 0..tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let current_storage =
                step.dynamic[base + 0];

            let reserve =
                step.dynamic[base + 1];

            let requested =
                step.dynamic[base + 2];

            dynamic_flat.push(
                current_storage,
            );

            dynamic_flat.push(
                reserve,
            );

            dynamic_flat.push(
                requested,
            );

            let tank =
                tanks[tank_index];

            let clamped_reserve =
                reserve
                    .max(0.0)
                    .min(
                        tank.target_storage_energy,
                    );

            let compressed =
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                clamped_reserve >
                    1.0e-6;

            if !compressed {
                gpu_serial_fallback_jobs += 1;
            }
        }
    }

    let gpu_compressed_jobs =
        job_count as u32 -
        gpu_serial_fallback_jobs;

    let mut direct_cuts =
        vec![0.0_f64; job_count];

    let mut direct_status =
        vec![0_u32; job_count];

    let mut direct_evaluations =
        vec![0_u32; job_count];

    let mut direct_static_h2d_ms = 0.0_f64;
    let mut direct_gpu_total_raw_ms = 0.0_f64;
    let mut direct_host_wait_raw_ms = 0.0_f64;
    let mut direct_wall_raw_ms = 0.0_f64;

    let direct_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                direct_cuts.as_mut_ptr(),
                direct_status.as_mut_ptr(),
                direct_evaluations.as_mut_ptr(),
                &mut direct_static_h2d_ms,
                &mut direct_gpu_total_raw_ms,
                &mut direct_host_wait_raw_ms,
                &mut direct_wall_raw_ms,
            )
        };

    if direct_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_candidate",
                direct_code,
            ),
        );
    }

    let mut graph_cuts =
        vec![0.0_f64; job_count];

    let mut graph_status =
        vec![0_u32; job_count];

    let mut graph_evaluations =
        vec![0_u32; job_count];

    let mut graph_static_h2d_ms = 0.0_f64;
    let mut graph_build_ms = 0.0_f64;
    let mut graph_gpu_total_raw_ms = 0.0_f64;
    let mut graph_host_wait_raw_ms = 0.0_f64;
    let mut graph_wall_raw_ms = 0.0_f64;

    let graph_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_graph_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                graph_cuts.as_mut_ptr(),
                graph_status.as_mut_ptr(),
                graph_evaluations.as_mut_ptr(),
                &mut graph_static_h2d_ms,
                &mut graph_build_ms,
                &mut graph_gpu_total_raw_ms,
                &mut graph_host_wait_raw_ms,
                &mut graph_wall_raw_ms,
            )
        };

    if graph_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_graph_candidate",
                graph_code,
            ),
        );
    }

    let divisor =
        repetition_count as f64;

    let direct_gpu_total_ms =
        direct_gpu_total_raw_ms / divisor;

    let direct_host_wait_ms =
        direct_host_wait_raw_ms / divisor;

    let direct_wall_ms =
        direct_wall_raw_ms / divisor;

    let graph_gpu_total_ms =
        graph_gpu_total_raw_ms / divisor;

    let graph_host_wait_ms =
        graph_host_wait_raw_ms / divisor;

    let graph_wall_ms =
        graph_wall_raw_ms / divisor;

    let wall_saving_ms =
        direct_wall_ms -
        graph_wall_ms;

    let wall_speedup =
        if graph_wall_ms > 0.0 {
            direct_wall_ms /
            graph_wall_ms
        } else {
            f64::INFINITY
        };

    let wall_reduction_percent =
        if direct_wall_ms > 0.0 {
            wall_saving_ms /
            direct_wall_ms *
            100.0
        } else {
            0.0
        };

    let mut direct_graph_status_mismatches =
        0_u32;

    let mut direct_graph_cut_mismatches =
        0_u32;

    let mut direct_graph_max_cut_error =
        0.0_f64;

    for job in 0..job_count {
        if (
            direct_status[job] !=
            graph_status[job]
        ) {
            direct_graph_status_mismatches += 1;
        }

        let error =
            (
                direct_cuts[job] -
                graph_cuts[job]
            )
            .abs();

        direct_graph_max_cut_error =
            direct_graph_max_cut_error.max(
                error,
            );

        if error > 1.0e-9 {
            direct_graph_cut_mismatches += 1;
        }
    }

    let parity_sample_steps =
        192_usize.min(step_count);

    let mut parity_status_mismatches =
        0_u32;

    let mut parity_cut_mismatches =
        0_u32;

    let mut parity_max_cut_error =
        0.0_f64;

    for sample in 0..parity_sample_steps {
        let step_index =
            if parity_sample_steps <= 1 {
                0
            } else {
                sample *
                (step_count - 1) /
                (parity_sample_steps - 1)
            };

        let step =
            &snapshot.steps[step_index];

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        for tank_index in 0..tank_count {
            let job =
                step_index *
                tank_count +
                tank_index;

            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reference =
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    step.dynamic[base + 0],
                    step.dynamic[base + 1],
                    step.dynamic[base + 2],
                    tanks[tank_index],
                );

            let expected_status =
                if reference.normal_plan_survives {
                    0_u32
                } else if !reference.maximum_pv_plan_survives {
                    1_u32
                } else {
                    2_u32
                };

            if (
                graph_status[job] !=
                expected_status
            ) {
                parity_status_mismatches += 1;
            }

            let error =
                (
                    graph_cuts[job] -
                    reference.cut_level_power
                )
                .abs();

            parity_max_cut_error =
                parity_max_cut_error.max(
                    error,
                );

            if error > 1.0e-9 {
                parity_cut_mismatches += 1;
            }
        }
    }

    let mut checksum =
        1469598103934665603u64;

    for value in &graph_cuts {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    Ok(
        BlackwellDigitalGraphAbResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            search_iterations,

            annual_step_count:
                step_count as u32,

            tank_count:
                tank_count as u32,

            causal_jobs:
                job_count as u32,

            gpu_compressed_jobs,

            gpu_serial_fallback_jobs,

            direct_static_h2d_ms,
            direct_gpu_total_ms,
            direct_host_wait_ms,
            direct_wall_ms,

            graph_static_h2d_ms,
            graph_build_ms,
            graph_gpu_total_ms,
            graph_host_wait_ms,
            graph_wall_ms,

            wall_saving_ms,
            wall_speedup,
            wall_reduction_percent,

            direct_graph_status_mismatches,
            direct_graph_cut_mismatches,
            direct_graph_max_cut_error,

            parity_sample_steps:
                parity_sample_steps as u32,

            parity_status_mismatches,
            parity_cut_mismatches,
            parity_max_cut_error,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_FN_END


// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_presort_ab(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalPresortAbResult> {
    let path = snapshot_path.map(PathBuf::from).unwrap_or_else(blackwell_default_snapshot_path);
    let snapshot = read_blackwell_snapshot(&path).map_err(|error| {
        Error::from_reason(format!("Unable to read Blackwell snapshot {}: {error}", path.display()))
    })?;

    if snapshot.steps.is_empty() || snapshot.tank_count == 0 ||
       snapshot.export_power.is_empty() || snapshot.steps.len() != snapshot.data_length {
        return Err(Error::from_reason("Blackwell snapshot is not a complete annual snapshot."));
    }

    let selected_device = device_id.unwrap_or(0);
    let repetition_count = repetitions.unwrap_or(1).clamp(1, 3);
    let search_iterations = 8_u32;
    let step_count = snapshot.steps.len();
    let tank_count = snapshot.tank_count;
    let job_count = step_count.checked_mul(tank_count).ok_or_else(|| Error::from_reason("job count overflow"))?;

    let tanks = (0..tank_count)
        .map(|tank| research_tank_from_packed(&snapshot.packed_config, tank))
        .collect::<Vec<_>>();

    let mut packed_tanks = Vec::<f64>::with_capacity(tank_count * 7);
    for tank in &tanks {
        let pmax = tank.maximum_heat_pump_electrical_power;
        packed_tanks.push(tank.target_storage_energy);
        packed_tanks.push(pmax);
        packed_tanks.push(pmax * tank.cop * tank.time_step_hours);
        packed_tanks.push(tank.heat_loss_retention);
        packed_tanks.push(tank.heat_loss_intercept);
        packed_tanks.push(if tank.affine_supported { 1.0 } else { 0.0 });
        packed_tanks.push(pmax * tank.time_step_hours);
    }

    let mut current_indices = Vec::<u32>::with_capacity(step_count);
    let mut horizon_steps = Vec::<u32>::with_capacity(step_count);
    let mut dynamic_flat = Vec::<f64>::with_capacity(job_count * 3);

    for step in &snapshot.steps {
        if step.horizon_steps == 0 || step.horizon_steps > 256 {
            return Err(Error::from_reason(format!("unsupported horizon {}", step.horizon_steps)));
        }
        current_indices.push(step.current_index);
        horizon_steps.push(step.horizon_steps);
        for tank in 0..tank_count {
            let base = tank * BLACKWELL_DYNAMIC_STRIDE;
            dynamic_flat.push(step.dynamic[base + 0]);
            dynamic_flat.push(step.dynamic[base + 1]);
            dynamic_flat.push(step.dynamic[base + 2]);
        }
    }

    let presort_started = Instant::now();
    let mut presorted = vec![f64::NEG_INFINITY; step_count * 256];

    for (step_index, step) in snapshot.steps.iter().enumerate() {
        let horizon = step.horizon_steps as usize;
        let base = step_index * 256;
        let first = base + 256 - horizon;
        for offset in 0..horizon {
            let index = (step.current_index as usize + offset) % snapshot.export_power.len();
            presorted[first + offset] = snapshot.export_power[index];
        }
        presorted[first..base + 256].sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let presort_cpu_ms = presort_started.elapsed().as_secs_f64() * 1000.0;
    let presorted_mib = (presorted.len() * std::mem::size_of::<f64>()) as f64 / 1048576.0;

    let mut direct_cuts = vec![0.0_f64; job_count];
    let mut direct_status = vec![0_u32; job_count];
    let mut direct_evals = vec![0_u32; job_count];
    let mut direct_static = 0.0_f64;
    let mut direct_gpu_raw = 0.0_f64;
    let mut direct_wait_raw = 0.0_f64;
    let mut direct_wall_raw = 0.0_f64;

    let code = unsafe {
        sunopt_cuda_run_digital_lockstep_candidate(
            selected_device,
            snapshot.export_power.as_ptr(),
            snapshot.demand_energy.as_ptr(),
            snapshot.export_power.len(),
            packed_tanks.as_ptr(),
            tank_count,
            current_indices.as_ptr(),
            horizon_steps.as_ptr(),
            dynamic_flat.as_ptr(),
            step_count,
            search_iterations,
            repetition_count,
            direct_cuts.as_mut_ptr(),
            direct_status.as_mut_ptr(),
            direct_evals.as_mut_ptr(),
            &mut direct_static,
            &mut direct_gpu_raw,
            &mut direct_wait_raw,
            &mut direct_wall_raw,
        )
    };
    if code != 0 { return Err(cuda_error("direct digital lockstep", code)); }

    let mut candidate_cuts = vec![0.0_f64; job_count];
    let mut candidate_status = vec![0_u32; job_count];
    let mut candidate_evals = vec![0_u32; job_count];
    let mut candidate_static = 0.0_f64;
    let mut presorted_h2d = 0.0_f64;
    let mut candidate_gpu_raw = 0.0_f64;
    let mut candidate_wait_raw = 0.0_f64;
    let mut candidate_wall_raw = 0.0_f64;

    let code = unsafe {
        sunopt_cuda_run_digital_lockstep_presorted_candidate(
            selected_device,
            snapshot.export_power.as_ptr(),
            snapshot.demand_energy.as_ptr(),
            snapshot.export_power.len(),
            packed_tanks.as_ptr(),
            tank_count,
            presorted.as_ptr(),
            current_indices.as_ptr(),
            horizon_steps.as_ptr(),
            dynamic_flat.as_ptr(),
            step_count,
            search_iterations,
            repetition_count,
            candidate_cuts.as_mut_ptr(),
            candidate_status.as_mut_ptr(),
            candidate_evals.as_mut_ptr(),
            &mut candidate_static,
            &mut presorted_h2d,
            &mut candidate_gpu_raw,
            &mut candidate_wait_raw,
            &mut candidate_wall_raw,
        )
    };
    if code != 0 { return Err(cuda_error("presorted digital lockstep", code)); }

    let divisor = repetition_count as f64;
    let direct_gpu_total_ms = direct_gpu_raw / divisor;
    let direct_wall_ms = direct_wall_raw / divisor;
    let presorted_gpu_total_ms = candidate_gpu_raw / divisor;
    let presorted_wall_ms = candidate_wall_raw / divisor;

    let mut status_mismatches = 0_u32;
    let mut cut_mismatches = 0_u32;
    let mut maximum_cut_error = 0.0_f64;

    for job in 0..job_count {
        if direct_status[job] != candidate_status[job] { status_mismatches += 1; }
        let error = (direct_cuts[job] - candidate_cuts[job]).abs();
        maximum_cut_error = maximum_cut_error.max(error);
        if error > 1.0e-9 { cut_mismatches += 1; }
    }

    let wall_saving_ms = direct_wall_ms - presorted_wall_ms;
    let wall_speedup = if presorted_wall_ms > 0.0 { direct_wall_ms / presorted_wall_ms } else { f64::INFINITY };
    let wall_reduction_percent = if direct_wall_ms > 0.0 { wall_saving_ms / direct_wall_ms * 100.0 } else { 0.0 };

    let mut checksum = 1469598103934665603u64;
    for value in &candidate_cuts { blackwell_fnv1a_update(&mut checksum, *value); }

    Ok(BlackwellDigitalPresortAbResult {
        snapshot_path: path.display().to_string(),
        device_id: selected_device,
        repetitions: repetition_count,
        search_iterations,
        annual_step_count: step_count as u32,
        tank_count: tank_count as u32,
        causal_jobs: job_count as u32,
        presorted_values: presorted.len() as u32,
        presorted_mib,
        presort_cpu_ms,
        direct_gpu_total_ms,
        direct_wall_ms,
        presorted_window_h2d_ms: presorted_h2d,
        presorted_gpu_total_ms,
        presorted_wall_ms,
        wall_saving_ms,
        wall_speedup,
        wall_reduction_percent,
        status_mismatches,
        cut_mismatches,
        maximum_cut_error,
        checksum_hex: format!("{checksum:016x}"),
    })
}
// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FN_END


// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_warp_tree_ab(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalWarpTreeAbResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(&path)
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
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device =
        device_id.unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(1, 3);

    let search_iterations =
        8_u32;

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );
        packed_tanks.push(pmax);
        packed_tanks.push(
            pmax *
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
        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(tank_count)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Digital warp-tree A/B job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count * 3,
        );

    for step in &snapshot.steps {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Digital warp-tree A/B supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank_index in 0..tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(
                step.dynamic[base + 0],
            );
            dynamic_flat.push(
                step.dynamic[base + 1],
            );
            dynamic_flat.push(
                step.dynamic[base + 2],
            );
        }
    }

    let presort_started =
        Instant::now();

    let presorted_count =
        step_count
            .checked_mul(256)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Presorted export matrix size overflow.",
                    )
                },
            )?;

    let mut presorted =
        vec![
            f64::NEG_INFINITY;
            presorted_count
        ];

    for (
        step_index,
        step,
    ) in
        snapshot.steps
            .iter()
            .enumerate()
    {
        let horizon =
            step.horizon_steps as usize;

        let base =
            step_index * 256;

        let first =
            base +
            (256 - horizon);

        for offset in 0..horizon {
            let index =
                (
                    step.current_index as usize +
                    offset
                ) %
                snapshot.export_power.len();

            presorted[
                first + offset
            ] =
                snapshot.export_power[index];
        }

        presorted[
            first..
            base + 256
        ]
        .sort_by(
            |a, b| {
                a.partial_cmp(b)
                    .unwrap_or(
                        std::cmp::Ordering::Equal,
                    )
            },
        );
    }

    let presort_cpu_ms =
        presort_started
            .elapsed()
            .as_secs_f64() *
        1000.0;

    let presorted_mib =
        (
            presorted.len() *
            std::mem::size_of::<f64>()
        ) as f64 /
        1048576.0;

    let mut block_cuts =
        vec![0.0_f64; job_count];

    let mut block_status =
        vec![0_u32; job_count];

    let mut block_evaluations =
        vec![0_u32; job_count];

    let mut block_static_h2d_ms =
        0.0_f64;

    let mut block_presorted_h2d_ms =
        0.0_f64;

    let mut block_gpu_total_raw_ms =
        0.0_f64;

    let mut block_host_wait_raw_ms =
        0.0_f64;

    let mut block_wall_raw_ms =
        0.0_f64;

    let block_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_presorted_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                block_cuts.as_mut_ptr(),
                block_status.as_mut_ptr(),
                block_evaluations.as_mut_ptr(),
                &mut block_static_h2d_ms,
                &mut block_presorted_h2d_ms,
                &mut block_gpu_total_raw_ms,
                &mut block_host_wait_raw_ms,
                &mut block_wall_raw_ms,
            )
        };

    if block_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_presorted_candidate",
                block_code,
            ),
        );
    }

    let mut warp_cuts =
        vec![0.0_f64; job_count];

    let mut warp_status =
        vec![0_u32; job_count];

    let mut warp_evaluations =
        vec![0_u32; job_count];

    let mut warp_static_h2d_ms =
        0.0_f64;

    let mut warp_presorted_h2d_ms =
        0.0_f64;

    let mut warp_gpu_total_raw_ms =
        0.0_f64;

    let mut warp_host_wait_raw_ms =
        0.0_f64;

    let mut warp_wall_raw_ms =
        0.0_f64;

    let warp_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                warp_cuts.as_mut_ptr(),
                warp_status.as_mut_ptr(),
                warp_evaluations.as_mut_ptr(),
                &mut warp_static_h2d_ms,
                &mut warp_presorted_h2d_ms,
                &mut warp_gpu_total_raw_ms,
                &mut warp_host_wait_raw_ms,
                &mut warp_wall_raw_ms,
            )
        };

    if warp_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",
                warp_code,
            ),
        );
    }

    let divisor =
        repetition_count as f64;

    let block_tree_gpu_total_ms =
        block_gpu_total_raw_ms /
        divisor;

    let block_tree_wall_ms =
        block_wall_raw_ms /
        divisor;

    let warp_tree_gpu_total_ms =
        warp_gpu_total_raw_ms /
        divisor;

    let warp_tree_wall_ms =
        warp_wall_raw_ms /
        divisor;

    let wall_saving_ms =
        block_tree_wall_ms -
        warp_tree_wall_ms;

    let wall_speedup =
        if warp_tree_wall_ms > 0.0 {
            block_tree_wall_ms /
            warp_tree_wall_ms
        } else {
            f64::INFINITY
        };

    let wall_reduction_percent =
        if block_tree_wall_ms > 0.0 {
            wall_saving_ms /
            block_tree_wall_ms *
            100.0
        } else {
            0.0
        };

    let mut block_warp_status_mismatches =
        0_u32;

    let mut block_warp_cut_mismatches =
        0_u32;

    let mut block_warp_max_cut_error =
        0.0_f64;

    let mut evaluation_mismatches =
        0_u32;

    for job in 0..job_count {
        if (
            block_status[job] !=
            warp_status[job]
        ) {
            block_warp_status_mismatches +=
                1;
        }

        if (
            block_evaluations[job] !=
            warp_evaluations[job]
        ) {
            evaluation_mismatches +=
                1;
        }

        let error =
            (
                block_cuts[job] -
                warp_cuts[job]
            )
            .abs();

        block_warp_max_cut_error =
            block_warp_max_cut_error.max(
                error,
            );

        if error > 1.0e-9 {
            block_warp_cut_mismatches +=
                1;
        }
    }

    let parity_sample_steps =
        192_usize.min(step_count);

    let mut parity_status_mismatches =
        0_u32;

    let mut parity_cut_mismatches =
        0_u32;

    let mut parity_max_cut_error =
        0.0_f64;

    for sample in 0..parity_sample_steps {
        let step_index =
            if parity_sample_steps <= 1 {
                0
            } else {
                sample *
                (step_count - 1) /
                (parity_sample_steps - 1)
            };

        let step =
            &snapshot.steps[step_index];

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        for tank_index in 0..tank_count {
            let job =
                step_index *
                tank_count +
                tank_index;

            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reference =
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    step.dynamic[base + 0],
                    step.dynamic[base + 1],
                    step.dynamic[base + 2],
                    tanks[tank_index],
                );

            let expected_status =
                if reference.normal_plan_survives {
                    0_u32
                } else if !reference.maximum_pv_plan_survives {
                    1_u32
                } else {
                    2_u32
                };

            if (
                warp_status[job] !=
                expected_status
            ) {
                parity_status_mismatches +=
                    1;
            }

            let error =
                (
                    warp_cuts[job] -
                    reference.cut_level_power
                )
                .abs();

            parity_max_cut_error =
                parity_max_cut_error.max(
                    error,
                );

            if error > 1.0e-9 {
                parity_cut_mismatches +=
                    1;
            }
        }
    }

    let mut checksum =
        1469598103934665603u64;

    for value in &warp_cuts {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    Ok(
        BlackwellDigitalWarpTreeAbResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            search_iterations,

            annual_step_count:
                step_count as u32,

            tank_count:
                tank_count as u32,

            causal_jobs:
                job_count as u32,

            presorted_mib,
            presort_cpu_ms,

            block_tree_gpu_total_ms,
            block_tree_wall_ms,

            warp_tree_gpu_total_ms,
            warp_tree_wall_ms,

            wall_saving_ms,
            wall_speedup,
            wall_reduction_percent,

            block_warp_status_mismatches,
            block_warp_cut_mismatches,
            block_warp_max_cut_error,
            evaluation_mismatches,

            parity_sample_steps:
                parity_sample_steps as u32,

            parity_status_mismatches,
            parity_cut_mismatches,
            parity_max_cut_error,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_FN_END


// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_piecewise_zero_ab(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalPiecewiseZeroAbResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(&path)
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
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device =
        device_id.unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(1, 3);

    let search_iterations =
        8_u32;

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(pmax);

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(tank_count)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Piecewise-zero A/B job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count * 3,
        );

    let mut legacy_serial_fallback_jobs =
        0_u32;

    let mut piecewise_serial_fallback_jobs =
        0_u32;

    for step in &snapshot.steps {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Piecewise-zero A/B supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank_index in 0..tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reserve =
                step.dynamic[
                    base + 1
                ]
                .max(0.0)
                .min(
                    tanks[
                        tank_index
                    ]
                    .target_storage_energy,
                );

            dynamic_flat.push(
                step.dynamic[base + 0],
            );

            dynamic_flat.push(
                reserve,
            );

            dynamic_flat.push(
                step.dynamic[base + 2],
            );

            let tank =
                tanks[tank_index];

            let legacy_supported =
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0 &&
                reserve >
                    1.0e-6;

            let piecewise_supported =
                tank.affine_supported &&
                tank.heat_loss_retention >
                    0.0;

            if !legacy_supported {
                legacy_serial_fallback_jobs +=
                    1;
            }

            if !piecewise_supported {
                piecewise_serial_fallback_jobs +=
                    1;
            }
        }
    }

    let legacy_compressed_jobs =
        job_count as u32 -
        legacy_serial_fallback_jobs;

    let piecewise_compressed_jobs =
        job_count as u32 -
        piecewise_serial_fallback_jobs;

    let presorted_count =
        step_count
            .checked_mul(256)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Presorted export matrix size overflow.",
                    )
                },
            )?;

    let mut presorted =
        vec![
            f64::NEG_INFINITY;
            presorted_count
        ];

    for (
        step_index,
        step,
    ) in
        snapshot.steps
            .iter()
            .enumerate()
    {
        let horizon =
            step.horizon_steps as usize;

        let base =
            step_index * 256;

        let first =
            base +
            (256 - horizon);

        for offset in 0..horizon {
            let index =
                (
                    step.current_index as usize +
                    offset
                ) %
                snapshot.export_power.len();

            presorted[
                first + offset
            ] =
                snapshot.export_power[index];
        }

        presorted[
            first..
            base + 256
        ]
        .sort_by(
            |a, b| {
                a.partial_cmp(b)
                    .unwrap_or(
                        std::cmp::Ordering::Equal,
                    )
            },
        );
    }

    let mut legacy_cuts =
        vec![0.0_f64; job_count];

    let mut legacy_status =
        vec![0_u32; job_count];

    let mut legacy_evaluations =
        vec![0_u32; job_count];

    let mut legacy_static_h2d_ms =
        0.0_f64;

    let mut legacy_presorted_h2d_ms =
        0.0_f64;

    let mut legacy_gpu_total_raw_ms =
        0.0_f64;

    let mut legacy_host_wait_raw_ms =
        0.0_f64;

    let mut legacy_wall_raw_ms =
        0.0_f64;

    let legacy_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                legacy_cuts.as_mut_ptr(),
                legacy_status.as_mut_ptr(),
                legacy_evaluations.as_mut_ptr(),
                &mut legacy_static_h2d_ms,
                &mut legacy_presorted_h2d_ms,
                &mut legacy_gpu_total_raw_ms,
                &mut legacy_host_wait_raw_ms,
                &mut legacy_wall_raw_ms,
            )
        };

    if legacy_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",
                legacy_code,
            ),
        );
    }

    let mut piecewise_cuts =
        vec![0.0_f64; job_count];

    let mut piecewise_status =
        vec![0_u32; job_count];

    let mut piecewise_evaluations =
        vec![0_u32; job_count];

    let mut piecewise_static_h2d_ms =
        0.0_f64;

    let mut piecewise_presorted_h2d_ms =
        0.0_f64;

    let mut piecewise_gpu_total_raw_ms =
        0.0_f64;

    let mut piecewise_host_wait_raw_ms =
        0.0_f64;

    let mut piecewise_wall_raw_ms =
        0.0_f64;

    let piecewise_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                piecewise_cuts.as_mut_ptr(),
                piecewise_status.as_mut_ptr(),
                piecewise_evaluations.as_mut_ptr(),
                &mut piecewise_static_h2d_ms,
                &mut piecewise_presorted_h2d_ms,
                &mut piecewise_gpu_total_raw_ms,
                &mut piecewise_host_wait_raw_ms,
                &mut piecewise_wall_raw_ms,
            )
        };

    if piecewise_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate",
                piecewise_code,
            ),
        );
    }

    let divisor =
        repetition_count as f64;

    let legacy_gpu_total_ms =
        legacy_gpu_total_raw_ms /
        divisor;

    let legacy_wall_ms =
        legacy_wall_raw_ms /
        divisor;

    let piecewise_gpu_total_ms =
        piecewise_gpu_total_raw_ms /
        divisor;

    let piecewise_wall_ms =
        piecewise_wall_raw_ms /
        divisor;

    let wall_saving_ms =
        legacy_wall_ms -
        piecewise_wall_ms;

    let wall_speedup =
        if piecewise_wall_ms > 0.0 {
            legacy_wall_ms /
            piecewise_wall_ms
        } else {
            f64::INFINITY
        };

    let wall_reduction_percent =
        if legacy_wall_ms > 0.0 {
            wall_saving_ms /
            legacy_wall_ms *
            100.0
        } else {
            0.0
        };

    let mut legacy_piecewise_status_mismatches =
        0_u32;

    let mut legacy_piecewise_cut_mismatches =
        0_u32;

    let mut legacy_piecewise_max_cut_error =
        0.0_f64;

    let mut evaluation_mismatches =
        0_u32;

    for job in 0..job_count {
        if (
            legacy_status[job] !=
            piecewise_status[job]
        ) {
            legacy_piecewise_status_mismatches +=
                1;
        }

        if (
            legacy_evaluations[job] !=
            piecewise_evaluations[job]
        ) {
            evaluation_mismatches +=
                1;
        }

        let error =
            (
                legacy_cuts[job] -
                piecewise_cuts[job]
            )
            .abs();

        legacy_piecewise_max_cut_error =
            legacy_piecewise_max_cut_error.max(
                error,
            );

        if error > 1.0e-9 {
            legacy_piecewise_cut_mismatches +=
                1;
        }
    }

    let parity_sample_steps =
        192_usize.min(step_count);

    let mut parity_status_mismatches =
        0_u32;

    let mut parity_cut_mismatches =
        0_u32;

    let mut parity_max_cut_error =
        0.0_f64;

    for sample in 0..parity_sample_steps {
        let step_index =
            if parity_sample_steps <= 1 {
                0
            } else {
                sample *
                (step_count - 1) /
                (parity_sample_steps - 1)
            };

        let step =
            &snapshot.steps[step_index];

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        let demand_window =
            research_window(
                &snapshot.demand_energy,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        for tank_index in 0..tank_count {
            let job =
                step_index *
                tank_count +
                tank_index;

            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let reference =
                digital_run_scalar(
                    &export_window,
                    &demand_window,
                    step.dynamic[base + 0],
                    step.dynamic[base + 1],
                    step.dynamic[base + 2],
                    tanks[tank_index],
                );

            let expected_status =
                if reference.normal_plan_survives {
                    0_u32
                } else if !reference.maximum_pv_plan_survives {
                    1_u32
                } else {
                    2_u32
                };

            if (
                piecewise_status[job] !=
                expected_status
            ) {
                parity_status_mismatches +=
                    1;
            }

            let error =
                (
                    piecewise_cuts[job] -
                    reference.cut_level_power
                )
                .abs();

            parity_max_cut_error =
                parity_max_cut_error.max(
                    error,
                );

            if error > 1.0e-9 {
                parity_cut_mismatches +=
                    1;
            }
        }
    }

    let mut checksum =
        1469598103934665603u64;

    for value in &piecewise_cuts {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    Ok(
        BlackwellDigitalPiecewiseZeroAbResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            search_iterations,

            annual_step_count:
                step_count as u32,

            tank_count:
                tank_count as u32,

            causal_jobs:
                job_count as u32,

            legacy_compressed_jobs,
            legacy_serial_fallback_jobs,
            piecewise_compressed_jobs,
            piecewise_serial_fallback_jobs,

            legacy_gpu_total_ms,
            legacy_wall_ms,

            piecewise_gpu_total_ms,
            piecewise_wall_ms,

            wall_saving_ms,
            wall_speedup,
            wall_reduction_percent,

            legacy_piecewise_status_mismatches,
            legacy_piecewise_cut_mismatches,
            legacy_piecewise_max_cut_error,
            evaluation_mismatches,

            parity_sample_steps:
                parity_sample_steps as u32,

            parity_status_mismatches,
            parity_cut_mismatches,
            parity_max_cut_error,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_FN_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_resident_chunk(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalResidentChunkResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(&path)
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
        snapshot.demand_energy.len() != snapshot.export_power.len() ||
        snapshot.steps.len() != snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device = device_id.unwrap_or(0);
    let repetition_count = repetitions.unwrap_or(1).clamp(1, 3);
    let search_iterations = 8_u32;

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(tank.target_storage_energy);
        packed_tanks.push(pmax);
        packed_tanks.push(
            pmax * tank.cop * tank.time_step_hours,
        );
        packed_tanks.push(tank.heat_loss_retention);
        packed_tanks.push(tank.heat_loss_intercept);
        packed_tanks.push(
            if tank.affine_supported { 1.0 } else { 0.0 },
        );
        packed_tanks.push(
            pmax * tank.time_step_hours,
        );
    }

    let step_count = snapshot.steps.len();
    let tank_count = snapshot.tank_count;
    let job_count =
        step_count
            .checked_mul(tank_count)
            .ok_or_else(
                || Error::from_reason(
                    "Resident-chunk job count overflow.",
                ),
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(step_count);
    let mut horizon_steps =
        Vec::<u32>::with_capacity(step_count);
    let mut dynamic_flat =
        Vec::<f64>::with_capacity(job_count * 3);

    for step in &snapshot.steps {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Resident-chunk benchmark supports horizon 1..=256; got {}.",
                        step.horizon_steps,
                    ),
                ),
            );
        }

        current_indices.push(step.current_index);
        horizon_steps.push(step.horizon_steps);

        for tank_index in 0..tank_count {
            let base =
                tank_index * BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(step.dynamic[base + 0]);
            dynamic_flat.push(step.dynamic[base + 1]);
            dynamic_flat.push(step.dynamic[base + 2]);
        }
    }

    let presorted_count =
        step_count
            .checked_mul(256)
            .ok_or_else(
                || Error::from_reason(
                    "Presorted export matrix size overflow.",
                ),
            )?;

    let mut presorted =
        vec![f64::NEG_INFINITY; presorted_count];

    for (step_index, step) in snapshot.steps.iter().enumerate() {
        let horizon = step.horizon_steps as usize;
        let base = step_index * 256;
        let first = base + (256 - horizon);

        for offset in 0..horizon {
            let index =
                (step.current_index as usize + offset) %
                snapshot.export_power.len();

            presorted[first + offset] =
                snapshot.export_power[index];
        }

        presorted[first..base + 256]
            .sort_by(
                |a, b| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                },
            );
    }

    let presorted_mib =
        (
            presorted.len() * std::mem::size_of::<f64>()
        ) as f64 /
        1048576.0;

    let mut reference_cuts = vec![0.0_f64; job_count];
    let mut reference_status = vec![0_u32; job_count];
    let mut reference_evaluations = vec![0_u32; job_count];

    let mut ref_static_h2d_ms = 0.0_f64;
    let mut ref_presorted_h2d_ms = 0.0_f64;
    let mut ref_gpu_total_raw_ms = 0.0_f64;
    let mut ref_host_wait_raw_ms = 0.0_f64;
    let mut ref_wall_raw_ms = 0.0_f64;

    let reference_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                reference_cuts.as_mut_ptr(),
                reference_status.as_mut_ptr(),
                reference_evaluations.as_mut_ptr(),
                &mut ref_static_h2d_ms,
                &mut ref_presorted_h2d_ms,
                &mut ref_gpu_total_raw_ms,
                &mut ref_host_wait_raw_ms,
                &mut ref_wall_raw_ms,
            )
        };

    if reference_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",
                reference_code,
            ),
        );
    }

    let divisor = repetition_count as f64;
    let lockstep_gpu_total_ms = ref_gpu_total_raw_ms / divisor;
    let lockstep_wall_ms = ref_wall_raw_ms / divisor;

    let mut reference_checksum = 1469598103934665603u64;
    for value in &reference_cuts {
        blackwell_fnv1a_update(&mut reference_checksum, *value);
    }

    let chunk_sizes: [u32; 7] =
        [1, 4, 16, 64, 256, 1024, 4096];

    let mut variants =
        Vec::<BlackwellDigitalResidentChunkVariant>::with_capacity(
            chunk_sizes.len(),
        );

    for chunk_steps in chunk_sizes {
        let mut cuts = vec![0.0_f64; job_count];
        let mut status = vec![0_u32; job_count];
        let mut evaluations = vec![0_u32; job_count];

        let mut input_h2d_ms = 0.0_f64;
        let mut kernel_ms = 0.0_f64;
        let mut result_d2h_ms = 0.0_f64;
        let mut wall_ms = 0.0_f64;

        let code =
            unsafe {
                sunopt_cuda_run_digital_resident_chunk_replay(
                    selected_device,
                    snapshot.export_power.as_ptr(),
                    snapshot.demand_energy.as_ptr(),
                    snapshot.export_power.len(),
                    packed_tanks.as_ptr(),
                    tank_count,
                    presorted.as_ptr(),
                    current_indices.as_ptr(),
                    horizon_steps.as_ptr(),
                    dynamic_flat.as_ptr(),
                    step_count,
                    search_iterations,
                    chunk_steps,
                    repetition_count,
                    cuts.as_mut_ptr(),
                    status.as_mut_ptr(),
                    evaluations.as_mut_ptr(),
                    &mut input_h2d_ms,
                    &mut kernel_ms,
                    &mut result_d2h_ms,
                    &mut wall_ms,
                )
            };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_digital_resident_chunk_replay",
                    code,
                ),
            );
        }

        let mut status_mismatches = 0_u32;
        let mut cut_mismatches = 0_u32;
        let mut maximum_cut_error = 0.0_f64;
        let mut evaluation_mismatches = 0_u32;

        for job in 0..job_count {
            if status[job] != reference_status[job] {
                status_mismatches += 1;
            }

            if evaluations[job] != reference_evaluations[job] {
                evaluation_mismatches += 1;
            }

            let error =
                (cuts[job] - reference_cuts[job]).abs();

            maximum_cut_error =
                maximum_cut_error.max(error);

            if error > 1.0e-9 {
                cut_mismatches += 1;
            }
        }

        let mut checksum = 1469598103934665603u64;
        for value in &cuts {
            blackwell_fnv1a_update(&mut checksum, *value);
        }

        let speedup_vs_lockstep =
            if wall_ms > 0.0 {
                lockstep_wall_ms / wall_ms
            } else {
                f64::INFINITY
            };

        let reduction_vs_lockstep_percent =
            if lockstep_wall_ms > 0.0 {
                (lockstep_wall_ms - wall_ms) /
                    lockstep_wall_ms *
                    100.0
            } else {
                0.0
            };

        let launches_per_year =
            (
                (step_count as u64 + chunk_steps as u64 - 1) /
                chunk_steps as u64
            ) as u32;

        variants.push(
            BlackwellDigitalResidentChunkVariant {
                chunk_steps,
                launches_per_year,
                input_h2d_ms,
                kernel_ms,
                result_d2h_ms,
                wall_ms,
                speedup_vs_lockstep,
                reduction_vs_lockstep_percent,
                status_mismatches,
                cut_mismatches,
                maximum_cut_error,
                evaluation_mismatches,
                checksum_hex: format!("{checksum:016x}"),
            },
        );
    }

    Ok(
        BlackwellDigitalResidentChunkResult {
            snapshot_path: path.display().to_string(),
            device_id: selected_device,
            repetitions: repetition_count,
            search_iterations,
            annual_step_count: step_count as u32,
            tank_count: tank_count as u32,
            causal_jobs: job_count as u32,
            presorted_mib,
            lockstep_gpu_total_ms,
            lockstep_wall_ms,
            lockstep_checksum_hex:
                format!("{reference_checksum:016x}"),
            variants,
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_FN_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_resident_job_batch(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalResidentJobBatchResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
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
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device =
        device_id.unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(
                1,
                3,
            );

    let search_iterations =
        8_u32;

    let tanks =
        (
            0..
            snapshot.tank_count
        )
        .map(
            |tank| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank,
                )
            },
        )
        .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count *
            7,
        );

    for tank in
        &tanks
    {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            pmax,
        );

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Resident-job-batch job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count *
            3,
        );

    for step in
        &snapshot.steps
    {
        if (
            step.horizon_steps ==
                0 ||
            step.horizon_steps >
                256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Resident-job-batch supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank_index in
            0..
            tank_count
        {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(
                step.dynamic[
                    base + 0
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 1
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 2
                ],
            );
        }
    }

    let presorted_count =
        step_count
            .checked_mul(
                256
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Presorted export matrix size overflow.",
                    )
                },
            )?;

    let mut presorted =
        vec![
            f64::NEG_INFINITY;
            presorted_count
        ];

    for (
        step_index,
        step,
    ) in
        snapshot.steps
            .iter()
            .enumerate()
    {
        let horizon =
            step.horizon_steps
                as usize;

        let base =
            step_index *
            256;

        let first =
            base +
            (
                256 -
                horizon
            );

        for offset in
            0..
            horizon
        {
            let index =
                (
                    step.current_index
                        as usize +
                    offset
                ) %
                snapshot.export_power
                    .len();

            presorted[
                first +
                offset
            ] =
                snapshot.export_power[
                    index
                ];
        }

        presorted[
            first..
            base + 256
        ]
        .sort_by(
            |a, b| {
                a.partial_cmp(
                    b
                )
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
            },
        );
    }

    let presorted_mib =
        (
            presorted.len() *
            std::mem::size_of::<f64>()
        ) as f64 /
        1048576.0;

    /*
     * Reference = current B8 resident chunk=1 behaviour.
     * This has the same 28 jobs per launch as the true lockstep but with
     * all annual inputs/results resident.
     */
    let mut reference_cuts =
        vec![
            0.0_f64;
            job_count
        ];

    let mut reference_status =
        vec![
            0_u32;
            job_count
        ];

    let mut reference_evaluations =
        vec![
            0_u32;
            job_count
        ];

    let mut ref_input_h2d_ms =
        0.0_f64;

    let mut ref_kernel_ms =
        0.0_f64;

    let mut ref_result_d2h_ms =
        0.0_f64;

    let mut ref_wall_ms =
        0.0_f64;

    let reference_code =
        unsafe {
            sunopt_cuda_run_digital_resident_chunk_replay(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                tank_count,
                presorted
                    .as_ptr(),
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_flat
                    .as_ptr(),
                step_count,
                search_iterations,
                1,
                repetition_count,
                reference_cuts
                    .as_mut_ptr(),
                reference_status
                    .as_mut_ptr(),
                reference_evaluations
                    .as_mut_ptr(),
                &mut ref_input_h2d_ms,
                &mut ref_kernel_ms,
                &mut ref_result_d2h_ms,
                &mut ref_wall_ms,
            )
        };

    if (
        reference_code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_resident_chunk_replay",
                reference_code,
            ),
        );
    }

    /*
     * For context, retain the most recently measured lockstep shape by
     * running the actual pre-sorted warp lockstep once in this same process.
     */
    let mut lock_cuts =
        vec![
            0.0_f64;
            job_count
        ];

    let mut lock_status =
        vec![
            0_u32;
            job_count
        ];

    let mut lock_eval =
        vec![
            0_u32;
            job_count
        ];

    let mut lock_static_h2d =
        0.0_f64;

    let mut lock_presorted_h2d =
        0.0_f64;

    let mut lock_gpu_total =
        0.0_f64;

    let mut lock_host_wait =
        0.0_f64;

    let mut lock_wall_raw =
        0.0_f64;

    let lock_code =
        unsafe {
            sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                tank_count,
                presorted
                    .as_ptr(),
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_flat
                    .as_ptr(),
                step_count,
                search_iterations,
                repetition_count,
                lock_cuts
                    .as_mut_ptr(),
                lock_status
                    .as_mut_ptr(),
                lock_eval
                    .as_mut_ptr(),
                &mut lock_static_h2d,
                &mut lock_presorted_h2d,
                &mut lock_gpu_total,
                &mut lock_host_wait,
                &mut lock_wall_raw,
            )
        };

    if (
        lock_code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_lockstep_presorted_warp_candidate",
                lock_code,
            ),
        );
    }

    let lockstep_wall_ms =
        lock_wall_raw /
        repetition_count
            as f64;

    let mut reference_checksum =
        1469598103934665603u64;

    for value in
        &reference_cuts
    {
        blackwell_fnv1a_update(
            &mut reference_checksum,
            *value,
        );
    }

    /*
     * 28 jobs == one annual time slice. Larger values deliberately expose
     * more independent snapshot jobs to the GPU. They are a throughput lower
     * bound, NOT a production-causal execution order.
     */
    let batch_sizes:
        [u32; 7] =
        [
            28,
            112,
            448,
            1792,
            7168,
            28672,
            114688,
        ];

    let mut variants =
        Vec::<BlackwellDigitalResidentJobBatchVariant>::with_capacity(
            batch_sizes.len(),
        );

    for jobs_per_launch in
        batch_sizes
    {
        let mut cuts =
            vec![
                0.0_f64;
                job_count
            ];

        let mut status =
            vec![
                0_u32;
                job_count
            ];

        let mut evaluations =
            vec![
                0_u32;
                job_count
            ];

        let mut input_h2d_ms =
            0.0_f64;

        let mut kernel_ms =
            0.0_f64;

        let mut result_d2h_ms =
            0.0_f64;

        let mut wall_ms =
            0.0_f64;

        let code =
            unsafe {
                sunopt_cuda_run_digital_resident_job_batch(
                    selected_device,
                    snapshot.export_power
                        .as_ptr(),
                    snapshot.demand_energy
                        .as_ptr(),
                    snapshot.export_power
                        .len(),
                    packed_tanks
                        .as_ptr(),
                    tank_count,
                    presorted
                        .as_ptr(),
                    current_indices
                        .as_ptr(),
                    horizon_steps
                        .as_ptr(),
                    dynamic_flat
                        .as_ptr(),
                    step_count,
                    search_iterations,
                    jobs_per_launch,
                    repetition_count,
                    cuts
                        .as_mut_ptr(),
                    status
                        .as_mut_ptr(),
                    evaluations
                        .as_mut_ptr(),
                    &mut input_h2d_ms,
                    &mut kernel_ms,
                    &mut result_d2h_ms,
                    &mut wall_ms,
                )
            };

        if (
            code !=
            0
        ) {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_digital_resident_job_batch",
                    code,
                ),
            );
        }

        let mut status_mismatches =
            0_u32;

        let mut cut_mismatches =
            0_u32;

        let mut maximum_cut_error =
            0.0_f64;

        let mut evaluation_mismatches =
            0_u32;

        for job in
            0..
            job_count
        {
            if (
                status[
                    job
                ] !=
                reference_status[
                    job
                ]
            ) {
                status_mismatches +=
                    1;
            }

            if (
                evaluations[
                    job
                ] !=
                reference_evaluations[
                    job
                ]
            ) {
                evaluation_mismatches +=
                    1;
            }

            let error =
                (
                    cuts[
                        job
                    ] -
                    reference_cuts[
                        job
                    ]
                )
                .abs();

            maximum_cut_error =
                maximum_cut_error.max(
                    error,
                );

            if (
                error >
                1.0e-9
            ) {
                cut_mismatches +=
                    1;
            }
        }

        let mut checksum =
            1469598103934665603u64;

        for value in
            &cuts
        {
            blackwell_fnv1a_update(
                &mut checksum,
                *value,
            );
        }

        let speedup_vs_chunk_28 =
            if (
                wall_ms >
                0.0
            ) {
                ref_wall_ms /
                wall_ms
            } else {
                f64::INFINITY
            };

        let speedup_vs_lockstep =
            if (
                wall_ms >
                0.0
            ) {
                lockstep_wall_ms /
                wall_ms
            } else {
                f64::INFINITY
            };

        let reduction_vs_lockstep_percent =
            if (
                lockstep_wall_ms >
                0.0
            ) {
                (
                    lockstep_wall_ms -
                    wall_ms
                ) /
                lockstep_wall_ms *
                100.0
            } else {
                0.0
            };

        let launches_per_year =
            (
                (
                    job_count
                        as u64 +
                    jobs_per_launch
                        as u64 -
                    1
                ) /
                jobs_per_launch
                    as u64
            ) as u32;

        variants.push(
            BlackwellDigitalResidentJobBatchVariant {
                jobs_per_launch,
                launches_per_year,
                input_h2d_ms,
                kernel_ms,
                result_d2h_ms,
                wall_ms,
                speedup_vs_chunk_28,
                speedup_vs_lockstep,
                reduction_vs_lockstep_percent,
                status_mismatches,
                cut_mismatches,
                maximum_cut_error,
                evaluation_mismatches,
                checksum_hex:
                    format!(
                        "{checksum:016x}",
                    ),
            },
        );
    }

    Ok(
        BlackwellDigitalResidentJobBatchResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            search_iterations,

            annual_step_count:
                step_count
                    as u32,

            tank_count:
                tank_count
                    as u32,

            causal_jobs:
                job_count
                    as u32,

            presorted_mib,

            lockstep_wall_ms,

            chunk_28_wall_ms:
                ref_wall_ms,

            reference_checksum_hex:
                format!(
                    "{reference_checksum:016x}",
                ),

            variants,
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_FN_END


// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_multilane_search(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalMultilaneResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
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
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device =
        device_id.unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(
                1,
                3,
            );

    let tanks =
        (
            0..
            snapshot.tank_count
        )
        .map(
            |tank| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank,
                )
            },
        )
        .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count *
            7,
        );

    for tank in
        &tanks
    {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            pmax,
        );

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Multilane-search job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count *
            3,
        );

    for step in
        &snapshot.steps
    {
        if (
            step.horizon_steps ==
                0 ||
            step.horizon_steps >
                256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Multilane-search supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank_index in
            0..
            tank_count
        {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(
                step.dynamic[
                    base + 0
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 1
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base + 2
                ],
            );
        }
    }

    let presorted_count =
        step_count
            .checked_mul(
                256
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Presorted export matrix size overflow.",
                    )
                },
            )?;

    let mut presorted =
        vec![
            f64::NEG_INFINITY;
            presorted_count
        ];

    for (
        step_index,
        step,
    ) in
        snapshot.steps
            .iter()
            .enumerate()
    {
        let horizon =
            step.horizon_steps
                as usize;

        let base =
            step_index *
            256;

        let first =
            base +
            (
                256 -
                horizon
            );

        for offset in
            0..
            horizon
        {
            let index =
                (
                    step.current_index
                        as usize +
                    offset
                ) %
                snapshot.export_power
                    .len();

            presorted[
                first +
                offset
            ] =
                snapshot.export_power[
                    index
                ];
        }

        presorted[
            first..
            base + 256
        ]
        .sort_by(
            |a, b| {
                a.partial_cmp(
                    b
                )
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
            },
        );
    }

    let presorted_mib =
        (
            presorted.len() *
            std::mem::size_of::<f64>()
        ) as f64 /
        1048576.0;

    /*
     * Binary reference = B8 resident chunk=1.
     */
    let mut reference_cuts =
        vec![
            0.0_f64;
            job_count
        ];

    let mut reference_status =
        vec![
            0_u32;
            job_count
        ];

    let mut reference_evaluations =
        vec![
            0_u32;
            job_count
        ];

    let mut ref_input_h2d_ms =
        0.0_f64;

    let mut binary_kernel_ms =
        0.0_f64;

    let mut ref_result_d2h_ms =
        0.0_f64;

    let mut binary_wall_ms =
        0.0_f64;

    let reference_code =
        unsafe {
            sunopt_cuda_run_digital_resident_chunk_replay(
                selected_device,
                snapshot.export_power
                    .as_ptr(),
                snapshot.demand_energy
                    .as_ptr(),
                snapshot.export_power
                    .len(),
                packed_tanks
                    .as_ptr(),
                tank_count,
                presorted
                    .as_ptr(),
                current_indices
                    .as_ptr(),
                horizon_steps
                    .as_ptr(),
                dynamic_flat
                    .as_ptr(),
                step_count,
                8,
                1,
                repetition_count,
                reference_cuts
                    .as_mut_ptr(),
                reference_status
                    .as_mut_ptr(),
                reference_evaluations
                    .as_mut_ptr(),
                &mut ref_input_h2d_ms,
                &mut binary_kernel_ms,
                &mut ref_result_d2h_ms,
                &mut binary_wall_ms,
            )
        };

    if (
        reference_code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_resident_chunk_replay",
                reference_code,
            ),
        );
    }

    let mut reference_checksum =
        1469598103934665603u64;

    for value in
        &reference_cuts
    {
        blackwell_fnv1a_update(
            &mut reference_checksum,
            *value,
        );
    }

    let lane_counts:
        [u32; 3] =
        [
            2,
            4,
            8,
        ];

    let mut variants =
        Vec::<BlackwellDigitalMultilaneVariant>::with_capacity(
            lane_counts.len(),
        );

    for lane_count in
        lane_counts
    {
        let mut cuts =
            vec![
                0.0_f64;
                job_count
            ];

        let mut status =
            vec![
                0_u32;
                job_count
            ];

        let mut evaluations =
            vec![
                0_u32;
                job_count
            ];

        let mut input_h2d_ms =
            0.0_f64;

        let mut kernel_ms =
            0.0_f64;

        let mut result_d2h_ms =
            0.0_f64;

        let mut wall_ms =
            0.0_f64;

        let code =
            unsafe {
                sunopt_cuda_run_digital_multilane_lockstep(
                    selected_device,
                    snapshot.export_power
                        .as_ptr(),
                    snapshot.demand_energy
                        .as_ptr(),
                    snapshot.export_power
                        .len(),
                    packed_tanks
                        .as_ptr(),
                    tank_count,
                    presorted
                        .as_ptr(),
                    current_indices
                        .as_ptr(),
                    horizon_steps
                        .as_ptr(),
                    dynamic_flat
                        .as_ptr(),
                    step_count,
                    lane_count,
                    repetition_count,
                    cuts
                        .as_mut_ptr(),
                    status
                        .as_mut_ptr(),
                    evaluations
                        .as_mut_ptr(),
                    &mut input_h2d_ms,
                    &mut kernel_ms,
                    &mut result_d2h_ms,
                    &mut wall_ms,
                )
            };

        if (
            code !=
            0
        ) {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_digital_multilane_lockstep",
                    code,
                ),
            );
        }

        let mut status_mismatches =
            0_u32;

        let mut cut_mismatches =
            0_u32;

        let mut maximum_cut_error =
            0.0_f64;

        let mut evaluation_sum =
            0_u64;

        let mut maximum_survival_evaluations =
            0_u32;

        for job in
            0..
            job_count
        {
            if (
                status[
                    job
                ] !=
                reference_status[
                    job
                ]
            ) {
                status_mismatches +=
                    1;
            }

            let error =
                (
                    cuts[
                        job
                    ] -
                    reference_cuts[
                        job
                    ]
                )
                .abs();

            maximum_cut_error =
                maximum_cut_error.max(
                    error,
                );

            if (
                error >
                1.0e-9
            ) {
                cut_mismatches +=
                    1;
            }

            evaluation_sum +=
                evaluations[
                    job
                ] as u64;

            maximum_survival_evaluations =
                maximum_survival_evaluations.max(
                    evaluations[
                        job
                    ],
                );
        }

        let average_survival_evaluations =
            if job_count >
                0
            {
                evaluation_sum
                    as f64 /
                job_count
                    as f64
            } else {
                0.0
            };

        let mut checksum =
            1469598103934665603u64;

        for value in
            &cuts
        {
            blackwell_fnv1a_update(
                &mut checksum,
                *value,
            );
        }

        let speedup_vs_binary =
            if wall_ms >
                0.0
            {
                binary_wall_ms /
                wall_ms
            } else {
                f64::INFINITY
            };

        let reduction_vs_binary_percent =
            if binary_wall_ms >
                0.0
            {
                (
                    binary_wall_ms -
                    wall_ms
                ) /
                binary_wall_ms *
                100.0
            } else {
                0.0
            };

        variants.push(
            BlackwellDigitalMultilaneVariant {
                lane_count,

                threads_per_block:
                    lane_count *
                    32,

                input_h2d_ms,
                kernel_ms,
                result_d2h_ms,
                wall_ms,
                speedup_vs_binary,
                reduction_vs_binary_percent,
                status_mismatches,
                cut_mismatches,
                maximum_cut_error,
                average_survival_evaluations,
                maximum_survival_evaluations,

                checksum_hex:
                    format!(
                        "{checksum:016x}",
                    ),
            },
        );
    }

    Ok(
        BlackwellDigitalMultilaneResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            annual_step_count:
                step_count
                    as u32,

            tank_count:
                tank_count
                    as u32,

            causal_jobs:
                job_count
                    as u32,

            presorted_mib,

            binary_kernel_ms,
            binary_wall_ms,

            binary_checksum_hex:
                format!(
                    "{reference_checksum:016x}",
                ),

            variants,
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_FN_END


// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FN_BEGIN
#[napi]
pub fn run_blackwell_digital_cluster4(
    snapshot_path: Option<String>,
    repetitions: Option<u32>,
    device_id: Option<i32>,
) -> Result<BlackwellDigitalCluster4Result> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );

    let snapshot =
        read_blackwell_snapshot(&path)
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
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty or not a full annual lockstep snapshot.",
            ),
        );
    }

    let selected_device =
        device_id.unwrap_or(0);

    let repetition_count =
        repetitions
            .unwrap_or(1)
            .clamp(1, 3);

    let tanks =
        (0..snapshot.tank_count)
            .map(
                |tank| {
                    research_tank_from_packed(
                        &snapshot.packed_config,
                        tank,
                    )
                },
            )
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 7,
        );

    for tank in &tanks {
        let pmax =
            tank.maximum_heat_pump_electrical_power;

        packed_tanks.push(
            tank.target_storage_energy,
        );

        packed_tanks.push(
            pmax,
        );

        packed_tanks.push(
            pmax *
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

        packed_tanks.push(
            pmax *
            tank.time_step_hours,
        );
    }

    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;

    let job_count =
        step_count
            .checked_mul(tank_count)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Cluster4 job count overflow.",
                    )
                },
            )?;

    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<f64>::with_capacity(
            job_count * 3,
        );

    for step in &snapshot.steps {
        if (
            step.horizon_steps == 0 ||
            step.horizon_steps > 256
        ) {
            return Err(
                Error::from_reason(
                    format!(
                        "Cluster4 benchmark supports horizon 1..=256; got {}.",
                        step.horizon_steps,
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

        for tank_index in 0..tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic_flat.push(
                step.dynamic[base + 0],
            );

            dynamic_flat.push(
                step.dynamic[base + 1],
            );

            dynamic_flat.push(
                step.dynamic[base + 2],
            );
        }
    }

    let presorted_count =
        step_count
            .checked_mul(256)
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Presorted export matrix size overflow.",
                    )
                },
            )?;

    let mut presorted =
        vec![
            f64::NEG_INFINITY;
            presorted_count
        ];

    for (
        step_index,
        step,
    ) in
        snapshot.steps
            .iter()
            .enumerate()
    {
        let horizon =
            step.horizon_steps as usize;

        let base =
            step_index * 256;

        let first =
            base +
            (
                256 -
                horizon
            );

        for offset in 0..horizon {
            let index =
                (
                    step.current_index as usize +
                    offset
                ) %
                snapshot.export_power.len();

            presorted[
                first + offset
            ] =
                snapshot.export_power[index];
        }

        presorted[
            first..
            base + 256
        ]
        .sort_by(
            |a, b| {
                a.partial_cmp(b)
                    .unwrap_or(
                        std::cmp::Ordering::Equal,
                    )
            },
        );
    }

    /*
     * B8 exact resident reference: same annual data, same binary breakpoint
     * search, one 256-thread block per tank.
     */
    let mut binary_cuts =
        vec![0.0_f64; job_count];

    let mut binary_status =
        vec![0_u32; job_count];

    let mut binary_evaluations =
        vec![0_u32; job_count];

    let mut binary_input_h2d_ms =
        0.0_f64;

    let mut binary_kernel_ms =
        0.0_f64;

    let mut binary_result_d2h_ms =
        0.0_f64;

    let mut binary_wall_ms =
        0.0_f64;

    let binary_code =
        unsafe {
            sunopt_cuda_run_digital_resident_chunk_replay(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                8,
                1,
                repetition_count,
                binary_cuts.as_mut_ptr(),
                binary_status.as_mut_ptr(),
                binary_evaluations.as_mut_ptr(),
                &mut binary_input_h2d_ms,
                &mut binary_kernel_ms,
                &mut binary_result_d2h_ms,
                &mut binary_wall_ms,
            )
        };

    if binary_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_resident_chunk_replay",
                binary_code,
            ),
        );
    }

    let mut cluster_cuts =
        vec![0.0_f64; job_count];

    let mut cluster_status =
        vec![0_u32; job_count];

    let mut cluster_evaluations =
        vec![0_u32; job_count];

    let mut cluster_input_h2d_ms =
        0.0_f64;

    let mut cluster_kernel_ms =
        0.0_f64;

    let mut cluster_result_d2h_ms =
        0.0_f64;

    let mut cluster_wall_ms =
        0.0_f64;

    let cluster_code =
        unsafe {
            sunopt_cuda_run_digital_cluster4_lockstep(
                selected_device,
                snapshot.export_power.as_ptr(),
                snapshot.demand_energy.as_ptr(),
                snapshot.export_power.len(),
                packed_tanks.as_ptr(),
                tank_count,
                presorted.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                repetition_count,
                cluster_cuts.as_mut_ptr(),
                cluster_status.as_mut_ptr(),
                cluster_evaluations.as_mut_ptr(),
                &mut cluster_input_h2d_ms,
                &mut cluster_kernel_ms,
                &mut cluster_result_d2h_ms,
                &mut cluster_wall_ms,
            )
        };

    if cluster_code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_cluster4_lockstep",
                cluster_code,
            ),
        );
    }

    let mut status_mismatches =
        0_u32;

    let mut cut_mismatches =
        0_u32;

    let mut maximum_cut_error =
        0.0_f64;

    let mut evaluation_mismatches =
        0_u32;

    for job in 0..job_count {
        if (
            cluster_status[job] !=
            binary_status[job]
        ) {
            status_mismatches +=
                1;
        }

        if (
            cluster_evaluations[job] !=
            binary_evaluations[job]
        ) {
            evaluation_mismatches +=
                1;
        }

        let error =
            (
                cluster_cuts[job] -
                binary_cuts[job]
            )
            .abs();

        maximum_cut_error =
            maximum_cut_error.max(
                error,
            );

        if error > 1.0e-9 {
            cut_mismatches +=
                1;
        }
    }

    let speedup_vs_binary =
        if cluster_wall_ms > 0.0 {
            binary_wall_ms /
            cluster_wall_ms
        } else {
            f64::INFINITY
        };

    let reduction_vs_binary_percent =
        if binary_wall_ms > 0.0 {
            (
                binary_wall_ms -
                cluster_wall_ms
            ) /
            binary_wall_ms *
            100.0
        } else {
            0.0
        };

    let mut binary_checksum =
        1469598103934665603u64;

    for value in &binary_cuts {
        blackwell_fnv1a_update(
            &mut binary_checksum,
            *value,
        );
    }

    let mut cluster_checksum =
        1469598103934665603u64;

    for value in &cluster_cuts {
        blackwell_fnv1a_update(
            &mut cluster_checksum,
            *value,
        );
    }

    Ok(
        BlackwellDigitalCluster4Result {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            repetitions:
                repetition_count,

            annual_step_count:
                step_count as u32,

            tank_count:
                tank_count as u32,

            causal_jobs:
                job_count as u32,

            binary_kernel_ms,
            binary_wall_ms,

            cluster_blocks_per_tank:
                4,

            cluster_threads_per_block:
                64,

            cluster_total_blocks_per_step:
                (tank_count * 4) as u32,

            cluster_input_h2d_ms,
            cluster_kernel_ms,
            cluster_result_d2h_ms,
            cluster_wall_ms,

            speedup_vs_binary,
            reduction_vs_binary_percent,

            status_mismatches,
            cut_mismatches,
            maximum_cut_error,
            evaluation_mismatches,

            binary_checksum_hex:
                format!(
                    "{binary_checksum:016x}",
                ),

            cluster_checksum_hex:
                format!(
                    "{cluster_checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_FN_END


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



// SUNOPT_BLACKWELL_B14_EXACT_ENVELOPE_FN_BEGIN
#[napi]
pub fn run_blackwell_b14_exact_envelope(
    snapshot_path: Option<String>, sample_steps: Option<u32>
) -> Result<BlackwellB14ExactEnvelopeResult> {
    let path=snapshot_path.map(PathBuf::from).unwrap_or_else(blackwell_default_snapshot_path);
    let snapshot=read_blackwell_snapshot(&path).map_err(|e|Error::from_reason(format!("Unable to read Blackwell snapshot {}: {e}",path.display())))?;
    let requested=sample_steps.unwrap_or(192).clamp(2,snapshot.steps.len() as u32) as usize;
    let mut positions=Vec::with_capacity(requested); for i in 0..requested {let p=i*(snapshot.steps.len()-1)/(requested-1);if positions.last().copied()!=Some(p){positions.push(p)}}
    let tanks=(0..snapshot.tank_count).map(|i|research_tank_from_packed(&snapshot.packed_config,i)).collect::<Vec<_>>();
    let started=Instant::now(); let mut supported=0u32;let mut unsupported=0u32;let mut correction=0u32;let mut status_mis=0u32;
    let mut e12=0u32;let mut e9=0u32;let mut e6=0u32;let mut maxerr=0.0f64;let mut sumerr=0.0;let mut segs=Vec::<u32>::new();let mut bps=Vec::<u32>::new();
    for &p in &positions {let step=&snapshot.steps[p];let h=step.horizon_steps as usize;let ew=research_window(&snapshot.export_power,step.current_index as usize,h);let dw=research_window(&snapshot.demand_energy,step.current_index as usize,h);
      for t in 0..snapshot.tank_count {let b=t*BLACKWELL_DYNAMIC_STRIDE;let reference=research_run_scalar(&ew,&dw,step.dynamic[b],step.dynamic[b+1],step.dynamic[b+2],tanks[t]);
        match b14_exact_cut(&ew,&dw,step.dynamic[b],step.dynamic[b+1],step.dynamic[b+2],tanks[t]) {None=>unsupported+=1,Some((candidate,nseg,nbp))=>{supported+=1;if !reference.normal_plan_survives&&reference.maximum_pv_plan_survives{correction+=1}
          if candidate.normal_plan_survives!=reference.normal_plan_survives||candidate.maximum_pv_plan_survives!=reference.maximum_pv_plan_survives{status_mis+=1}
          let e=(candidate.cut_level_power-reference.cut_level_power).abs();maxerr=maxerr.max(e);sumerr+=e;if e>1e-12{e12+=1}if e>1e-9{e9+=1}if e>1e-6{e6+=1}segs.push(nseg as u32);bps.push(nbp as u32);}}
      }}
    segs.sort_unstable();bps.sort_unstable(); let quant=|v:&Vec<u32>,q:f64|->u32{if v.is_empty(){0}else{v[((v.len()-1) as f64*q).round() as usize]}};
    let mean=|v:&Vec<u32>|->f64{if v.is_empty(){0.0}else{v.iter().map(|&x|x as f64).sum::<f64>()/v.len() as f64}};
    let extra=segs.iter().zip(bps.iter()).map(|(&s,&b)|s.saturating_sub(b+1)).sum::<u32>();
    Ok(BlackwellB14ExactEnvelopeResult{snapshot_path:path.display().to_string(),sample_step_count:positions.len() as u32,sample_job_count:(positions.len()*snapshot.tank_count) as u32,supported_jobs:supported,unsupported_jobs:unsupported,correction_search_jobs:correction,elapsed_ms:started.elapsed().as_secs_f64()*1000.0,status_mismatches:status_mis,cut_mismatches_1e12:e12,cut_mismatches_1e9:e9,cut_mismatches_1e6:e6,max_abs_cut_error:maxerr,mean_abs_cut_error:if supported>0{sumerr/supported as f64}else{0.0},min_segments:segs.first().copied().unwrap_or(0),median_segments:quant(&segs,0.5) as f64,p95_segments:quant(&segs,0.95),max_segments:segs.last().copied().unwrap_or(0),mean_segments:mean(&segs),min_hp_breakpoints:bps.first().copied().unwrap_or(0),median_hp_breakpoints:quant(&bps,0.5) as f64,p95_hp_breakpoints:quant(&bps,0.95),max_hp_breakpoints:bps.last().copied().unwrap_or(0),mean_hp_breakpoints:mean(&bps),extra_segments_total:extra})
}
// SUNOPT_BLACKWELL_B14_EXACT_ENVELOPE_FN_END


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
