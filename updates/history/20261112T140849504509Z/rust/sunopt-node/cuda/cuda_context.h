#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define SUNOPT_CUDA_EXPORT __declspec(dllexport)
#else
#define SUNOPT_CUDA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SunoptCudaContext SunoptCudaContext;

typedef struct SunoptCausalParameters {
    size_t data_length;
    size_t current_index;
    size_t horizon_steps;

    double current_storage_energy;
    double minimum_reserve_energy;
    double target_storage_energy;
    double maximum_heat_pump_electrical_power;
    double cop;

    /*
     * >= 0:
     *   CUDA computes the canonical 32-iteration energy Waterline
     *   before causal survival testing.
     *
     * < 0:
     *   use host-provided energy_cut_level (legacy/debug path).
     */
    double requested_electrical_energy;

    double energy_cut_level;
    double time_step_hours;
    double storage_energy_per_kelvin;
    double storage_conductance_w_per_kelvin;
    double storage_min_temperature;
    double ambient_temperature;
    double daily_heat_loss;
} SunoptCausalParameters;

typedef struct SunoptCausalDynamic {
    double current_storage_energy;
    double minimum_reserve_energy;
    double requested_electrical_energy;
} SunoptCausalDynamic;


typedef struct SunoptCausalResult {
    double cut_level_power;
    int normal_plan_survives;
    int maximum_pv_plan_survives;
    unsigned int survival_checks;
    unsigned int binary_search_iterations;
} SunoptCausalResult;


typedef struct SunoptNoSunshineTankParameters {
    size_t horizon_steps;
    size_t protected_steps;

    double time_step_hours;

    double maximum_heat_pump_electrical_power;
    double cop;

    double initial_waterline_power;

    double target_storage_energy;

    double storage_energy_per_kelvin;
    double storage_conductance_w_per_kelvin;
    double storage_min_temperature;
    double ambient_temperature;
    double daily_heat_loss;
} SunoptNoSunshineTankParameters;

typedef struct SunoptCudaProfile {
    uint64_t causal_calls;
    uint64_t causal_jobs;

    double parameter_h2d_ms;
    double kernel_ms;
    double result_d2h_ms;
    double gpu_total_ms;

    double host_wait_ms;
    double wall_ms;

    uint64_t profiled_search_jobs;

    uint64_t energy_phase_cycles;
    uint64_t boundary_phase_cycles;
    uint64_t radix_round_0_cycles;
    uint64_t radix_round_1_cycles;
    uint64_t radix_round_2_cycles;
    uint64_t radix_round_3_cycles;

    uint64_t boundary_storage_steps;
    uint64_t radix_round_0_storage_steps;
    uint64_t radix_round_1_storage_steps;
    uint64_t radix_round_2_storage_steps;
    uint64_t radix_round_3_storage_steps;
} SunoptCudaProfile;

SUNOPT_CUDA_EXPORT int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_destroy(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_reserve(
    SunoptCudaContext* context,
    size_t input_bytes,
    size_t workspace_bytes,
    size_t result_bytes
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_capacities(
    const SunoptCudaContext* context,
    size_t* out_input_bytes,
    size_t* out_workspace_bytes,
    size_t* out_result_bytes
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_upload_causal_arrays(
    SunoptCudaContext* context,
    const double* export_power,
    const double* demand_energy,
    size_t length
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal(
    SunoptCudaContext* context,
    const SunoptCausalParameters* parameters,
    SunoptCausalResult* host_result
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_batch(
    SunoptCudaContext* context,
    const SunoptCausalParameters* host_parameters,
    size_t job_count,
    SunoptCausalResult* host_results
);


SUNOPT_CUDA_EXPORT int sunopt_cuda_configure_causal_static_parameters(
    SunoptCudaContext* context,
    const SunoptCausalParameters* host_parameters,
    size_t job_count
);


SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_dynamic_batch(
    SunoptCudaContext* context,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
);


SUNOPT_CUDA_EXPORT int sunopt_cuda_precompute_no_sunshine_waterlines(
    SunoptCudaContext* context,
    const SunoptNoSunshineTankParameters* host_parameters,
    size_t tank_count,
    double* host_waterlines
);

// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_BEGIN
/*
 * Benchmark-only annual CUDA graph replay.
 *
 * The recorded Blackwell snapshot already contains the exact dynamic
 * inputs for every annual step. This API bakes those 35,040 dynamic
 * batches into one CUDA graph so we can measure the dispatch/host-wait
 * cost independently of the causal mathematics.
 *
 * Production lockstep is intentionally unchanged.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_graph_replay(
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
);
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_END


// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_BEGIN
/*
 * Benchmark-only B3 piecewise-affine cut solver.
 *
 * tank_parameters: 6 doubles per tank
 *   0 target_storage_energy
 *   1 maximum_heat_pump_electrical_power
 *   2 hp_thermal_energy_per_electrical_power (= COP * dt_hours)
 *   3 heat_loss_retention
 *   4 heat_loss_intercept
 *   5 affine_supported (1.0 / 0.0)
 *
 * dynamic_values: 2 doubles per job
 *   0 current_storage_energy
 *   1 minimum_reserve_energy
 *
 * energy_cut_levels: 1 double per job
 *
 * status per job:
 *   0 normal energy-cut survives
 *   1 maximum-PV cut=0 still fails
 *   2 corrected cut found
 *   3 exact max-affine branch unsupported
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_piecewise_affine_b3_benchmark(
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
);
// SUNOPT_BLACKWELL_PIECEWISE_AFFINE_B3_END


// SUNOPT_BLACKWELL_B14C_WARP32_BEGIN
/*
 * Research-only exact continuous piecewise-affine envelope prototype.
 *
 * One 32-thread CUDA block (one warp) owns one causal job.  The current
 * correctness-first kernel performs the ordered segment algebra in lane 0
 * using fixed shared-memory scratch and reports any envelope that exceeds
 * the intended 32-segment representation instead of truncating it.
 *
 * tank_parameters: 6 doubles per tank, identical to B3
 * dynamic_values: 2 doubles per job, identical to B3
 * energy_cut_levels: 1 double per job
 *
 * status per job:
 *   0 normal energy-cut survives
 *   1 maximum-PV cut=0 still fails
 *   2 corrected cut found
 *   3 unsupported affine branch
 *   4 exact envelope exceeded 32 final segments
 *   5 internal fixed scratch capacity exceeded
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14c_warp32_benchmark(
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
);
// SUNOPT_BLACKWELL_B14C_WARP32_END



// SUNOPT_BLACKWELL_B14D_SORTED_MERGE_BEGIN
/*
 * Research-only B14D benchmark.  Same exact envelope/status semantics as
 * B14C, but PWA addition merges the two already ordered segment lists in
 * O(n) rather than rebuilding/sorting boundaries and searching segments.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14d_warp32_merge_benchmark(
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
);
// SUNOPT_BLACKWELL_B14D_SORTED_MERGE_END



// SUNOPT_BLACKWELL_B14E_WARP_PARALLEL_BEGIN
/*
 * Research-only B14E benchmark. Same exact B14D envelope/status semantics,
 * with warp-parallel affine updates, feasibility reduction, charge addition,
 * and capacity clipping. Operations fall back to B14D/B14C serial helpers
 * when an intermediate list exceeds one warp rather than truncating it.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14e_warp_parallel_benchmark(
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
);
// SUNOPT_BLACKWELL_B14E_WARP_PARALLEL_END


// SUNOPT_BLACKWELL_B14F_WARP_PRECHECK_BEGIN
/*
 * Research-only B14F benchmark. B14E exact PWA envelope with the two
 * fixed-cut prechecks replaced by a 32-lane ordered max-affine reduction.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14f_warp_precheck_benchmark(
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
);
// SUNOPT_BLACKWELL_B14F_WARP_PRECHECK_END


// SUNOPT_BLACKWELL_B14G_WARP_BINARY_BEGIN
/*
 * Research-only B14G benchmark. Production-equivalent 20-step lower-bound
 * binary search, with every fixed-cut survival check evaluated by B14F's
 * exact ordered max-affine warp reduction. No PWA envelope is constructed.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14g_warp_binary_benchmark(
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
);
// SUNOPT_BLACKWELL_B14G_WARP_BINARY_END

// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark(
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
);
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_END

// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14j_small_batch_benchmark(
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
);
// SUNOPT_BLACKWELL_B14J_SMALL_BATCH_END

// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14k_speculative_tree_benchmark(
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
    uint32_t tree_depth,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_dynamic_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14K_SPECULATIVE_TREE_END

// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14l_packed_transport_benchmark(
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
    uint32_t pinned_host_memory,
    double* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_host_pack_ms,
    double* out_input_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14L_PACKED_TRANSPORT_END

// SUNOPT_BLACKWELL_B14M_FP32_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14m_fp32_benchmark(
    int device_id,
    const float* host_export_power,
    const float* host_demand_energy,
    size_t data_length,
    const float* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const float* host_dynamic_values,
    const float* host_energy_cut_levels,
    size_t step_count,
    float* host_cut_results,
    uint32_t* host_status,
    double* out_static_h2d_ms,
    double* out_input_enqueue_ms,
    double* out_kernel_ms,
    double* out_result_wait_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14M_FP32_END

// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN
/*
 * Benchmark-only exact max-affine survival tree.
 *
 * tank_parameters: 6 doubles per tank
 *   0 target_storage_energy
 *   1 maximum_heat_pump_electrical_power
 *   2 hp_thermal_energy_per_electrical_power (= COP * dt_hours)
 *   3 heat_loss_retention
 *   4 heat_loss_intercept
 *   5 affine_supported (1.0 / 0.0)
 *
 * dynamic_values: 2 doubles per job
 *   0 current_storage_energy
 *   1 minimum_reserve_energy
 *
 * cut_levels: 2 doubles per job
 *   0 normal energy-cut level
 *   1 maximum-PV cut level (= 0)
 *
 * host_survival: 2 uint32 values per job
 *   0 = does not survive
 *   1 = survives
 *   2 = unsupported exact max-affine branch
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_max_affine_tree_benchmark(
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
);
// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_END


// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN
/*
 * Research-only ideal digital-HP breakpoint solver.
 *
 * tank_parameters: 7 doubles/tank
 *   0 target_storage_energy
 *   1 maximum_heat_pump_electrical_power
 *   2 hp_thermal_energy_per_on_bucket (= Pmax * COP * dt)
 *   3 heat_loss_retention
 *   4 heat_loss_intercept
 *   5 affine_supported (1/0)
 *   6 hp_electrical_energy_per_on_bucket (= Pmax * dt)
 *
 * dynamic_values: 3 doubles/job
 *   0 current_storage_energy
 *   1 minimum_reserve_energy
 *   2 requested_electrical_energy
 *
 * status:
 *   0 = normal digital energy cut survives
 *   1 = even maximum-PV digital plan fails
 *   2 = correction breakpoint selected
 *   3 = unsupported exact max-affine branch
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_hp_breakpoints(
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
);
// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_END


// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN
/*
 * Production-shaped direct lockstep benchmark for the ideal digital HP.
 *
 * Static annual arrays and tank parameters are uploaded once.
 * Every causal step then performs:
 *   dynamic H2D -> 28-block digital breakpoint kernel -> result D2H -> sync.
 *
 * This deliberately models the host/GPU boundary of the real annual
 * simulation rather than batching thousands of independent recorded jobs.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_candidate(
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
);
// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_END


// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_graph_candidate(
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
);
// SUNOPT_BLACKWELL_DIGITAL_GRAPH_AB_END


// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_presorted_candidate(
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
);
// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_END


// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_presorted_warp_candidate(
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
);
// SUNOPT_BLACKWELL_DIGITAL_WARP_TREE_AB_END


// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_lockstep_piecewise_zero_candidate(
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
);
// SUNOPT_BLACKWELL_DIGITAL_PIECEWISE_ZERO_AB_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_resident_chunk_replay(
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
);
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_CHUNK_END


// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_resident_job_batch(
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
);
// SUNOPT_BLACKWELL_DIGITAL_RESIDENT_JOB_BATCH_END


// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_multilane_lockstep(
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
    uint32_t lane_count,
    uint32_t repetitions,
    double* host_cut_results,
    uint32_t* host_status,
    uint32_t* host_survival_evaluations,
    double* out_input_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_DIGITAL_MULTILANE_SEARCH_END


// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_digital_cluster4_lockstep(
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
);
// SUNOPT_BLACKWELL_DIGITAL_CLUSTER4_END


SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile(
    const SunoptCudaContext* context,
    SunoptCudaProfile* out_profile
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_reset_profile(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT const char* sunopt_cuda_error_string(
    int error_code
);

#ifdef __cplusplus
}
#endif
