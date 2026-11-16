// rust/sunopt-node/src/b12_phase3c.rs
//
// B12 Phase 3C — B side.
//
// This module is deliberately parameter-only.
//
// A-side responsibilities:
//   - read files / domain objects
//   - map current SunOpt representation to the fixed B ABI
//   - build the presorted export matrix
//
// B-side responsibilities:
//   - validate flat numeric buffers
//   - call the resident CUDA solver
//   - compare returned cut levels against flat reference buffers
//
// No ThermalStorage, HeatPump, controller objects, TypeScript source,
// or file paths cross this boundary.

use napi::bindgen_prelude::{
    Error,
    Float64Array,
    Result,
    Status,
};

use napi_derive::napi;

use std::ffi::c_int;


const TANK_STRIDE:
    usize =
    7;

const DYNAMIC_STRIDE:
    usize =
    3;

const PRESORT_STRIDE:
    usize =
    256;


unsafe extern "C" {

    fn sunopt_cuda_run_digital_resident_chunk_replay(
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

        host_presorted_export_windows:
            *const f64,

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

        chunk_steps:
            u32,

        repetitions:
            u32,

        host_cut_results:
            *mut f64,

        host_status:
            *mut u32,

        host_survival_evaluations:
            *mut u32,

        out_input_h2d_ms:
            *mut f64,

        out_kernel_ms:
            *mut f64,

        out_result_d2h_ms:
            *mut f64,

        out_wall_ms:
            *mut f64,
    ) -> c_int;
}


#[derive(Default)]
struct CrossStats {

    mismatches_1e12:
        u32,

    mismatches_1e9:
        u32,

    maximum_abs_error:
        f64,

    maximum_error_job:
        u32,
}


impl CrossStats {

    fn add(
        &mut self,

        job:
            usize,

        actual:
            f64,

        expected:
            f64,
    ) {

        let error =
            (
                actual -
                expected
            )
            .abs();


        if error >
            1.0e-12
        {
            self.mismatches_1e12 +=
                1;
        }


        if error >
            1.0e-9
        {
            self.mismatches_1e9 +=
                1;
        }


        if error >
            self.maximum_abs_error
        {
            self.maximum_abs_error =
                error;

            self.maximum_error_job =
                job as u32;
        }
    }
}


#[napi(object)]
pub struct B12Phase3cCrossParityBResult {

    #[napi(js_name = "annualStepCount")]
    pub annual_step_count:
        u32,

    #[napi(js_name = "tankCount")]
    pub tank_count:
        u32,

    #[napi(js_name = "causalJobs")]
    pub causal_jobs:
        u32,

    #[napi(js_name = "searchIterations")]
    pub search_iterations:
        u32,

    #[napi(js_name = "chunkSteps")]
    pub chunk_steps:
        u32,

    #[napi(js_name = "presortedMiB")]
    pub presorted_mib:
        f64,

    #[napi(js_name = "inputH2DMs")]
    pub input_h2d_ms:
        f64,

    #[napi(js_name = "kernelMs")]
    pub kernel_ms:
        f64,

    #[napi(js_name = "resultD2HMs")]
    pub result_d2h_ms:
        f64,

    #[napi(js_name = "wallMs")]
    pub wall_ms:
        f64,

    #[napi(js_name = "checksumHex")]
    pub checksum_hex:
        String,

    #[napi(js_name = "nonFiniteCuts")]
    pub non_finite_cuts:
        u32,

    #[napi(js_name = "nonFinitePhysical")]
    pub non_finite_physical:
        u32,

    #[napi(js_name = "cutVsUserForecastMismatches1e12")]
    pub cut_vs_user_forecast_mismatches_1e12:
        u32,

    #[napi(js_name = "cutVsUserForecastMismatches1e9")]
    pub cut_vs_user_forecast_mismatches_1e9:
        u32,

    #[napi(js_name = "cutVsUserForecastMaxAbsError")]
    pub cut_vs_user_forecast_max_abs_error:
        f64,

    #[napi(js_name = "cutVsUserForecastMaxErrorJob")]
    pub cut_vs_user_forecast_max_error_job:
        u32,

    #[napi(js_name = "cutVsNoSunshineMismatches1e12")]
    pub cut_vs_no_sunshine_mismatches_1e12:
        u32,

    #[napi(js_name = "cutVsNoSunshineMismatches1e9")]
    pub cut_vs_no_sunshine_mismatches_1e9:
        u32,

    #[napi(js_name = "cutVsNoSunshineMaxAbsError")]
    pub cut_vs_no_sunshine_max_abs_error:
        f64,

    #[napi(js_name = "cutVsNoSunshineMaxErrorJob")]
    pub cut_vs_no_sunshine_max_error_job:
        u32,

    #[napi(js_name = "cutVsEffectiveMismatches1e12")]
    pub cut_vs_effective_mismatches_1e12:
        u32,

    #[napi(js_name = "cutVsEffectiveMismatches1e9")]
    pub cut_vs_effective_mismatches_1e9:
        u32,

    #[napi(js_name = "cutVsEffectiveMaxAbsError")]
    pub cut_vs_effective_max_abs_error:
        f64,

    #[napi(js_name = "cutVsEffectiveMaxErrorJob")]
    pub cut_vs_effective_max_error_job:
        u32,

    #[napi(js_name = "phase3dSummary")]
    pub phase3d_summary:
        String,

    #[napi(js_name = "phase3dTankMismatchCounts")]
    pub phase3d_tank_mismatch_counts:
        String,

    #[napi(js_name = "phase3dTopMismatches")]
    pub phase3d_top_mismatches:
        String,
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


fn cuda_error(
    function_name:
        &str,

    code:
        c_int,
) -> Error {

    Error::new(
        Status::GenericFailure,
        format!(
            "{function_name} failed with CUDA error code {code}.",
        ),
    )
}


fn f64_to_u32_array(
    values:
        &Float64Array,

    label:
        &str,
) -> Result<Vec<u32>> {

    let mut converted =
        Vec::<u32>::with_capacity(
            values.len(),
        );


    for (
        index,
        value,
    ) in
        values
            .iter()
            .copied()
            .enumerate()
    {
        if (
            !value.is_finite() ||
            value <
                0.0 ||
            value >
                u32::MAX as f64 ||
            value.fract() !=
                0.0
        ) {
            return Err(
                invalid_argument(
                    format!(
                        "{label}[{index}]={value} is not an exact u32.",
                    ),
                ),
            );
        }


        converted.push(
            value as u32,
        );
    }


    Ok(
        converted,
    )
}


fn fnv1a_f64(
    values:
        &[f64],
) -> u64 {

    let mut checksum =
        1469598103934665603u64;


    for value in
        values
    {
        for byte in
            value
                .to_bits()
                .to_le_bytes()
        {
            checksum ^=
                byte as u64;

            checksum =
                checksum
                    .wrapping_mul(
                        1099511628211,
                    );
        }
    }


    checksum
}


/*
 * Stable B-side research entry point.
 *
 * Every input is either:
 *   - a flat numeric buffer, or
 *   - a primitive scalar option.
 *
 * The function intentionally does not know anything about files,
 * TypeScript objects, or SunOpt domain object types.
 */
#[napi(js_name = "runB12Phase3cCrossParityB")]
pub fn run_b12_phase3c_cross_parity_b(
    export_power:
        Float64Array,

    demand_energy:
        Float64Array,

    tank_parameters:
        Float64Array,

    presorted_export_windows:
        Float64Array,

    current_indices:
        Float64Array,

    horizon_steps:
        Float64Array,

    dynamic_values:
        Float64Array,

    user_forecast_waterline:
        Float64Array,

    no_sunshine_waterline:
        Float64Array,

    effective_waterline:
        Float64Array,

    search_iterations:
        Option<u32>,

    chunk_steps:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<
    B12Phase3cCrossParityBResult
> {

    let data_length =
        export_power.len();


    if data_length ==
        0
    {
        return Err(
            invalid_argument(
                "exportPower must not be empty.",
            ),
        );
    }


    if demand_energy.len() !=
        data_length
    {
        return Err(
            invalid_argument(
                format!(
                    "demandEnergy length {} does not match exportPower length {data_length}.",
                    demand_energy.len(),
                ),
            ),
        );
    }


    if (
        tank_parameters.is_empty() ||
        tank_parameters.len() %
            TANK_STRIDE !=
            0
    ) {
        return Err(
            invalid_argument(
                "tankParameters length must be a non-zero multiple of 7.",
            ),
        );
    }


    let tank_count =
        tank_parameters.len() /
        TANK_STRIDE;


    if tank_count >
        32
    {
        return Err(
            invalid_argument(
                format!(
                    "B resident solver supports at most 32 tanks; got {tank_count}.",
                ),
            ),
        );
    }


    let step_count =
        current_indices.len();


    if step_count ==
        0
    {
        return Err(
            invalid_argument(
                "currentIndices must not be empty.",
            ),
        );
    }


    if step_count !=
        data_length
    {
        return Err(
            invalid_argument(
                format!(
                    "B12 annual cross-parity requires stepCount == dataLength; got {step_count} vs {data_length}.",
                ),
            ),
        );
    }


    if horizon_steps.len() !=
        step_count
    {
        return Err(
            invalid_argument(
                format!(
                    "horizonSteps length {} does not match stepCount {step_count}.",
                    horizon_steps.len(),
                ),
            ),
        );
    }


    let job_count =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    invalid_argument(
                        "B12 Phase-3C job count overflow.",
                    )
                },
            )?;


    let expected_dynamic =
        job_count
            .checked_mul(
                DYNAMIC_STRIDE,
            )
            .ok_or_else(
                || {
                    invalid_argument(
                        "B12 Phase-3C dynamic buffer size overflow.",
                    )
                },
            )?;


    if dynamic_values.len() !=
        expected_dynamic
    {
        return Err(
            invalid_argument(
                format!(
                    "dynamicValues length {} does not match expected {expected_dynamic}.",
                    dynamic_values.len(),
                ),
            ),
        );
    }


    let expected_presorted =
        step_count
            .checked_mul(
                PRESORT_STRIDE,
            )
            .ok_or_else(
                || {
                    invalid_argument(
                        "B12 Phase-3C presorted buffer size overflow.",
                    )
                },
            )?;


    if presorted_export_windows.len() !=
        expected_presorted
    {
        return Err(
            invalid_argument(
                format!(
                    "presortedExportWindows length {} does not match expected {expected_presorted}.",
                    presorted_export_windows.len(),
                ),
            ),
        );
    }


    for (
        label,
        length,
    ) in [
        (
            "userForecastWaterline",
            user_forecast_waterline.len(),
        ),
        (
            "noSunshineWaterline",
            no_sunshine_waterline.len(),
        ),
        (
            "effectiveWaterline",
            effective_waterline.len(),
        ),
    ] {
        if length !=
            job_count
        {
            return Err(
                invalid_argument(
                    format!(
                        "{label} length {length} does not match jobCount {job_count}.",
                    ),
                ),
            );
        }
    }


    let current_indices_u32 =
        f64_to_u32_array(
            &current_indices,
            "currentIndices",
        )?;


    let horizon_steps_u32 =
        f64_to_u32_array(
            &horizon_steps,
            "horizonSteps",
        )?;


    for step in
        0..
        step_count
    {
        if current_indices_u32[
            step
        ] as usize >=
            data_length
        {
            return Err(
                invalid_argument(
                    format!(
                        "currentIndices[{step}]={} is outside dataLength {data_length}.",
                        current_indices_u32[
                            step
                        ],
                    ),
                ),
            );
        }


        let horizon =
            horizon_steps_u32[
                step
            ];


        if horizon ==
                0 ||
            horizon >
                256
        {
            return Err(
                invalid_argument(
                    format!(
                        "horizonSteps[{step}]={horizon}; B resident solver supports 1..=256.",
                    ),
                ),
            );
        }
    }


    let search_iterations =
        search_iterations
            .unwrap_or(
                8,
            );


    if search_iterations ==
        0
    {
        return Err(
            invalid_argument(
                "searchIterations must be greater than zero.",
            ),
        );
    }


    let chunk_steps =
        chunk_steps
            .unwrap_or(
                4096,
            );


    if chunk_steps ==
        0
    {
        return Err(
            invalid_argument(
                "chunkSteps must be greater than zero.",
            ),
        );
    }


    let selected_device =
        device_id
            .unwrap_or(
                0,
            );


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
            sunopt_cuda_run_digital_resident_chunk_replay(
                selected_device,
                export_power.as_ptr(),
                demand_energy.as_ptr(),
                data_length,
                tank_parameters.as_ptr(),
                tank_count,
                presorted_export_windows.as_ptr(),
                current_indices_u32.as_ptr(),
                horizon_steps_u32.as_ptr(),
                dynamic_values.as_ptr(),
                step_count,
                search_iterations,
                chunk_steps,
                1,
                cuts.as_mut_ptr(),
                status.as_mut_ptr(),
                evaluations.as_mut_ptr(),
                &mut input_h2d_ms,
                &mut kernel_ms,
                &mut result_d2h_ms,
                &mut wall_ms,
            )
        };


    if code !=
        0
    {
        return Err(
            cuda_error(
                "sunopt_cuda_run_digital_resident_chunk_replay",
                code,
            ),
        );
    }


    let mut user_stats =
        CrossStats::default();

    let mut no_sunshine_stats =
        CrossStats::default();

    let mut effective_stats =
        CrossStats::default();


    let mut non_finite_cuts =
        0_u32;

    let mut non_finite_physical =
        0_u32;


    /*
     * Phase 3D mismatch classification.
     *
     * Critical comparison:
     *   B digital cut vs production userForecastWaterline.
     *
     * We deliberately classify here on the B side so the full cut array
     * does not cross N-API merely for diagnostics.
     */
    let mut phase3d_positive =
        0_u32;

    let mut phase3d_negative =
        0_u32;

    let mut phase3d_abs_le_001 =
        0_u32;

    let mut phase3d_abs_le_010 =
        0_u32;

    let mut phase3d_abs_le_025 =
        0_u32;

    let mut phase3d_abs_le_050 =
        0_u32;

    let mut phase3d_abs_le_100 =
        0_u32;

    let mut phase3d_abs_gt_100 =
        0_u32;

    let mut phase3d_near_quarter_pmax =
        0_u32;

    let mut phase3d_near_half_pmax =
        0_u32;

    let mut phase3d_near_three_quarter_pmax =
        0_u32;

    let mut phase3d_near_full_pmax =
        0_u32;

    let mut phase3d_tank_mismatches =
        vec![
            0_u32;
            tank_count
        ];

    let mut phase3d_top:
        Vec<(
            f64,
            usize,
            f64,
            f64,
            f64,
            f64,
            f64,
        )> =
        Vec::new();


    for job in
        0..
        job_count
    {
        let cut =
            cuts[
                job
            ];

        let user =
            user_forecast_waterline[
                job
            ];

        let no_sunshine =
            no_sunshine_waterline[
                job
            ];

        let effective =
            effective_waterline[
                job
            ];


        if !cut.is_finite() {
            non_finite_cuts +=
                1;

            continue;
        }


        if (
            !user.is_finite() ||
            !no_sunshine.is_finite() ||
            !effective.is_finite()
        ) {
            non_finite_physical +=
                1;

            continue;
        }


        user_stats.add(
            job,
            cut,
            user,
        );


        let delta =
            cut -
            user;

        let abs_delta =
            delta.abs();


        if abs_delta >
            1.0e-9
        {
            let tank =
                job %
                tank_count;

            phase3d_tank_mismatches[
                tank
            ] +=
                1;


            if delta >
                0.0
            {
                phase3d_positive +=
                    1;
            } else {
                phase3d_negative +=
                    1;
            }


            if abs_delta <=
                0.01
            {
                phase3d_abs_le_001 +=
                    1;
            } else if abs_delta <=
                0.10
            {
                phase3d_abs_le_010 +=
                    1;
            } else if abs_delta <=
                0.25
            {
                phase3d_abs_le_025 +=
                    1;
            } else if abs_delta <=
                0.50
            {
                phase3d_abs_le_050 +=
                    1;
            } else if abs_delta <=
                1.00
            {
                phase3d_abs_le_100 +=
                    1;
            } else {
                phase3d_abs_gt_100 +=
                    1;
            }


            let pmax =
                tank_parameters[
                    tank *
                    TANK_STRIDE +
                    1
                ];

            if pmax >
                0.0
            {
                let ratio =
                    abs_delta /
                    pmax;

                let ratio_tolerance =
                    1.0e-9;


                if (
                    ratio -
                    0.25
                )
                .abs() <=
                    ratio_tolerance
                {
                    phase3d_near_quarter_pmax +=
                        1;
                }


                if (
                    ratio -
                    0.50
                )
                .abs() <=
                    ratio_tolerance
                {
                    phase3d_near_half_pmax +=
                        1;
                }


                if (
                    ratio -
                    0.75
                )
                .abs() <=
                    ratio_tolerance
                {
                    phase3d_near_three_quarter_pmax +=
                        1;
                }


                if (
                    ratio -
                    1.00
                )
                .abs() <=
                    ratio_tolerance
                {
                    phase3d_near_full_pmax +=
                        1;
                }
            }


            let dynamic_base =
                job *
                DYNAMIC_STRIDE;

            let entry =
                (
                    abs_delta,
                    job,
                    delta,
                    cut,
                    user,
                    dynamic_values[
                        dynamic_base +
                        0
                    ],
                    dynamic_values[
                        dynamic_base +
                        1
                    ],
                );


            phase3d_top.push(
                entry,
            );
        }


        no_sunshine_stats.add(
            job,
            cut,
            no_sunshine,
        );

        effective_stats.add(
            job,
            cut,
            effective,
        );
    }


    let checksum =
        fnv1a_f64(
            &cuts,
        );


    phase3d_top.sort_by(
        |left, right| {
            right
                .0
                .partial_cmp(
                    &left.0,
                )
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
        },
    );

    phase3d_top.truncate(
        12,
    );


    let phase3d_summary =
        format!(
            "mismatches={};positive={};negative={};abs<=0.01={};0.01<abs<=0.10={};0.10<abs<=0.25={};0.25<abs<=0.50={};0.50<abs<=1.00={};abs>1.00={};near0.25Pmax={};near0.50Pmax={};near0.75Pmax={};near1.00Pmax={}",
            user_stats.mismatches_1e9,
            phase3d_positive,
            phase3d_negative,
            phase3d_abs_le_001,
            phase3d_abs_le_010,
            phase3d_abs_le_025,
            phase3d_abs_le_050,
            phase3d_abs_le_100,
            phase3d_abs_gt_100,
            phase3d_near_quarter_pmax,
            phase3d_near_half_pmax,
            phase3d_near_three_quarter_pmax,
            phase3d_near_full_pmax,
        );


    let phase3d_tank_mismatch_counts =
        phase3d_tank_mismatches
            .iter()
            .enumerate()
            .map(
                |(
                    tank,
                    count,
                )| {
                    format!(
                        "{}:{}",
                        tank,
                        count,
                    )
                },
            )
            .collect::<Vec<String>>()
            .join(
                ",",
            );


    let phase3d_top_mismatches =
        phase3d_top
            .iter()
            .map(
                |entry| {
                    let job =
                        entry.1;

                    let step =
                        job /
                        tank_count;

                    let tank =
                        job %
                        tank_count;

                    let dynamic_base =
                        job *
                        DYNAMIC_STRIDE;

                    format!(
                        "job={}|step={}|tank={}|delta={:.17e}|abs={:.17e}|bCut={:.17e}|production={:.17e}|currentStorage={:.17e}|minimumReserve={:.17e}|requestedElectrical={:.17e}|pmax={:.17e}",
                        job,
                        step,
                        tank,
                        entry.2,
                        entry.0,
                        entry.3,
                        entry.4,
                        entry.5,
                        entry.6,
                        dynamic_values[
                            dynamic_base +
                            2
                        ],
                        tank_parameters[
                            tank *
                            TANK_STRIDE +
                            1
                        ],
                    )
                },
            )
            .collect::<Vec<String>>()
            .join(
                "\n",
            );


    Ok(
        B12Phase3cCrossParityBResult {

            annual_step_count:
                step_count as u32,

            tank_count:
                tank_count as u32,

            causal_jobs:
                job_count as u32,

            search_iterations,

            chunk_steps,

            presorted_mib:
                (
                    presorted_export_windows.len() *
                    std::mem::size_of::<f64>()
                )
                as f64 /
                1048576.0,

            input_h2d_ms,

            kernel_ms,

            result_d2h_ms,

            wall_ms,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),

            non_finite_cuts,

            non_finite_physical,

            cut_vs_user_forecast_mismatches_1e12:
                user_stats
                    .mismatches_1e12,

            cut_vs_user_forecast_mismatches_1e9:
                user_stats
                    .mismatches_1e9,

            cut_vs_user_forecast_max_abs_error:
                user_stats
                    .maximum_abs_error,

            cut_vs_user_forecast_max_error_job:
                user_stats
                    .maximum_error_job,

            cut_vs_no_sunshine_mismatches_1e12:
                no_sunshine_stats
                    .mismatches_1e12,

            cut_vs_no_sunshine_mismatches_1e9:
                no_sunshine_stats
                    .mismatches_1e9,

            cut_vs_no_sunshine_max_abs_error:
                no_sunshine_stats
                    .maximum_abs_error,

            cut_vs_no_sunshine_max_error_job:
                no_sunshine_stats
                    .maximum_error_job,

            cut_vs_effective_mismatches_1e12:
                effective_stats
                    .mismatches_1e12,

            cut_vs_effective_mismatches_1e9:
                effective_stats
                    .mismatches_1e9,

            cut_vs_effective_max_abs_error:
                effective_stats
                    .maximum_abs_error,

            cut_vs_effective_max_error_job:
                effective_stats
                    .maximum_error_job,

            phase3d_summary,

            phase3d_tank_mismatch_counts,

            phase3d_top_mismatches,
        },
    )
}

