// rust/sunopt-node/research/b12_phase3c_cross_parity_extension.rs
//
// B12 Phase 3C:
// Run the CURRENT B8 resident digital solver on the synchronized Blackwell
// snapshot and compare every returned cut level against the waterlines recorded
// by the SAME B12 annual production run.
//
// Critical comparison:
//   B8 cut_results[job] vs physical.userForecastWaterlinePower
//
// Diagnostic comparisons:
//   B8 cut_results[job] vs physical.noSunshineWaterlinePower
//   B8 cut_results[job] vs physical.effectiveWaterlinePower
//
// Research only. Production source is not modified.

const B12_PHY_MAGIC:
    &str =
    "SUNOPT_B12_PHY1";

const B12_PHY_VERSION:
    u32 =
    1;

const B12_PHY_FIELD_COUNT:
    usize =
    24;

const B12_PHY_FIELD_USER_FORECAST_WATERLINE:
    usize =
    10;

const B12_PHY_FIELD_NO_SUNSHINE_WATERLINE:
    usize =
    11;

const B12_PHY_FIELD_EFFECTIVE_WATERLINE:
    usize =
    12;


#[derive(Default)]
struct B12CrossStats {
    mismatches_1e12:
        u64,

    mismatches_1e9:
        u64,

    maximum_abs_error:
        f64,

    maximum_error_job:
        usize,
}


impl B12CrossStats {

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
                job;
        }
    }
}


struct B12PhysicalReference {
    bytes:
        Vec<u8>,

    step_count:
        usize,

    tank_count:
        usize,

    field_count:
        usize,

    time_step_hours:
        f64,

    data_offset:
        usize,
}


impl B12PhysicalReference {

    fn read_u32(
        bytes:
            &[u8],

        offset:
            usize,
    ) -> std::result::Result<u32, String> {

        let slice =
            bytes
                .get(
                    offset..
                    offset + 4,
                )
                .ok_or_else(
                    || {
                        format!(
                            "B12 physical header truncated at u32 offset {offset}.",
                        )
                    },
                )?;


        Ok(
            u32::from_le_bytes(
                slice
                    .try_into()
                    .map_err(
                        |_| {
                            format!(
                                "Invalid B12 physical u32 at offset {offset}.",
                            )
                        },
                    )?,
            ),
        )
    }


    fn read_f64_raw(
        bytes:
            &[u8],

        offset:
            usize,
    ) -> std::result::Result<f64, String> {

        let slice =
            bytes
                .get(
                    offset..
                    offset + 8,
                )
                .ok_or_else(
                    || {
                        format!(
                            "B12 physical file truncated at f64 offset {offset}.",
                        )
                    },
                )?;


        Ok(
            f64::from_le_bytes(
                slice
                    .try_into()
                    .map_err(
                        |_| {
                            format!(
                                "Invalid B12 physical f64 at offset {offset}.",
                            )
                        },
                    )?,
            ),
        )
    }


    fn open(
        path:
            &std::path::Path,
    ) -> std::result::Result<Self, String> {

        let bytes =
            std::fs::read(
                path,
            )
            .map_err(
                |error| {
                    format!(
                        "Unable to read B12 physical reference {}: {error}",
                        path.display(),
                    )
                },
            )?;


        if bytes.len() <
            40
        {
            return Err(
                "B12 physical reference is smaller than its fixed header."
                    .to_string(),
            );
        }


        let magic =
            std::str::from_utf8(
                &bytes[0..15],
            )
            .map_err(
                |_| {
                    "B12 physical magic is not UTF-8."
                        .to_string()
                },
            )?;


        if magic !=
            B12_PHY_MAGIC
        {
            return Err(
                format!(
                    "Unexpected B12 physical magic {magic:?}.",
                ),
            );
        }


        let version =
            Self::read_u32(
                &bytes,
                16,
            )?;


        if version !=
            B12_PHY_VERSION
        {
            return Err(
                format!(
                    "Unsupported B12 physical version {version}.",
                ),
            );
        }


        let step_count =
            Self::read_u32(
                &bytes,
                20,
            )?
            as usize;

        let tank_count =
            Self::read_u32(
                &bytes,
                24,
            )?
            as usize;

        let field_count =
            Self::read_u32(
                &bytes,
                28,
            )?
            as usize;

        let time_step_hours =
            Self::read_f64_raw(
                &bytes,
                32,
            )?;


        if step_count == 0 ||
            tank_count == 0
        {
            return Err(
                "B12 physical reference is empty."
                    .to_string(),
            );
        }


        if field_count !=
            B12_PHY_FIELD_COUNT
        {
            return Err(
                format!(
                    "B12 physical field count is {field_count}; expected {B12_PHY_FIELD_COUNT}.",
                ),
            );
        }


        let data_offset =
            40 +
            tank_count *
            8;


        let expected_bytes =
            data_offset +
            tank_count *
            step_count *
            field_count *
            8;


        if expected_bytes !=
            bytes.len()
        {
            return Err(
                format!(
                    "B12 physical size mismatch: expected {expected_bytes}, got {}.",
                    bytes.len(),
                ),
            );
        }


        Ok(
            Self {
                bytes,
                step_count,
                tank_count,
                field_count,
                time_step_hours,
                data_offset,
            },
        )
    }


    fn value(
        &self,

        tank:
            usize,

        step:
            usize,

        field:
            usize,
    ) -> f64 {

        /*
         * Physical reference is TANK-MAJOR:
         *
         *   [tank][step][field]
         */
        let flat_index =
            (
                (
                    tank *
                    self.step_count +
                    step
                ) *
                self.field_count +
                field
            );

        let offset =
            self.data_offset +
            flat_index *
            8;


        f64::from_le_bytes(
            self.bytes[
                offset..
                offset +
                    8
            ]
            .try_into()
            .expect(
                "validated B12 physical bounds",
            ),
        )
    }
}


fn emit_cross_stats(
    name:
        &str,

    stats:
        &B12CrossStats,
) {

    print!(
        "\"{}Mismatches1e12\":{},",
        name,
        stats.mismatches_1e12,
    );

    print!(
        "\"{}Mismatches1e9\":{},",
        name,
        stats.mismatches_1e9,
    );

    print!(
        "\"{}MaxAbsError\":{:.17e},",
        name,
        stats.maximum_abs_error,
    );

    print!(
        "\"{}MaxErrorJob\":{}",
        name,
        stats.maximum_error_job,
    );
}


fn b12_fnv1a_f64(
    values:
        &[f64],
) -> u64 {

    let mut checksum =
        1469598103934665603u64;


    for value in
        values
    {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }


    checksum
}


pub fn run_b12_phase3c_cross_parity(
    snapshot_path:
        &str,

    physical_path:
        &str,

    device_id:
        i32,

    chunk_steps:
        u32,
) -> std::result::Result<(), String> {

    let snapshot_path_buf =
        std::path::PathBuf::from(
            snapshot_path,
        );


    let snapshot =
        read_blackwell_snapshot(
            &snapshot_path_buf,
        )
        .map_err(
            |error| {
                format!(
                    "Unable to read synchronized Blackwell snapshot {}: {error}",
                    snapshot_path_buf.display(),
                )
            },
        )?;


    let physical =
        B12PhysicalReference::open(
            std::path::Path::new(
                physical_path,
            ),
        )?;


    if snapshot.steps.is_empty() ||
        snapshot.tank_count == 0 ||
        snapshot.export_power.is_empty() ||
        snapshot.demand_energy.len() !=
            snapshot.export_power.len() ||
        snapshot.steps.len() !=
            snapshot.data_length
    {
        return Err(
            "Synchronized Blackwell snapshot is empty or inconsistent."
                .to_string(),
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
                    "B12 Phase-3C job count overflow."
                        .to_string()
                },
            )?;


    if physical.step_count !=
            step_count ||
        physical.tank_count !=
            tank_count
    {
        return Err(
            format!(
                "B12 synchronized/physical shape mismatch: snapshot={}x{}, physical={}x{}.",
                step_count,
                tank_count,
                physical.step_count,
                physical.tank_count,
            ),
        );
    }


    if (
        physical.time_step_hours -
        0.25
    )
    .abs() >
        1.0e-15
    {
        return Err(
            format!(
                "Unexpected B12 physical timestep {}.",
                physical.time_step_hours,
            ),
        );
    }


    if chunk_steps == 0 {
        return Err(
            "chunk_steps must be greater than zero."
                .to_string(),
        );
    }


    let tanks =
        (
            0..
            tank_count
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
            tank_count *
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
        if step.horizon_steps == 0 ||
            step.horizon_steps >
                256
        {
            return Err(
                format!(
                    "B12 Phase-3C supports horizon 1..=256; got {}.",
                    step.horizon_steps,
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
                    base +
                    0
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base +
                    1
                ],
            );

            dynamic_flat.push(
                step.dynamic[
                    base +
                    2
                ],
            );
        }
    }


    let presorted_count =
        step_count
            .checked_mul(
                256,
            )
            .ok_or_else(
                || {
                    "B12 Phase-3C presorted matrix size overflow."
                        .to_string()
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
        snapshot
            .steps
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
            base +
            256
        ]
        .sort_by(
            |a, b| {
                a.partial_cmp(
                    b,
                )
                .unwrap_or(
                    std::cmp::Ordering::Equal,
                )
            },
        );
    }


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
                device_id,
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
            format!(
                "sunopt_cuda_run_digital_resident_chunk_replay failed with CUDA error code {code}.",
            ),
        );
    }


    let mut user_stats =
        B12CrossStats::default();

    let mut no_sunshine_stats =
        B12CrossStats::default();

    let mut effective_stats =
        B12CrossStats::default();


    let mut non_finite_cuts =
        0_u64;

    let mut non_finite_physical =
        0_u64;


    /*
     * B8 result layout is STEP-MAJOR:
     *
     *   job = step * tank_count + tank
     *
     * Physical reference is tank-major, so map explicitly.
     */
    for step in
        0..
        step_count
    {
        for tank in
            0..
            tank_count
        {
            let job =
                step *
                tank_count +
                tank;

            let cut =
                cuts[
                    job
                ];

            let user_waterline =
                physical.value(
                    tank,
                    step,
                    B12_PHY_FIELD_USER_FORECAST_WATERLINE,
                );

            let no_sunshine_waterline =
                physical.value(
                    tank,
                    step,
                    B12_PHY_FIELD_NO_SUNSHINE_WATERLINE,
                );

            let effective_waterline =
                physical.value(
                    tank,
                    step,
                    B12_PHY_FIELD_EFFECTIVE_WATERLINE,
                );


            if !cut.is_finite() {
                non_finite_cuts +=
                    1;

                continue;
            }


            if !user_waterline.is_finite() ||
                !no_sunshine_waterline.is_finite() ||
                !effective_waterline.is_finite()
            {
                non_finite_physical +=
                    1;

                continue;
            }


            user_stats.add(
                job,
                cut,
                user_waterline,
            );

            no_sunshine_stats.add(
                job,
                cut,
                no_sunshine_waterline,
            );

            effective_stats.add(
                job,
                cut,
                effective_waterline,
            );
        }
    }


    let checksum =
        b12_fnv1a_f64(
            &cuts,
        );


    println!(
        "SUNOPT_B12_PHASE3C_CROSS_PARITY_JSON={{\"stepCount\":{},\"tankCount\":{},\"jobs\":{},\"chunkSteps\":{},\"searchIterations\":8,\"inputH2DMs\":{:.17e},\"kernelMs\":{:.17e},\"resultD2HMs\":{:.17e},\"wallMs\":{:.17e},\"checksumHex\":\"{:016x}\",\"nonFiniteCuts\":{},\"nonFinitePhysical\":{},",
        step_count,
        tank_count,
        job_count,
        chunk_steps,
        input_h2d_ms,
        kernel_ms,
        result_d2h_ms,
        wall_ms,
        checksum,
        non_finite_cuts,
        non_finite_physical,
    );

    emit_cross_stats(
        "cutVsUserForecast",
        &user_stats,
    );

    print!(
        ","
    );

    emit_cross_stats(
        "cutVsNoSunshine",
        &no_sunshine_stats,
    );

    print!(
        ","
    );

    emit_cross_stats(
        "cutVsEffective",
        &effective_stats,
    );

    println!(
        "}}"
    );


    println!();
    println!(
        "--- B12 PHASE 3C CROSS PARITY ---"
    );

    println!(
        "steps={step_count}"
    );

    println!(
        "tanks={tank_count}"
    );

    println!(
        "jobs={job_count}"
    );

    println!(
        "chunkSteps={chunk_steps}"
    );

    println!(
        "inputH2DMs={:.3}",
        input_h2d_ms,
    );

    println!(
        "kernelMs={:.3}",
        kernel_ms,
    );

    println!(
        "resultD2HMs={:.3}",
        result_d2h_ms,
    );

    println!(
        "wallMs={:.3}",
        wall_ms,
    );

    println!(
        "checksum={:016x}",
        checksum,
    );

    println!(
        "cut vs userForecast mismatches1e-12={} mismatches1e-9={} maxError={:.6e} maxJob={}",
        user_stats.mismatches_1e12,
        user_stats.mismatches_1e9,
        user_stats.maximum_abs_error,
        user_stats.maximum_error_job,
    );

    println!(
        "cut vs noSunshine mismatches1e-12={} mismatches1e-9={} maxError={:.6e} maxJob={}",
        no_sunshine_stats.mismatches_1e12,
        no_sunshine_stats.mismatches_1e9,
        no_sunshine_stats.maximum_abs_error,
        no_sunshine_stats.maximum_error_job,
    );

    println!(
        "cut vs effective mismatches1e-12={} mismatches1e-9={} maxError={:.6e} maxJob={}",
        effective_stats.mismatches_1e12,
        effective_stats.mismatches_1e9,
        effective_stats.maximum_abs_error,
        effective_stats.maximum_error_job,
    );

    println!(
        "nonFiniteCuts={non_finite_cuts}"
    );

    println!(
        "nonFinitePhysical={non_finite_physical}"
    );


    Ok(())
}

