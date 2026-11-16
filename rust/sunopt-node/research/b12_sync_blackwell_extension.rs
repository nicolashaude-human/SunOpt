// rust/sunopt-node/research/b12_sync_blackwell_extension.rs
//
// B12 Phase 3B bridge:
// Convert the synchronized B12 coupled snapshot into the CURRENT Blackwell
// replay format by using BlackwellCapture from cuda_context.rs itself.
//
// This code is appended inside the same Rust module as a cleaned temporary
// copy of cuda_context.rs. Production source is never modified.

const B12_COUPLED_MAGIC: &str =
    "SUNOPT_B12_CPL1";

const B12_COUPLED_VERSION: u32 =
    1;

const B12_COUPLED_FIELD_COUNT: usize =
    8;

const B12_COUPLED_CURRENT_STORAGE: usize =
    0;

const B12_COUPLED_MINIMUM_RESERVE: usize =
    1;

const B12_COUPLED_REQUESTED_ELECTRICAL: usize =
    2;

const B12_COUPLED_TARGET_STORAGE: usize =
    3;

const B12_COUPLED_MAXIMUM_HP_ELECTRICAL: usize =
    4;

const B12_COUPLED_COP: usize =
    5;


struct B12CoupledReference {
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

    horizon_steps:
        u32,

    volumes:
        Vec<f64>,

    export_offset:
        usize,

    load_offset:
        usize,

    dynamic_offset:
        usize,
}


impl B12CoupledReference {

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
                            "B12 coupled header truncated at u32 offset {offset}.",
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
                                "Invalid u32 at B12 coupled offset {offset}.",
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
                            "B12 coupled file truncated at f64 offset {offset}.",
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
                                "Invalid f64 at B12 coupled offset {offset}.",
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
                        "Unable to read B12 coupled snapshot {}: {error}",
                        path.display(),
                    )
                },
            )?;


        if bytes.len() < 48 {
            return Err(
                "B12 coupled snapshot is smaller than its fixed header."
                    .to_string(),
            );
        }


        let magic =
            std::str::from_utf8(
                &bytes[0..15],
            )
            .map_err(
                |_| {
                    "B12 coupled magic is not UTF-8."
                        .to_string()
                },
            )?;


        if magic !=
            B12_COUPLED_MAGIC
        {
            return Err(
                format!(
                    "Unexpected B12 coupled magic {magic:?}.",
                ),
            );
        }


        let version =
            Self::read_u32(
                &bytes,
                16,
            )?;


        if version !=
            B12_COUPLED_VERSION
        {
            return Err(
                format!(
                    "Unsupported B12 coupled version {version}.",
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

        let horizon_steps =
            Self::read_u32(
                &bytes,
                40,
            )?;


        if step_count == 0 ||
            tank_count == 0
        {
            return Err(
                "B12 coupled snapshot is empty."
                    .to_string(),
            );
        }


        if field_count !=
            B12_COUPLED_FIELD_COUNT
        {
            return Err(
                format!(
                    "B12 coupled field count is {field_count}; expected {B12_COUPLED_FIELD_COUNT}.",
                ),
            );
        }


        if time_step_hours <=
            0.0
        {
            return Err(
                format!(
                    "Invalid B12 time step {time_step_hours}.",
                ),
            );
        }


        if horizon_steps == 0 ||
            horizon_steps as usize >
                step_count
        {
            return Err(
                format!(
                    "Invalid B12 horizon {horizon_steps} for {step_count} steps.",
                ),
            );
        }


        let volume_offset =
            48usize;

        let export_offset =
            volume_offset +
            tank_count *
                8;

        let load_offset =
            export_offset +
            step_count *
                8;

        let dynamic_offset =
            load_offset +
            step_count *
                8;

        let expected_bytes =
            dynamic_offset +
            tank_count *
            step_count *
            field_count *
            8;


        if expected_bytes !=
            bytes.len()
        {
            return Err(
                format!(
                    "B12 coupled size mismatch: expected {expected_bytes}, got {}.",
                    bytes.len(),
                ),
            );
        }


        let mut volumes =
            Vec::<f64>::with_capacity(
                tank_count,
            );


        for tank in
            0..
            tank_count
        {
            volumes.push(
                Self::read_f64_raw(
                    &bytes,
                    volume_offset +
                        tank *
                            8,
                )?,
            );
        }


        Ok(
            Self {
                bytes,
                step_count,
                tank_count,
                field_count,
                time_step_hours,
                horizon_steps,
                volumes,
                export_offset,
                load_offset,
                dynamic_offset,
            },
        )
    }


    fn static_value(
        &self,
        offset:
            usize,

        index:
            usize,
    ) -> f64 {

        f64::from_le_bytes(
            self.bytes[
                offset +
                    index *
                        8..
                offset +
                    index *
                        8 +
                    8
            ]
            .try_into()
            .expect(
                "validated B12 static array bounds",
            ),
        )
    }


    fn export_power(
        &self,
    ) -> Vec<f64> {

        (
            0..
            self.step_count
        )
        .map(
            |index| {
                self.static_value(
                    self.export_offset,
                    index,
                )
            },
        )
        .collect()
    }


    fn load_energy(
        &self,
    ) -> Vec<f64> {

        (
            0..
            self.step_count
        )
        .map(
            |index| {
                self.static_value(
                    self.load_offset,
                    index,
                )
            },
        )
        .collect()
    }


    fn dynamic_value(
        &self,
        tank:
            usize,

        step:
            usize,

        field:
            usize,
    ) -> f64 {

        let flat =
            (
                (
                    tank *
                    self.step_count +
                    step
                ) *
                self.field_count +
                field
            ) *
            8;

        let offset =
            self.dynamic_offset +
            flat;


        f64::from_le_bytes(
            self.bytes[
                offset..
                offset +
                    8
            ]
            .try_into()
            .expect(
                "validated B12 dynamic bounds",
            ),
        )
    }
}


fn b12_storage_capacity(
    volume_litres:
        f64,
) -> f64 {

    volume_litres *
    4.186 *
    15.0 /
    3600.0
}


fn b12_storage_energy_per_kelvin(
    volume_litres:
        f64,
) -> f64 {

    volume_litres *
    4.186 /
    3600.0
}


fn b12_storage_conductance(
    volume_litres:
        f64,
) -> f64 {

    let volume_m3 =
        volume_litres /
        1000.0;

    let height =
        1.70_f64;

    let conductivity =
        0.030_f64;

    let thickness =
        0.10_f64;

    let radius =
        (
            volume_m3 /
            (
                std::f64::consts::PI *
                height
            )
        )
        .sqrt();

    let side =
        2.0 *
        std::f64::consts::PI *
        conductivity *
        height /
        (
            (
                radius +
                thickness
            ) /
            radius
        )
        .ln();

    /*
     * Current canonical B12 geometry:
     * endArea already contains BOTH circular ends.
     */
    let end_area =
        2.0 *
        std::f64::consts::PI *
        radius *
        radius;

    let ends =
        conductivity *
        end_area /
        thickness;

    side +
    ends
}


fn b12_close(
    a:
        f64,

    b:
        f64,

    tolerance:
        f64,
) -> bool {

    (
        a -
        b
    )
    .abs() <=
    tolerance
}


pub fn run_b12_sync_blackwell(
    coupled_path:
        &str,

    output_path:
        &str,
) -> std::result::Result<(), String> {

    let coupled =
        B12CoupledReference::open(
            std::path::Path::new(
                coupled_path,
            ),
        )?;


    let export_power =
        coupled.export_power();

    let load_energy =
        coupled.load_energy();


    /*
     * BLACKWELL_CONFIG_STRIDE is defined by the current cuda_context.rs.
     *
     * Current research consumers deliberately read the physical/static
     * tank values from offsets 2..11. Offsets 0,1,5 are legacy/dynamic
     * positions and are initialized to zero here.
     *
     * We validate the resulting packed configuration through the CURRENT
     * blackwell_parameters_from_packed() before writing anything.
     */
    let mut packed =
        vec![
            0.0_f64;
            coupled.tank_count *
            BLACKWELL_CONFIG_STRIDE
        ];


    for tank in
        0..
        coupled.tank_count
    {
        let base =
            tank *
            BLACKWELL_CONFIG_STRIDE;

        let volume =
            coupled.volumes[
                tank
            ];

        let target =
            coupled.dynamic_value(
                tank,
                0,
                B12_COUPLED_TARGET_STORAGE,
            );

        let maximum_hp =
            coupled.dynamic_value(
                tank,
                0,
                B12_COUPLED_MAXIMUM_HP_ELECTRICAL,
            );

        let cop =
            coupled.dynamic_value(
                tank,
                0,
                B12_COUPLED_COP,
            );

        let expected_target =
            b12_storage_capacity(
                volume,
            );


        if !b12_close(
            target,
            expected_target,
            1.0e-12,
        ) {
            return Err(
                format!(
                    "Tank {tank} target mismatch: coupled={target:.17e}, derived={expected_target:.17e}.",
                ),
            );
        }


        if !b12_close(
            maximum_hp,
            5.0 /
                3.0,
            1.0e-12,
        ) {
            return Err(
                format!(
                    "Tank {tank} maximum HP electrical power mismatch: {maximum_hp:.17e}.",
                ),
            );
        }


        if !b12_close(
            cop,
            3.0,
            1.0e-12,
        ) {
            return Err(
                format!(
                    "Tank {tank} COP mismatch: {cop:.17e}.",
                ),
            );
        }


        /*
         * Confirm static values are truly static for every annual step.
         */
        for step in
            1..
            coupled.step_count
        {
            let step_target =
                coupled.dynamic_value(
                    tank,
                    step,
                    B12_COUPLED_TARGET_STORAGE,
                );

            let step_maximum_hp =
                coupled.dynamic_value(
                    tank,
                    step,
                    B12_COUPLED_MAXIMUM_HP_ELECTRICAL,
                );

            let step_cop =
                coupled.dynamic_value(
                    tank,
                    step,
                    B12_COUPLED_COP,
                );


            if !b12_close(
                step_target,
                target,
                0.0,
            ) ||
                !b12_close(
                    step_maximum_hp,
                    maximum_hp,
                    0.0,
                ) ||
                !b12_close(
                    step_cop,
                    cop,
                    0.0,
                )
            {
                return Err(
                    format!(
                        "B12 static tank parameters changed at tank {tank}, step {step}.",
                    ),
                );
            }
        }


        packed[
            base +
            2
        ] =
            target;

        packed[
            base +
            3
        ] =
            maximum_hp;

        packed[
            base +
            4
        ] =
            cop;

        packed[
            base +
            6
        ] =
            coupled
                .time_step_hours;

        packed[
            base +
            7
        ] =
            b12_storage_energy_per_kelvin(
                volume,
            );

        packed[
            base +
            8
        ] =
            b12_storage_conductance(
                volume,
            );

        packed[
            base +
            9
        ] =
            60.0;

        packed[
            base +
            10
        ] =
            20.0;

        packed[
            base +
            11
        ] =
            -1.0;
    }


    let native_parameters =
        blackwell_parameters_from_packed(
            &packed,
            coupled.step_count,
        )
        .map_err(
            |error| {
                format!(
                    "Current blackwell_parameters_from_packed rejected synchronized B12 config: {error}",
                )
            },
        )?;


    if native_parameters.len() !=
        coupled.tank_count
    {
        return Err(
            format!(
                "Current Blackwell parameter decoder returned {} tanks; expected {}.",
                native_parameters.len(),
                coupled.tank_count,
            ),
        );
    }


    let output =
        std::path::PathBuf::from(
            output_path,
        );


    let mut capture =
        BlackwellCapture::new(
            output.clone(),
            &export_power,
            &load_energy,
        );


    capture.configure(
        &packed,
    );


    let mut dynamic =
        vec![
            0.0_f64;
            coupled.tank_count *
            BLACKWELL_DYNAMIC_STRIDE
        ];


    for step in
        0..
        coupled.step_count
    {
        for tank in
            0..
            coupled.tank_count
        {
            let base =
                tank *
                BLACKWELL_DYNAMIC_STRIDE;

            dynamic[
                base +
                0
            ] =
                coupled.dynamic_value(
                    tank,
                    step,
                    B12_COUPLED_CURRENT_STORAGE,
                );

            dynamic[
                base +
                1
            ] =
                coupled.dynamic_value(
                    tank,
                    step,
                    B12_COUPLED_MINIMUM_RESERVE,
                );

            dynamic[
                base +
                2
            ] =
                coupled.dynamic_value(
                    tank,
                    step,
                    B12_COUPLED_REQUESTED_ELECTRICAL,
                );
        }


        capture.push_step(
            step as u32,
            coupled.horizon_steps,
            &dynamic,
        );
    }


    if !capture.complete() {
        return Err(
            "Current BlackwellCapture reports the synchronized B12 capture incomplete."
                .to_string(),
        );
    }


    capture
        .write_snapshot()
        .map_err(
            |error| {
                format!(
                    "Unable to write synchronized Blackwell snapshot {}: {error}",
                    output.display(),
                )
            },
        )?;


    let bytes =
        std::fs::metadata(
            &output,
        )
        .map_err(
            |error| {
                format!(
                    "Unable to stat synchronized Blackwell snapshot {}: {error}",
                    output.display(),
                )
            },
        )?
        .len();


    println!(
        "SUNOPT_B12_SYNC_BLACKWELL_JSON={{\"ok\":true,\"stepCount\":{},\"tankCount\":{},\"horizonSteps\":{},\"dynamicStride\":{},\"configStride\":{},\"binaryBytes\":{},\"outputPath\":{:?}}}",
        coupled.step_count,
        coupled.tank_count,
        coupled.horizon_steps,
        BLACKWELL_DYNAMIC_STRIDE,
        BLACKWELL_CONFIG_STRIDE,
        bytes,
        output.display().to_string(),
    );


    println!();
    println!(
        "--- B12 SYNCHRONIZED BLACKWELL ---"
    );

    println!(
        "steps={}",
        coupled.step_count,
    );

    println!(
        "tanks={}",
        coupled.tank_count,
    );

    println!(
        "horizonSteps={}",
        coupled.horizon_steps,
    );

    println!(
        "dynamicStride={}",
        BLACKWELL_DYNAMIC_STRIDE,
    );

    println!(
        "configStride={}",
        BLACKWELL_CONFIG_STRIDE,
    );

    println!(
        "binaryBytes={bytes}"
    );

    println!(
        "outputPath={}",
        output.display(),
    );


    Ok(())
}

