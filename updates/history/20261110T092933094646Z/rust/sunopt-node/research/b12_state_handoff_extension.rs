// rust/sunopt-node/research/b12_state_handoff_extension.rs
// Appended inside a module containing the current cuda_context.rs.
// Research only.

const PHY_FIELD_STORAGE_START: usize = 0;
    const PHY_FIELD_STORAGE_AFTER_DEMAND: usize = 1;
    const PHY_FIELD_MINIMUM_RESERVE: usize = 3;
    const PHY_FIELD_STORAGE_AFTER_MANDATORY: usize = 5;
    const PHY_FIELD_REQUIRED_48H: usize = 8;

    #[derive(Default)]
    struct Stats {
        mismatches_1e12: u64,
        mismatches_1e9: u64,
        max_abs_error: f64,
        max_job: usize,
    }

    impl Stats {
        fn add(
            &mut self,
            job: usize,
            actual: f64,
            expected: f64,
        ) {
            let error =
                (actual - expected).abs();

            if error > self.max_abs_error {
                self.max_abs_error = error;
                self.max_job = job;
            }

            if error > 1.0e-12 {
                self.mismatches_1e12 += 1;
            }

            if error > 1.0e-9 {
                self.mismatches_1e9 += 1;
            }
        }
    }

    struct PhysicalReference {
        bytes: Vec<u8>,
        step_count: usize,
        tank_count: usize,
        field_count: usize,
        data_offset: usize,
    }

    impl PhysicalReference {
        fn read_u32(
            bytes: &[u8],
            offset: usize,
        ) -> std::result::Result<u32, String> {
            let slice =
                bytes.get(
                    offset..
                    offset + 4,
                )
                .ok_or_else(
                    || {
                        "Physical reference header is truncated."
                            .to_string()
                    },
                )?;

            Ok(
                u32::from_le_bytes(
                    slice.try_into()
                        .map_err(
                            |_| {
                                "Invalid u32 field in physical reference."
                                    .to_string()
                            },
                        )?,
                ),
            )
        }

        fn open(
            path: &std::path::Path,
        ) -> std::result::Result<Self, String> {
            let bytes =
                std::fs::read(path)
                    .map_err(
                        |error| {
                            format!(
                                "Unable to read physical reference {}: {error}",
                                path.display(),
                            )
                        },
                    )?;

            if bytes.len() < 40 {
                return Err(
                    "Physical reference is too small."
                        .to_string(),
                );
            }

            let magic =
                std::str::from_utf8(
                    &bytes[0..15],
                )
                .map_err(
                    |_| {
                        "Physical reference magic is not UTF-8."
                            .to_string()
                    },
                )?;

            if magic != "SUNOPT_B12_PHY1" {
                return Err(
                    format!(
                        "Unexpected physical reference magic: {magic:?}",
                    ),
                );
            }

            let version =
                Self::read_u32(
                    &bytes,
                    16,
                )?;

            if version != 1 {
                return Err(
                    format!(
                        "Unsupported physical reference version {version}.",
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

            if field_count <= PHY_FIELD_REQUIRED_48H {
                return Err(
                    format!(
                        "Physical reference has only {field_count} fields.",
                    ),
                );
            }

            let data_offset =
                40 +
                tank_count *
                8;

            let expected =
                data_offset +
                step_count *
                tank_count *
                field_count *
                8;

            if expected != bytes.len() {
                return Err(
                    format!(
                        "Physical reference size mismatch: expected {expected}, got {}.",
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
                    data_offset,
                },
            )
        }

        fn value(
            &self,
            tank: usize,
            step: usize,
            field: usize,
        ) -> f64 {
            let job =
                tank *
                self.step_count +
                step;

            let offset =
                self.data_offset +
                (
                    job *
                    self.field_count +
                    field
                ) *
                8;

            f64::from_le_bytes(
                self.bytes[
                    offset..
                    offset + 8
                ]
                .try_into()
                .expect(
                    "validated physical-reference range",
                ),
            )
        }
    }

    fn emit_stats(
        name: &str,
        stats: &Stats,
    ) {
        println!(
            "\"{name}Mismatches1e12\":{},\"{name}Mismatches1e9\":{},\"{name}MaxAbsError\":{:.17e},\"{name}MaxErrorJob\":{}",
            stats.mismatches_1e12,
            stats.mismatches_1e9,
            stats.max_abs_error,
            stats.max_job,
        );
    }

    pub fn run_b12_state_handoff(
        snapshot_path: &str,
        physical_path: &str,
    ) -> std::result::Result<(), String> {
        let snapshot =
            read_blackwell_snapshot(
                std::path::Path::new(
                    snapshot_path,
                ),
            )
            .map_err(
                |error| {
                    format!(
                        "Unable to read Blackwell snapshot {snapshot_path}: {error}",
                    )
                },
            )?;

        let physical =
            PhysicalReference::open(
                std::path::Path::new(
                    physical_path,
                ),
            )?;

        if snapshot.steps.len() != physical.step_count {
            return Err(
                format!(
                    "Step-count mismatch: Blackwell={} physical={}.",
                    snapshot.steps.len(),
                    physical.step_count,
                ),
            );
        }

        if snapshot.tank_count != physical.tank_count {
            return Err(
                format!(
                    "Tank-count mismatch: Blackwell={} physical={}.",
                    snapshot.tank_count,
                    physical.tank_count,
                ),
            );
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

        let step_count =
            snapshot.steps.len();

        let tank_count =
            snapshot.tank_count;

        let job_count =
            step_count *
            tank_count;

        let mut current_vs_start =
            Stats::default();

        let mut current_vs_after_demand =
            Stats::default();

        let mut current_vs_after_mandatory =
            Stats::default();

        let mut reserve_vs_physical =
            Stats::default();

        let mut requested_vs_start =
            Stats::default();

        let mut requested_vs_after_demand =
            Stats::default();

        let mut requested_vs_after_mandatory =
            Stats::default();

        for (
            step_index,
            step,
        ) in snapshot
            .steps
            .iter()
            .enumerate()
        {
            if step.dynamic.len() <
                tank_count *
                BLACKWELL_DYNAMIC_STRIDE
            {
                return Err(
                    format!(
                        "Dynamic row {step_index} is too short: {}.",
                        step.dynamic.len(),
                    ),
                );
            }

            for tank_index in 0..tank_count {
                let job =
                    step_index *
                    tank_count +
                    tank_index;

                let base =
                    tank_index *
                    BLACKWELL_DYNAMIC_STRIDE;

                let dynamic_current =
                    step.dynamic[
                        base + 0
                    ];

                let dynamic_reserve =
                    step.dynamic[
                        base + 1
                    ];

                let dynamic_requested =
                    step.dynamic[
                        base + 2
                    ];

                let storage_start =
                    physical.value(
                        tank_index,
                        step_index,
                        PHY_FIELD_STORAGE_START,
                    );

                let storage_after_demand =
                    physical.value(
                        tank_index,
                        step_index,
                        PHY_FIELD_STORAGE_AFTER_DEMAND,
                    );

                let storage_after_mandatory =
                    physical.value(
                        tank_index,
                        step_index,
                        PHY_FIELD_STORAGE_AFTER_MANDATORY,
                    );

                let minimum_reserve =
                    physical.value(
                        tank_index,
                        step_index,
                        PHY_FIELD_MINIMUM_RESERVE,
                    );

                let required_48h =
                    physical.value(
                        tank_index,
                        step_index,
                        PHY_FIELD_REQUIRED_48H,
                    );

                current_vs_start.add(
                    job,
                    dynamic_current,
                    storage_start,
                );

                current_vs_after_demand.add(
                    job,
                    dynamic_current,
                    storage_after_demand,
                );

                current_vs_after_mandatory.add(
                    job,
                    dynamic_current,
                    storage_after_mandatory,
                );

                reserve_vs_physical.add(
                    job,
                    dynamic_reserve,
                    minimum_reserve,
                );

                let tank =
                    &tanks[tank_index];

                let expected_requested =
                    |state: f64| {
                        let current =
                            state
                                .max(0.0)
                                .min(
                                    tank.target_storage_energy,
                                );

                        let missing =
                            (
                                required_48h -
                                current
                            )
                            .max(
                                0.0,
                            );

                        let capacity =
                            (
                                tank.target_storage_energy -
                                current
                            )
                            .max(
                                0.0,
                            );

                        missing
                            .min(
                                capacity,
                            ) /
                            tank.cop
                    };

                requested_vs_start.add(
                    job,
                    dynamic_requested,
                    expected_requested(
                        storage_start,
                    ),
                );

                requested_vs_after_demand.add(
                    job,
                    dynamic_requested,
                    expected_requested(
                        storage_after_demand,
                    ),
                );

                requested_vs_after_mandatory.add(
                    job,
                    dynamic_requested,
                    expected_requested(
                        storage_after_mandatory,
                    ),
                );
            }
        }

        println!(
            "SUNOPT_B12_STATE_HANDOFF_JSON={{\"stepCount\":{step_count},\"tankCount\":{tank_count},\"jobs\":{job_count},"
        );

        emit_stats(
            "currentVsStorageStart",
            &current_vs_start,
        );
        println!(",");

        emit_stats(
            "currentVsStorageAfterDemand",
            &current_vs_after_demand,
        );
        println!(",");

        emit_stats(
            "currentVsStorageAfterMandatory",
            &current_vs_after_mandatory,
        );
        println!(",");

        emit_stats(
            "reserveVsMinimumReserve",
            &reserve_vs_physical,
        );
        println!(",");

        emit_stats(
            "requestedVsStartDerived",
            &requested_vs_start,
        );
        println!(",");

        emit_stats(
            "requestedVsAfterDemandDerived",
            &requested_vs_after_demand,
        );
        println!(",");

        emit_stats(
            "requestedVsAfterMandatoryDerived",
            &requested_vs_after_mandatory,
        );

        println!("}}");

        println!();
        println!(
            "--- B12 STATE HANDOFF ---"
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

        macro_rules! line {
            ($label:expr, $stats:expr) => {
                println!(
                    "{} mismatches1e-12={} mismatches1e-9={} maxError={:.6e} maxJob={}",
                    $label,
                    $stats.mismatches_1e12,
                    $stats.mismatches_1e9,
                    $stats.max_abs_error,
                    $stats.max_job,
                );
            };
        }

        line!(
            "current vs storageStart",
            current_vs_start
        );

        line!(
            "current vs storageAfterDemand",
            current_vs_after_demand
        );

        line!(
            "current vs storageAfterMandatory",
            current_vs_after_mandatory
        );

        line!(
            "reserve vs minimumReserve",
            reserve_vs_physical
        );

        line!(
            "requested vs start-derived",
            requested_vs_start
        );

        line!(
            "requested vs afterDemand-derived",
            requested_vs_after_demand
        );

        line!(
            "requested vs afterMandatory-derived",
            requested_vs_after_mandatory
        );

        Ok(())
    }
