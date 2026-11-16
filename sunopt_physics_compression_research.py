#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path.cwd()
INBOX = ROOT / "updates" / "inbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"

RS_PATH = ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs"
MJS_PATH = ROOT / "rust" / "sunopt-node" / "scripts" / "blackwell-compression-research.mjs"

RS_REL = "rust/sunopt-node/src/cuda_context.rs"
MJS_REL = "rust/sunopt-node/scripts/blackwell-compression-research.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

BEGIN = "// SUNOPT_BLACKWELL_PHYSICS_COMPRESSION_RESEARCH_BEGIN"
END = "// SUNOPT_BLACKWELL_PHYSICS_COMPRESSION_RESEARCH_END"

RS_BLOCK = r'''
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
'''.strip()

MJS = r'''import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");

const sampleSteps = Number.parseInt(process.argv[3] ?? "192", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

if (!Number.isFinite(sampleSteps) || sampleSteps < 2) {
  throw new Error("sampleSteps must be >= 2");
}

const result = addon.runBlackwellPhysicsCompressionResearch(
  snapshot,
  sampleSteps,
  deviceId,
);

console.log(`SUNOPT_PHYSICS_COMPRESSION_JSON=${JSON.stringify(result)}`);
console.log();
console.log("--- PHYSICS COMPRESSION SUMMARY ---");
console.log(`snapshot=${result.snapshotPath}`);
console.log(`sampleSteps=${result.sampleStepCount}`);
console.log(`sampleJobs=${result.sampleJobCount}`);
console.log(`gpuReferenceMs=${result.gpuReferenceMs.toFixed(3)}`);
console.log(`scalarReferenceMs=${result.scalarReferenceMs.toFixed(3)}`);
console.log(`scalarGpuMaxAbsCutError=${result.scalarGpuMaxAbsCutError.toExponential(6)}`);
console.log(`scalarGpuCutErrorOver1e6=${result.scalarGpuCutErrorOver1e6}`);
console.log();
console.log("MAX-AFFINE EXACT-COMPRESSION CANDIDATE");
console.log(`maxAffineMs=${result.maxAffineMs.toFixed(3)}`);
console.log(`treeLevels=${result.maxAffineTreeLevels}`);
console.log(`supportedJobs=${result.maxAffineSupportedJobs}/${result.sampleJobCount}`);
console.log(`maxAbsCutError=${result.maxAffineScalarMaxAbsCutError.toExponential(6)}`);
console.log(`cutErrorOver1e6=${result.maxAffineScalarCutErrorOver1e6}`);
console.log(`normalSurvivalMismatches=${result.maxAffineNormalSurvivalMismatches}`);
console.log(`maxPvSurvivalMismatches=${result.maxAffineMaxpvSurvivalMismatches}`);
console.log();
console.log("FOURIER COMPRESSION");
console.log(
  [
    "harm",
    "coeff",
    "fraction",
    "ms",
    "exportRMSE",
    "demandRMSE",
    "cutMAE",
    "cutRMSE",
    "cutMax",
    ">0.01kW",
    ">0.1kW",
    "normalMis",
    "maxPvMis",
  ].join("\t"),
);

for (let i = 0; i < result.fourierHarmonics.length; i += 1) {
  console.log(
    [
      result.fourierHarmonics[i],
      result.fourierCoefficients[i],
      (100 * result.fourierSignalFraction[i]).toFixed(1) + "%",
      result.fourierMs[i].toFixed(1),
      result.fourierExportRmse[i].toFixed(6),
      result.fourierDemandRmse[i].toFixed(6),
      result.fourierCutMae[i].toFixed(6),
      result.fourierCutRmse[i].toFixed(6),
      result.fourierCutMaxAbsError[i].toFixed(6),
      result.fourierCutErrorOver001[i],
      result.fourierCutErrorOver01[i],
      result.fourierNormalSurvivalMismatches[i],
      result.fourierMaxpvSurvivalMismatches[i],
    ].join("\t"),
  );
}

const outputPath = path.join(root, "blackwell-compression-research.json");
writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log();
console.log(`report=${outputPath}`);
'''.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="ascii").strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid hash in {HASH_FILE}")

    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid sha256 in {RELEASE_FILE}")

    return GENESIS


def insert_or_replace(source: str) -> str:
    has_begin = BEGIN in source
    has_end = END in source

    if has_begin != has_end:
        raise RuntimeError("Only one physics-compression marker is present.")

    if has_begin:
        start = source.index(BEGIN)
        finish = source.index(END, start) + len(END)
        return source[:start] + RS_BLOCK + source[finish:]

    required = [
        "struct BlackwellReplaySnapshot",
        "fn read_blackwell_snapshot(",
        "fn blackwell_parameters_from_packed(",
        "fn blackwell_dynamic_from_values(",
        "pub fn run_blackwell_replay(",
        "BLACKWELL_CONFIG_STRIDE",
        "BLACKWELL_DYNAMIC_STRIDE",
        "run_causal_dynamic_batch_into(",
    ]

    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Current cuda_context.rs is not the expected Blackwell generation; missing {token!r}."
            )

    anchors = [
        "#[napi]\npub fn run_blackwell_replay(",
        "pub fn run_blackwell_replay(",
    ]

    position = -1
    for anchor in anchors:
        position = source.find(anchor)
        if position >= 0:
            break

    if position < 0:
        raise RuntimeError("Could not find run_blackwell_replay insertion anchor.")

    return source[:position] + RS_BLOCK + "\n\n\n" + source[position:]


def build_package() -> Path:
    if not RS_PATH.is_file():
        raise RuntimeError(f"Required current project file missing: {RS_PATH}")

    current = RS_PATH.read_text(encoding="utf-8")
    patched = insert_or_replace(current)

    payloads = {
        RS_REL: patched.encode("utf-8"),
        MJS_REL: MJS.encode("utf-8"),
    }

    parent = read_parent_hash()
    files = {name: sha256_bytes(data) for name, data in payloads.items()}

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": files,
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    release_hash = sha256_bytes(manifest_bytes)

    INBOX.mkdir(parents=True, exist_ok=True)
    output = INBOX / "20261106-blackwell-physics-compression-research.zip"

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr(
            "manifest.sha256",
            (release_hash + "\n").encode("ascii"),
        )

        for name, data in payloads.items():
            archive.writestr(name, data)

    print("SunOpt physics-compression research package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release_hash}")
    print("  files:")
    for name in payloads:
        print(f"    - {name}")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(r"  cd .\rust\sunopt-node")
    print(r"  npm run build")
    print(r"  node .\scripts\blackwell-compression-research.mjs .\blackwell-benchmark.bin 192 0")

    return output


if __name__ == "__main__":
    build_package()
