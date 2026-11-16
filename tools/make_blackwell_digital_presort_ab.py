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

H_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.h"
CU_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.cu"
RS_PATH = ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs"

H_REL = "rust/sunopt-node/cuda/cuda_context.h"
CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
RS_REL = "rust/sunopt-node/src/cuda_context.rs"
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-presort-ab.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_BEGIN"
H_END = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_END"
CU_BEGIN = H_BEGIN
CU_END = H_END
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_PRESORT_AB_FN_END"

H_BLOCK = r'''
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
'''.strip()

RUNNER = r'''
extern "C" int sunopt_cuda_run_digital_lockstep_presorted_candidate(
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
) {
    using Clock = std::chrono::steady_clock;

    if (
        host_export_power == nullptr ||
        host_demand_energy == nullptr ||
        host_tank_parameters == nullptr ||
        host_presorted_export_windows == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_values == nullptr ||
        host_cut_results == nullptr ||
        host_status == nullptr ||
        host_survival_evaluations == nullptr ||
        out_static_h2d_ms == nullptr ||
        out_presorted_h2d_ms == nullptr ||
        out_gpu_total_ms == nullptr ||
        out_host_wait_ms == nullptr ||
        out_wall_ms == nullptr ||
        data_length == 0 || tank_count == 0 || tank_count > 32 ||
        step_count == 0 || search_iterations == 0 || repetitions == 0
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t gpu_start = nullptr;
    cudaEvent_t gpu_stop = nullptr;
    double* d_export = nullptr;
    double* d_demand = nullptr;
    double* d_tanks = nullptr;
    double* d_presorted = nullptr;
    uint32_t* d_current = nullptr;
    uint32_t* d_horizon = nullptr;
    double* d_dynamic = nullptr;
    double* d_cuts = nullptr;
    uint32_t* d_status = nullptr;
    uint32_t* d_evaluations = nullptr;

    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 7 * sizeof(double);
    const size_t presorted_bytes = step_count * SUNOPT_DIGITAL_THREADS * sizeof(double);
    const size_t dynamic_step_bytes = tank_count * 3 * sizeof(double);
    const size_t cut_step_bytes = tank_count * sizeof(double);
    const size_t u32_step_bytes = tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_evaluations) cudaFree(d_evaluations);
        if (d_status) cudaFree(d_status);
        if (d_cuts) cudaFree(d_cuts);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_presorted) cudaFree(d_presorted);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (gpu_stop) cudaEventDestroy(gpu_stop);
        if (gpu_start) cudaEventDestroy(gpu_start);
        if (stream) cudaStreamDestroy(stream);
    };

#define CHECK(call) do { err = (call); if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } } while(false)
#define MALLOC(ptr, bytes) CHECK(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)))

    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&gpu_start));
    CHECK(cudaEventCreate(&gpu_stop));
    MALLOC(d_export, annual_bytes);
    MALLOC(d_demand, annual_bytes);
    MALLOC(d_tanks, tank_bytes);
    MALLOC(d_presorted, presorted_bytes);
    MALLOC(d_current, sizeof(uint32_t));
    MALLOC(d_horizon, sizeof(uint32_t));
    MALLOC(d_dynamic, dynamic_step_bytes);
    MALLOC(d_cuts, cut_step_bytes);
    MALLOC(d_status, u32_step_bytes);
    MALLOC(d_evaluations, u32_step_bytes);
#undef MALLOC

    const auto static_start = Clock::now();
    CHECK(cudaMemcpy(d_export, host_export_power, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_demand, host_demand_energy, annual_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tanks, host_tank_parameters, tank_bytes, cudaMemcpyHostToDevice));
    const auto static_stop = Clock::now();
    *out_static_h2d_ms = std::chrono::duration<double, std::milli>(static_stop-static_start).count();

    const auto presort_start = Clock::now();
    CHECK(cudaMemcpy(d_presorted, host_presorted_export_windows, presorted_bytes, cudaMemcpyHostToDevice));
    const auto presort_stop = Clock::now();
    *out_presorted_h2d_ms = std::chrono::duration<double, std::milli>(presort_stop-presort_start).count();

    double host_wait_ms = 0.0;
    const auto wall_start = Clock::now();
    CHECK(cudaEventRecord(gpu_start, stream));

    const unsigned int blocks = static_cast<unsigned int>(tank_count);

    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        for (size_t step = 0; step < step_count; ++step) {
            const size_t dynamic_offset = step * tank_count * 3;
            const size_t result_offset = step * tank_count;

            CHECK(cudaMemcpyAsync(d_current, host_current_indices + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_horizon, host_horizon_steps + step, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
            CHECK(cudaMemcpyAsync(d_dynamic, host_dynamic_values + dynamic_offset, dynamic_step_bytes, cudaMemcpyHostToDevice, stream));

            sunopt_digital_hp_presorted_kernel<<<blocks, SUNOPT_DIGITAL_THREADS, 0, stream>>>(
                d_export,
                d_demand,
                d_presorted + step * SUNOPT_DIGITAL_THREADS,
                data_length,
                d_tanks,
                tank_count,
                d_current,
                d_horizon,
                d_dynamic,
                1,
                search_iterations,
                d_cuts,
                d_status,
                d_evaluations
            );
            CHECK(cudaGetLastError());

            CHECK(cudaMemcpyAsync(host_cut_results + result_offset, d_cuts, cut_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_status + result_offset, d_status, u32_step_bytes, cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(host_survival_evaluations + result_offset, d_evaluations, u32_step_bytes, cudaMemcpyDeviceToHost, stream));

            const auto wait_start = Clock::now();
            CHECK(cudaStreamSynchronize(stream));
            const auto wait_stop = Clock::now();
            host_wait_ms += std::chrono::duration<double, std::milli>(wait_stop-wait_start).count();
        }
    }

    CHECK(cudaEventRecord(gpu_stop, stream));
    CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    const auto wall_stop = Clock::now();

    *out_gpu_total_ms = static_cast<double>(gpu_ms);
    *out_host_wait_ms = host_wait_ms;
    *out_wall_ms = std::chrono::duration<double, std::milli>(wall_stop-wall_start).count();

#undef CHECK
    cleanup();
    return static_cast<int>(cudaSuccess);
}
'''.strip()

RS_FFI_BLOCK = r'''
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
'''.strip()

RS_RESULT_BLOCK = r'''
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
'''.strip()

RS_FN_BLOCK = r'''
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
'''.strip()

MJS = r'''import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshot = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, "blackwell-benchmark.bin");
const repetitions = Number.parseInt(process.argv[3] ?? "1", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

const result = addon.runBlackwellDigitalPresortAb(snapshot, repetitions, deviceId);
console.log(`SUNOPT_DIGITAL_PRESORT_AB_JSON=${JSON.stringify(result)}`);
console.log("--- DIGITAL LOCKSTEP: BITONIC vs PRE-SORTED ---");
console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);
console.log(`presortCpuMs=${result.presortCpuMs.toFixed(3)}`);
console.log(`presortedWindowH2DMs=${result.presortedWindowH2DMs.toFixed(3)}`);
console.log();
console.log(`directGpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`);
console.log(`directWallMs=${result.directWallMs.toFixed(3)}`);
console.log(`presortedGpuTotalMs=${result.presortedGpuTotalMs.toFixed(3)}`);
console.log(`presortedWallMs=${result.presortedWallMs.toFixed(3)}`);
console.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);
console.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);
console.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);
console.log();
console.log(`statusMismatches=${result.statusMismatches}`);
console.log(`cutMismatches=${result.cutMismatches}`);
console.log(`maximumCutError=${result.maximumCutError.toExponential(6)}`);
console.log(`checksum=${result.checksumHex}`);

const report = path.join(root, "blackwell-digital-presort-ab.json");
writeFileSync(report, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log(`report=${report}`);
'''.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="utf-8").strip().lower()
        if HASH_RE.fullmatch(value): return value
        raise RuntimeError(f"Invalid SHA-256 in {HASH_FILE}")
    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).strip().lower()
        if HASH_RE.fullmatch(value): return value
        raise RuntimeError(f"Invalid sha256 in {RELEASE_FILE}")
    return GENESIS


def replace_between(source: str, begin: str, end: str, block: str) -> str:
    has_begin = begin in source
    has_end = end in source
    if has_begin != has_end: raise RuntimeError(f"Only one marker present: {begin!r} / {end!r}")
    if not has_begin: raise RuntimeError(f"Marker pair missing: {begin!r} / {end!r}")
    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[:start] + block + source[finish:]


def insert_or_replace(source: str, begin: str, end: str, block: str, anchors: list[str]) -> str:
    if begin in source or end in source:
        return replace_between(source, begin, end, block)
    for anchor in anchors:
        position = source.find(anchor)
        if position >= 0: return source[:position] + block + "\n\n\n" + source[position:]
    raise RuntimeError("Could not find insertion anchor")


def extract_braced_function(source: str, token: str) -> str:
    start = source.find(token)
    if start < 0: raise RuntimeError(f"Missing function {token!r}")
    brace = source.find("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{": depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0: return source[start:index + 1]
    raise RuntimeError("Unterminated CUDA kernel")


def build_presorted_kernel(source: str) -> str:
    token = "__global__ void\nsunopt_digital_hp_breakpoint_kernel("
    kernel = extract_braced_function(source, token)
    kernel = kernel.replace("sunopt_digital_hp_breakpoint_kernel(", "sunopt_digital_hp_presorted_kernel(", 1)

    old_params = "    const double* __restrict__ export_power,\n    const double* __restrict__ demand_energy,\n    size_t data_length,\n"
    new_params = "    const double* __restrict__ export_power,\n    const double* __restrict__ demand_energy,\n    const double* __restrict__ presorted_export_window,\n    size_t data_length,\n"
    if old_params not in kernel: raise RuntimeError("Kernel parameter pattern changed")
    kernel = kernel.replace(old_params, new_params, 1)

    start_marker = "    /*\n     * A breakpoint is relevant only when the HP can run at cut=0.\n"
    count_marker = "    if (\n        tid ==\n        0\n    ) {\n        unsigned int count =\n"
    a = kernel.find(start_marker)
    b = kernel.find(count_marker, a)
    if a < 0 or b < 0: raise RuntimeError("Could not locate bitonic sort region")

    replacement = r'''    /* B5: shared sorted export order, reused by every tank block. */
    const double sorted_export_value = presorted_export_window[tid];

    breakpoints[tid] =
        (
            maximum_hp_power > 0.0 &&
            sorted_export_value >= maximum_hp_power
        ) ?
            (sorted_export_value - maximum_hp_power) :
            -CUDART_INF;

    __syncthreads();

'''
    kernel = kernel[:a] + replacement + kernel[b:]
    return kernel


def patch_header(source: str) -> str:
    return insert_or_replace(source, H_BEGIN, H_END, H_BLOCK, ["SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile(", "#ifdef __cplusplus\n}"])


def patch_cuda(source: str) -> str:
    for token in ("SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN", "sunopt_digital_hp_breakpoint_kernel(", "sunopt_cuda_run_digital_lockstep_candidate("):
        if token not in source: raise RuntimeError(f"Expected current CUDA generation missing {token!r}")
    kernel = build_presorted_kernel(source)
    block = CU_BEGIN + "\n" + kernel + "\n\n" + RUNNER + "\n" + CU_END
    return insert_or_replace(source, CU_BEGIN, CU_END, block, ['extern "C" int sunopt_cuda_get_profile(', 'extern "C" const char* sunopt_cuda_error_string('])


def patch_rust(source: str) -> str:
    for token in ("fn sunopt_cuda_run_digital_lockstep_candidate(", "fn research_tank_from_packed(", "fn blackwell_fnv1a_update(", "BLACKWELL_DYNAMIC_STRIDE"):
        if token not in source: raise RuntimeError(f"Expected current Rust generation missing {token!r}")
    source = insert_or_replace(source, RS_FFI_BEGIN, RS_FFI_END, RS_FFI_BLOCK, ["    fn sunopt_cuda_get_profile(", "    fn sunopt_cuda_error_string("])
    source = insert_or_replace(source, RS_RESULT_BEGIN, RS_RESULT_END, RS_RESULT_BLOCK, ["#[napi]\npub struct CudaCausalWorker {"])
    source = insert_or_replace(source, RS_FN_BEGIN, RS_FN_END, RS_FN_BLOCK, ["#[napi]\npub fn run_blackwell_replay(", "pub fn run_blackwell_replay("])
    return source


def build_package() -> Path:
    for path in (H_PATH, CU_PATH, RS_PATH):
        if not path.is_file(): raise RuntimeError(f"Missing project file: {path}")

    payloads = {
        H_REL: patch_header(H_PATH.read_text(encoding="utf-8")).encode("utf-8"),
        CU_REL: patch_cuda(CU_PATH.read_text(encoding="utf-8")).encode("utf-8"),
        RS_REL: patch_rust(RS_PATH.read_text(encoding="utf-8")).encode("utf-8"),
        MJS_REL: (MJS + "\n").encode("utf-8"),
    }

    parent = read_parent_hash()
    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {name: sha256_bytes(data) for name, data in payloads.items()},
        "delete": [],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    release = sha256_bytes(manifest_bytes)

    INBOX.mkdir(parents=True, exist_ok=True)
    output = INBOX / "20261109-blackwell-digital-presort-ab.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("manifest.sha256", (release + "\n").encode("ascii"))
        for name, data in payloads.items(): archive.writestr(name, data)

    print("SunOpt digital pre-sort A/B package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print("  production worker: UNCHANGED")
    print("  parity: current all-GPU vs pre-sorted all-GPU over all annual jobs")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(r'  python .\tools\update_watcher.py --capture "npm run build && node .\scripts\blackwell-digital-presort-ab.mjs .\blackwell-benchmark.bin 1 0" --cwd rust\sunopt-node')
    return output


if __name__ == "__main__":
    build_package()
