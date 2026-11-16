#!/usr/bin/env python3
# tools/make_blackwell_digital_gpu_fallback.py
#
# Run from:
#   C:\NextJS\SunOpt\sunopt
#
# Requires the installed digital-HP breakpoint package and the production-
# shaped digital lockstep candidate.
#
# Candidate-only change:
#   - exact max-affine breakpoint solver for the common branch
#   - exact SERIAL GPU fallback for reserve ~= 0 / max-affine edge cases
#   - NO CPU causal fallback
#   - 8 breakpoint search iterations (exact for <= 256 candidates)
#
# The existing continuous production worker remains untouched.

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

CU_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.cu"
RS_PATH = ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs"

CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
RS_REL = "rust/sunopt-node/src/cuda_context.rs"
MJS_REL = "rust/sunopt-node/scripts/blackwell-digital-lockstep-gpu-fallback.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

SERIAL_BEGIN = "// SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN"
SERIAL_END = "// SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_END"

RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_FN_END"

SERIAL_HELPER = r'''
// SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_BEGIN
/*
 * Exact serial digital-HP survival evaluator.
 *
 * This deliberately does not use the max-affine reduction. It mirrors the
 * scalar digital reference step-by-step and therefore handles reserve == 0
 * and the empty-storage clipping boundary exactly.
 */
__device__ __forceinline__ bool
sunopt_digital_survives_serial_exact(
    const double* export_power,
    const double* demand_energy,
    size_t data_length,
    const double* tank_parameters,
    size_t tank_index,
    size_t current_index,
    size_t horizon,
    double current_storage_energy,
    double minimum_reserve_energy,
    double cut
) {
    const size_t tank_base =
        tank_index *
        7;

    const double target =
        tank_parameters[
            tank_base + 0
        ];

    const double maximum_hp_power =
        tank_parameters[
            tank_base + 1
        ];

    const double full_charge =
        tank_parameters[
            tank_base + 2
        ];

    const double retention =
        tank_parameters[
            tank_base + 3
        ];

    const double loss_intercept =
        tank_parameters[
            tank_base + 4
        ];

    const double reserve =
        fmin(
            target,
            fmax(
                0.0,
                minimum_reserve_energy
            )
        );

    double stored =
        fmin(
            target,
            fmax(
                0.0,
                current_storage_energy
            )
        );

    for (
        size_t offset = 0;
        offset < horizon;
        ++offset
    ) {
        const size_t index =
            (
                current_index +
                offset
            ) %
            data_length;

        const double charge =
            sunopt_digital_hp_on(
                export_power[
                    index
                ],
                cut,
                maximum_hp_power
            ) ?
                full_charge :
                0.0;

        /* Current demand/loss already happened before currentStorageEnergy. */
        if (
            offset == 0
        ) {
            stored =
                fmin(
                    target,
                    stored +
                    charge
                );

            continue;
        }

        /*
         * Mirrors digital_survives_scalar():
         * 1. passive loss, clipped at zero and at stored energy
         * 2. thermal demand
         * 3. reserve check BEFORE forecast HP charging
         * 4. all-or-nothing HP charge
         */
        const double requested_loss =
            fmax(
                0.0,
                (
                    1.0 -
                    retention
                ) *
                stored +
                loss_intercept
            );

        const double heat_loss =
            fmin(
                requested_loss,
                stored
            );

        const double after_loss =
            fmax(
                0.0,
                stored -
                heat_loss
            );

        const double demand =
            fmax(
                0.0,
                demand_energy[
                    index
                ]
            );

        const double discharged =
            fmin(
                demand,
                after_loss
            );

        const double before_charge =
            after_loss -
            discharged;

        const double unmet =
            fmax(
                0.0,
                demand -
                discharged
            );

        if (
            unmet >
            SUNOPT_DIGITAL_EPSILON
        ) {
            return false;
        }

        if (
            reserve -
            before_charge >
            SUNOPT_DIGITAL_EPSILON
        ) {
            return false;
        }

        stored =
            fmin(
                target,
                before_charge +
                charge
            );
    }

    return true;
}
// SUNOPT_DIGITAL_GPU_SERIAL_FALLBACK_END
'''.strip()

OLD_UNSUPPORTED_BLOCK = r'''    if (
        tid ==
        0
    ) {
        eval_count +=
            1;

        if (
            !shared_supported
        ) {
            final_status =
                3;
            final_cut =
                0.0;
        } else if (
            normal_survives
        ) {
            final_status =
                0;
            final_cut =
                normal_cut;
        }
    }

    __syncthreads();

    if (
        final_status ==
            0 ||
        final_status ==
            3
    ) {
        if (
            tid ==
            0
        ) {
            cut_results[
                job_index
            ] =
                final_cut;

            status[
                job_index
            ] =
                final_status;

            survival_evaluations[
                job_index
            ] =
                eval_count;
        }

        return;
    }
'''

NEW_UNSUPPORTED_BLOCK = r'''    if (
        tid ==
        0
    ) {
        if (
            !shared_supported
        ) {
            /*
             * Exact GPU fallback for the max-affine edge cases.
             * The breakpoints are already sorted in shared memory.
             */
            eval_count =
                1;

            const bool serial_normal_survives =
                sunopt_digital_survives_serial_exact(
                    export_power,
                    demand_energy,
                    data_length,
                    tank_parameters,
                    tank_index,
                    current_index,
                    horizon,
                    current_storage,
                    reserve,
                    normal_cut
                );

            if (
                serial_normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            } else {
                eval_count +=
                    1;

                const bool serial_maxpv_survives =
                    sunopt_digital_survives_serial_exact(
                        export_power,
                        demand_energy,
                        data_length,
                        tank_parameters,
                        tank_index,
                        current_index,
                        horizon,
                        current_storage,
                        reserve,
                        0.0
                    );

                if (
                    !serial_maxpv_survives
                ) {
                    final_status =
                        1;

                    final_cut =
                        0.0;
                } else {
                    final_status =
                        2;

                    double best_cut =
                        0.0;

                    int left =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            valid_count
                        );

                    int right =
                        static_cast<int>(
                            SUNOPT_DIGITAL_THREADS -
                            1
                        );

                    for (
                        uint32_t iteration = 0;
                        iteration <
                            search_iterations &&
                        left <=
                            right;
                        ++iteration
                    ) {
                        const int middle =
                            left +
                            (
                                right -
                                left
                            ) /
                            2;

                        const double candidate =
                            fmax(
                                0.0,
                                breakpoints[
                                    middle
                                ]
                            );

                        eval_count +=
                            1;

                        if (
                            sunopt_digital_survives_serial_exact(
                                export_power,
                                demand_energy,
                                data_length,
                                tank_parameters,
                                tank_index,
                                current_index,
                                horizon,
                                current_storage,
                                reserve,
                                candidate
                            )
                        ) {
                            best_cut =
                                candidate;

                            left =
                                middle +
                                1;
                        } else {
                            right =
                                middle -
                                1;
                        }
                    }

                    final_cut =
                        best_cut;
                }
            }
        } else {
            eval_count +=
                1;

            if (
                normal_survives
            ) {
                final_status =
                    0;

                final_cut =
                    normal_cut;
            }
        }
    }

    __syncthreads();

    /*
     * Max-affine-unsupported jobs are already fully solved by the serial GPU
     * branch. No status=3 leaves this kernel.
     */
    if (
        final_status ==
            0 ||
        !shared_supported
    ) {
        if (
            tid ==
            0
        ) {
            cut_results[
                job_index
            ] =
                final_cut;

            status[
                job_index
            ] =
                final_status;

            survival_evaluations[
                job_index
            ] =
                eval_count;
        }

        return;
    }
'''

RS_RESULT_BLOCK = r'''
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
const repetitions = Number.parseInt(process.argv[3] ?? "1", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

const result = addon.runBlackwellDigitalLockstepCandidate(
  snapshot,
  repetitions,
  deviceId,
);

console.log(`SUNOPT_DIGITAL_GPU_FALLBACK_JSON=${JSON.stringify(result)}`);
console.log();
console.log("--- BLACKWELL DIGITAL-HP LOCKSTEP / ALL GPU ---");
console.log(`snapshot=${result.snapshotPath}`);
console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);
console.log(`gpuCompressedJobs=${result.gpuCompressedJobs}`);
console.log(`gpuSerialFallbackJobs=${result.gpuSerialFallbackJobs}`);
console.log(`unexpectedUnsupportedJobs=${result.unexpectedUnsupportedJobs}`);
console.log();
console.log("TIMING");
console.log(`staticH2DMs=${result.staticH2DMs.toFixed(3)}`);
console.log(`directGpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`);
console.log(`directHostWaitMs=${result.directHostWaitMs.toFixed(3)}`);
console.log(`directWallMs=${result.directWallMs.toFixed(3)}`);
console.log(`candidateCausalMs=${result.candidateCausalMs.toFixed(3)}`);
console.log(`candidateAverageUsPerJob=${result.candidateAverageUsPerJob.toFixed(3)}`);
console.log();
console.log("PARITY SAMPLE (includes serial GPU fallback)");
console.log(`sampleSteps=${result.paritySampleSteps}`);
console.log(`statusMismatches=${result.parityStatusMismatches}`);
console.log(`cutMismatches=${result.parityCutMismatches}`);
console.log(`maxCutError=${result.parityMaxCutError.toExponential(6)}`);
console.log();
console.log("SEARCH WORK");
console.log(`averageSurvivalEvaluations=${result.averageSurvivalEvaluations.toFixed(3)}`);
console.log(`maximumSurvivalEvaluations=${result.maximumSurvivalEvaluations}`);
console.log(`checksum=${result.checksumHex}`);

const report = path.join(
  root,
  "blackwell-digital-lockstep-gpu-fallback.json",
);
writeFileSync(report, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log();
console.log(`report=${report}`);
'''.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="utf-8").strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid SHA-256 in {HASH_FILE}")

    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid sha256 in {RELEASE_FILE}")

    return GENESIS


def replace_between(source: str, begin: str, end: str, block: str) -> str:
    has_begin = begin in source
    has_end = end in source
    if has_begin != has_end:
        raise RuntimeError(f"Only one marker present: {begin!r} / {end!r}")
    if not has_begin:
        raise RuntimeError(f"Marker pair missing: {begin!r} / {end!r}")
    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[:start] + block + source[finish:]


def extract_between(source: str, begin: str, end: str) -> tuple[int, int, str]:
    start = source.find(begin)
    if start < 0:
        raise RuntimeError(f"Missing begin marker: {begin}")
    finish_start = source.find(end, start)
    if finish_start < 0:
        raise RuntimeError(f"Missing end marker: {end}")
    finish = finish_start + len(end)
    return start, finish, source[start:finish]


def insert_before(source: str, anchor: str, block: str) -> str:
    position = source.find(anchor)
    if position < 0:
        raise RuntimeError(f"Required insertion anchor missing: {anchor!r}")
    return source[:position] + block + "\n\n\n" + source[position:]


def patch_cuda(source: str) -> str:
    required = [
        "SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_BEGIN",
        "SUNOPT_BLACKWELL_DIGITAL_LOCKSTEP_CANDIDATE_BEGIN",
        "sunopt_digital_survives(",
        "sunopt_digital_hp_breakpoint_kernel(",
    ]
    for token in required:
        if token not in source:
            raise RuntimeError(f"Expected installed CUDA generation missing {token!r}")

    if SERIAL_BEGIN in source or SERIAL_END in source:
        source = replace_between(source, SERIAL_BEGIN, SERIAL_END, SERIAL_HELPER)
    else:
        source = insert_before(
            source,
            "__global__ void\nsunopt_digital_hp_breakpoint_kernel(",
            SERIAL_HELPER,
        )

    if OLD_UNSUPPORTED_BLOCK in source:
        source = source.replace(OLD_UNSUPPORTED_BLOCK, NEW_UNSUPPORTED_BLOCK, 1)
    elif NEW_UNSUPPORTED_BLOCK not in source:
        raise RuntimeError(
            "Could not find the expected digital unsupported branch. "
            "The installed CUDA source differs from the previous candidate."
        )

    return source


def patch_function_block(source: str) -> str:
    start, finish, block = extract_between(source, RS_FN_BEGIN, RS_FN_END)

    old_search = """    let search_iterations =\n        6_u32;"""
    new_search = """    let search_iterations =\n        8_u32;"""
    if old_search not in block:
        raise RuntimeError("Expected search_iterations=6 not found.")
    block = block.replace(old_search, new_search, 1)

    fallback_start = """    /*\n     * Preserve the raw GPU status for parity statistics before replacing\n     * unsupported jobs by the exact scalar digital fallback.\n     */"""
    parity_anchor = """    /*\n     * Production-shaped parity sample:"""
    a = block.find(fallback_start)
    b = block.find(parity_anchor, a)
    if a < 0 or b < 0:
        raise RuntimeError("Could not locate the old CPU fallback section.")

    replacement = '''    /*
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

'''
    block = block[:a] + replacement + block[b:]

    block = block.replace(
        """     * annual steps. Unsupported GPU jobs are excluded from GPU parity because\n     * they intentionally use the scalar fallback.\n""",
        """     * annual steps. This includes the serial GPU fallback jobs.\n""",
        1,
    )

    parity_skip = '''            if (
                gpu_statuses[
                    job
                ] ==
                3
            ) {
                continue;
            }

'''
    if parity_skip not in block:
        raise RuntimeError("Expected parity status=3 skip not found.")
    block = block.replace(parity_skip, "", 1)

    eval_skip = '''        if (
            gpu_statuses[
                job
            ] ==
            3
        ) {
            continue;
        }

'''
    if eval_skip not in block:
        raise RuntimeError("Expected evaluation status=3 skip not found.")
    block = block.replace(eval_skip, "", 1)

    old_average = '''    let average_survival_evaluations =
        if (
            gpu_supported_jobs >
            0
        ) {
            evaluation_sum
                as f64 /
            gpu_supported_jobs
                as f64
        } else {
            0.0
        };

    let candidate_causal_ms =
        direct_wall_ms +
        fallback_cpu_ms;
'''
    new_average = '''    let average_survival_evaluations =
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
'''
    if old_average not in block:
        raise RuntimeError("Expected old average/candidate timing section not found.")
    block = block.replace(old_average, new_average, 1)

    old_initializer = '''            gpu_supported_jobs,

            fallback_jobs,

            static_h2d_ms,

            direct_gpu_total_ms,

            direct_host_wait_ms,

            direct_wall_ms,

            fallback_cpu_ms,

            candidate_causal_ms,
'''
    new_initializer = '''            gpu_compressed_jobs,

            gpu_serial_fallback_jobs,

            unexpected_unsupported_jobs,

            static_h2d_ms,

            direct_gpu_total_ms,

            direct_host_wait_ms,

            direct_wall_ms,

            candidate_causal_ms,
'''
    if old_initializer not in block:
        raise RuntimeError("Expected old result initializer fields not found.")
    block = block.replace(old_initializer, new_initializer, 1)

    for token in (
        "fallback_started",
        "fallback_cpu_ms",
        "let mut fallback_jobs",
        "gpu_supported_jobs",
    ):
        if token in block:
            raise RuntimeError(f"CPU fallback token still present after patch: {token}")

    return source[:start] + block + source[finish:]


def patch_rust(source: str) -> str:
    required = [
        RS_RESULT_BEGIN,
        RS_FN_BEGIN,
        "fn digital_run_scalar(",
        "struct ResearchTankParameters",
        "BLACKWELL_DYNAMIC_STRIDE",
    ]
    for token in required:
        if token not in source:
            raise RuntimeError(f"Expected installed Rust generation missing {token!r}")

    source = replace_between(source, RS_RESULT_BEGIN, RS_RESULT_END, RS_RESULT_BLOCK)
    source = patch_function_block(source)
    return source


def build_package() -> Path:
    for path in (CU_PATH, RS_PATH):
        if not path.is_file():
            raise RuntimeError(f"Missing project file: {path}")

    cu = patch_cuda(CU_PATH.read_text(encoding="utf-8"))
    rs = patch_rust(RS_PATH.read_text(encoding="utf-8"))

    payloads = {
        CU_REL: cu.encode("utf-8"),
        RS_REL: rs.encode("utf-8"),
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
    output = INBOX / "20261109-blackwell-digital-gpu-fallback.zip"

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("manifest.sha256", (release + "\n").encode("ascii"))
        for name, data in payloads.items():
            archive.writestr(name, data)

    print("SunOpt all-GPU digital fallback package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print("  CPU causal fallback: REMOVED from candidate")
    print("  breakpoint search: 8 iterations max")
    print("  continuous production worker: UNCHANGED")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(r"  cd .\rust\sunopt-node")
    print(r"  npm run build")
    print(r"  node .\scripts\blackwell-digital-lockstep-gpu-fallback.mjs .\blackwell-benchmark.bin 1 0")
    return output


if __name__ == "__main__":
    build_package()
