#!/usr/bin/env python3
# tools/make_blackwell_max_affine_b1_analysis.py
#
# Patches the already-installed B1 max-affine CUDA-tree benchmark:
#   - fixes N-API acronym property names in the .mjs reporter
#   - adds job-category counts
#   - projects the naive "tree + existing 20-step correction search" kernel cost
#
# Run from:
#   C:\NextJS\SunOpt\sunopt

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
MJS_PATH = ROOT / "rust" / "sunopt-node" / "scripts" / "blackwell-max-affine-cuda-tree.mjs"

RS_REL = "rust/sunopt-node/src/cuda_context.rs"
MJS_REL = "rust/sunopt-node/scripts/blackwell-max-affine-cuda-tree.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


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


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return source.replace(old, new, 1)


def patch_rust(source: str) -> str:
    required = [
        "SUNOPT_BLACKWELL_MAX_AFFINE_TREE_RESULT_BEGIN",
        "SUNOPT_BLACKWELL_MAX_AFFINE_TREE_FN_BEGIN",
        "pub supported_jobs: u32,",
        "let mut normal_survival_mismatches =",
        "candidate_kernel_ms,",
    ]

    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Current cuda_context.rs is not the installed B1 generation; missing {token!r}"
            )

    if "pub correction_search_jobs: u32," in source:
        return source

    source = replace_once(
        source,
        '''    pub supported_jobs: u32,
    pub unsupported_jobs: u32,

    pub reference_full_causal_gpu_ms: f64,
''',
        '''    pub supported_jobs: u32,
    pub unsupported_jobs: u32,

    pub normal_survives_jobs: u32,
    pub normal_fails_jobs: u32,
    pub maxpv_fails_jobs: u32,
    pub correction_search_jobs: u32,

    pub projected_survival_check_count: u32,
    pub projected_tree_kernel_ms: f64,

    pub reference_full_causal_gpu_ms: f64,
''',
        "result struct fields",
    )

    source = replace_once(
        source,
        '''    let mut normal_survival_mismatches =
        0_u32;

    let mut maxpv_survival_mismatches =
        0_u32;
''',
        '''    let mut normal_survives_jobs =
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
''',
        "counter declarations",
    )

    source = replace_once(
        source,
        '''        let gpu_maxpv =
            gpu_survival[
                maxpv_index
            ];

        if (
            expected_normal != 2 &&
            gpu_normal == 2
        ) {
''',
        '''        let gpu_maxpv =
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
''',
        "job category counting",
    )

    source = replace_once(
        source,
        '''    let candidate_ns_per_survival_check =
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
''',
        '''    let candidate_ns_per_survival_check =
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
''',
        "projection calculation",
    )

    source = replace_once(
        source,
        '''            unsupported_jobs:
                job_count
                    as u32 -
                supported_jobs,

            reference_full_causal_gpu_ms,
''',
        '''            unsupported_jobs:
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
''',
        "result literal fields",
    )

    return source


def patch_mjs(source: str) -> str:
    if "SUNOPT_MAX_AFFINE_CUDA_TREE_JSON" not in source:
        raise RuntimeError(
            "Current blackwell-max-affine-cuda-tree.mjs is not the expected B1 reporter."
        )

    source = source.replace(
        "result.candidateH2dMs",
        "result.candidateH2DMs",
    )

    source = source.replace(
        "result.candidateD2hMs",
        "result.candidateD2HMs",
    )

    if "correctionSearchJobs=" not in source:
        anchor = '''console.log(
  `unsupportedJobs=${result.unsupportedJobs}`,
);

console.log();
console.log(
  "REFERENCE (existing complete causal kernel)",
);
'''

        addition = '''console.log(
  `unsupportedJobs=${result.unsupportedJobs}`,
);

console.log();
console.log(
  "JOB CLASSIFICATION",
);

console.log(
  `normalSurvivesJobs=${result.normalSurvivesJobs}`,
);

console.log(
  `normalFailsJobs=${result.normalFailsJobs}`,
);

console.log(
  `maxPvFailsJobs=${result.maxpvFailsJobs}`,
);

console.log(
  `correctionSearchJobs=${result.correctionSearchJobs}`,
);

console.log(
  `projectedSurvivalCheckCount=${result.projectedSurvivalCheckCount}`,
);

console.log(
  `projectedTreeKernelMs=${result.projectedTreeKernelMs.toFixed(6)}`,
);

console.log();
console.log(
  "REFERENCE (existing complete causal kernel)",
);
'''

        if anchor not in source:
            raise RuntimeError(
                "Could not find B1 reporter insertion anchor."
            )

        source = source.replace(
            anchor,
            addition,
            1,
        )

    return source


def build_package() -> Path:
    if not RS_PATH.is_file():
        raise RuntimeError(f"Missing {RS_PATH}")

    if not MJS_PATH.is_file():
        raise RuntimeError(f"Missing {MJS_PATH}")

    rs = patch_rust(
        RS_PATH.read_text(
            encoding="utf-8",
        )
    )

    mjs = patch_mjs(
        MJS_PATH.read_text(
            encoding="utf-8",
        )
    )

    payloads = {
        RS_REL: rs.encode("utf-8"),
        MJS_REL: mjs.encode("utf-8"),
    }

    parent = read_parent_hash()

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {
            name: sha256_bytes(data)
            for name, data in payloads.items()
        },
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    release = sha256_bytes(
        manifest_bytes
    )

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        INBOX
        / "20261109-blackwell-max-affine-b1-analysis.zip"
    )

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "manifest.json",
            manifest_bytes,
        )

        archive.writestr(
            "manifest.sha256",
            (release + "\n").encode("ascii"),
        )

        for name, data in payloads.items():
            archive.writestr(
                name,
                data,
            )

    print("SunOpt B1 max-affine analysis package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(r"  cd .\rust\sunopt-node")
    print(r"  npm run build")
    print(r"  node .\scripts\blackwell-max-affine-cuda-tree.mjs .\blackwell-benchmark.bin 192 20 0")

    return output


if __name__ == "__main__":
    build_package()
