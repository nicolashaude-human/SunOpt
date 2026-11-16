#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, zipfile

ROOT = Path(__file__).resolve().parents[3]
INBOX = ROOT / "updates" / "inbox"
OUTBOX = ROOT / "updates" / "outbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"

CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
TS_REL = "src/simulation/tank-comparison-gpu-lockstep.ts"
CU_PATH = ROOT / CU_REL
TS_PATH = ROOT / TS_REL

EXPECTED_TS_SHA = "a63e072b5519bfbfc10738acdd55b534d881e24559dd0cfd9b6bab673138c753"
EXPECTED_OLD_AG_BLOCK_SHA = "b5373d691657e47fbc3582128e5b6ac09b8840efd6913954f2510b41366ff08c"

BEGIN = "// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN"
END = "// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END"
CREATE_TOKEN = 'extern "C" int sunopt_cuda_create_causal_step_graph('
RUN_TOKEN = 'extern "C" int sunopt_cuda_run_causal_step_graph('

TEMPLATE_PATH = Path(__file__).with_name(
    "approach-a-step-graph-create-capture.inc"
)

TS_OLD = '  const previousValue =\n    process.env[\n      environmentKey\n    ];\n\n  process.env[\n    environmentKey\n  ] =\n    causalPath;'
TS_NEW = '  const previousValue =\n    process.env[\n      environmentKey\n    ];\n\n  /*\n   * Final Approach-A benchmark override.\n   *\n   * A/AG run in separate processes and set\n   * SUNOPT_APPROACH_A_STEP_GRAPH to 0 or 1. In those benchmark\n   * processes preserve process-level SUNOPT_CUDA_CAUSAL_PATH=A\n   * instead of overwriting it with the UI default B.\n   */\n  const benchmarkForcesExactA =\n    process.env[\n      "SUNOPT_APPROACH_A_STEP_GRAPH"\n    ] !==\n      undefined &&\n    previousValue\n      ?.trim()\n      .toUpperCase() ===\n      "A";\n\n  process.env[\n    environmentKey\n  ] =\n    benchmarkForcesExactA\n      ? "A"\n      : causalPath;'

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def fail(message: str):
    OUTBOX.mkdir(parents=True, exist_ok=True)
    (OUTBOX / "failed-prompt.txt").write_text(
        "stage=APPROACH_A_FINAL_CAPTURE_HOTFIX\n"
        + f"time={datetime.now(timezone.utc).isoformat()}\n"
        + f"error={message}\n",
        encoding="utf-8",
    )
    raise RuntimeError(message)

def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)

    if not HASH_FILE.is_file():
        fail(f"Missing current release hash: {HASH_FILE}")

    parent = HASH_FILE.read_text(
        encoding="utf-8"
    ).strip().lower()

    if not re.fullmatch(
        r"[0-9a-f]{64}",
        parent,
    ):
        fail(
            f"Invalid current release hash: {parent!r}"
        )

    for p in (CU_PATH, TS_PATH, TEMPLATE_PATH):
        if not p.is_file():
            fail(f"Missing required file: {p}")

    cu = CU_PATH.read_text(encoding="utf-8")
    ts = TS_PATH.read_text(encoding="utf-8")

    ts_sha = sha(ts.encode("utf-8"))
    if ts_sha != EXPECTED_TS_SHA:
        fail(
            f"TS precondition failed: expected {EXPECTED_TS_SHA}, got {ts_sha}"
        )

    begin = cu.find(BEGIN)
    end = cu.find(END, begin)
    if begin < 0 or end < 0:
        fail("AG graph markers missing")

    end += len(END)
    old_block = cu[begin:end]
    old_block_sha = sha(old_block.encode("utf-8"))

    if old_block_sha != EXPECTED_OLD_AG_BLOCK_SHA:
        fail(
            f"AG block precondition failed: expected "
            f"{EXPECTED_OLD_AG_BLOCK_SHA}, got {old_block_sha}"
        )

    create_start = old_block.find(CREATE_TOKEN)
    run_start = old_block.find(RUN_TOKEN, create_start)

    if create_start < 0 or run_start < 0:
        fail("Could not locate AG create/run functions")

    new_create = TEMPLATE_PATH.read_text(
        encoding="utf-8"
    ).strip()

    new_block = (
        old_block[:create_start]
        + new_create
        + "\n\n\n"
        + old_block[run_start:]
    )

    cu_new = cu[:begin] + new_block + cu[end:]

    if ts.count(TS_OLD) != 1:
        fail(
            f"TS benchmark anchor count={ts.count(TS_OLD)}; expected 1"
        )

    ts_new = ts.replace(TS_OLD, TS_NEW, 1)

    files = {
        CU_REL: cu_new.encode("utf-8"),
        TS_REL: ts_new.encode("utf-8"),
    }

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {
            name: sha(data)
            for name, data in files.items()
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

    release = sha(manifest_bytes)
    package = INBOX / (
        "20261113-approach-a-final-capture-hotfix.zip"
    )

    with zipfile.ZipFile(
        package,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr("manifest.json", manifest_bytes)
        z.writestr("manifest.sha256", release + "\n")
        for name, data in files.items():
            z.writestr(name, data)

    with zipfile.ZipFile(package, "r") as z:
        if sha(z.read("manifest.json")) != z.read(
            "manifest.sha256"
        ).decode().strip():
            fail("Manifest verification failed")

        for name, expected in manifest["files"].items():
            if sha(z.read(name)) != expected:
                fail(f"Payload hash mismatch: {name}")

    report = {
        "stage": "APPROACH_A_FINAL_CAPTURE_HOTFIX",
        "time": datetime.now(timezone.utc).isoformat(),
        "parent": parent,
        "candidateRelease": release,
        "candidatePackage": str(package),
        "tsPreconditionSha256": ts_sha,
        "oldAgBlockSha256": old_block_sha,
        "newAgBlockSha256": sha(new_block.encode("utf-8")),
        "changes": [
            "benchmark preserves exact A instead of UI default B",
            "AG graph create uses one-kernel CUDA stream capture",
        ],
    }

    report_path = OUTBOX / (
        "approach-a-final-capture-hotfix.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    llm = OUTBOX / "llm.txt"
    with llm.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            + f"=== {datetime.now(timezone.utc).isoformat()} "
              "APPROACH_A_FINAL_CAPTURE_HOTFIX ===\n"
            + f"parent={parent}\n"
            + f"candidateRelease={release}\n"
            + f"candidatePackage={package}\n"
            + f"tsPrecondition={ts_sha}\n"
            + f"oldAgBlockSha256={old_block_sha}\n"
            + f"newAgBlockSha256={sha(new_block.encode('utf-8'))}\n"
            + f"report={report_path}\n"
        )

    print("APPROACH_A_FINAL_CAPTURE_HOTFIX=OK")
    print(f"parent={parent}")
    print(f"candidate_release={release}")
    print(f"candidate_package={package}")
    print(f"report={report_path}")
    print()
    print("After candidate installation:")
    print(r"node .\rust\sunopt-node\scripts\build-native-fresh.mjs")
    print(r"node .\rust\sunopt-node\scripts\approach-a-final-production-ab.mjs")

if __name__ == "__main__":
    main()
