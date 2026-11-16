#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess, zipfile

ROOT = Path(__file__).resolve().parents[3]
INBOX = ROOT / "updates" / "inbox"
OUTBOX = ROOT / "updates" / "outbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"

CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
PAIR_REL = "rust/sunopt-node/scripts/approach-a-final-pair.mjs"
CU_PATH = ROOT / CU_REL
PAIR_PATH = ROOT / PAIR_REL
RUN_TEMPLATE = Path(__file__).with_name(
    "approach-a-step-graph-run-nodeparams.inc"
)

BEGIN = "// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN"
END = "// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END"
RUN_TOKEN = 'extern "C" int sunopt_cuda_run_causal_step_graph('
DESTROY_TOKEN = 'extern "C" int sunopt_cuda_destroy_causal_step_graph('

EXPECTED_BLOCK_SHA = "ec85c5ca26ad50d692bafe5415df60f47063cceeeb377144ff898f70c280e0d1"
EXPECTED_RUN_SHA = "4b72f1690ba82e0320da9bfe56a8b329296cf009e865b9ecd69470394e819147"

READY_OLD = """        await fetch(
          origin,
          {
            cache:
              "no-store",
          }
        );"""

READY_NEW = """        await fetch(
          `${origin}/api/weather`,
          {
            cache:
              "no-store",
          }
        );"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fail(message):
    OUTBOX.mkdir(parents=True, exist_ok=True)
    (OUTBOX / "failed-prompt.txt").write_text(
        "stage=APPROACH_A_FINAL_RUN_HOTFIX\n"
        f"time={datetime.now(timezone.utc).isoformat()}\n"
        f"error={message}\n",
        encoding="utf-8",
    )
    raise RuntimeError(message)


def current_release():
    if not HASH_FILE.is_file():
        fail(f"Missing release hash: {HASH_FILE}")
    value = HASH_FILE.read_text(
        encoding="utf-8"
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        fail(f"Invalid release hash: {value!r}")
    return value


def patch_cuda(source):
    begin = source.find(BEGIN)
    end = source.find(END, begin)
    if begin < 0 or end < 0:
        fail("Approach-A graph markers missing")
    end += len(END)

    block = source[begin:end]
    block_sha = sha(block.encode("utf-8"))
    if block_sha != EXPECTED_BLOCK_SHA:
        fail(
            f"AG block hash mismatch: expected "
            f"{EXPECTED_BLOCK_SHA}, got {block_sha}"
        )

    run_start = block.find(RUN_TOKEN)
    destroy_start = block.find(
        DESTROY_TOKEN,
        run_start,
    )
    if run_start < 0 or destroy_start < 0:
        fail("AG run/destroy anchors missing")

    old_run = block[
        run_start:destroy_start
    ].rstrip()
    run_sha = sha(old_run.encode("utf-8"))
    if run_sha != EXPECTED_RUN_SHA:
        fail(
            f"AG run hash mismatch: expected "
            f"{EXPECTED_RUN_SHA}, got {run_sha}"
        )

    new_run = RUN_TEMPLATE.read_text(
        encoding="utf-8"
    ).strip()

    new_block = (
        block[:run_start]
        + new_run
        + "\n\n\n"
        + block[destroy_start:]
    )

    return (
        source[:begin] + new_block + source[end:],
        block_sha,
        sha(new_block.encode("utf-8")),
    )


def patch_pair(source):
    count = source.count(READY_OLD)
    if count != 1:
        fail(
            f"Readiness anchor mismatch: "
            f"expected 1, found {count}"
        )

    patched = source.replace(
        READY_OLD,
        READY_NEW,
        1,
    )

    check = OUTBOX / "_pair-check.mjs"
    check.write_text(
        patched,
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", "--check", str(check)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    check.unlink(missing_ok=True)

    if result.returncode != 0:
        fail(
            "Pair runner syntax error:\n"
            + result.stdout
            + "\n"
            + result.stderr
        )

    return patched


def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)

    for name in (
        "failed-prompt.txt",
        "failed-build.txt",
        "failed-run.txt",
    ):
        try:
            (OUTBOX / name).unlink()
        except FileNotFoundError:
            pass

    parent = current_release()

    for path in (
        CU_PATH,
        PAIR_PATH,
        RUN_TEMPLATE,
    ):
        if not path.is_file():
            fail(f"Missing required file: {path}")

    cu = CU_PATH.read_text(encoding="utf-8")
    pair = PAIR_PATH.read_text(encoding="utf-8")

    cu_new, old_block_sha, new_block_sha = patch_cuda(cu)
    pair_new = patch_pair(pair)

    files = {
        CU_REL: cu_new.encode("utf-8"),
        PAIR_REL: pair_new.encode("utf-8"),
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
        "20261113-approach-a-final-run-hotfix.zip"
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
        "stage": "APPROACH_A_FINAL_RUN_HOTFIX",
        "time": datetime.now(timezone.utc).isoformat(),
        "parent": parent,
        "candidateRelease": release,
        "candidatePackage": str(package),
        "oldAgBlockSha256": old_block_sha,
        "newAgBlockSha256": new_block_sha,
        "nodeSyntaxCheck": "PASS",
        "changes": [
            (
                "AG run obtains captured node metadata with "
                "cudaGraphKernelNodeGetParams before exec update"
            ),
            (
                "readiness probes /api/weather instead of / "
                "to avoid the UI/Turbopack .vs panic"
            ),
        ],
    }

    report_path = OUTBOX / (
        "approach-a-final-run-hotfix.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    llm = OUTBOX / "llm.txt"
    with llm.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            f"=== {datetime.now(timezone.utc).isoformat()} "
            "APPROACH_A_FINAL_RUN_HOTFIX ===\n"
            f"parent={parent}\n"
            f"candidateRelease={release}\n"
            f"candidatePackage={package}\n"
            f"oldAgBlockSha256={old_block_sha}\n"
            f"newAgBlockSha256={new_block_sha}\n"
            "nodeSyntaxCheck=PASS\n"
            f"report={report_path}\n"
        )

    print("APPROACH_A_FINAL_RUN_HOTFIX=OK")
    print(f"parent={parent}")
    print(f"candidate_release={release}")
    print(f"candidate_package={package}")
    print(f"report={report_path}")
    print()
    print("After candidate installation:")
    print(
        r"node .\rust\sunopt-node\scripts\build-native-fresh.mjs"
    )
    print(
        r"node .\rust\sunopt-node\scripts\approach-a-final-production-ab.mjs"
    )


if __name__ == "__main__":
    main()
