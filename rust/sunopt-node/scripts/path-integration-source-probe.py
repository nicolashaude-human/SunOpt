#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
OUTBOX = ROOT / "updates" / "outbox"

TARGETS = [
    "src/app/api/tank-comparison/route.ts",
    "src/simulation/tank-comparison-gpu-lockstep.ts",
    "src/app/tank-comparison/page.tsx",
    "src/types/tank-comparison.ts",
]

PATTERNS = [
    "causalPath",
    "TankComparisonGpuCausalPath",
    "createCudaCausalWorker",
    "new CudaCausalWorker",
    "/api/tank-comparison",
    "Approach A",
    "Approach B",
    "B14P",
    "FP64",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    OUTBOX.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=== PATH_INTEGRATION_SOURCE_PROBE ===",
        f"time={datetime.now(timezone.utc).isoformat()}",
        f"root={ROOT}",
        "",
    ]

    summary = {
        "stage": "PATH_INTEGRATION_SOURCE_PROBE",
        "time": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "hits": [],
    }

    for rel in TARGETS:
        path = ROOT / rel
        report_lines.append(f"--- FILE {rel} ---")

        if not path.is_file():
            report_lines.append("missing=true")
            report_lines.append("")
            summary["files"][rel] = {"missing": True}
            continue

        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        digest = sha256(data)

        summary["files"][rel] = {
            "sha256": digest,
            "bytes": len(data),
            "lines": text.count("\n") + 1,
        }

        report_lines.append(f"sha256={digest}")
        report_lines.append(f"bytes={len(data)}")
        report_lines.append(f"lines={text.count(chr(10)) + 1}")
        report_lines.append(">>> BEGIN FULL FILE")
        report_lines.append(text)
        report_lines.append("<<< END FULL FILE")
        report_lines.append("")

    # Search all project TypeScript/TSX sources for the current path wiring.
    src = ROOT / "src"
    if src.is_dir():
        for path in sorted(
            p for p in src.rglob("*")
            if p.is_file() and p.suffix.lower() in {".ts", ".tsx"}
        ):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue

            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if any(pattern in line for pattern in PATTERNS):
                    hit = {
                        "file": rel,
                        "line": line_no,
                        "text": line,
                    }
                    summary["hits"].append(hit)

    report_lines.append("=== GLOBAL HITS ===")
    for hit in summary["hits"]:
        report_lines.append(
            f'{hit["file"]}:{hit["line"]}: {hit["text"]}'
        )
    report_lines.append("")

    txt_path = OUTBOX / "path-integration-source-probe.txt"
    json_path = OUTBOX / "path-integration-source-probe.json"
    llm_path = OUTBOX / "llm.txt"

    txt_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    with llm_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            f"=== {datetime.now(timezone.utc).isoformat()} "
            "PATH_INTEGRATION_SOURCE_PROBE ===\n"
            f"report={txt_path}\n"
            f"json={json_path}\n"
        )
        for rel, meta in summary["files"].items():
            f.write(
                f"{rel}="
                + (
                    "MISSING"
                    if meta.get("missing")
                    else meta["sha256"]
                )
                + "\n"
            )
        f.write(f"globalHits={len(summary['hits'])}\n")

    print("PATH_INTEGRATION_SOURCE_PROBE=OK")
    print(f"report={txt_path}")
    print(f"json={json_path}")
    print(f"llm={llm_path}")


if __name__ == "__main__":
    main()
