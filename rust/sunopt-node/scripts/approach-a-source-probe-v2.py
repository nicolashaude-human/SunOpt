#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib


def find_project_root(start: Path) -> Path:
    current = start.resolve()

    for candidate in [current, *current.parents]:
        if (
            (candidate / "package.json").is_file()
            and
            (candidate / "rust" / "sunopt-node").is_dir()
        ):
            return candidate

    raise RuntimeError(
        f"SunOpt project root not found from {start}",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def extract_around(
    text: str,
    token: str,
    before: int = 600,
    after: int = 8000,
) -> str | None:
    index = text.find(token)

    if index < 0:
        return None

    lo = max(
        0,
        index - before,
    )

    hi = min(
        len(text),
        index + after,
    )

    return text[lo:hi]


def main() -> None:
    root = find_project_root(
        Path(__file__).parent,
    )

    targets = [
        root
        / "rust"
        / "sunopt-node"
        / "src"
        / "cuda_context.rs",

        root
        / "src"
        / "simulation"
        / "tank-comparison-gpu-lockstep.ts",
    ]

    symbols = [
        "pub struct CudaCausalWorker",
        "pub fn new(",
        "pub fn configure_lockstep_packed(",
        "pub fn run_lockstep_dynamic_into(",
        "SUNOPT_CUDA_CAUSAL_PATH",
        "use_b14p",
        "b14p_context",
        "run_causal_dynamic_batch_into",
        "sunopt_cuda_run_causal_dynamic_batch",
        "sunopt_cuda_run_causal_graph_replay",
        "causal_batch_kernel",
    ]

    now = datetime.now(
        timezone.utc,
    ).isoformat()

    lines: list[str] = [
        "",
        f"=== APPROACH_A_SOURCE_PROBE_V2 {now} ===",
        f"project_root={root}",
        "",
    ]

    for path in targets:
        rel = path.relative_to(
            root,
        )

        lines.append(
            f"--- FILE {rel} ---",
        )

        if not path.is_file():
            lines.append(
                "missing=true",
            )
            lines.append("")
            continue

        lines.append(
            f"sha256={sha256_file(path)}",
        )

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        for symbol in symbols:
            snippet = extract_around(
                text,
                symbol,
            )

            if snippet is None:
                continue

            lines.append(
                f">>> BEGIN {symbol}",
            )
            lines.append(
                snippet.rstrip(),
            )
            lines.append(
                f"<<< END {symbol}",
            )
            lines.append("")

    lines.append(
        "=== END APPROACH_A_SOURCE_PROBE_V2 ===",
    )
    lines.append("")

    payload = "\n".join(
        lines,
    )

    dedicated = (
        root /
        "llm-source-probe.txt"
    )

    dedicated.write_text(
        payload,
        encoding="utf-8",
    )

    llm = (
        root /
        "llm.txt"
    )

    with llm.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            payload,
        )

    print(
        "APPROACH_A_SOURCE_PROBE_V2=OK",
    )
    print(
        f"project_root={root}",
    )
    print(
        f"dedicated_output={dedicated}",
    )
    print(
        f"llm_output={llm}",
    )
    print(
        f"dedicated_exists={dedicated.is_file()}",
    )
    print(
        f"llm_exists={llm.is_file()}",
    )
    print(
        f"dedicated_size={dedicated.stat().st_size}",
    )
    print(
        f"llm_size={llm.stat().st_size}",
    )


if __name__ == "__main__":
    main()
