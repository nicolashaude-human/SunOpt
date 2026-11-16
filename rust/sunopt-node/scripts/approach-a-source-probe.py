#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
LLM = ROOT / "llm.txt"

TARGETS = [
    ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs",
    ROOT / "src" / "simulation" / "tank-comparison-gpu-lockstep.ts",
]

SEARCH_ROOTS = [
    ROOT / "rust" / "sunopt-node",
]

SYMBOLS = [
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

EXTENSIONS = {".rs", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".ts", ".tsx", ".mjs"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    in_string = False
    escape = False
    quote = ""

    for i in range(open_index, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == quote:
                in_string = False
            continue

        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

    return None


def extract_block(text: str, token: str, max_chars: int = 30000) -> str | None:
    start = text.find(token)
    if start < 0:
        return None

    brace = text.find("{", start)
    if brace < 0 or brace - start > 1200:
        lo = max(0, start - 1200)
        hi = min(len(text), start + max_chars)
        return text[lo:hi]

    end = find_matching_brace(text, brace)
    if end is None:
        lo = max(0, start - 1200)
        hi = min(len(text), start + max_chars)
        return text[lo:hi]

    lo = max(0, start - 300)
    hi = min(len(text), end + 1)
    return text[lo:hi]


def collect_candidate_files() -> list[Path]:
    found = set()

    for path in TARGETS:
        if path.is_file():
            found.add(path)

    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if any(symbol in text for symbol in SYMBOLS):
                    found.add(path)

    return sorted(found)


def main() -> None:
    files = collect_candidate_files()

    if not files:
        raise RuntimeError("No relevant SunOpt source files found.")

    now = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("")
    lines.append(f"=== APPROACH_A_SOURCE_PROBE {now} ===")
    lines.append(f"root={ROOT}")
    lines.append(f"candidate_file_count={len(files)}")
    lines.append("")

    for path in files:
        rel = path.relative_to(ROOT)
        digest = sha256_file(path)

        lines.append(f"--- FILE {rel} ---")
        lines.append(f"sha256={digest}")

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            lines.append(f"read_error={exc!r}")
            lines.append("")
            continue

        hits = [symbol for symbol in SYMBOLS if symbol in text]
        lines.append("symbols=" + ", ".join(hits))
        lines.append("")

        emitted = set()

        for symbol in hits:
            block = extract_block(text, symbol)
            if not block:
                continue

            block_key = hashlib.sha256(
                block.encode("utf-8", errors="replace")
            ).hexdigest()

            if block_key in emitted:
                continue

            emitted.add(block_key)
            lines.append(f">>> BEGIN SYMBOL {symbol}")
            lines.append(block.rstrip())
            lines.append(f"<<< END SYMBOL {symbol}")
            lines.append("")

    lines.append("=== END APPROACH_A_SOURCE_PROBE ===")
    lines.append("")

    with LLM.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("APPROACH_A_SOURCE_PROBE=OK")
    print(f"llm={LLM}")
    print(f"files={len(files)}")

    for path in files:
        print(f"{path.relative_to(ROOT)} sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
