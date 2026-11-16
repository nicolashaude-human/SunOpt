#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path.cwd()

BUILD_RS = (
    ROOT /
    "rust" /
    "sunopt-node" /
    "build.rs"
)

INBOX = (
    ROOT /
    "updates" /
    "inbox"
)

HASH_FILE = (
    ROOT /
    "updates" /
    "history" /
    "hash.txt"
)

RELEASE_FILE = (
    ROOT /
    "updates" /
    "installed-release.json"
)

BUILD_REL = "rust/sunopt-node/build.rs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

PREPROCESSOR_MARKER = (
    "// SUNOPT_MSVC_STANDARD_PREPROCESSOR"
)

CPP17_MARKER = (
    "// SUNOPT_CUDA_CPP17"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(
            encoding="utf-8",
        ).strip().lower()

        if HASH_RE.fullmatch(value):
            return value

        raise RuntimeError(
            f"Invalid SHA-256 in {HASH_FILE}"
        )

    if RELEASE_FILE.is_file():
        state = json.loads(
            RELEASE_FILE.read_text(
                encoding="utf-8",
            )
        )

        value = str(
            state.get(
                "sha256",
                "",
            )
        ).strip().lower()

        if HASH_RE.fullmatch(value):
            return value

        raise RuntimeError(
            f"Invalid sha256 in {RELEASE_FILE}"
        )

    return GENESIS


def patch_build_rs(source: str) -> str:
    if CPP17_MARKER in source:
        print(
            "build.rs already contains the "
            "CUDA C++17 fix"
        )
        return source

    if PREPROCESSOR_MARKER not in source:
        raise RuntimeError(
            "The previous /Zc:preprocessor fix "
            "is not present. Refusing to patch "
            "an unexpected build.rs generation."
        )

    if "Command::new(&nvcc)" not in source:
        raise RuntimeError(
            "Could not find Command::new(&nvcc) "
            "in rust/sunopt-node/build.rs"
        )

    if (
        '-std=c++17' in source or
        '--std=c++17' in source
    ):
        raise RuntimeError(
            "build.rs already contains a C++17 "
            "nvcc flag, but without the SunOpt marker. "
            "Refusing to patch blindly."
        )

    anchor = '.arg("-O3")'

    position = source.find(anchor)

    if position < 0:
        raise RuntimeError(
            'Could not find `.arg("-O3")` '
            "in the nvcc command chain."
        )

    command_position = source.rfind(
        "Command::new(&nvcc)",
        0,
        position,
    )

    if command_position < 0:
        raise RuntimeError(
            "The -O3 argument was not found "
            "after Command::new(&nvcc)."
        )

    insertion = (
        anchor +
        "\n"
        "        " +
        CPP17_MARKER +
        "\n"
        '        .arg("-std=c++17")'
    )

    patched = (
        source[:position] +
        insertion +
        source[
            position +
            len(anchor):
        ]
    )

    if patched.count("-std=c++17") != 1:
        raise RuntimeError(
            "Unexpected number of -std=c++17 "
            "occurrences after patch."
        )

    if "/Zc:preprocessor" not in patched:
        raise RuntimeError(
            "Previous /Zc:preprocessor fix "
            "disappeared unexpectedly."
        )

    return patched


def build_package() -> Path:
    if not BUILD_RS.is_file():
        raise RuntimeError(
            f"Missing project file: {BUILD_RS}"
        )

    original = BUILD_RS.read_text(
        encoding="utf-8"
    )

    patched = patch_build_rs(original)

    payload = patched.encode("utf-8")

    parent = read_parent_hash()

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {
            BUILD_REL:
                sha256_bytes(payload),
        },
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) +
        "\n"
    ).encode("utf-8")

    release = sha256_bytes(
        manifest_bytes
    )

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        INBOX /
        "20261109-cuda-cpp17.zip"
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
            (release + "\n").encode(
                "ascii"
            ),
        )

        archive.writestr(
            BUILD_REL,
            payload,
        )

    print(
        "SunOpt CUDA C++17 build fix created"
    )

    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {release}")

    print(
        "  change : nvcc -> -std=c++17"
    )

    print(
        "  keeps  : -Xcompiler /Zc:preprocessor"
    )

    print(
        "  CUDA/B11 source: UNCHANGED"
    )

    return output


if __name__ == "__main__":
    build_package()
