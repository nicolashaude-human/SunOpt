# tools/make_update_package.py
"""Create a self-contained SunOpt hash-chain update package.

Example:
    python tools/make_update_package.py --name project-status \
        src/app/project-status/page.tsx

Format 2 writes ONE ZIP directly to updates/inbox.  The ZIP contains:
    manifest.json
    manifest.sha256
    <payload files>

No external .sha256 sidecar is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "updates" / "inbox"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"
GENESIS_HASH = "0" * 64
HASH_RE = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def installed_hash() -> str:
    if not RELEASE_FILE.exists():
        return GENESIS_HASH
    state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    value = state.get("sha256")
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError("Corrupt updates/installed-release.json")
    return value


def safe_rel(value: str) -> str:
    if "\\" in value:
        value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Unsafe project-relative path: {value}")
    return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="short package name")
    parser.add_argument("--delete", action="append", default=[], help="project-relative file to delete")
    parser.add_argument("files", nargs="*", help="project-relative files to include")
    args = parser.parse_args()

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", args.name).strip("-")
    if not slug:
        raise ValueError("Package name is empty after sanitizing")

    files: dict[str, bytes] = {}
    for value in args.files:
        rel = safe_rel(value)
        source = ROOT / Path(*PurePosixPath(rel).parts)
        if not source.is_file():
            raise FileNotFoundError(source)
        files[rel] = source.read_bytes()

    deletes = [safe_rel(value) for value in args.delete]
    if set(files).intersection(deletes):
        raise ValueError("A path cannot be both included and deleted")
    if not files and not deletes:
        raise ValueError("No changes specified")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_name = f"sunopt-update-{stamp}-{slug}.zip"
    INBOX.mkdir(parents=True, exist_ok=True)
    archive = INBOX / archive_name

    manifest = {
        "format": 2,
        "package": slug,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_sha256": installed_hash(),
        "files": {name: sha256_bytes(data) for name, data in files.items()},
    }
    if deletes:
        manifest["delete"] = deletes

    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_hash = sha256_bytes(manifest_bytes)
    embedded_hash = f"{manifest_hash}  manifest.json\n".encode("ascii")

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("manifest.json", manifest_bytes)
        z.writestr("manifest.sha256", embedded_hash)

    print(archive)
    print(f"parent_sha256={manifest['parent_sha256']}")
    print(f"release_sha256={manifest_hash}")
    print("format=2 (single ZIP; embedded manifest.sha256)")


if __name__ == "__main__":
    main()
