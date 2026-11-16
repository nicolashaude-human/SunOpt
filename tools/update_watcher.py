# tools/update_watcher.py
"""Verified hash-chain ZIP inbox installer for SunOpt.

Usage:
    python tools/update_watcher.py --once
    python tools/update_watcher.py --watch
    python tools/update_watcher.py --status
    python tools/update_watcher.py --capture "COMMAND" [--cwd PROJECT_RELATIVE_DIR]

The outbox is written to ``updates/outbox/llm.txt``.  --capture executes only
the command explicitly supplied by the local user; update ZIPs can never inject
or auto-run commands.

Update package format 2 is self-contained: each ZIP contains ``manifest.json``
and ``manifest.sha256``.  The manifest contains the SHA-256 of every payload
file plus ``parent_sha256``.  The manifest hash therefore commits to the entire
payload and becomes the release/hash-chain ID.

Legacy format-1 packages with a sibling ``.zip.sha256`` sidecar remain readable
for migration, but new packages no longer require or create a sidecar.

The watcher is intentionally NOT self-updatable through normal packages.
SHA-256 protects integrity and ordering; it does not authenticate the author.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
UPDATES = ROOT / "updates"
INBOX = UPDATES / "inbox"
PROCESSING = UPDATES / "processing"
DONE = UPDATES / "installed"
REJECTED = UPDATES / "rejected"
HISTORY = UPDATES / "history"
RELEASE_FILE = UPDATES / "installed-release.json"
HISTORY_HASH_FILE = HISTORY / "hash.txt"
WATCHER_LOCK_FILE = UPDATES / "watcher.lock"
OUTBOX = UPDATES / "outbox"
LLM_OUTBOX_FILE = OUTBOX / "llm.txt"

GENESIS_HASH = "0" * 64
MAX_FILES = 500
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024

ALLOWED_PREFIXES = (
    "src/",
    "public/",
    "rust/",
    "data/localities/",
)
ALLOWED_TOP_LEVEL = {
    "package.json",
    "package-lock.json",
    "next.config.ts",
    "tsconfig.json",
    "eslint.config.mjs",
    "postcss.config.mjs",
    "README.md",
    "README.txt",
    "README_INSTALL.txt",
    "notes.md",
}
DENIED_PARTS = {
    ".git",
    ".next",
    "node_modules",
    "target",
    "updates",
    "tools",
    "cache",
    "__pycache__",
}

HASH_RE = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".updating",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def installed_hash() -> str:
    if not RELEASE_FILE.exists():
        return GENESIS_HASH

    state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    value = state.get("sha256")
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError("Corrupt updates/installed-release.json")
    return value


def write_expected_parent_hash(value: str) -> None:
    """Persist the hash that the next format-2 package must use as parent_sha256."""
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError("Invalid expected parent SHA-256")
    write_atomic(HISTORY_HASH_FILE, (value + "\n").encode("ascii"))



def outbox_append(text: str) -> None:
    """Append UTF-8 text to updates/outbox/llm.txt."""
    OUTBOX.mkdir(parents=True, exist_ok=True)
    with LLM_OUTBOX_FILE.open("a", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()


def outbox_replace(text: str) -> None:
    """Atomically replace updates/outbox/llm.txt with UTF-8 text."""
    OUTBOX.mkdir(parents=True, exist_ok=True)
    write_atomic(
        LLM_OUTBOX_FILE,
        text.encode("utf-8"),
    )


def outbox_event(kind: str, message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    outbox_append(
        f"\n=== {kind} {stamp} ===\n"
        f"{message.rstrip()}\n"
    )


def resolve_capture_cwd(value: str | None) -> Path:
    candidate = ROOT if not value else (ROOT / value)
    resolved = candidate.resolve()

    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"--cwd must stay inside the SunOpt project root: {value!r}"
        ) from exc

    if not resolved.is_dir():
        raise ValueError(
            f"--cwd does not name an existing project directory: {resolved}"
        )

    return resolved


def capture_command(
    command: str,
    cwd_value: str | None,
    append: bool,
) -> int:
    """Run an explicitly user-supplied command and tee stdout/stderr to llm.txt."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("--capture requires a non-empty command string")

    cwd = resolve_capture_cwd(cwd_value)
    stamp = datetime.now(timezone.utc).isoformat()

    try:
        current_release = installed_hash()
    except Exception as exc:
        current_release = f"<unavailable: {exc}>"

    header = (
        "# SunOpt LLM outbox\n"
        f"timestamp_utc: {stamp}\n"
        f"project_root: {ROOT}\n"
        f"cwd: {cwd}\n"
        f"installed_release: {current_release}\n"
        f"command: {command}\n"
        "\n"
        "=== COMMAND OUTPUT ===\n"
    )

    if append:
        outbox_append("\n\n" + header)
    else:
        outbox_replace(header)

    print(f"Capturing to {LLM_OUTBOX_FILE}", flush=True)
    print(f"cwd: {cwd}", flush=True)
    print(f"command: {command}", flush=True)

    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=os.environ.copy(),
    )

    assert process.stdout is not None

    OUTBOX.mkdir(parents=True, exist_ok=True)
    with LLM_OUTBOX_FILE.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        return_code = process.wait()

        footer = (
            "\n=== COMMAND COMPLETE ===\n"
            f"exit_code: {return_code}\n"
            f"completed_utc: {datetime.now(timezone.utc).isoformat()}\n"
        )

        log.write(footer)
        log.flush()

    print(
        f"Command exited with {return_code}; outbox: {LLM_OUTBOX_FILE}",
        flush=True,
    )

    return return_code


def validate_project_path(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Empty update path")
    if "\\" in name:
        raise ValueError(f"Backslashes forbidden in package path: {name}")

    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise ValueError(f"Unsafe update path: {name}")
    if any(part in DENIED_PARTS for part in posix.parts):
        raise ValueError(f"Protected path: {name}")

    allowed = (
        name in ALLOWED_TOP_LEVEL
        or any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES)
    )
    if not allowed:
        raise ValueError(f"Path is outside SunOpt update allow-list: {name}")

    target = (ROOT / Path(*posix.parts)).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escaped project root: {name}")
    return name


def parse_hash_text(data: bytes, label: str) -> str:
    try:
        value = data.decode("ascii").split()[0].lower()
    except (UnicodeDecodeError, IndexError) as exc:
        raise ValueError(f"Invalid {label}") from exc
    if not HASH_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}")
    return value


def read_legacy_sidecar(checksum: Path) -> str:
    return parse_hash_text(checksum.read_bytes(), "ZIP SHA-256 sidecar")


def move_to_done(archive: Path, checksum: Path | None = None) -> None:
    DONE.mkdir(parents=True, exist_ok=True)
    archive_target = DONE / archive.name

    if archive_target.exists() and sha256_file(archive_target) == sha256_file(archive):
        archive.unlink()
    else:
        os.replace(archive, archive_target)

    if checksum is not None and checksum.exists():
        checksum_target = DONE / checksum.name
        if checksum_target.exists() and checksum_target.read_bytes() == checksum.read_bytes():
            checksum.unlink()
        else:
            os.replace(checksum, checksum_target)


def is_transient_file_lock(exc: BaseException) -> bool:
    """Return True for transient Windows sharing/lock violations."""
    return isinstance(exc, PermissionError) or (
        isinstance(exc, OSError) and getattr(exc, "winerror", None) in (32, 33)
    )


def file_is_stable(path: Path, delay: float = 0.35) -> bool:
    """Check that size/mtime stop changing before the watcher touches a new file."""
    try:
        first = path.stat()
        if first.st_size <= 0:
            return False
        time.sleep(delay)
        second = path.stat()
    except OSError:
        return False
    return (
        first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
    )


def claim_archive(archive: Path) -> Path | None:
    """Atomically claim an inbox ZIP by moving it into updates/processing.

    On Windows, an active copier normally prevents this rename with WinError
    32/33.  That is treated as "not ready yet", not as a rejected package.
    """
    PROCESSING.mkdir(parents=True, exist_ok=True)
    claimed = PROCESSING / archive.name
    if claimed.exists():
        return None
    try:
        os.replace(archive, claimed)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if is_transient_file_lock(exc):
            return None
        raise
    return claimed


def restore_claimed_file(path: Path) -> None:
    """Best-effort return of a deferred processing file to the inbox."""
    if not path.exists():
        return
    INBOX.mkdir(parents=True, exist_ok=True)
    destination = INBOX / path.name
    if destination.exists():
        return
    try:
        os.replace(path, destination)
    except OSError:
        pass


def recover_processing() -> None:
    """Recover packages left in updates/processing by a previous crash."""
    PROCESSING.mkdir(parents=True, exist_ok=True)
    for path in sorted(PROCESSING.iterdir()):
        if path.is_file():
            restore_claimed_file(path)


def acquire_instance_lock():
    """Prevent two watcher processes from installing different packages concurrently."""
    UPDATES.mkdir(parents=True, exist_ok=True)
    handle = WATCHER_LOCK_FILE.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)

    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "Another update_watcher instance is already active"
        ) from exc

    return handle


def package_has_embedded_hash(archive: Path) -> bool:
    try:
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            return "manifest.json" in names and "manifest.sha256" in names
    except (OSError, zipfile.BadZipFile):
        # Let install/reject produce the actual error once the file is stable.
        return True


def install(archive: Path) -> None:
    legacy_checksum = archive.with_suffix(archive.suffix + ".sha256")

    with zipfile.ZipFile(archive) as z:
        members = [item for item in z.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        names_set = set(names)

        if len(names) != len(names_set):
            raise ValueError("Duplicate ZIP members")
        if "manifest.json" not in names_set:
            raise ValueError("Missing manifest.json")

        manifest_bytes = z.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        package_format = manifest.get("format", 1)

        if package_format == 2:
            if "manifest.sha256" not in names_set:
                raise ValueError("Missing embedded manifest.sha256")
            expected_manifest_hash = parse_hash_text(
                z.read("manifest.sha256"),
                "embedded manifest SHA-256",
            )
            actual_manifest_hash = sha256_bytes(manifest_bytes)
            if actual_manifest_hash != expected_manifest_hash:
                raise ValueError("Embedded manifest SHA-256 mismatch")
            release_hash = actual_manifest_hash
            checksum_for_move = None
            metadata_members = {"manifest.json", "manifest.sha256"}
        elif package_format == 1:
            if not legacy_checksum.is_file():
                raise ValueError(
                    "Legacy format-1 package requires sibling .zip.sha256; "
                    "new packages should use embedded format 2"
                )
            expected_zip_hash = read_legacy_sidecar(legacy_checksum)
            actual_zip_hash = sha256_file(archive)
            if actual_zip_hash != expected_zip_hash:
                raise ValueError("ZIP SHA-256 mismatch")
            release_hash = actual_zip_hash
            checksum_for_move = legacy_checksum
            metadata_members = {"manifest.json"}
        else:
            raise ValueError(f"Unsupported update package format: {package_format}")

        current_hash = installed_hash()

        release_file_existed = RELEASE_FILE.is_file()
        release_file_before = RELEASE_FILE.read_bytes() if release_file_existed else None
        history_hash_existed = HISTORY_HASH_FILE.is_file()
        history_hash_before = (
            HISTORY_HASH_FILE.read_bytes() if history_hash_existed else None
        )

        if release_hash == current_hash:
            write_expected_parent_hash(current_hash)
            # Windows cannot reliably rename an archive while ZipFile still
            # owns the handle.  Close it explicitly before moving the ZIP.
            z.close()
            move_to_done(archive, checksum_for_move)
            print(
                f"Already installed {archive.name}; moved duplicate to updates/installed",
                flush=True,
            )
            return

        parent = manifest.get("parent_sha256")
        if not isinstance(parent, str) or not HASH_RE.fullmatch(parent):
            raise ValueError("Missing/invalid parent_sha256")
        if parent != current_hash:
            raise ValueError(
                "Update rejected: expected parent "
                f"{current_hash}, package parent is {parent} "
                "(old, repeated or out-of-order update)"
            )

        updates = manifest.get("files")
        deletes = manifest.get("delete", [])
        if not isinstance(updates, dict):
            raise ValueError("Manifest files must be an object")
        if not isinstance(deletes, list) or not all(isinstance(x, str) for x in deletes):
            raise ValueError("Manifest delete must be a string array")
        if not updates and not deletes:
            raise ValueError("Manifest contains no changes")
        if set(updates).intersection(deletes):
            raise ValueError("A path cannot be both updated and deleted")
        if len(updates) > MAX_FILES:
            raise ValueError(f"Too many payload files (max {MAX_FILES})")
        if names_set != set(updates) | metadata_members:
            raise ValueError("ZIP entries do not match manifest files")

        payload: dict[str, bytes] = {}
        total_bytes = 0

        for name, hash_value in updates.items():
            validate_project_path(name)
            if not isinstance(hash_value, str) or not HASH_RE.fullmatch(hash_value):
                raise ValueError(f"Invalid file hash in manifest: {name}")

            info = z.getinfo(name)
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError(f"File too large: {name}")
            total_bytes += info.file_size
            if total_bytes > MAX_TOTAL_BYTES:
                raise ValueError("Update payload exceeds total size limit")

            mode = (info.external_attr >> 16) & 0o170000
            if mode not in (0, 0o100000):
                raise ValueError(f"Symlink/special file forbidden: {name}")

            data = z.read(name)
            if sha256_bytes(data) != hash_value:
                raise ValueError(f"File hash mismatch: {name}")
            payload[name] = data

        for name in deletes:
            validate_project_path(name)

    # Everything is validated before the project is touched.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history_root = HISTORY / stamp
    touched = list(payload) + list(deletes)
    existed_before: dict[str, bool] = {}

    try:
        for name in touched:
            target = ROOT / Path(*PurePosixPath(name).parts)
            existed_before[name] = target.is_file()
            if target.exists() and not target.is_file():
                raise ValueError(f"Update target is not a regular file: {name}")
            if target.is_file():
                backup = history_root / Path(*PurePosixPath(name).parts)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)

        for name, data in payload.items():
            target = ROOT / Path(*PurePosixPath(name).parts)
            write_atomic(target, data)

        for name in deletes:
            target = ROOT / Path(*PurePosixPath(name).parts)
            if target.exists():
                target.unlink()

        state = {
            "sha256": release_hash,
            "hash_kind": "manifest-sha256" if package_format == 2 else "zip-sha256",
            "package_format": package_format,
            "package": archive.name,
            "installed_utc": datetime.now(timezone.utc).isoformat(),
            "parent_sha256": current_hash,
        }
        write_atomic(
            RELEASE_FILE,
            (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        write_expected_parent_hash(release_hash)
        move_to_done(archive, checksum_for_move)

    except Exception:
        for name in reversed(touched):
            target = ROOT / Path(*PurePosixPath(name).parts)
            backup = history_root / Path(*PurePosixPath(name).parts)
            try:
                if existed_before.get(name) and backup.is_file():
                    write_atomic(target, backup.read_bytes())
                elif not existed_before.get(name) and target.exists() and target.is_file():
                    target.unlink()
            except Exception:
                pass

        # Roll back update metadata as well. The expected parent hash must
        # always match the actually installed release.
        try:
            if release_file_existed and release_file_before is not None:
                write_atomic(RELEASE_FILE, release_file_before)
            elif RELEASE_FILE.exists():
                RELEASE_FILE.unlink()
        except Exception:
            pass

        try:
            if history_hash_existed and history_hash_before is not None:
                write_atomic(HISTORY_HASH_FILE, history_hash_before)
            elif HISTORY_HASH_FILE.exists():
                HISTORY_HASH_FILE.unlink()
        except Exception:
            pass

        raise

    summary = (
        f"Installed {archive.name}: {len(payload)} updated, "
        f"{len(deletes)} deleted; release {release_hash}; "
        f"originals in {history_root}"
    )
    print(summary, flush=True)
    outbox_event("INSTALLED", summary)


def reject(archive: Path, exc: Exception) -> None:
    legacy_checksum = archive.with_suffix(archive.suffix + ".sha256")

    if is_transient_file_lock(exc):
        message = f"DEFERRED {archive.name}: transient file lock: {exc}"
        print(message, flush=True)
        outbox_event("DEFERRED", message)
        restore_claimed_file(archive)
        restore_claimed_file(legacy_checksum)
        return

    message = f"REJECTED {archive.name}: {exc}"
    print(message, flush=True)
    outbox_event("REJECTED", message)
    REJECTED.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for file in (archive, legacy_checksum):
        if not file.exists():
            continue
        destination = REJECTED / file.name
        if destination.exists():
            destination = REJECTED / f"{stamp}-{file.name}"
        try:
            os.replace(file, destination)
        except OSError as move_exc:
            if is_transient_file_lock(move_exc):
                print(
                    f"DEFERRED {file.name}: could not move rejected file yet: {move_exc}",
                    flush=True,
                )
                restore_claimed_file(file)
                return
            raise


def scan() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    PROCESSING.mkdir(parents=True, exist_ok=True)

    for inbox_archive in sorted(INBOX.glob("*.zip")):
        legacy_checksum = inbox_archive.with_suffix(inbox_archive.suffix + ".sha256")

        # Do not inspect a destination file while Explorer/browser is still
        # copying it into the inbox.
        if not file_is_stable(inbox_archive):
            continue
        if legacy_checksum.is_file() and not file_is_stable(legacy_checksum):
            continue

        archive = claim_archive(inbox_archive)
        if archive is None:
            continue

        claimed_checksum = archive.with_suffix(archive.suffix + ".sha256")
        if legacy_checksum.is_file():
            try:
                os.replace(legacy_checksum, claimed_checksum)
            except OSError as exc:
                if is_transient_file_lock(exc):
                    restore_claimed_file(archive)
                    continue
                raise

        # Format-2 has its hash embedded.  Legacy format-1 must have the
        # sidecar that was claimed above.
        if not claimed_checksum.is_file() and not package_has_embedded_hash(archive):
            restore_claimed_file(archive)
            continue

        try:
            install(archive)
        except Exception as exc:
            reject(archive, exc)


def print_status() -> None:
    print(f"SunOpt root: {ROOT}")
    print(f"Inbox:      {INBOX}")
    print(f"Processing: {PROCESSING}")
    print(f"Outbox:     {LLM_OUTBOX_FILE}")
    current = installed_hash()
    print(f"Installed:  {current}")
    print(f"Next parent:{current}")
    if RELEASE_FILE.exists():
        print(RELEASE_FILE.read_text(encoding="utf-8").rstrip())
    else:
        print("State:      fresh/genesis")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument(
        "--capture",
        metavar="COMMAND",
        help=(
            "run an explicitly supplied local command and tee stdout/stderr "
            "to updates/outbox/llm.txt"
        ),
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help=(
            "project-relative working directory for --capture "
            "(default: project root)"
        ),
    )
    parser.add_argument(
        "--append-outbox",
        action="store_true",
        help="append a capture instead of replacing llm.txt",
    )
    args = parser.parse_args()

    if args.capture is not None:
        raise SystemExit(
            capture_command(
                args.capture,
                args.cwd,
                args.append_outbox,
            )
        )

    if args.cwd is not None or args.append_outbox:
        parser.error("--cwd/--append-outbox are only valid with --capture")

    if args.status:
        print_status()
        return

    watcher_lock = acquire_instance_lock()
    try:
        # Keep updates/history/hash.txt synchronized with the actually
        # installed release and recover any package left in processing by a
        # previous crash before scanning the inbox.
        write_expected_parent_hash(installed_hash())
        recover_processing()

        scan()
        if args.watch:
            print(
                f"Watching {INBOX} (Ctrl+C to stop); "
                f"outbox={LLM_OUTBOX_FILE}",
                flush=True,
            )
            try:
                while True:
                    time.sleep(1)
                    scan()
            except KeyboardInterrupt:
                print("Watcher stopped", flush=True)
    finally:
        watcher_lock.close()


if __name__ == "__main__":
    main()
