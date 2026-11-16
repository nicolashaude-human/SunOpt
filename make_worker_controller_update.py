# make_worker_controller_update.py
#
# Run from the SunOpt project root:
#   python make_worker_controller_update.py
#
# Creates a format-2 update package directly in updates/inbox.
# It reads the current hash-chain parent from updates/installed-release.json
# and patches the CURRENT rust/sunopt-node/src/lib.rs in-memory so the update
# package does not overwrite unrelated existing code.

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
INBOX = ROOT / "updates" / "inbox"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"
LIB_RS = ROOT / "rust" / "sunopt-node" / "src" / "lib.rs"

GENESIS_HASH = "0" * 64

CONTROLLER_PATH = "rust/sunopt-node/src/acceleration_controller.rs"
LIB_PATH = "rust/sunopt-node/src/lib.rs"

CONTROLLER_RS = r"""use napi::bindgen_prelude::*;
use napi_derive::napi;

/// Public N-API configuration.
/// napi-rs exposes snake_case Rust fields as camelCase JavaScript properties:
/// cpu_workers -> cpuWorkers
/// gpu_workers -> gpuWorkers
#[napi(object)]
pub struct AccelerationOptions {
    pub cpu_workers: u32,
    pub gpu_workers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccelerationConfig {
    pub cpu_workers: usize,
    pub gpu_workers: usize,
}

impl AccelerationConfig {
    pub fn from_options(options: AccelerationOptions) -> Result<Self> {
        let cpu_workers = options.cpu_workers as usize;
        let gpu_workers = options.gpu_workers as usize;

        if cpu_workers == 0 && gpu_workers == 0 {
            return Err(Error::from_reason(
                "At least one CPU worker or GPU worker must be enabled.",
            ));
        }

        Ok(Self {
            cpu_workers,
            gpu_workers,
        })
    }

    pub fn total_workers(&self) -> usize {
        self.cpu_workers + self.gpu_workers
    }
}

/// Central acceleration controller configuration.
///
/// `gpu_workers` means persistent GPU worker contexts / CUDA streams,
/// not CUDA hardware threads.
#[derive(Debug)]
pub struct AccelerationController {
    config: AccelerationConfig,
}

impl AccelerationController {
    pub fn new(config: AccelerationConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> AccelerationConfig {
        self.config
    }

    pub fn cpu_workers(&self) -> usize {
        self.config.cpu_workers
    }

    pub fn gpu_workers(&self) -> usize {
        self.config.gpu_workers
    }

    pub fn total_workers(&self) -> usize {
        self.config.total_workers()
    }
}

#[napi(object)]
pub struct AccelerationStatus {
    pub cpu_workers: u32,
    pub gpu_workers: u32,
    pub total_workers: u32,
}

/// Temporary N-API smoke-test export.
/// Keep this until the real controller is wired into the simulation path.
#[napi]
pub fn test_acceleration_config(
    options: AccelerationOptions,
) -> Result<AccelerationStatus> {
    let config = AccelerationConfig::from_options(options)?;
    let controller = AccelerationController::new(config);

    Ok(AccelerationStatus {
        cpu_workers: controller.cpu_workers() as u32,
        gpu_workers: controller.gpu_workers() as u32,
        total_workers: controller.total_workers() as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_cpu_and_gpu_workers() {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 6,
            gpu_workers: 2,
        })
        .unwrap();

        assert_eq!(config.cpu_workers, 6);
        assert_eq!(config.gpu_workers, 2);
        assert_eq!(config.total_workers(), 8);
    }

    #[test]
    fn accepts_gpu_only() {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 0,
            gpu_workers: 1,
        })
        .unwrap();

        assert_eq!(config.total_workers(), 1);
    }

    #[test]
    fn accepts_cpu_only() {
        let config = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 4,
            gpu_workers: 0,
        })
        .unwrap();

        assert_eq!(config.total_workers(), 4);
    }

    #[test]
    fn rejects_zero_workers() {
        let result = AccelerationConfig::from_options(AccelerationOptions {
            cpu_workers: 0,
            gpu_workers: 0,
        });

        assert!(result.is_err());
    }
}
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def installed_hash() -> str:
    if not RELEASE_FILE.is_file():
        return GENESIS_HASH

    state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    value = state.get("sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError("Invalid updates/installed-release.json sha256")
    return value.lower()


def patch_lib_rs(source: str) -> str:
    module_line = "mod acceleration_controller;"
    export_block = """pub use acceleration_controller::{
    test_acceleration_config,
    AccelerationOptions,
    AccelerationStatus,
};"""

    result = source.rstrip() + "\n"

    if module_line not in result:
        result += "\n// SunOpt CPU/GPU worker controller\n"
        result += module_line + "\n"

    if "pub use acceleration_controller::" not in result:
        result += "\n"
        result += export_block + "\n"

    return result


def main() -> None:
    if not LIB_RS.is_file():
        raise SystemExit(
            f"ERROR: {LIB_RS} not found.\n"
            "Run this script from C:\\NextJS\\SunOpt\\sunopt"
        )

    parent_sha256 = installed_hash()

    original_lib = LIB_RS.read_text(encoding="utf-8")
    patched_lib = patch_lib_rs(original_lib)

    payload = {
        CONTROLLER_PATH: CONTROLLER_RS.encode("utf-8"),
        LIB_PATH: patched_lib.encode("utf-8"),
    }

    files = {
        path: sha256_bytes(data)
        for path, data in payload.items()
    }

    manifest = {
        "format": 2,
        "parent_sha256": parent_sha256,
        "files": files,
        "delete": [],
    }

    # Keep JSON bytes stable because their SHA-256 is the release ID.
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    manifest_sha256 = sha256_bytes(manifest_bytes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_name = f"{stamp}-worker-controller.zip"

    INBOX.mkdir(parents=True, exist_ok=True)
    archive_path = INBOX / archive_name

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr("manifest.json", manifest_bytes)
        z.writestr(
            "manifest.sha256",
            (manifest_sha256 + "\n").encode("ascii"),
        )

        for path, data in payload.items():
            z.writestr(path, data)

    print()
    print("SunOpt format-2 update package created")
    print("--------------------------------------")
    print(f"Package: {archive_path}")
    print(f"Parent:  {parent_sha256}")
    print(f"Release: {manifest_sha256}")
    print()
    print("Payload:")
    for path in payload:
        print(f"  {path}")
    print()
    print("Your running update_watcher.py --watch should now install it.")


if __name__ == "__main__":
    main()
