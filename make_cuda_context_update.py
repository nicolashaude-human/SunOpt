# make_cuda_context_update.py
#
# Run from:
#   C:\NextJS\SunOpt\sunopt
#
# Creates a verified format-2 watcher package directly in updates/inbox.
# It reads the CURRENT hash-chain parent and CURRENT build.rs/lib.rs before
# patching. It fails closed if the expected structure is not found.

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
INBOX = ROOT / "updates" / "inbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"

BUILD_RS = ROOT / "rust" / "sunopt-node" / "build.rs"
LIB_RS = ROOT / "rust" / "sunopt-node" / "src" / "lib.rs"

BUILD_PATH = "rust/sunopt-node/build.rs"
LIB_PATH = "rust/sunopt-node/src/lib.rs"
CUDA_RS_PATH = "rust/sunopt-node/src/cuda_context.rs"
CUDA_H_PATH = "rust/sunopt-node/cuda/cuda_context.h"
CUDA_CU_PATH = "rust/sunopt-node/cuda/cuda_context.cu"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

CUDA_H = r'''#pragma once

#ifdef _WIN32
#define SUNOPT_CUDA_EXPORT __declspec(dllexport)
#else
#define SUNOPT_CUDA_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SunoptCudaContext SunoptCudaContext;

SUNOPT_CUDA_EXPORT int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_destroy(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
);

SUNOPT_CUDA_EXPORT const char* sunopt_cuda_error_string(
    int error_code
);

#ifdef __cplusplus
}
#endif
'''

CUDA_CU = r'''#include "cuda_context.h"

#include <cuda_runtime.h>

struct SunoptCudaContext {
    int device_id;
    cudaStream_t stream;
};

extern "C" int sunopt_cuda_create(
    int device_id,
    SunoptCudaContext** out_context
) {
    if (out_context == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_context = nullptr;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    SunoptCudaContext* context = new SunoptCudaContext{};
    context->device_id = device_id;
    context->stream = nullptr;

    err = cudaStreamCreateWithFlags(
        &context->stream,
        cudaStreamNonBlocking
    );

    if (err != cudaSuccess) {
        delete context;
        return static_cast<int>(err);
    }

    *out_context = context;
    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_destroy(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(cudaSuccess);
    }

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaStreamDestroy(context->stream);
    delete context;

    return static_cast<int>(err);
}

extern "C" int sunopt_cuda_device_id(
    const SunoptCudaContext* context,
    int* out_device_id
) {
    if (context == nullptr || out_device_id == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_device_id = context->device_id;
    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_stream_synchronize(
    SunoptCudaContext* context
) {
    if (context == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    return static_cast<int>(cudaStreamSynchronize(context->stream));
}

extern "C" const char* sunopt_cuda_error_string(
    int error_code
) {
    return cudaGetErrorString(static_cast<cudaError_t>(error_code));
}
'''

CUDA_RS = r'''use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::ffi::{c_char, c_int, CStr};
use std::ptr::NonNull;

#[repr(C)]
struct SunoptCudaContextOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn sunopt_cuda_create(
        device_id: c_int,
        out_context: *mut *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_destroy(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_device_id(
        context: *const SunoptCudaContextOpaque,
        out_device_id: *mut c_int,
    ) -> c_int;

    fn sunopt_cuda_stream_synchronize(
        context: *mut SunoptCudaContextOpaque,
    ) -> c_int;

    fn sunopt_cuda_error_string(
        error_code: c_int,
    ) -> *const c_char;
}

fn cuda_error(operation: &str, code: c_int) -> Error {
    let message = unsafe {
        let ptr = sunopt_cuda_error_string(code);

        if ptr.is_null() {
            format!("CUDA error {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Error::from_reason(format!(
        "{operation} failed: {message} (CUDA error {code})"
    ))
}

/// Persistent CUDA simulation context.
///
/// The CUDA stream is created once and remains alive until this Rust value
/// is dropped. Future packages will add persistent device buffers/workspace
/// to this same context.
#[derive(Debug)]
pub struct CudaSimulationContext {
    raw: NonNull<SunoptCudaContextOpaque>,
}

unsafe impl Send for CudaSimulationContext {}

impl CudaSimulationContext {
    pub fn new(device_id: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();

        let code = unsafe {
            sunopt_cuda_create(device_id as c_int, &mut raw)
        };

        if code != 0 {
            return Err(cuda_error("sunopt_cuda_create", code));
        }

        let raw = NonNull::new(raw).ok_or_else(|| {
            Error::from_reason(
                "sunopt_cuda_create returned success with a null context",
            )
        })?;

        Ok(Self { raw })
    }

    pub fn device_id(&self) -> Result<i32> {
        let mut device_id: c_int = -1;

        let code = unsafe {
            sunopt_cuda_device_id(self.raw.as_ptr(), &mut device_id)
        };

        if code != 0 {
            return Err(cuda_error("sunopt_cuda_device_id", code));
        }

        Ok(device_id as i32)
    }

    pub fn synchronize(&mut self) -> Result<()> {
        let code = unsafe {
            sunopt_cuda_stream_synchronize(self.raw.as_ptr())
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_stream_synchronize",
                code,
            ));
        }

        Ok(())
    }
}

impl Drop for CudaSimulationContext {
    fn drop(&mut self) {
        let _ = unsafe { sunopt_cuda_destroy(self.raw.as_ptr()) };
    }
}

#[napi(object)]
pub struct CudaContextStatus {
    pub device_id: i32,
    pub stream_ready: bool,
}

/// End-to-end smoke test for the persistent CUDA context.
/// JavaScript name: testCudaContext
#[napi]
pub fn test_cuda_context(
    device_id: Option<i32>,
) -> Result<CudaContextStatus> {
    let mut context = CudaSimulationContext::new(device_id.unwrap_or(0))?;
    context.synchronize()?;

    Ok(CudaContextStatus {
        device_id: context.device_id()?,
        stream_ready: true,
    })
}
'''

BUILD_HELPER = r'''
fn compile_sunopt_cuda() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    println!("cargo:rerun-if-changed=cuda/cuda_context.cu");
    println!("cargo:rerun-if-changed=cuda/cuda_context.h");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=SUNOPT_CUDA_ARCH");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os != "windows" {
        println!(
            "cargo:warning=SunOpt CUDA native context is currently built only on Windows; skipping CUDA build"
        );
        return;
    }

    let cuda_path = env::var("CUDA_PATH")
        .expect("CUDA_PATH must be set to build SunOpt CUDA support");

    let nvcc = Path::new(&cuda_path)
        .join("bin")
        .join("nvcc.exe");

    if !nvcc.is_file() {
        panic!("nvcc.exe not found at {}", nvcc.display());
    }

    let out_dir = PathBuf::from(
        env::var("OUT_DIR").expect("OUT_DIR is missing"),
    );

    let output_lib = out_dir.join("sunopt_cuda.lib");

    let arch = env::var("SUNOPT_CUDA_ARCH")
        .unwrap_or_else(|_| "sm_120".to_string());

    let status = Command::new(&nvcc)
        .arg("-lib")
        .arg("-O3")
        .arg(format!("-arch={arch}"))
        .arg("cuda/cuda_context.cu")
        .arg("-o")
        .arg(&output_lib)
        .status()
        .expect("failed to execute nvcc");

    if !status.success() {
        panic!("nvcc failed while building SunOpt CUDA context");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=sunopt_cuda");

    let cuda_lib = Path::new(&cuda_path).join("lib").join("x64");

    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
}
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def current_parent() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="ascii").strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid hash in {HASH_FILE}")

    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).lower()
        if HASH_RE.fullmatch(value):
            return value

    return GENESIS


def patch_build_rs(source: str) -> str:
    if "compile_sunopt_cuda();" not in source:
        patched, count = re.subn(
            r"(fn\s+main\s*\(\s*\)\s*\{)",
            r"\1\n    compile_sunopt_cuda();",
            source,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                "Could not find exactly one `fn main() {` in build.rs. "
                "No package created."
            )
    else:
        patched = source

    if "fn compile_sunopt_cuda()" not in patched:
        patched = patched.rstrip() + "\n" + BUILD_HELPER + "\n"

    return patched


def patch_lib_rs(source: str) -> str:
    patched = source

    if "mod cuda_context;" not in patched:
        if "mod acceleration_controller;" in patched:
            patched = patched.replace(
                "mod acceleration_controller;",
                "mod acceleration_controller;\nmod cuda_context;",
                1,
            )
        else:
            patched = "mod cuda_context;\n" + patched

    export_block = (
        "pub use cuda_context::{\n"
        "    test_cuda_context,\n"
        "    CudaContextStatus,\n"
        "};"
    )

    if "pub use cuda_context::" not in patched:
        patched = patched.rstrip() + "\n\n" + export_block + "\n"

    return patched


def main() -> None:
    if not BUILD_RS.is_file():
        raise SystemExit(f"Missing {BUILD_RS}")
    if not LIB_RS.is_file():
        raise SystemExit(f"Missing {LIB_RS}")

    parent = current_parent()

    current_build = BUILD_RS.read_text(encoding="utf-8")
    current_lib = LIB_RS.read_text(encoding="utf-8")

    patched_build = patch_build_rs(current_build)
    patched_lib = patch_lib_rs(current_lib)

    payload = {
        BUILD_PATH: patched_build.encode("utf-8"),
        LIB_PATH: patched_lib.encode("utf-8"),
        CUDA_RS_PATH: CUDA_RS.encode("utf-8"),
        CUDA_H_PATH: CUDA_H.encode("utf-8"),
        CUDA_CU_PATH: CUDA_CU.encode("utf-8"),
    }

    files = {path: sha256_bytes(data) for path, data in payload.items()}

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": files,
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    release = sha256_bytes(manifest_bytes)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = INBOX / f"{stamp}-cuda-persistent-context.zip"
    INBOX.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", manifest_bytes)
        z.writestr("manifest.sha256", (release + "\n").encode("ascii"))
        for path, data in payload.items():
            z.writestr(path, data)

    print()
    print("SunOpt CUDA persistent-context update created")
    print("------------------------------------------------")
    print(f"Package: {archive}")
    print(f"Parent:  {parent}")
    print(f"Release: {release}")
    print()
    print("Payload:")
    for path in payload:
        print(f"  {path}")
    print()
    print("Next:")
    print("  1. Let update_watcher.py --watch install the package.")
    print("  2. Rebuild sunopt-node in the VS x64 Developer PowerShell.")
    print("  3. Test native.testCudaContext(0).")


if __name__ == "__main__":
    main()
