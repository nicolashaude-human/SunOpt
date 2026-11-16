// rust/sunopt-node/build.rs

fn main() {
    compile_sunopt_cuda();
    napi_build::setup();
}

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
        // SUNOPT_MSVC_STANDARD_PREPROCESSOR
        .arg("-Xcompiler")
        .arg("/Zc:preprocessor")
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

