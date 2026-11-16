#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import re
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
BASE_GENERATOR = SCRIPT_DIR / "make-approach-a-final-single-step-graph.py"

if not BASE_GENERATOR.is_file():
    raise RuntimeError(f"Base generator missing: {BASE_GENERATOR}")

spec = importlib.util.spec_from_file_location(
    "sunopt_approach_a_base_generator",
    BASE_GENERATOR,
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the installed Approach-A generator.")

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ROBUST_HELPER_BLOCK = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPER_BEGIN
impl CudaCausalWorker {
    fn reset_approach_a_step_graph(&mut self) {
        if self.approach_a_step_graph == 0 {
            return;
        }

        let _ = unsafe {
            sunopt_cuda_destroy_causal_step_graph(
                self.approach_a_step_graph as *mut std::ffi::c_void,
            )
        };

        self.approach_a_step_graph = 0;
    }

    fn ensure_approach_a_step_graph(&mut self) -> Result<()> {
        if self.approach_a_step_graph != 0 {
            return Ok(());
        }

        if self.lockstep_dynamic.is_empty() {
            return Err(Error::from_reason(
                "Approach-A step graph requires configured lockstep parameters.",
            ));
        }

        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let code = unsafe {
            sunopt_cuda_create_causal_step_graph(
                self.context.raw.as_ptr(),
                self.lockstep_dynamic.len(),
                &mut graph,
            )
        };

        if code != 0 {
            return Err(cuda_error(
                "sunopt_cuda_create_causal_step_graph",
                code,
            ));
        }

        if graph.is_null() {
            return Err(Error::from_reason(
                "Approach-A step graph creation returned null.",
            ));
        }

        self.approach_a_step_graph = graph as usize;
        Ok(())
    }
}

impl Drop for CudaCausalWorker {
    fn drop(&mut self) {
        self.reset_approach_a_step_graph();
    }
}
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPER_END
'''.strip()


def find_matching_brace(source: str, open_index: int, label: str) -> int:
    if open_index < 0 or open_index >= len(source) or source[open_index] != "{":
        raise RuntimeError(f"{label}: invalid opening brace")

    depth = 0
    for index in range(open_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index

    raise RuntimeError(f"{label}: closing brace not found")


def find_worker_struct(source: str) -> tuple[int, int]:
    token = "pub struct CudaCausalWorker"
    start = source.find(token)
    if start < 0:
        raise RuntimeError("CudaCausalWorker struct not found")

    open_index = source.find("{", start + len(token))
    close_index = find_matching_brace(source, open_index, "CudaCausalWorker struct")
    return open_index, close_index


def insert_worker_field(source: str) -> str:
    open_index, close_index = find_worker_struct(source)
    struct_text = source[open_index + 1:close_index]

    if "approach_a_step_graph:" in struct_text:
        return source

    field = "\n    approach_a_step_graph:\n        usize,\n"
    return source[:close_index] + field + source[close_index:]


def find_napi_worker_impl(source: str) -> int:
    struct_pos = source.find("pub struct CudaCausalWorker")
    if struct_pos < 0:
        raise RuntimeError("CudaCausalWorker struct not found")

    pattern = re.compile(r"#\[napi\]\s*impl\s+CudaCausalWorker\s*\{")
    match = pattern.search(source, struct_pos)
    if match is None:
        raise RuntimeError("N-API CudaCausalWorker impl not found")
    return match.start()


def insert_constructor_field(source: str) -> str:
    impl_pos = find_napi_worker_impl(source)
    new_pos = source.find("pub fn new(", impl_pos)
    if new_pos < 0:
        raise RuntimeError("CudaCausalWorker new() not found")

    self_pos = source.find("Ok(Self", new_pos)
    if self_pos < 0:
        raise RuntimeError("CudaCausalWorker Ok(Self { ... }) not found")

    open_index = source.find("{", self_pos)
    close_index = find_matching_brace(source, open_index, "CudaCausalWorker constructor")
    object_text = source[open_index + 1:close_index]

    if "approach_a_step_graph:" in object_text:
        return source

    field = "\n\n            approach_a_step_graph:\n                0,"
    return source[:close_index] + field + "\n" + source[close_index:]


def insert_configure_reset(source: str) -> str:
    token = "pub fn configure_lockstep_packed("
    start = source.find(token)
    if start < 0:
        raise RuntimeError("configure_lockstep_packed() not found")

    open_index = source.find("{", start + len(token))
    close_index = find_matching_brace(source, open_index, "configure_lockstep_packed")
    body = source[open_index + 1:close_index]

    reset_call = "self.reset_approach_a_step_graph();"
    if reset_call in body:
        return source

    insertion = "\n        self.reset_approach_a_step_graph();\n"
    return source[:open_index + 1] + insertion + source[open_index + 1:]


def replace_lockstep_dispatch(source: str) -> str:
    token = "pub fn run_lockstep_dynamic_into("
    start = source.find(token)
    if start < 0:
        raise RuntimeError("run_lockstep_dynamic_into() not found")

    open_index = source.find("{", start + len(token))
    close_index = find_matching_brace(source, open_index, "run_lockstep_dynamic_into")
    body = source[open_index + 1:close_index]

    if "sunopt_cuda_run_causal_step_graph(" in body:
        return source

    method_pos = body.find(".run_causal_dynamic_batch_into(")
    if method_pos < 0:
        raise RuntimeError("Direct FP64 causal batch call not found inside run_lockstep_dynamic_into()")

    statement_start = body.rfind("self.context", 0, method_pos)
    if statement_start < 0:
        raise RuntimeError("Start of direct FP64 causal call not found")

    statement_end = body.find(")?;", method_pos)
    if statement_end < 0:
        raise RuntimeError("End of direct FP64 causal call not found")
    statement_end += len(")?;")

    line_start = body.rfind("\n", 0, statement_start) + 1
    indent = body[line_start:statement_start]

    replacement = r'''let use_approach_a_graph =
    std::env::var(
        "SUNOPT_CUDA_CAUSAL_PATH",
    )
    .ok()
    .as_deref() ==
    Some("AG");


if use_approach_a_graph {
    self.ensure_approach_a_step_graph()?;

    let code = unsafe {
        sunopt_cuda_run_causal_step_graph(
            self.approach_a_step_graph as *mut std::ffi::c_void,
            current_index as usize,
            horizon_steps as usize,
            self.lockstep_dynamic.as_ptr(),
            self.lockstep_dynamic.len(),
            output_slice.as_mut_ptr(),
        )
    };

    if code != 0 {
        return Err(cuda_error(
            "sunopt_cuda_run_causal_step_graph",
            code,
        ));
    }

} else {
    self.context
        .run_causal_dynamic_batch_into(
            current_index as usize,
            horizon_steps as usize,
            &self.lockstep_dynamic,
            output_slice,
        )?;
}'''

    replacement = replacement.replace("\n", "\n" + indent)
    new_body = body[:statement_start] + replacement + body[statement_end:]
    return source[:open_index + 1] + new_body + source[close_index:]


def robust_patch_rust(source: str) -> str:
    for token in (
        "pub struct CudaCausalWorker",
        "lockstep_dynamic:",
        "blackwell_capture:",
        "pub fn configure_lockstep_packed(",
        "pub fn run_lockstep_dynamic_into(",
        "fn sunopt_cuda_run_causal_dynamic_batch(",
    ):
        if token not in source:
            raise RuntimeError(f"Unexpected cuda_context.rs; missing {token!r}")

    source = base.insert_or_replace_before(
        source,
        base.RS_FFI_BEGIN,
        base.RS_FFI_END,
        base.RS_FFI_BLOCK,
        ["    fn sunopt_cuda_get_profile(", "    fn sunopt_cuda_error_string("],
    )

    source = insert_worker_field(source)
    source = insert_constructor_field(source)
    source = base.insert_or_replace_before(
        source,
        base.RS_HELPER_BEGIN,
        base.RS_HELPER_END,
        ROBUST_HELPER_BLOCK,
        ["#[napi]\nimpl CudaCausalWorker {", "#[napi]\r\nimpl CudaCausalWorker {"],
    )
    source = insert_configure_reset(source)
    source = replace_lockstep_dispatch(source)
    return source


base.RS_HELPER_BLOCK = ROBUST_HELPER_BLOCK
base.patch_rust = robust_patch_rust

# Self-test against the CURRENT Rust file before any candidate package is written.
current_rs = base.RS_PATH.read_text(encoding="utf-8")
probe = robust_patch_rust(current_rs)
for expected in (
    "approach_a_step_graph:",
    "self.reset_approach_a_step_graph();",
    "sunopt_cuda_run_causal_step_graph(",
    'Some("AG")',
):
    if expected not in probe:
        raise RuntimeError(f"Robust Rust patch self-test failed: missing {expected!r}")

package_path = base.build_package()

with zipfile.ZipFile(package_path, "r") as archive:
    manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    candidate_release = archive.read("manifest.sha256").decode("utf-8").strip()

llm_path = ROOT / "llm.txt"
with llm_path.open("a", encoding="utf-8") as llm:
    llm.write(
        "\n=== APPROACH_A_FINAL_GENERATOR_V4 ===\n"
        f"installed_parent={manifest['parent_sha256']}\n"
        f"candidate_release={candidate_release}\n"
        f"candidate_package={package_path}\n"
        "rust_patch_self_test=PASS\n\n"
    )

print("APPROACH_A_FINAL_GENERATOR_V4=OK")
print(f"installed_parent={manifest['parent_sha256']}")
print(f"candidate_release={candidate_release}")
print(f"candidate_package={package_path}")
print(f"llm={llm_path}")
