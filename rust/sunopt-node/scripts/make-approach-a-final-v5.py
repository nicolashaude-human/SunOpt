#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
INBOX = ROOT / "updates" / "inbox"
OUTBOX = ROOT / "updates" / "outbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"

H_REL = "rust/sunopt-node/cuda/cuda_context.h"
CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
RS_REL = "rust/sunopt-node/src/cuda_context.rs"
PAIR_REL = "rust/sunopt-node/scripts/approach-a-final-pair.mjs"
RUNNER_REL = "rust/sunopt-node/scripts/approach-a-final-production-ab.mjs"

H_PATH = ROOT / H_REL
CU_PATH = ROOT / CU_REL
RS_PATH = ROOT / RS_REL
B14Q_PATH = ROOT / "rust" / "sunopt-node" / "scripts" / "blackwell-b14q-production-year-parity.mjs"

EXPECTED = {
    H_REL: "b1ef5a44da080efab3552d5f249d9b538b6ec5c854cff36cb7ecced19c055646",
    CU_REL: "08999c8fbc15b2496ed6f3ab2a0a71082389ec6ad494615ec3a9d2a38dd2800f",
    RS_REL: "2bdfa331e0ea1dac574fcfc1f7394da245676a8e9c08209ca734aef5a78cd8fd",
    "rust/sunopt-node/scripts/blackwell-b14q-production-year-parity.mjs":
        "971eb53d97e15346577c328b9f8d5a21d595c8b702e88ba0978e8475515c2412",
}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())

def fail(message: str) -> None:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    (OUTBOX / "failed-prompt.txt").write_text(
        f"stage=APPROACH_A_FINAL_GENERATOR\n"
        f"time={datetime.now(timezone.utc).isoformat()}\n"
        f"error={message}\n",
        encoding="utf-8",
    )
    raise RuntimeError(message)

def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        fail(f"{label}: expected exactly one anchor; found {count}")
    return source.replace(old, new, 1)

def insert_before(source: str, anchor: str, block: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        fail(f"{label}: expected exactly one anchor; found {count}")
    return source.replace(anchor, block + "\n\n" + anchor, 1)

def current_release() -> str:
    if not HASH_FILE.is_file():
        fail(f"Missing current release hash: {HASH_FILE}")
    value = HASH_FILE.read_text(encoding="utf-8").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        fail(f"Invalid current release hash: {value!r}")
    return value

def verify_sources() -> dict[str, str]:
    paths = {
        H_REL: H_PATH,
        CU_REL: CU_PATH,
        RS_REL: RS_PATH,
        "rust/sunopt-node/scripts/blackwell-b14q-production-year-parity.mjs": B14Q_PATH,
    }
    actual = {}
    for rel, path in paths.items():
        if not path.is_file():
            fail(f"Missing source file: {rel}")
        digest = sha256_file(path)
        actual[rel] = digest
        if digest != EXPECTED[rel]:
            fail(
                f"Source precondition failed for {rel}: "
                f"expected {EXPECTED[rel]}, got {digest}"
            )
    return actual

H_API = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_create_causal_step_graph(
    SunoptCudaContext* context,
    size_t job_count,
    void** out_graph
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_step_graph(
    void* graph,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_destroy_causal_step_graph(
    void* graph
);
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END
'''.strip()

CU_BLOCK = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN
struct SunoptCausalStepGraph {
    SunoptCudaContext* context = nullptr;
    size_t job_count = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphNode_t kernel_node = nullptr;
};

static cudaError_t sunopt_fill_causal_step_graph_params(
    SunoptCausalStepGraph* step_graph,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    cudaKernelNodeParams* out_params,
    SunoptCausalDynamicBatch* out_dynamic_batch,
    int* out_use_dynamic_parameters,
    SunoptCausalResult** out_results,
    double** out_cut_level_results,
    const double** out_export_power,
    const double** out_demand_energy,
    const SunoptCausalParameters** out_device_parameters
) {
    if (
        step_graph == nullptr ||
        step_graph->context == nullptr ||
        host_dynamic_parameters == nullptr ||
        out_params == nullptr ||
        out_dynamic_batch == nullptr ||
        out_use_dynamic_parameters == nullptr ||
        out_results == nullptr ||
        out_cut_level_results == nullptr ||
        out_export_power == nullptr ||
        out_demand_energy == nullptr ||
        out_device_parameters == nullptr
    ) {
        return cudaErrorInvalidValue;
    }

    SunoptCudaContext* context = step_graph->context;

    if (
        step_graph->job_count == 0 ||
        step_graph->job_count > 32 ||
        context->causal_length == 0 ||
        context->causal_static_parameter_buffer == nullptr ||
        context->causal_static_job_count != step_graph->job_count ||
        current_index >= context->causal_length ||
        horizon_steps == 0 ||
        horizon_steps > context->causal_length
    ) {
        return cudaErrorInvalidValue;
    }

    for (size_t i = 0; i < step_graph->job_count; ++i) {
        out_dynamic_batch->values[i] = host_dynamic_parameters[i];
    }

    const size_t one_array_bytes =
        context->causal_length * sizeof(double);

    unsigned char* base =
        static_cast<unsigned char*>(context->input_buffer);

    *out_export_power =
        reinterpret_cast<const double*>(base);

    *out_demand_energy =
        reinterpret_cast<const double*>(base + one_array_bytes);

    *out_device_parameters =
        static_cast<const SunoptCausalParameters*>(
            context->causal_static_parameter_buffer
        );

    *out_use_dynamic_parameters = 1;
    *out_results = nullptr;
    *out_cut_level_results = context->causal_mapped_cut_device;

    void* kernel_args[] = {
        out_export_power,
        out_demand_energy,
        out_device_parameters,
        out_dynamic_batch,
        out_use_dynamic_parameters,
        &current_index,
        &horizon_steps,
        out_results,
        out_cut_level_results,
        &step_graph->job_count,
    };

    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(causal_batch_kernel);
    params.gridDim =
        dim3(static_cast<unsigned int>(step_graph->job_count), 1, 1);
    params.blockDim = dim3(32, 1, 1);
    params.sharedMemBytes = 0;
    params.kernelParams = kernel_args;
    params.extra = nullptr;

    *out_params = params;
    return cudaSuccess;
}

extern "C" int sunopt_cuda_create_causal_step_graph(
    SunoptCudaContext* context,
    size_t job_count,
    void** out_graph
) {
    if (
        context == nullptr ||
        out_graph == nullptr ||
        job_count == 0 ||
        job_count > 32 ||
        context->causal_length == 0 ||
        context->causal_static_parameter_buffer == nullptr ||
        context->causal_static_job_count != job_count
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_graph = nullptr;

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t result_bytes = job_count * sizeof(double);
    err = reserve_mapped_cut_buffer(context, result_bytes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    auto* step_graph =
        new (std::nothrow) SunoptCausalStepGraph();

    if (step_graph == nullptr) {
        return static_cast<int>(cudaErrorMemoryAllocation);
    }

    step_graph->context = context;
    step_graph->job_count = job_count;

    err = cudaGraphCreate(&step_graph->graph, 0);
    if (err != cudaSuccess) {
        delete step_graph;
        return static_cast<int>(err);
    }

    SunoptCausalDynamicBatch dynamic_batch{};
    size_t current_index = 0;
    size_t horizon_steps = 1;
    int use_dynamic_parameters = 1;
    SunoptCausalResult* results = nullptr;
    double* cut_level_results = nullptr;
    const double* export_power = nullptr;
    const double* demand_energy = nullptr;
    const SunoptCausalParameters* device_parameters = nullptr;
    cudaKernelNodeParams params{};

    err = sunopt_fill_causal_step_graph_params(
        step_graph,
        current_index,
        horizon_steps,
        dynamic_batch.values,
        &params,
        &dynamic_batch,
        &use_dynamic_parameters,
        &results,
        &cut_level_results,
        &export_power,
        &demand_energy,
        &device_parameters
    );

    if (err == cudaSuccess) {
        err = cudaGraphAddKernelNode(
            &step_graph->kernel_node,
            step_graph->graph,
            nullptr,
            0,
            &params
        );
    }

    if (err == cudaSuccess) {
        err = cudaGraphInstantiate(
            &step_graph->graph_exec,
            step_graph->graph,
            nullptr,
            nullptr,
            0
        );
    }

    if (err != cudaSuccess) {
        if (step_graph->graph_exec != nullptr) {
            cudaGraphExecDestroy(step_graph->graph_exec);
        }
        if (step_graph->graph != nullptr) {
            cudaGraphDestroy(step_graph->graph);
        }
        delete step_graph;
        return static_cast<int>(err);
    }

    *out_graph = static_cast<void*>(step_graph);
    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_run_causal_step_graph(
    void* graph,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
) {
    auto* step_graph =
        static_cast<SunoptCausalStepGraph*>(graph);

    if (
        step_graph == nullptr ||
        step_graph->context == nullptr ||
        host_dynamic_parameters == nullptr ||
        host_cut_level_results == nullptr ||
        job_count != step_graph->job_count
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    SunoptCudaContext* context = step_graph->context;

    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    const size_t result_bytes = job_count * sizeof(double);
    err = reserve_mapped_cut_buffer(context, result_bytes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    SunoptCausalDynamicBatch dynamic_batch{};
    int use_dynamic_parameters = 1;
    SunoptCausalResult* results = nullptr;
    double* cut_level_results = nullptr;
    const double* export_power = nullptr;
    const double* demand_energy = nullptr;
    const SunoptCausalParameters* device_parameters = nullptr;
    cudaKernelNodeParams params{};

    err = sunopt_fill_causal_step_graph_params(
        step_graph,
        current_index,
        horizon_steps,
        host_dynamic_parameters,
        &params,
        &dynamic_batch,
        &use_dynamic_parameters,
        &results,
        &cut_level_results,
        &export_power,
        &demand_energy,
        &device_parameters
    );

    if (err == cudaSuccess) {
        err = cudaGraphExecKernelNodeSetParams(
            step_graph->graph_exec,
            step_graph->kernel_node,
            &params
        );
    }

    if (err == cudaSuccess) {
        err = cudaGraphLaunch(
            step_graph->graph_exec,
            context->stream
        );
    }

    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(context->stream);
    }

    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    std::memcpy(
        host_cut_level_results,
        context->causal_mapped_cut_host,
        result_bytes
    );

    context->profile.causal_calls += 1;
    context->profile.causal_jobs +=
        static_cast<uint64_t>(job_count);

    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_destroy_causal_step_graph(
    void* graph
) {
    auto* step_graph =
        static_cast<SunoptCausalStepGraph*>(graph);

    if (step_graph == nullptr) {
        return static_cast<int>(cudaSuccess);
    }

    cudaError_t err = cudaSuccess;

    if (step_graph->graph_exec != nullptr) {
        err = cudaGraphExecDestroy(step_graph->graph_exec);
    }

    if (step_graph->graph != nullptr) {
        const cudaError_t graph_error =
            cudaGraphDestroy(step_graph->graph);

        if (
            err == cudaSuccess &&
            graph_error != cudaSuccess
        ) {
            err = graph_error;
        }
    }

    delete step_graph;
    return static_cast<int>(err);
}
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END
'''.strip()

RS_FFI = r'''
    // SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_FFI_BEGIN
    fn sunopt_cuda_create_causal_step_graph(
        context: *mut SunoptCudaContextOpaque,
        job_count: usize,
        out_graph: *mut *mut std::ffi::c_void,
    ) -> c_int;

    fn sunopt_cuda_run_causal_step_graph(
        graph: *mut std::ffi::c_void,
        current_index: usize,
        horizon_steps: usize,
        host_dynamic_parameters: *const SunoptCausalDynamic,
        job_count: usize,
        host_cut_level_results: *mut f64,
    ) -> c_int;

    fn sunopt_cuda_destroy_causal_step_graph(
        graph: *mut std::ffi::c_void,
    ) -> c_int;
    // SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_FFI_END
'''.strip()

RS_HELPERS = r'''
    // SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPERS_BEGIN
    fn approach_a_step_graph_requested() -> bool {
        std::env::var(
            "SUNOPT_APPROACH_A_STEP_GRAPH",
        )
        .map(
            |value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            },
        )
        .unwrap_or(false)
    }

    fn reset_approach_a_step_graph(
        &mut self,
    ) {
        if self.approach_a_step_graph == 0 {
            return;
        }

        let graph =
            self.approach_a_step_graph
                as *mut std::ffi::c_void;

        let _ = unsafe {
            sunopt_cuda_destroy_causal_step_graph(
                graph,
            )
        };

        self.approach_a_step_graph = 0;
    }

    fn ensure_approach_a_step_graph(
        &mut self,
    ) -> Result<()> {
        if self.approach_a_step_graph != 0 {
            return Ok(());
        }

        if self.lockstep_dynamic.is_empty() {
            return Err(
                Error::from_reason(
                    "Approach-A step graph requires configured lockstep parameters.",
                ),
            );
        }

        let mut graph:
            *mut std::ffi::c_void =
            std::ptr::null_mut();

        let code = unsafe {
            sunopt_cuda_create_causal_step_graph(
                self.context.raw.as_ptr(),
                self.lockstep_dynamic.len(),
                &mut graph,
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_create_causal_step_graph",
                    code,
                ),
            );
        }

        if graph.is_null() {
            return Err(
                Error::from_reason(
                    "sunopt_cuda_create_causal_step_graph returned null.",
                ),
            );
        }

        self.approach_a_step_graph =
            graph as usize;

        Ok(())
    }

    fn run_approach_a_direct_or_graph(
        &mut self,
        current_index: usize,
        horizon_steps: usize,
        output: &mut [f64],
    ) -> Result<()> {
        if !Self::approach_a_step_graph_requested() {
            return self.context
                .run_causal_dynamic_batch_into(
                    current_index,
                    horizon_steps,
                    &self.lockstep_dynamic,
                    output,
                );
        }

        self.ensure_approach_a_step_graph()?;

        let code = unsafe {
            sunopt_cuda_run_causal_step_graph(
                self.approach_a_step_graph
                    as *mut std::ffi::c_void,
                current_index,
                horizon_steps,
                self.lockstep_dynamic.as_ptr(),
                self.lockstep_dynamic.len(),
                output.as_mut_ptr(),
            )
        };

        if code != 0 {
            return Err(
                cuda_error(
                    "sunopt_cuda_run_causal_step_graph",
                    code,
                ),
            );
        }

        Ok(())
    }
    // SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPERS_END
'''.rstrip()

RS_DROP = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_DROP_BEGIN
impl Drop for CudaCausalWorker {
    fn drop(&mut self) {
        self.reset_approach_a_step_graph();
    }
}
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_DROP_END
'''.strip()

def patch_header(h: str) -> str:
    anchor = r'''SUNOPT_CUDA_EXPORT int sunopt_cuda_precompute_no_sunshine_waterlines(
    SunoptCudaContext* context,
    const SunoptNoSunshineTankParameters* host_parameters,
    size_t tank_count,
    double* host_waterlines
);'''
    return insert_before(h, anchor, H_API, "cuda_context.h Approach-A API")

def patch_cuda(cu: str) -> str:
    anchor = 'extern "C" int sunopt_cuda_precompute_no_sunshine_waterlines('
    return insert_before(cu, anchor, CU_BLOCK, "cuda_context.cu Approach-A graph")

def patch_rust(rs: str) -> str:
    ffi_anchor = r'''    fn sunopt_cuda_precompute_no_sunshine_waterlines(
        context: *mut SunoptCudaContextOpaque,
        host_parameters: *const SunoptNoSunshineTankParameters,
        tank_count: usize,
        host_waterlines: *mut f64,
    ) -> c_int;'''

    rs = insert_before(
        rs,
        ffi_anchor,
        RS_FFI,
        "cuda_context.rs Approach-A FFI",
    )

    rs = replace_once(
        rs,
        r'''    b14p_status:
        Vec<u32>,
}''',
        r'''    b14p_status:
        Vec<u32>,

    approach_a_step_graph:
        usize,
}''',
        "CudaCausalWorker graph field",
    )

    rs = replace_once(
        rs,
        r'''            b14p_status:
                Vec::new(),
        })''',
        r'''            b14p_status:
                Vec::new(),

            approach_a_step_graph:
                0,
        })''',
        "CudaCausalWorker graph constructor field",
    )

    rs = replace_once(
        rs,
        r'''impl CudaCausalWorker {
    fn run_lockstep_b14p_or_a(''',
        "impl CudaCausalWorker {\n"
        + RS_HELPERS
        + "\n\n    fn run_lockstep_b14p_or_a(",
        "CudaCausalWorker helper insertion",
    )

    rs = replace_once(
        rs,
        r'''        self.context
            .run_causal_dynamic_batch_into(
                current_index,
                horizon_steps,
                &self.lockstep_dynamic,
                output,
            )
    }

    fn validate_job(''',
        r'''        self
            .run_approach_a_direct_or_graph(
                current_index,
                horizon_steps,
                output,
            )
    }

    fn validate_job(''',
        "A direct/graph dispatch",
    )

    rs = replace_once(
        rs,
        r'''    pub fn configure_lockstep_packed(
        &mut self,
        packed:
            Float64Array,
    ) -> Result<()> {
        const STRIDE: usize =
            12;''',
        r'''    pub fn configure_lockstep_packed(
        &mut self,
        packed:
            Float64Array,
    ) -> Result<()> {
        self.reset_approach_a_step_graph();

        const STRIDE: usize =
            12;''',
        "configureLockstepPacked graph reset",
    )

    rs = insert_before(
        rs,
        "#[napi]\nimpl CudaCausalWorker {",
        RS_DROP,
        "CudaCausalWorker Drop",
    )

    return rs

def make_pair_runner() -> str:
    source = B14Q_PATH.read_text(encoding="utf-8")

    source = source.replace("blackwell-b14q", "approach-a-final-pair")
    source = source.replace("b14q-", "approach-a-final-pair-")
    source = source.replace("B14Q_", "APPROACH_A_FINAL_PAIR_")
    source = source.replace("B14Q", "APPROACH_A_FINAL_PAIR")

    source = replace_once(
        source,
        r'''          SUNOPT_CUDA_CAUSAL_PATH:
            causalPath,

          /*
           * Do not allow an earlier shell value to accidentally switch
           * the project to FastMath during the parity test.
           */
          SUNOPT_CUDA_FAST_MATH:
            "0",''',
        r'''          SUNOPT_CUDA_CAUSAL_PATH:
            "A",

          SUNOPT_APPROACH_A_STEP_GRAPH:
            causalPath === "AG"
              ? "1"
              : "0",

          /*
           * Do not allow an earlier shell value to accidentally switch
           * the project to FastMath during the parity test.
           */
          SUNOPT_CUDA_FAST_MATH:
            "0",''',
        "pair runner exact A/AG environment",
    )

    source = replace_once(
        source,
        r'''  const B =
    await runProduction(
      "B",
      "B",
      31472,
      requestBody
    );''',
        r'''  const B =
    await runProduction(
      "AG",
      "AG",
      31472,
      requestBody
    );''',
        "pair runner AG call",
    )

    source = replace_once(
        source,
        r'''  const report = {
    stage:''',
        r'''  const report = {
    comparisonSlots: {
      A:
        "Approach A direct exact FP64",
      B:
        "AG persistent single-step CUDA graph exact FP64",
    },

    stage:''',
        "pair runner slot marker",
    )

    return source

FINAL_RUNNER = r'''import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const pairRunner = path.join(here, "approach-a-final-pair.mjs");
const pairReport = path.join(
  outbox,
  "approach-a-final-pair-production-year-parity.json"
);
const hashFile = path.join(
  projectRoot,
  "updates",
  "history",
  "hash.txt"
);
const finalReport = path.join(
  outbox,
  "approach-a-final-production-ab.json"
);
const llmPath = path.join(outbox, "llm.txt");
const failedRun = path.join(outbox, "failed-run.txt");

fs.mkdirSync(outbox, { recursive: true });

function fail(message, extra = "") {
  fs.writeFileSync(
    failedRun,
    [
      "stage=APPROACH_A_FINAL_PRODUCTION_AB",
      `time=${new Date().toISOString()}`,
      `error=${message}`,
      extra,
      "",
    ].join("\n"),
    "utf8"
  );
  throw new Error(message);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function physicsProjection(result) {
  return {
    fake: result.fake,
    year: result.year,
    heatPumpThermalPower: result.heatPumpThermalPower,
    pvPeakPower: result.pvPeakPower,
    gridPriceChfPerKWh: result.gridPriceChfPerKWh,
    feedInPriceChfPerKWh: result.feedInPriceChfPerKWh,
    points: (result.points ?? []).map(point => ({
      storageVolume: point.storageVolume,
      gridImport: point.gridImport,
      feedIn: point.feedIn,
      selfConsumption: point.selfConsumption,
      storageLosses: point.storageLosses,
      unmetThermalDemand: point.unmetThermalDemand,
      gridCost: point.gridCost,
      feedInRevenue: point.feedInRevenue,
      netEnergyCost: point.netEnergyCost,
    })),
  };
}

function runPair(label) {
  try { fs.unlinkSync(pairReport); } catch {}

  const run = spawnSync(
    process.execPath,
    [pairRunner],
    {
      cwd: projectRoot,
      encoding: "utf8",
      windowsHide: true,
      env: {
        ...process.env,
        SUNOPT_CUDA_FAST_MATH: "0",
      },
    }
  );

  const rawLog = path.join(
    outbox,
    `approach-a-final-${label}-runner.txt`
  );

  fs.writeFileSync(
    rawLog,
    [
      `status=${run.status}`,
      `spawnError=${run.error?.stack ?? "none"}`,
      "",
      "=== STDOUT ===",
      run.stdout ?? "",
      "",
      "=== STDERR ===",
      run.stderr ?? "",
      "",
    ].join("\n"),
    "utf8"
  );

  if (run.error || run.status !== 0) {
    fail(
      `Pair runner failed for ${label}`,
      fs.readFileSync(rawLog, "utf8")
    );
  }

  if (!fs.existsSync(pairReport)) {
    fail(`Pair report missing for ${label}: ${pairReport}`);
  }

  const report = JSON.parse(
    fs.readFileSync(pairReport, "utf8")
  );

  const rawAPath = report.raw?.A;
  const rawAGPath = report.raw?.B;

  if (
    !rawAPath ||
    !rawAGPath ||
    !fs.existsSync(rawAPath) ||
    !fs.existsSync(rawAGPath)
  ) {
    fail(`Raw A/AG result files missing for ${label}`);
  }

  const rawA = JSON.parse(
    fs.readFileSync(rawAPath, "utf8")
  );
  const rawAG = JSON.parse(
    fs.readFileSync(rawAGPath, "utf8")
  );

  const exact =
    JSON.stringify(physicsProjection(rawA.result)) ===
    JSON.stringify(physicsProjection(rawAG.result));

  const saved = path.join(
    outbox,
    `approach-a-final-${label}-pair.json`
  );

  fs.writeFileSync(
    saved,
    JSON.stringify({ exact, report }, null, 2) + "\n",
    "utf8"
  );

  return {
    label,
    exact,
    requestWallA: report.performance.requestWallMs.A,
    requestWallAG: report.performance.requestWallMs.B,
    apiA: report.performance.apiTotalDurationMs.A,
    apiAG: report.performance.apiTotalDurationMs.B,
    sweepA: report.performance.sweepDurationMs.A,
    sweepAG: report.performance.sweepDurationMs.B,
    fixedClockA: report.fixedClock?.A,
    fixedClockAG: report.fixedClock?.B,
    saved,
  };
}

if (!fs.existsSync(hashFile)) {
  fail(`Missing release hash: ${hashFile}`);
}

const release = fs.readFileSync(hashFile, "utf8").trim();

runPair("warmup");

const rows = [];
for (let i = 0; i < 5; ++i) {
  rows.push(runPair(`measured-${i}`));
}

const parityExact = rows.every(row => row.exact);

const summary = {
  requestWallMs: {
    A: median(rows.map(row => row.requestWallA)),
    AG: median(rows.map(row => row.requestWallAG)),
  },
  apiTotalDurationMs: {
    A: median(rows.map(row => row.apiA)),
    AG: median(rows.map(row => row.apiAG)),
  },
  sweepDurationMs: {
    A: median(rows.map(row => row.sweepA)),
    AG: median(rows.map(row => row.sweepAG)),
  },
};

for (const metric of Object.values(summary)) {
  metric.deltaPct =
    100 * (metric.AG - metric.A) / metric.A;
  metric.speedupAOverAG =
    metric.A / metric.AG;
}

let decision;
if (!parityExact) {
  decision = "REJECT_AG_PARITY";
} else if (summary.sweepDurationMs.deltaPct <= -10) {
  decision = "AG_READY_FOR_APPROACH_A_DEFAULT";
} else {
  decision = "APPROACH_A_COMPLETE_KEEP_DIRECT_FP64";
}

const report = {
  stage: "APPROACH_A_FINAL_PRODUCTION_AB",
  release,
  variants: {
    A: "exact FP64 direct CUDA launch",
    AG: "exact FP64 persistent single-step CUDA graph",
  },
  measurementPolicy: {
    warmupPairs: 1,
    measuredPairs: 5,
    medianAuthoritative: true,
    noiseThresholdPercent: 10,
    freshNextProcessPerVariant: true,
  },
  parityExact,
  summary,
  decision,
  rows,
};

fs.writeFileSync(
  finalReport,
  JSON.stringify(report, null, 2) + "\n",
  "utf8"
);

fs.appendFileSync(
  llmPath,
  [
    "",
    `=== ${new Date().toISOString()} APPROACH_A_FINAL_PRODUCTION_AB ===`,
    `release=${release}`,
    `parityExact=${parityExact}`,
    `requestWall median A=${summary.requestWallMs.A.toFixed(3)} AG=${summary.requestWallMs.AG.toFixed(3)} deltaPct=${summary.requestWallMs.deltaPct.toFixed(3)}`,
    `apiTotal median A=${summary.apiTotalDurationMs.A.toFixed(3)} AG=${summary.apiTotalDurationMs.AG.toFixed(3)} deltaPct=${summary.apiTotalDurationMs.deltaPct.toFixed(3)}`,
    `sweep median A=${summary.sweepDurationMs.A.toFixed(3)} AG=${summary.sweepDurationMs.AG.toFixed(3)} deltaPct=${summary.sweepDurationMs.deltaPct.toFixed(3)}`,
    `decision=${decision}`,
    `report=${finalReport}`,
    "",
  ].join("\n"),
  "utf8"
);

console.log(JSON.stringify(report, null, 2));
console.log(`report=${finalReport}`);
console.log(`llm=${llmPath}`);
console.log("APPROACH_A_FINAL_PRODUCTION_AB=OK");
'''

def patch_all() -> tuple[str, str, str, str]:
    h = patch_header(H_PATH.read_text(encoding="utf-8"))
    cu = patch_cuda(CU_PATH.read_text(encoding="utf-8"))
    rs = patch_rust(RS_PATH.read_text(encoding="utf-8"))
    pair = make_pair_runner()
    return h, cu, rs, pair

def main() -> None:
    INBOX.mkdir(parents=True, exist_ok=True)
    OUTBOX.mkdir(parents=True, exist_ok=True)

    for name in ("failed-prompt.txt", "failed-build.txt", "failed-run.txt"):
        try:
            (OUTBOX / name).unlink()
        except FileNotFoundError:
            pass

    source_hashes = verify_sources()
    parent = current_release()
    h, cu, rs, pair = patch_all()

    temp_pair = OUTBOX / "_approach-a-final-pair-check.mjs"
    temp_runner = OUTBOX / "_approach-a-final-runner-check.mjs"
    temp_pair.write_text(pair, encoding="utf-8")
    temp_runner.write_text(FINAL_RUNNER, encoding="utf-8")

    for check in (temp_pair, temp_runner):
        result = subprocess.run(
            ["node", "--check", str(check)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fail(
                f"Node syntax check failed for {check.name}:\n"
                f"{result.stdout}\n{result.stderr}"
            )

    temp_pair.unlink(missing_ok=True)
    temp_runner.unlink(missing_ok=True)

    files = {
        H_REL: h.encode("utf-8"),
        CU_REL: cu.encode("utf-8"),
        RS_REL: rs.encode("utf-8"),
        PAIR_REL: pair.encode("utf-8"),
        RUNNER_REL: FINAL_RUNNER.encode("utf-8"),
    }

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": {
            name: sha256_bytes(data)
            for name, data in files.items()
        },
        "delete": [],
    }

    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    release = sha256_bytes(manifest_bytes)
    package = INBOX / "20261113-approach-a-final-single-step-graph-candidate.zip"

    with zipfile.ZipFile(
        package,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr("manifest.json", manifest_bytes)
        z.writestr("manifest.sha256", release + "\n")
        for name, data in files.items():
            z.writestr(name, data)

    with zipfile.ZipFile(package, "r") as z:
        expected_members = {
            "manifest.json",
            "manifest.sha256",
            *files.keys(),
        }
        if set(z.namelist()) != expected_members:
            fail("Candidate ZIP member set mismatch.")

        if sha256_bytes(z.read("manifest.json")) != z.read(
            "manifest.sha256"
        ).decode().strip():
            fail("Candidate manifest hash verification failed.")

        for name, expected in manifest["files"].items():
            if sha256_bytes(z.read(name)) != expected:
                fail(f"Candidate payload hash mismatch: {name}")

    report = {
        "stage": "APPROACH_A_FINAL_GENERATOR",
        "time": datetime.now(timezone.utc).isoformat(),
        "parent": parent,
        "candidateRelease": release,
        "candidatePackage": str(package),
        "sourcePreconditions": source_hashes,
        "nodeSyntaxCheck": "PASS",
        "outputs": "reports in updates/outbox; installable candidate in updates/inbox",
    }

    report_path = OUTBOX / "approach-a-final-generator.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    llm_path = OUTBOX / "llm.txt"
    with llm_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "",
                    f"=== {datetime.now(timezone.utc).isoformat()} APPROACH_A_FINAL_GENERATOR ===",
                    f"parent={parent}",
                    f"candidateRelease={release}",
                    f"candidatePackage={package}",
                    f"sourcePreconditions={json.dumps(source_hashes, sort_keys=True)}",
                    "nodeSyntaxCheck=PASS",
                    f"report={report_path}",
                    "",
                ]
            )
        )

    print("APPROACH_A_FINAL_GENERATOR=OK")
    print(f"parent={parent}")
    print(f"candidate_release={release}")
    print(f"candidate_package={package}")
    print(f"report={report_path}")
    print(f"llm={llm_path}")
    print()
    print("After candidate installation:")
    print(r"node .\rust\sunopt-node\scripts\build-native-fresh.mjs")
    print(r"node .\rust\sunopt-node\scripts\approach-a-final-production-ab.mjs")

if __name__ == "__main__":
    main()
