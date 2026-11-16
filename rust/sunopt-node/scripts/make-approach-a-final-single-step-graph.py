#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] if 'rust' in str(Path(__file__)) else Path.cwd()
# When installed at rust/sunopt-node/scripts, parents[3] is project root.
if not (ROOT / 'rust' / 'sunopt-node').exists():
    ROOT = Path.cwd()

H_PATH = ROOT / 'rust' / 'sunopt-node' / 'cuda' / 'cuda_context.h'
CU_PATH = ROOT / 'rust' / 'sunopt-node' / 'cuda' / 'cuda_context.cu'
RS_PATH = ROOT / 'rust' / 'sunopt-node' / 'src' / 'cuda_context.rs'
H_REL = 'rust/sunopt-node/cuda/cuda_context.h'
CU_REL = 'rust/sunopt-node/cuda/cuda_context.cu'
RS_REL = 'rust/sunopt-node/src/cuda_context.rs'
RUNNER_REL = 'rust/sunopt-node/scripts/approach-a-final-single-step-graph-production-ab.mjs'
INBOX = ROOT / 'updates' / 'inbox'
RELEASE_FILE = ROOT / 'updates' / 'installed-release.json'
HASH_FILE = ROOT / 'updates' / 'history' / 'hash.txt'
GENESIS = '0' * 64
HASH_RE = re.compile(r'^[0-9a-f]{64}$')

H_BEGIN = '// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN'
H_END = '// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END'
CU_BEGIN = H_BEGIN
CU_END = H_END
RS_FFI_BEGIN = '// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_FFI_BEGIN'
RS_FFI_END = '// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_FFI_END'
RS_HELPER_BEGIN = '// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPER_BEGIN'
RS_HELPER_END = '// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPER_END'

H_BLOCK = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN
/* Production-safe final Approach-A graph candidate.
 * ONE causal timestep per graph launch; annual host/controller feedback
 * remains strictly sequential and unchanged.
 */
typedef struct SunoptCausalStepGraph SunoptCausalStepGraph;

SUNOPT_CUDA_EXPORT int sunopt_cuda_create_causal_step_graph(
    SunoptCudaContext* context,
    size_t tank_count,
    SunoptCausalStepGraph** out_graph
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_step_graph(
    SunoptCausalStepGraph* graph,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
);

SUNOPT_CUDA_EXPORT int sunopt_cuda_destroy_causal_step_graph(
    SunoptCausalStepGraph* graph
);
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END
'''.strip()

CU_BLOCK = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_BEGIN
struct SunoptCausalStepGraph {
    SunoptCudaContext* context = nullptr;
    size_t tank_count = 0;
    const double* export_power = nullptr;
    const double* demand_energy = nullptr;
    const SunoptCausalParameters* device_parameters = nullptr;
    double* device_results = nullptr;
    double* pinned_results = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphNode_t kernel_node = nullptr;
    cudaGraphNode_t copy_node = nullptr;
};

static void sunopt_cuda_cleanup_causal_step_graph(
    SunoptCausalStepGraph* graph
) {
    if (graph == nullptr) return;
    if (graph->graph_exec != nullptr) {
        cudaGraphExecDestroy(graph->graph_exec);
        graph->graph_exec = nullptr;
    }
    if (graph->graph != nullptr) {
        cudaGraphDestroy(graph->graph);
        graph->graph = nullptr;
    }
    if (graph->device_results != nullptr) {
        cudaFree(graph->device_results);
        graph->device_results = nullptr;
    }
    if (graph->pinned_results != nullptr) {
        cudaFreeHost(graph->pinned_results);
        graph->pinned_results = nullptr;
    }
}

extern "C" int sunopt_cuda_create_causal_step_graph(
    SunoptCudaContext* context,
    size_t tank_count,
    SunoptCausalStepGraph** out_graph
) {
    if (
        context == nullptr || out_graph == nullptr || tank_count == 0 ||
        tank_count > 32 || context->causal_length == 0 ||
        context->input_buffer == nullptr ||
        context->causal_static_parameter_buffer == nullptr ||
        context->causal_static_job_count != tank_count
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    *out_graph = nullptr;
    cudaError_t err = cudaSetDevice(context->device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    auto* state = new SunoptCausalStepGraph{};
    state->context = context;
    state->tank_count = tank_count;

    const size_t one_array_bytes = context->causal_length * sizeof(double);
    auto* base = static_cast<unsigned char*>(context->input_buffer);
    state->export_power = reinterpret_cast<const double*>(base);
    state->demand_energy = reinterpret_cast<const double*>(base + one_array_bytes);
    state->device_parameters = static_cast<const SunoptCausalParameters*>(
        context->causal_static_parameter_buffer
    );

    const size_t result_bytes = tank_count * sizeof(double);
    err = cudaMalloc(reinterpret_cast<void**>(&state->device_results), result_bytes);
    if (err == cudaSuccess) {
        err = cudaMallocHost(reinterpret_cast<void**>(&state->pinned_results), result_bytes);
    }
    if (err == cudaSuccess) err = cudaGraphCreate(&state->graph, 0);
    if (err != cudaSuccess) {
        sunopt_cuda_cleanup_causal_step_graph(state);
        delete state;
        return static_cast<int>(err);
    }

    SunoptCausalDynamicBatch dynamic_batch{};
    int cut_only = 1;
    size_t current_index = 0;
    size_t horizon_steps = 1;
    SunoptCausalResult* full_results = nullptr;
    const double* export_power = state->export_power;
    const double* demand_energy = state->demand_energy;
    const SunoptCausalParameters* device_parameters = state->device_parameters;
    double* device_results = state->device_results;
    size_t job_count = tank_count;

    void* kernel_arguments[] = {
        &export_power,
        &demand_energy,
        &device_parameters,
        &dynamic_batch,
        &cut_only,
        &current_index,
        &horizon_steps,
        &full_results,
        &device_results,
        &job_count,
    };

    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(causal_batch_kernel);
    params.gridDim = dim3(static_cast<unsigned int>(tank_count), 1, 1);
    params.blockDim = dim3(32, 1, 1);
    params.sharedMemBytes = 0;
    params.kernelParams = kernel_arguments;
    params.extra = nullptr;

    err = cudaGraphAddKernelNode(
        &state->kernel_node, state->graph, nullptr, 0, &params
    );
    if (err == cudaSuccess) {
        err = cudaGraphAddMemcpyNode1D(
            &state->copy_node,
            state->graph,
            &state->kernel_node,
            1,
            state->pinned_results,
            state->device_results,
            result_bytes,
            cudaMemcpyDeviceToHost
        );
    }
    if (err == cudaSuccess) {
        err = cudaGraphInstantiate(
            &state->graph_exec, state->graph, nullptr, nullptr, 0
        );
    }
    if (err != cudaSuccess) {
        sunopt_cuda_cleanup_causal_step_graph(state);
        delete state;
        return static_cast<int>(err);
    }

    *out_graph = state;
    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_run_causal_step_graph(
    SunoptCausalStepGraph* graph,
    size_t current_index,
    size_t horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t job_count,
    double* host_cut_level_results
) {
    if (
        graph == nullptr || graph->context == nullptr ||
        graph->graph_exec == nullptr || graph->kernel_node == nullptr ||
        graph->device_results == nullptr || graph->pinned_results == nullptr ||
        host_dynamic_parameters == nullptr || host_cut_level_results == nullptr ||
        job_count == 0 || job_count != graph->tank_count || job_count > 32 ||
        current_index >= graph->context->causal_length ||
        horizon_steps == 0 || horizon_steps > graph->context->causal_length
    ) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(graph->context->device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    SunoptCausalDynamicBatch dynamic_batch{};
    for (size_t tank = 0; tank < job_count; ++tank) {
        dynamic_batch.values[tank] = host_dynamic_parameters[tank];
    }

    int cut_only = 1;
    SunoptCausalResult* full_results = nullptr;
    const double* export_power = graph->export_power;
    const double* demand_energy = graph->demand_energy;
    const SunoptCausalParameters* device_parameters = graph->device_parameters;
    double* device_results = graph->device_results;

    void* kernel_arguments[] = {
        &export_power,
        &demand_energy,
        &device_parameters,
        &dynamic_batch,
        &cut_only,
        &current_index,
        &horizon_steps,
        &full_results,
        &device_results,
        &job_count,
    };

    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(causal_batch_kernel);
    params.gridDim = dim3(static_cast<unsigned int>(job_count), 1, 1);
    params.blockDim = dim3(32, 1, 1);
    params.sharedMemBytes = 0;
    params.kernelParams = kernel_arguments;
    params.extra = nullptr;

    err = cudaGraphExecKernelNodeSetParams(
        graph->graph_exec, graph->kernel_node, &params
    );
    if (err == cudaSuccess) {
        err = cudaGraphLaunch(graph->graph_exec, graph->context->stream);
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(graph->context->stream);
    }
    if (err != cudaSuccess) return static_cast<int>(err);

    for (size_t tank = 0; tank < job_count; ++tank) {
        host_cut_level_results[tank] = graph->pinned_results[tank];
    }

    return static_cast<int>(cudaSuccess);
}

extern "C" int sunopt_cuda_destroy_causal_step_graph(
    SunoptCausalStepGraph* graph
) {
    if (graph == nullptr) return static_cast<int>(cudaSuccess);
    cudaError_t err = cudaSetDevice(
        graph->context != nullptr ? graph->context->device_id : 0
    );
    sunopt_cuda_cleanup_causal_step_graph(graph);
    delete graph;
    return static_cast<int>(err);
}
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_END
'''.strip()

RS_FFI_BLOCK = r'''
    // SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_FFI_BEGIN
    fn sunopt_cuda_create_causal_step_graph(
        context: *mut SunoptCudaContextOpaque,
        tank_count: usize,
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
'''.rstrip()

RS_HELPER_BLOCK = r'''
// SUNOPT_APPROACH_A_SINGLE_STEP_GRAPH_HELPER_BEGIN
impl CudaCausalWorker {
    fn reset_approach_a_step_graph(&mut self) {
        if self.approach_a_step_graph.is_null() {
            return;
        }
        let _ = unsafe {
            sunopt_cuda_destroy_causal_step_graph(
                self.approach_a_step_graph,
            )
        };
        self.approach_a_step_graph = std::ptr::null_mut();
    }

    fn ensure_approach_a_step_graph(&mut self) -> Result<()> {
        if !self.approach_a_step_graph.is_null() {
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
        self.approach_a_step_graph = graph;
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

RUNNER = r'''import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const nativeRoot = path.resolve(path.dirname(scriptPath), "..");
const root = path.resolve(nativeRoot, "..", "..");
const port = 3111;
const origin = `http://127.0.0.1:${port}`;
const outbox = path.join(root, "updates", "outbox");
fs.mkdirSync(outbox, { recursive: true });
const reportPath = path.join(outbox, "approach-a-final-single-step-graph-production-ab.json");
const llmPath = path.join(root, "llm.txt");
const requestBody = {
  heatPumpThermalPower: 5,
  pvPeakPower: 10,
  gridPriceChfPerKWh: 0.3,
  feedInPriceChfPerKWh: 0.08,
  causalPath: "A",
};
const sequence = ["A","AG","AG","A","A","AG","AG","A","A","AG"];
const median = (v) => { const s=[...v].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; };
const stats = (v) => ({ values:v, min:Math.min(...v), max:Math.max(...v), median:median(v) });
function probeTcpPort(timeoutMs=1000){return new Promise((resolve)=>{const socket=net.createConnection({host:"127.0.0.1",port});let done=false;const finish=(ok)=>{if(done)return;done=true;socket.destroy();resolve(ok);};socket.setTimeout(timeoutMs);socket.once("connect",()=>finish(true));socket.once("timeout",()=>finish(false));socket.once("error",()=>finish(false));});}
async function waitForServer(child){const deadline=Date.now()+120000;while(Date.now()<deadline){if(child.exitCode!==null)throw new Error(`Next dev exited early code=${child.exitCode}`);if(await probeTcpPort()){await new Promise(r=>setTimeout(r,600));return;}await new Promise(r=>setTimeout(r,250));}throw new Error("Timed out waiting for Next dev");}
function stopTree(child){if(!child?.pid)return;if(process.platform==="win32"){spawnSync("taskkill",["/PID",String(child.pid),"/T","/F"],{cwd:root,stdio:"ignore",windowsHide:true});}else{try{process.kill(-child.pid,"SIGTERM");}catch{try{child.kill("SIGTERM");}catch{}}}}
async function requestOnce(){const start=performance.now();const response=await fetch(`${origin}/api/tank-comparison`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(requestBody)});const wallMs=performance.now()-start;const text=await response.text();if(!response.ok)throw new Error(`HTTP ${response.status}: ${text.slice(0,2000)}`);const json=JSON.parse(text);if(!Array.isArray(json.points)||json.points.length!==28)throw new Error(`Expected 28 points, got ${json.points?.length}`);return {wallMs,response:json};}
function physics(response){return response.points.map((p)=>({storageVolume:p.storageVolume,gridImport:p.gridImport,feedIn:p.feedIn,selfConsumption:p.selfConsumption,storageLosses:p.storageLosses,unmetThermalDemand:p.unmetThermalDemand,gridCost:p.gridCost,feedInRevenue:p.feedInRevenue,netEnergyCost:p.netEnergyCost}));}
const exact=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function telemetry(){const r=spawnSync("nvidia-smi",["--query-gpu=clocks.gr,temperature.gpu,pstate,power.draw,utilization.gpu","--format=csv,noheader,nounits"],{cwd:root,encoding:"utf8",windowsHide:true});if(r.status!==0)return null;const p=String(r.stdout??"").trim().split(/\r?\n/)[0]?.split(",").map(x=>x.trim());if(!p?.length)return null;return {clockMHz:Number(p[0]),temperatureC:Number(p[1]),pstate:p[2],powerW:Number(p[3]),utilizationPercent:Number(p[4])};}
async function runVariant(variant,index){const label=variant==="AG"?"graph":"reference";const logPath=path.join(outbox,`approach-a-final-${String(index+1).padStart(2,"0")}-${label}-server.log`);const log=fs.openSync(logPath,"w");const env={...process.env,SUNOPT_CUDA_CAUSAL_PATH:variant,SUNOPT_CUDA_FAST_MATH:"0"};const command=process.platform==="win32"?"cmd.exe":"npm";const args=process.platform==="win32"?["/d","/s","/c",`npm run dev -- --hostname 127.0.0.1 --port ${port}`]:["run","dev","--","--hostname","127.0.0.1","--port",String(port)];const child=spawn(command,args,{cwd:root,env,detached:process.platform!=="win32",stdio:["ignore",log,log],windowsHide:true});try{await waitForServer(child);await requestOnce();const before=telemetry();const measured=await requestOnce();const after=telemetry();return {variant,logPath,before,after,wallMs:measured.wallMs,apiTotalMs:measured.response.totalDurationMs,sweepMs:measured.response.sweepDurationMs,preparationMs:measured.response.preparationDurationMs,cudaBatchWallMs:measured.response.gpuLockstep?.cudaBatchWallMs??null,response:measured.response};}finally{stopTree(child);fs.closeSync(log);await new Promise(r=>setTimeout(r,900));}}
const runs=[];for(let i=0;i<sequence.length;i++){const v=sequence[i];process.stdout.write(`[${i+1}/${sequence.length}] ${v} ... `);const run=await runVariant(v,i);runs.push(run);console.log(`wall=${run.wallMs.toFixed(3)}ms sweep=${run.sweepMs.toFixed(3)}ms`);}
const A=runs.filter(r=>r.variant==="A");const AG=runs.filter(r=>r.variant==="AG");if(A.length!==5||AG.length!==5)throw new Error(`Expected 5+5 runs, got ${A.length}+${AG.length}`);const canonical=physics(A[0].response);const parity={referenceExact:A.every(r=>exact(canonical,physics(r.response))),graphExact:AG.every(r=>exact(canonical,physics(r.response)))};const metrics={};for(const [name,key] of [["wallMs","wallMs"],["apiTotalMs","apiTotalMs"],["sweepMs","sweepMs"],["preparationMs","preparationMs"],["cudaBatchWallMs","cudaBatchWallMs"]]){const av=A.map(r=>r[key]).filter(v=>typeof v==="number");const gv=AG.map(r=>r[key]).filter(v=>typeof v==="number");const a=stats(av),g=stats(gv);metrics[name]={A:a,AG:g,deltaPct:(g.median-a.median)/a.median*100,speedup:a.median/g.median};}
let decision;if(!parity.referenceExact||!parity.graphExact)decision="REJECT_AG_PARITY";else if(metrics.sweepMs.deltaPct<=-10)decision="AG_READY_FOR_APPROACH_A_DEFAULT";else decision="APPROACH_A_COMPLETE_KEEP_DIRECT_FP64";
const report={stage:"APPROACH_A_FINAL_SINGLE_STEP_GRAPH_PRODUCTION_AB",requestBody,sequence,parity,metrics,interpretation:{oldAnnualReplay:"Not used; pre-recorded annual dynamics removed feedback synchronization.",candidate:"Persistent one-step CUDA graph; exact FP64 kernel; every host/controller feedback boundary preserved.",thresholdPercent:10},decision,runs:runs.map(r=>({variant:r.variant,wallMs:r.wallMs,apiTotalMs:r.apiTotalMs,sweepMs:r.sweepMs,preparationMs:r.preparationMs,cudaBatchWallMs:r.cudaBatchWallMs,before:r.before,after:r.after,logPath:r.logPath}))};fs.writeFileSync(reportPath,JSON.stringify(report,null,2)+"\n","utf8");const summary=["",`=== ${new Date().toISOString()} APPROACH_A_FINAL_SINGLE_STEP_GRAPH_PRODUCTION_AB ===`,`parity referenceExact=${parity.referenceExact} graphExact=${parity.graphExact}`,`sweep A=${metrics.sweepMs.A.median.toFixed(3)}ms AG=${metrics.sweepMs.AG.median.toFixed(3)}ms deltaPct=${metrics.sweepMs.deltaPct.toFixed(3)} speedup=${metrics.sweepMs.speedup.toFixed(4)}x`,`wall A=${metrics.wallMs.A.median.toFixed(3)}ms AG=${metrics.wallMs.AG.median.toFixed(3)}ms deltaPct=${metrics.wallMs.deltaPct.toFixed(3)}`,`cudaBatch A=${metrics.cudaBatchWallMs.A.median.toFixed(3)}ms AG=${metrics.cudaBatchWallMs.AG.median.toFixed(3)}ms deltaPct=${metrics.cudaBatchWallMs.deltaPct.toFixed(3)}`,`decision=${decision}`,`report=${reportPath}`,""].join("\n");fs.appendFileSync(llmPath,summary,"utf8");console.log(summary);
'''

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def current_parent() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding='utf-8').strip().lower()
        if HASH_RE.fullmatch(value):
            return value
    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding='utf-8'))
        value = str(state.get('sha256', '')).strip().lower()
        if HASH_RE.fullmatch(value):
            return value
    return GENESIS

def insert_or_replace_before(source: str, begin: str, end: str, block: str, anchors: list[str]) -> str:
    hb, he = begin in source, end in source
    if hb != he:
        raise RuntimeError(f'Only one marker present: {begin} / {end}')
    if hb:
        start = source.index(begin)
        finish = source.index(end, start) + len(end)
        return source[:start] + block + source[finish:]
    for anchor in anchors:
        pos = source.find(anchor)
        if pos >= 0:
            return source[:pos] + block + '\n\n\n' + source[pos:]
    raise RuntimeError('Required insertion anchor not found: ' + ' | '.join(anchors))

def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one anchor; found {count}')
    return source.replace(old, new, 1)

def patch_header(source: str) -> str:
    for token in ('typedef struct SunoptCudaContext', 'SunoptCausalDynamic', 'sunopt_cuda_run_causal_dynamic_batch'):
        if token not in source:
            raise RuntimeError(f'Unexpected cuda_context.h; missing {token!r}')
    return insert_or_replace_before(source, H_BEGIN, H_END, H_BLOCK, ['SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile(', '#ifdef __cplusplus\n}'])

def patch_cuda(source: str) -> str:
    for token in ('struct SunoptCausalDynamicBatch', 'causal_batch_kernel', 'causal_static_parameter_buffer', 'causal_static_job_count', 'context->input_buffer', 'context->stream'):
        if token not in source:
            raise RuntimeError(f'Unexpected cuda_context.cu; missing {token!r}')
    return insert_or_replace_before(source, CU_BEGIN, CU_END, CU_BLOCK, ['extern "C" int sunopt_cuda_get_profile(', 'extern "C" const char* sunopt_cuda_error_string('])

def patch_rust(source: str) -> str:
    for token in ('pub struct CudaCausalWorker', 'blackwell_capture:', 'pub fn configure_lockstep_packed(', 'pub fn run_lockstep_dynamic_into(', 'fn sunopt_cuda_run_causal_dynamic_batch('):
        if token not in source:
            raise RuntimeError(f'Unexpected cuda_context.rs; missing {token!r}')
    source = insert_or_replace_before(source, RS_FFI_BEGIN, RS_FFI_END, RS_FFI_BLOCK, ['    fn sunopt_cuda_get_profile(', '    fn sunopt_cuda_error_string('])

    if 'approach_a_step_graph:' not in source:
        struct_pattern = re.compile(
            r'(?s)(#\[napi\]\s+pub struct CudaCausalWorker\s*\{.*?\n\s*blackwell_capture:\s*\n\s*Option<BlackwellCapture>,)(\s*\n\})'
        )
        source, count = struct_pattern.subn(
            r'\1\n\n    approach_a_step_graph:\n        *mut std::ffi::c_void,\2',
            source,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f'CudaCausalWorker field: expected one struct insertion point; found {count}'
            )

    if 'approach_a_step_graph:\n                std::ptr::null_mut()' not in source:
        constructor_pattern = re.compile(
            r'(?s)(Ok\(Self\s*\{.*?\n\s*blackwell_capture,)(\s*\n\s*\}\)\s*\n\s*\})'
        )
        source, count = constructor_pattern.subn(
            r'\1\n\n            approach_a_step_graph:\n                std::ptr::null_mut(),\2',
            source,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f'CudaCausalWorker constructor: expected one insertion point; found {count}'
            )

    source = insert_or_replace_before(source, RS_HELPER_BEGIN, RS_HELPER_END, RS_HELPER_BLOCK, ['#[napi]\nimpl CudaCausalWorker {'])

    configure_anchor = '''    pub fn configure_lockstep_packed(
        &mut self,
        packed:
            Float64Array,
    ) -> Result<()> {'''
    configure_replacement = configure_anchor + '''

        self.reset_approach_a_step_graph();'''
    if configure_replacement not in source:
        source = replace_once(source, configure_anchor, configure_replacement, 'configure reset')

    old_call = '''        self.context
            .run_causal_dynamic_batch_into(
                current_index as usize,
                horizon_steps as usize,
                &self.lockstep_dynamic,
                output_slice,
            )?;'''
    new_call = '''        let use_approach_a_graph =
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
                    self.approach_a_step_graph,
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
    run_pos = source.find('pub fn run_lockstep_dynamic_into(')
    if run_pos < 0:
        raise RuntimeError('runLockstepDynamicInto not found')
    if 'sunopt_cuda_run_causal_step_graph(' not in source[run_pos:]:
        source = replace_once(source, old_call, new_call, 'graph dispatch')
    return source

def build_package() -> Path:
    for path in (H_PATH, CU_PATH, RS_PATH):
        if not path.is_file():
            raise RuntimeError(f'Required current project file missing: {path}')
    h = patch_header(H_PATH.read_text(encoding='utf-8'))
    cu = patch_cuda(CU_PATH.read_text(encoding='utf-8'))
    rs = patch_rust(RS_PATH.read_text(encoding='utf-8'))
    payloads = {
        H_REL: h.encode('utf-8'),
        CU_REL: cu.encode('utf-8'),
        RS_REL: rs.encode('utf-8'),
        RUNNER_REL: (RUNNER + '\n').encode('utf-8'),
    }
    parent = current_parent()
    manifest = {
        'format': 2,
        'parent_sha256': parent,
        'files': {name: sha256_bytes(data) for name, data in payloads.items()},
        'delete': [],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8')
    release = sha256_bytes(manifest_bytes)
    INBOX.mkdir(parents=True, exist_ok=True)
    package_path = INBOX / '20261113-approach-a-final-single-step-graph-candidate.zip'
    with zipfile.ZipFile(package_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', manifest_bytes)
        z.writestr('manifest.sha256', release + '\n')
        for name, data in payloads.items():
            z.writestr(name, data)
    with zipfile.ZipFile(package_path, 'r') as z:
        if sha256_bytes(z.read('manifest.json')) != z.read('manifest.sha256').decode().strip():
            raise RuntimeError('Manifest hash verification failed')
        for name, expected in manifest['files'].items():
            if sha256_bytes(z.read(name)) != expected:
                raise RuntimeError(f'Payload hash mismatch: {name}')
    print('APPROACH_A_FINAL_GRAPH_GENERATOR=OK')
    print(f'parent={parent}')
    print(f'candidate_release={release}')
    print(f'package={package_path}')
    print('After watcher install:')
    print(r'node .\rust\sunopt-node\scripts\build-native-fresh.mjs')
    print(r'node .\rust\sunopt-node\scripts\approach-a-final-single-step-graph-production-ab.mjs')
    return package_path

if __name__ == '__main__':
    build_package()
