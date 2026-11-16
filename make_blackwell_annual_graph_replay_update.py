#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path.cwd()
INBOX = ROOT / "updates" / "inbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"

H_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.h"
CU_PATH = ROOT / "rust" / "sunopt-node" / "cuda" / "cuda_context.cu"
RS_PATH = ROOT / "rust" / "sunopt-node" / "src" / "cuda_context.rs"
MJS_PATH = ROOT / "rust" / "sunopt-node" / "scripts" / "blackwell-graph-replay.mjs"

H_REL = "rust/sunopt-node/cuda/cuda_context.h"
CU_REL = "rust/sunopt-node/cuda/cuda_context.cu"
RS_REL = "rust/sunopt-node/src/cuda_context.rs"
MJS_REL = "rust/sunopt-node/scripts/blackwell-graph-replay.mjs"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

H_BEGIN = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_BEGIN"
H_END = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_END"
CU_BEGIN = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_BEGIN"
CU_END = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_END"
RS_FFI_BEGIN = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FFI_BEGIN"
RS_FFI_END = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FFI_END"
RS_RESULT_BEGIN = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_RESULT_BEGIN"
RS_RESULT_END = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_RESULT_END"
RS_FN_BEGIN = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FN_BEGIN"
RS_FN_END = "// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FN_END"

H_BLOCK = r'''
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_BEGIN
/*
 * Benchmark-only annual CUDA graph replay.
 *
 * The recorded Blackwell snapshot already contains the exact dynamic
 * inputs for every annual step. This API bakes those 35,040 dynamic
 * batches into one CUDA graph so we can measure the dispatch/host-wait
 * cost independently of the causal mathematics.
 *
 * Production lockstep is intentionally unchanged.
 */
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_causal_graph_replay(
    SunoptCudaContext* context,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t step_count,
    size_t tank_count,
    uint32_t repetitions,
    double* host_results,
    double* host_wall_ms,
    double* host_gpu_ms,
    double* out_graph_build_ms,
    double* out_result_d2h_ms
);
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_END
'''.strip()

CU_BLOCK = r'''
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_BEGIN
extern "C" int sunopt_cuda_run_causal_graph_replay(
    SunoptCudaContext* context,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const SunoptCausalDynamic* host_dynamic_parameters,
    size_t step_count,
    size_t tank_count,
    uint32_t repetitions,
    double* host_results,
    double* host_wall_ms,
    double* host_gpu_ms,
    double* out_graph_build_ms,
    double* out_result_d2h_ms
) {
    using Clock =
        std::chrono::steady_clock;

    if (
        context == nullptr ||
        host_current_indices == nullptr ||
        host_horizon_steps == nullptr ||
        host_dynamic_parameters == nullptr ||
        host_results == nullptr ||
        host_wall_ms == nullptr ||
        host_gpu_ms == nullptr ||
        out_graph_build_ms == nullptr ||
        out_result_d2h_ms == nullptr ||
        step_count == 0 ||
        tank_count == 0 ||
        tank_count > 32 ||
        repetitions == 0 ||
        context->causal_length == 0 ||
        context->causal_static_parameter_buffer == nullptr ||
        context->causal_static_job_count != tank_count
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    if (
        step_count >
        SIZE_MAX /
        tank_count
    ) {
        return static_cast<int>(
            cudaErrorInvalidValue
        );
    }

    for (
        size_t step = 0;
        step < step_count;
        ++step
    ) {
        if (
            static_cast<size_t>(
                host_current_indices[step]
            ) >= context->causal_length ||
            host_horizon_steps[step] == 0 ||
            static_cast<size_t>(
                host_horizon_steps[step]
            ) > context->causal_length
        ) {
            return static_cast<int>(
                cudaErrorInvalidValue
            );
        }
    }

    cudaError_t err =
        cudaSetDevice(
            context->device_id
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    const size_t result_count =
        step_count *
        tank_count;

    const size_t result_bytes =
        result_count *
        sizeof(double);

    double* device_results =
        nullptr;

    cudaGraph_t graph =
        nullptr;

    cudaGraphExec_t graph_exec =
        nullptr;

    cudaEvent_t replay_start =
        nullptr;

    cudaEvent_t replay_end =
        nullptr;

    Clock::time_point graph_build_start{};
    Clock::time_point graph_build_end{};
    Clock::time_point d2h_start{};
    Clock::time_point d2h_end{};

    *out_graph_build_ms =
        0.0;

    *out_result_d2h_ms =
        0.0;

    err =
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_results
            ),
            result_bytes
        );

    if (
        err !=
        cudaSuccess
    ) {
        return static_cast<int>(
            err
        );
    }

    const size_t one_array_bytes =
        context->causal_length *
        sizeof(double);

    unsigned char* base =
        static_cast<unsigned char*>(
            context->input_buffer
        );

    const double* export_power =
        reinterpret_cast<const double*>(
            base
        );

    const double* demand_energy =
        reinterpret_cast<const double*>(
            base +
            one_array_bytes
        );

    const auto* device_parameters =
        static_cast<
            const SunoptCausalParameters*
        >(
            context
              ->causal_static_parameter_buffer
        );

    constexpr unsigned int THREADS =
        32;

    const unsigned int blocks =
        static_cast<unsigned int>(
            tank_count
        );

    graph_build_start =
        Clock::now();

    err =
        cudaStreamBeginCapture(
            context->stream,
            cudaStreamCaptureModeThreadLocal
        );

    if (
        err !=
        cudaSuccess
    ) {
        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    for (
        size_t step = 0;
        step < step_count;
        ++step
    ) {
        SunoptCausalDynamicBatch dynamic_batch{};

        const size_t dynamic_offset =
            step *
            tank_count;

        for (
            size_t tank = 0;
            tank < tank_count;
            ++tank
        ) {
            dynamic_batch.values[tank] =
                host_dynamic_parameters[
                    dynamic_offset +
                    tank
                ];
        }

        causal_batch_kernel<<<
            blocks,
            THREADS,
            0,
            context->stream
        >>>(
            export_power,
            demand_energy,
            device_parameters,
            dynamic_batch,
            1,
            static_cast<size_t>(
                host_current_indices[step]
            ),
            static_cast<size_t>(
                host_horizon_steps[step]
            ),
            nullptr,
            device_results +
                dynamic_offset,
            tank_count
        );

        err =
            cudaGetLastError();

        if (
            err !=
            cudaSuccess
        ) {
            break;
        }
    }

    {
        cudaError_t capture_end_error =
            cudaStreamEndCapture(
                context->stream,
                &graph
            );

        if (
            err ==
            cudaSuccess
        ) {
            err =
                capture_end_error;
        }
    }

    if (
        err !=
        cudaSuccess
    ) {
        if (
            graph !=
            nullptr
        ) {
            cudaGraphDestroy(
                graph
            );
        }

        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    err =
        cudaGraphInstantiate(
            &graph_exec,
            graph,
            nullptr,
            nullptr,
            0
        );

    {
        cudaError_t graph_destroy_error =
            cudaGraphDestroy(
                graph
            );

        graph =
            nullptr;

        if (
            err ==
            cudaSuccess &&
            graph_destroy_error !=
            cudaSuccess
        ) {
            err =
                graph_destroy_error;
        }
    }

    if (
        err !=
        cudaSuccess
    ) {
        if (
            graph_exec !=
            nullptr
        ) {
            cudaGraphExecDestroy(
                graph_exec
            );
        }

        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    graph_build_end =
        Clock::now();

    *out_graph_build_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            graph_build_end -
            graph_build_start
        ).count();

    err =
        cudaEventCreate(
            &replay_start
        );

    if (
        err ==
        cudaSuccess
    ) {
        err =
            cudaEventCreate(
                &replay_end
            );
    }

    if (
        err !=
        cudaSuccess
    ) {
        if (
            replay_start !=
            nullptr
        ) {
            cudaEventDestroy(
                replay_start
            );
        }

        if (
            replay_end !=
            nullptr
        ) {
            cudaEventDestroy(
                replay_end
            );
        }

        cudaGraphExecDestroy(
            graph_exec
        );

        cudaFree(
            device_results
        );

        return static_cast<int>(
            err
        );
    }

    /*
     * Untimed warm-up: graph execution objects and code paths are
     * resident before the measured repetitions begin.
     */
    err =
        cudaGraphLaunch(
            graph_exec,
            context->stream
        );

    if (
        err ==
        cudaSuccess
    ) {
        err =
            cudaStreamSynchronize(
                context->stream
            );
    }

    if (
        err ==
        cudaSuccess
    ) {
        for (
            uint32_t repetition = 0;
            repetition < repetitions;
            ++repetition
        ) {
            const auto wall_start =
                Clock::now();

            err =
                cudaEventRecord(
                    replay_start,
                    context->stream
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            err =
                cudaGraphLaunch(
                    graph_exec,
                    context->stream
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            err =
                cudaEventRecord(
                    replay_end,
                    context->stream
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            err =
                cudaEventSynchronize(
                    replay_end
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            const auto wall_end =
                Clock::now();

            float gpu_ms =
                0.0f;

            err =
                cudaEventElapsedTime(
                    &gpu_ms,
                    replay_start,
                    replay_end
                );

            if (
                err !=
                cudaSuccess
            ) {
                break;
            }

            host_gpu_ms[
                repetition
            ] =
                static_cast<double>(
                    gpu_ms
                );

            host_wall_ms[
                repetition
            ] =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    wall_end -
                    wall_start
                ).count();
        }
    }

    if (
        err ==
        cudaSuccess
    ) {
        d2h_start =
            Clock::now();

        err =
            cudaMemcpyAsync(
                host_results,
                device_results,
                result_bytes,
                cudaMemcpyDeviceToHost,
                context->stream
            );

        if (
            err ==
            cudaSuccess
        ) {
            err =
                cudaStreamSynchronize(
                    context->stream
                );
        }

        d2h_end =
            Clock::now();

        if (
            err ==
            cudaSuccess
        ) {
            *out_result_d2h_ms =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    d2h_end -
                    d2h_start
                ).count();
        }
    }

    cudaEventDestroy(
        replay_start
    );

    cudaEventDestroy(
        replay_end
    );

    cudaGraphExecDestroy(
        graph_exec
    );

    {
        cudaError_t free_error =
            cudaFree(
                device_results
            );

        if (
            err ==
            cudaSuccess &&
            free_error !=
            cudaSuccess
        ) {
            err =
                free_error;
        }
    }

    return static_cast<int>(
        err
    );
}
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_END
'''.strip()

RS_FFI_BLOCK = r'''
    // SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FFI_BEGIN
    fn sunopt_cuda_run_causal_graph_replay(
        context:
            *mut SunoptCudaContextOpaque,

        host_current_indices:
            *const u32,

        host_horizon_steps:
            *const u32,

        host_dynamic_parameters:
            *const SunoptCausalDynamic,

        step_count:
            usize,

        tank_count:
            usize,

        repetitions:
            u32,

        host_results:
            *mut f64,

        host_wall_ms:
            *mut f64,

        host_gpu_ms:
            *mut f64,

        out_graph_build_ms:
            *mut f64,

        out_result_d2h_ms:
            *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FFI_END
'''.strip('\n')

RS_RESULT_BLOCK = r'''
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellGraphReplayResult {
    pub snapshot_path:
        String,

    pub device_id:
        i32,

    pub data_length:
        u32,

    pub tank_count:
        u32,

    pub step_count:
        u32,

    pub causal_jobs:
        u32,

    pub repetitions:
        u32,

    pub graph_build_ms:
        f64,

    pub result_d2h_ms:
        f64,

    pub wall_ms:
        Vec<f64>,

    pub gpu_ms:
        Vec<f64>,

    pub median_wall_ms:
        f64,

    pub median_gpu_ms:
        f64,

    pub min_wall_ms:
        f64,

    pub max_wall_ms:
        f64,

    pub checksum_hex:
        String,
}
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_RESULT_END
'''.strip()

RS_FN_BLOCK = r'''
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FN_BEGIN
#[napi]
pub fn run_blackwell_graph_replay(
    snapshot_path:
        Option<String>,

    repetitions:
        Option<u32>,

    device_id:
        Option<i32>,
) -> Result<BlackwellGraphReplayResult> {
    let path =
        snapshot_path
            .map(
                PathBuf::from,
            )
            .unwrap_or_else(
                blackwell_default_snapshot_path,
            );


    let snapshot =
        read_blackwell_snapshot(
            &path,
        )
        .map_err(
            |error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell benchmark snapshot {}: {error}",
                        path.display(),
                    ),
                )
            },
        )?;


    if (
        snapshot.steps.len() !=
        snapshot.data_length
    ) {
        return Err(
            Error::from_reason(
                format!(
                    "Blackwell snapshot contains {} steps but annual data length is {}.",
                    snapshot.steps.len(),
                    snapshot.data_length,
                ),
            ),
        );
    }


    let parameters =
        blackwell_parameters_from_packed(
            &snapshot.packed_config,
            snapshot.data_length,
        )?;


    if (
        parameters.len() !=
        snapshot.tank_count
    ) {
        return Err(
            Error::from_reason(
                "Blackwell snapshot tank count does not match packed configuration.",
            ),
        );
    }


    let selected_device =
        device_id
            .unwrap_or(0);


    let mut context =
        CudaSimulationContext::new(
            selected_device,
        )?;


    context
        .upload_causal_arrays(
            &snapshot.export_power,
            &snapshot.demand_energy,
        )?;


    context
        .configure_causal_static_parameters(
            &parameters,
        )?;


    let repetition_count =
        repetitions
            .unwrap_or(3)
            .clamp(
                1,
                20,
            );


    let step_count =
        snapshot.steps.len();

    let tank_count =
        snapshot.tank_count;


    let total_jobs =
        step_count
            .checked_mul(
                tank_count,
            )
            .ok_or_else(
                || {
                    Error::from_reason(
                        "Blackwell graph replay job count overflow.",
                    )
                },
            )?;


    if (
        total_jobs >
        u32::MAX as usize
    ) {
        return Err(
            Error::from_reason(
                "Blackwell graph replay job count exceeds u32 result capacity.",
            ),
        );
    }


    let mut current_indices =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut horizon_steps =
        Vec::<u32>::with_capacity(
            step_count,
        );

    let mut dynamic_flat =
        Vec::<SunoptCausalDynamic>::with_capacity(
            total_jobs,
        );

    let mut dynamic_step =
        vec![
            SunoptCausalDynamic::default();
            tank_count
        ];


    for step in
        &snapshot.steps
    {
        blackwell_dynamic_from_values(
            &step.dynamic,
            tank_count,
            &mut dynamic_step,
        )?;


        current_indices.push(
            step.current_index,
        );

        horizon_steps.push(
            step.horizon_steps,
        );

        dynamic_flat.extend_from_slice(
            &dynamic_step,
        );
    }


    let mut results =
        vec![
            0.0_f64;
            total_jobs
        ];

    let mut wall_ms =
        vec![
            0.0_f64;
            repetition_count as usize
        ];

    let mut gpu_ms =
        vec![
            0.0_f64;
            repetition_count as usize
        ];

    let mut graph_build_ms =
        0.0_f64;

    let mut result_d2h_ms =
        0.0_f64;


    let code =
        unsafe {
            sunopt_cuda_run_causal_graph_replay(
                context.raw.as_ptr(),
                current_indices.as_ptr(),
                horizon_steps.as_ptr(),
                dynamic_flat.as_ptr(),
                step_count,
                tank_count,
                repetition_count,
                results.as_mut_ptr(),
                wall_ms.as_mut_ptr(),
                gpu_ms.as_mut_ptr(),
                &mut graph_build_ms,
                &mut result_d2h_ms,
            )
        };


    if (
        code !=
        0
    ) {
        return Err(
            cuda_error(
                "sunopt_cuda_run_causal_graph_replay",
                code,
            ),
        );
    }


    let mut checksum =
        1469598103934665603u64;


    for value in
        &results
    {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }


    let median_wall_ms =
        median_f64(
            &wall_ms,
        );

    let median_gpu_ms =
        median_f64(
            &gpu_ms,
        );

    let min_wall_ms =
        wall_ms
            .iter()
            .copied()
            .fold(
                f64::INFINITY,
                f64::min,
            );

    let max_wall_ms =
        wall_ms
            .iter()
            .copied()
            .fold(
                f64::NEG_INFINITY,
                f64::max,
            );


    Ok(
        BlackwellGraphReplayResult {
            snapshot_path:
                path.display()
                    .to_string(),

            device_id:
                selected_device,

            data_length:
                snapshot.data_length
                    as u32,

            tank_count:
                tank_count
                    as u32,

            step_count:
                step_count
                    as u32,

            causal_jobs:
                total_jobs
                    as u32,

            repetitions:
                repetition_count,

            graph_build_ms,

            result_d2h_ms,

            wall_ms,
            gpu_ms,

            median_wall_ms,
            median_gpu_ms,
            min_wall_ms,
            max_wall_ms,

            checksum_hex:
                format!(
                    "{checksum:016x}",
                ),
        },
    )
}
// SUNOPT_BLACKWELL_ANNUAL_GRAPH_REPLAY_FN_END
'''.strip()

MJS = r'''import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshotPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");

const repetitions = Number.parseInt(process.argv[3] ?? "3", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

if (typeof addon.runBlackwellGraphReplay !== "function") {
  throw new Error(
    "runBlackwellGraphReplay is not exported by sunopt-node. Rebuild the native addon first.",
  );
}

const result = addon.runBlackwellGraphReplay(
  snapshotPath,
  repetitions,
  deviceId,
);

console.log(`SUNOPT_BLACKWELL_GRAPH_JSON=${JSON.stringify(result)}`);
console.log("--- GRAPH REPLAY SUMMARY ---");
console.log(`graphBuildMs=${result.graphBuildMs.toFixed(3)}`);
console.log(`medianWallMs=${result.medianWallMs.toFixed(3)}`);
console.log(`medianGpuMs=${result.medianGpuMs.toFixed(3)}`);
console.log(`resultD2HMs=${result.resultD2HMs.toFixed(3)}`);
console.log(`wallMs=[${result.wallMs.map((value) => value.toFixed(3)).join(", ")}]`);
console.log(`gpuMs=[${result.gpuMs.map((value) => value.toFixed(3)).join(", ")}]`);
console.log(`checksumHex=${result.checksumHex}`);
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="utf-8").strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid SHA-256 in {HASH_FILE}")

    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).strip().lower()
        if HASH_RE.fullmatch(value):
            return value
        raise RuntimeError(f"Invalid sha256 in {RELEASE_FILE}")

    return GENESIS


def replace_between_markers(source: str, begin: str, end: str, block: str) -> str:
    has_begin = begin in source
    has_end = end in source
    if has_begin != has_end:
        raise RuntimeError(f"Only one marker present: {begin!r} / {end!r}")
    if not has_begin:
        raise RuntimeError(f"Marker pair not present: {begin!r} / {end!r}")
    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[:start] + block + source[finish:]


def insert_or_replace_before(source: str, begin: str, end: str, block: str, anchor: str) -> str:
    if begin in source or end in source:
        return replace_between_markers(source, begin, end, block)
    position = source.find(anchor)
    if position < 0:
        raise RuntimeError(f"Required insertion anchor not found: {anchor!r}")
    return source[:position] + block + "\n\n\n" + source[position:]


def patch_header(source: str) -> str:
    return insert_or_replace_before(
        source,
        H_BEGIN,
        H_END,
        H_BLOCK,
        "SUNOPT_CUDA_EXPORT int sunopt_cuda_get_profile(",
    )


def patch_cuda(source: str) -> str:
    required = [
        "struct SunoptCausalDynamicBatch",
        "causal_batch_kernel(",
        "context->causal_static_parameter_buffer",
        "context->causal_static_job_count",
    ]
    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Current cuda_context.cu is not the expected lockstep generation; missing {token!r}."
            )

    return insert_or_replace_before(
        source,
        CU_BEGIN,
        CU_END,
        CU_BLOCK,
        'extern "C" int sunopt_cuda_get_profile(',
    )


def patch_rust(source: str) -> str:
    required = [
        "struct BlackwellReplaySnapshot",
        "fn read_blackwell_snapshot(",
        "fn blackwell_parameters_from_packed(",
        "fn blackwell_dynamic_from_values(",
        "pub fn run_blackwell_replay(",
        "struct SunoptCausalDynamic",
        "fn configure_causal_static_parameters(",
    ]
    for token in required:
        if token not in source:
            raise RuntimeError(
                f"Current cuda_context.rs is not the expected Blackwell replay generation; missing {token!r}."
            )

    source = insert_or_replace_before(
        source,
        RS_FFI_BEGIN,
        RS_FFI_END,
        RS_FFI_BLOCK,
        "    fn sunopt_cuda_get_profile(",
    )

    source = insert_or_replace_before(
        source,
        RS_RESULT_BEGIN,
        RS_RESULT_END,
        RS_RESULT_BLOCK,
        "#[napi]\npub struct CudaCausalWorker {",
    )

    source = insert_or_replace_before(
        source,
        RS_FN_BEGIN,
        RS_FN_END,
        RS_FN_BLOCK,
        "#[napi]\npub fn run_blackwell_replay(",
    )

    return source


def build_package() -> Path:
    for path in (H_PATH, CU_PATH, RS_PATH):
        if not path.is_file():
            raise RuntimeError(f"Required current project file missing: {path}")

    h = patch_header(H_PATH.read_text(encoding="utf-8"))
    cu = patch_cuda(CU_PATH.read_text(encoding="utf-8"))
    rs = patch_rust(RS_PATH.read_text(encoding="utf-8"))

    payloads = {
        H_REL: h.encode("utf-8"),
        CU_REL: cu.encode("utf-8"),
        RS_REL: rs.encode("utf-8"),
        MJS_REL: MJS.encode("utf-8"),
    }

    parent = read_parent_hash()
    files = {name: sha256_bytes(data) for name, data in payloads.items()}

    manifest = {
        "format": 2,
        "parent_sha256": parent,
        "files": files,
        "delete": [],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_hash = sha256_bytes(manifest_bytes)

    INBOX.mkdir(parents=True, exist_ok=True)
    output = INBOX / "20261106-blackwell-annual-graph-replay.zip"

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("manifest.sha256", (manifest_hash + "\n").encode("ascii"))
        for name, data in payloads.items():
            archive.writestr(name, data)

    print("SunOpt update package created")
    print(f"  package: {output}")
    print(f"  parent : {parent}")
    print(f"  release: {manifest_hash}")
    print("  files:")
    for name in payloads:
        print(f"    - {name}")
    print()
    print("Next:")
    print(r"  python .\tools\update_watcher.py --once")
    print(r"  cd .\rust\sunopt-node")
    print(r"  npm run build")
    print(r"  node .\scripts\blackwell-graph-replay.mjs .\blackwell-benchmark.bin 3 0")

    return output


if __name__ == "__main__":
    build_package()
