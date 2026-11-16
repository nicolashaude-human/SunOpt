// rust/sunopt-node/scripts/make-blackwell-b14i1-lockstep.mjs
//
// Creates a verified format-2 watcher package from the CURRENT SunOpt sources.
// Run from project root:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14i1-lockstep.mjs
//
// Output package:
//   updates\inbox\20261112-blackwell-b14i1-production-shaped-lockstep.zip
//
// Benchmark output after install/build:
//   updates\outbox\blackwell-b14i1-lockstep.json
//
// This generator does NOT modify source files directly.

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");

const inbox = path.join(root, "updates", "inbox");
const hashFile = path.join(root, "updates", "history", "hash.txt");

const hPath = path.join(root, "rust", "sunopt-node", "cuda", "cuda_context.h");
const cuPath = path.join(root, "rust", "sunopt-node", "cuda", "cuda_context.cu");
const rsPath = path.join(root, "rust", "sunopt-node", "src", "cuda_context.rs");

const hRel = "rust/sunopt-node/cuda/cuda_context.h";
const cuRel = "rust/sunopt-node/cuda/cuda_context.cu";
const rsRel = "rust/sunopt-node/src/cuda_context.rs";
const runnerRel = "rust/sunopt-node/scripts/blackwell-b14i1-lockstep.mjs";

function sha256Buffer(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function readText(file) {
  return fs.readFileSync(file, "utf8");
}

function insertBefore(text, anchor, block, label) {
  const firstLine = block.split(/\r?\n/, 1)[0];
  if (text.includes(firstLine)) {
    throw new Error(`${label}: block already exists`);
  }
  const pos = text.indexOf(anchor);
  if (pos < 0) {
    throw new Error(`${label}: anchor not found: ${anchor}`);
  }
  return text.slice(0, pos) + block.trimEnd() + "\n\n" + text.slice(pos);
}

const parent = fs.existsSync(hashFile)
  ? fs.readFileSync(hashFile, "utf8").trim().toLowerCase()
  : "0".repeat(64);

if (!/^[0-9a-f]{64}$/.test(parent)) {
  throw new Error(`Invalid current hash: ${JSON.stringify(parent)}`);
}

const H_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_BEGIN
SUNOPT_CUDA_EXPORT int sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    const double* host_energy_cut_levels,
    size_t step_count,
    double* host_cut_results,
    double* out_static_h2d_ms,
    double* out_dynamic_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
);
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_END
`.trim();

const CU_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_BEGIN
extern "C" int sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark(
    int device_id,
    const double* host_export_power,
    const double* host_demand_energy,
    size_t data_length,
    const double* host_tank_parameters,
    size_t tank_count,
    const uint32_t* host_current_indices,
    const uint32_t* host_horizon_steps,
    const double* host_dynamic_values,
    const double* host_energy_cut_levels,
    size_t step_count,
    double* host_cut_results,
    double* out_static_h2d_ms,
    double* out_dynamic_h2d_ms,
    double* out_kernel_ms,
    double* out_result_d2h_ms,
    double* out_wall_ms
) {
    using Clock = std::chrono::steady_clock;

    if (!host_export_power || !host_demand_energy || !host_tank_parameters ||
        !host_current_indices || !host_horizon_steps || !host_dynamic_values ||
        !host_energy_cut_levels || !host_cut_results ||
        !out_static_h2d_ms || !out_dynamic_h2d_ms || !out_kernel_ms ||
        !out_result_d2h_ms || !out_wall_ms ||
        data_length == 0 || tank_count == 0 || step_count == 0)
    {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return static_cast<int>(err);

    cudaStream_t stream = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;

    double *d_export = nullptr;
    double *d_demand = nullptr;
    double *d_tanks = nullptr;
    double *d_dynamic = nullptr;
    double *d_energy_cut = nullptr;
    double *d_cut_results = nullptr;

    uint32_t *d_current = nullptr;
    uint32_t *d_horizon = nullptr;
    uint32_t *d_status = nullptr;
    uint32_t *d_max_segments = nullptr;
    uint32_t *d_max_work_segments = nullptr;

    const size_t annual_bytes = data_length * sizeof(double);
    const size_t tank_bytes = tank_count * 6 * sizeof(double);
    const size_t dynamic_step_bytes = tank_count * 2 * sizeof(double);
    const size_t cut_step_bytes = tank_count * sizeof(double);
    const size_t u32_step_bytes = tank_count * sizeof(uint32_t);

    auto cleanup = [&]() {
        if (d_max_work_segments) cudaFree(d_max_work_segments);
        if (d_max_segments) cudaFree(d_max_segments);
        if (d_status) cudaFree(d_status);
        if (d_cut_results) cudaFree(d_cut_results);
        if (d_energy_cut) cudaFree(d_energy_cut);
        if (d_dynamic) cudaFree(d_dynamic);
        if (d_horizon) cudaFree(d_horizon);
        if (d_current) cudaFree(d_current);
        if (d_tanks) cudaFree(d_tanks);
        if (d_demand) cudaFree(d_demand);
        if (d_export) cudaFree(d_export);
        if (kernel_stop) cudaEventDestroy(kernel_stop);
        if (kernel_start) cudaEventDestroy(kernel_start);
        if (stream) cudaStreamDestroy(stream);
    };

    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaEventCreate(&kernel_start);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaEventCreate(&kernel_stop);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

#define B14I1_MALLOC(ptr, bytes) do { \
    err = cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)); \
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); } \
} while(false)

    B14I1_MALLOC(d_export, annual_bytes);
    B14I1_MALLOC(d_demand, annual_bytes);
    B14I1_MALLOC(d_tanks, tank_bytes);
    B14I1_MALLOC(d_current, sizeof(uint32_t));
    B14I1_MALLOC(d_horizon, sizeof(uint32_t));
    B14I1_MALLOC(d_dynamic, dynamic_step_bytes);
    B14I1_MALLOC(d_energy_cut, cut_step_bytes);
    B14I1_MALLOC(d_cut_results, cut_step_bytes);
    B14I1_MALLOC(d_status, u32_step_bytes);
    B14I1_MALLOC(d_max_segments, u32_step_bytes);
    B14I1_MALLOC(d_max_work_segments, u32_step_bytes);

#undef B14I1_MALLOC

    const auto wall_start = Clock::now();
    const auto static_start = Clock::now();

    err = cudaMemcpyAsync(
        d_export, host_export_power, annual_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaMemcpyAsync(
        d_demand, host_demand_energy, annual_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaMemcpyAsync(
        d_tanks, host_tank_parameters, tank_bytes,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

    *out_static_h2d_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - static_start
        ).count();

    double dynamic_h2d_ms = 0.0;
    double result_d2h_ms = 0.0;
    double kernel_ms_total = 0.0;

    for (size_t step = 0; step < step_count; ++step) {
        const auto h2d_start = Clock::now();

        err = cudaMemcpyAsync(
            d_current,
            host_current_indices + step,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaMemcpyAsync(
            d_horizon,
            host_horizon_steps + step,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaMemcpyAsync(
            d_dynamic,
            host_dynamic_values + step * tank_count * 2,
            dynamic_step_bytes,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaMemcpyAsync(
            d_energy_cut,
            host_energy_cut_levels + step * tank_count,
            cut_step_bytes,
            cudaMemcpyHostToDevice,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        dynamic_h2d_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - h2d_start
            ).count();

        err = cudaEventRecord(kernel_start, stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        sunopt_b14g_warp_binary_kernel<<<
            static_cast<unsigned int>(tank_count),
            SUNOPT_B14G_THREADS,
            0,
            stream
        >>>(
            d_export,
            d_demand,
            data_length,
            d_tanks,
            tank_count,
            d_current,
            d_horizon,
            d_dynamic,
            d_energy_cut,
            1,
            d_cut_results,
            d_status,
            d_max_segments,
            d_max_work_segments
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaEventRecord(kernel_stop, stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        const auto d2h_start = Clock::now();

        err = cudaMemcpyAsync(
            host_cut_results + step * tank_count,
            d_cut_results,
            cut_step_bytes,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        result_d2h_ms +=
            std::chrono::duration<double,std::milli>(
                Clock::now() - d2h_start
            ).count();

        float step_kernel_ms = 0.0f;
        err = cudaEventElapsedTime(
            &step_kernel_ms,
            kernel_start,
            kernel_stop
        );
        if (err != cudaSuccess) { cleanup(); return static_cast<int>(err); }

        kernel_ms_total += static_cast<double>(step_kernel_ms);
    }

    *out_dynamic_h2d_ms = dynamic_h2d_ms;
    *out_kernel_ms = kernel_ms_total;
    *out_result_d2h_ms = result_d2h_ms;
    *out_wall_ms =
        std::chrono::duration<double,std::milli>(
            Clock::now() - wall_start
        ).count();

    cleanup();
    return static_cast<int>(cudaSuccess);
}
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_END
`.trim();

const RS_FFI_BLOCK = String.raw`
    // SUNOPT_BLACKWELL_B14I1_LOCKSTEP_FFI_BEGIN
    fn sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark(
        device_id: c_int,
        host_export_power: *const f64,
        host_demand_energy: *const f64,
        data_length: usize,
        host_tank_parameters: *const f64,
        tank_count: usize,
        host_current_indices: *const u32,
        host_horizon_steps: *const u32,
        host_dynamic_values: *const f64,
        host_energy_cut_levels: *const f64,
        step_count: usize,
        host_cut_results: *mut f64,
        out_static_h2d_ms: *mut f64,
        out_dynamic_h2d_ms: *mut f64,
        out_kernel_ms: *mut f64,
        out_result_d2h_ms: *mut f64,
        out_wall_ms: *mut f64,
    ) -> c_int;
    // SUNOPT_BLACKWELL_B14I1_LOCKSTEP_FFI_END
`.trim();

const RS_RESULT_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_RESULT_BEGIN
#[napi(object)]
pub struct BlackwellB14i1LockstepResult {
    pub snapshot_path: String,
    pub device_id: i32,
    pub annual_step_count: u32,
    pub tank_count: u32,
    pub causal_jobs: u32,
    pub static_h2d_ms: f64,
    pub dynamic_h2d_ms: f64,
    pub kernel_ms: f64,
    pub result_d2h_ms: f64,
    pub wall_ms: f64,
    pub checksum_hex: String,
}
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_RESULT_END
`.trim();

const RS_FN_BLOCK = String.raw`
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_FN_BEGIN
#[napi]
pub fn run_blackwell_b14i1_lockstep(
    snapshot_path: Option<String>,
    device_id: Option<i32>,
) -> Result<BlackwellB14i1LockstepResult> {
    let path =
        snapshot_path
            .map(PathBuf::from)
            .unwrap_or_else(blackwell_default_snapshot_path);

    let snapshot =
        read_blackwell_snapshot(&path)
            .map_err(|error| {
                Error::from_reason(
                    format!(
                        "Unable to read Blackwell snapshot {}: {error}",
                        path.display(),
                    ),
                )
            })?;

    if snapshot.steps.is_empty() || snapshot.tank_count == 0 {
        return Err(
            Error::from_reason(
                "Blackwell snapshot is empty.",
            ),
        );
    }

    let tank_parameters =
        (0..snapshot.tank_count)
            .map(|tank| {
                research_tank_from_packed(
                    &snapshot.packed_config,
                    tank,
                )
            })
            .collect::<Vec<_>>();

    let mut packed_tanks =
        Vec::<f64>::with_capacity(
            snapshot.tank_count * 6,
        );

    for tank in &tank_parameters {
        packed_tanks.extend_from_slice(&[
            tank.target_storage_energy,
            tank.maximum_heat_pump_electrical_power,
            tank.cop * tank.time_step_hours,
            tank.heat_loss_retention,
            tank.heat_loss_intercept,
            if tank.affine_supported { 1.0 } else { 0.0 },
        ]);
    }

    let step_count = snapshot.steps.len();

    let jobs =
        step_count
            .checked_mul(snapshot.tank_count)
            .ok_or_else(|| {
                Error::from_reason(
                    "B14I1 job count overflow.",
                )
            })?;

    let mut current_indices =
        Vec::<u32>::with_capacity(step_count);

    let mut horizons =
        Vec::<u32>::with_capacity(step_count);

    let mut dynamic_values =
        Vec::<f64>::with_capacity(jobs * 2);

    let mut energy_cut_levels =
        Vec::<f64>::with_capacity(jobs);

    for step in &snapshot.steps {
        current_indices.push(step.current_index);
        horizons.push(step.horizon_steps);

        let export_window =
            research_window(
                &snapshot.export_power,
                step.current_index as usize,
                step.horizon_steps as usize,
            );

        for tank_index in 0..snapshot.tank_count {
            let base =
                tank_index *
                BLACKWELL_DYNAMIC_STRIDE;

            let tank =
                tank_parameters[tank_index];

            let current =
                step.dynamic[base + 0];

            let reserve =
                step.dynamic[base + 1];

            let requested =
                step.dynamic[base + 2];

            dynamic_values.push(current);
            dynamic_values.push(reserve);

            energy_cut_levels.push(
                research_energy_cut_level(
                    &export_window,
                    requested,
                    tank,
                ),
            );
        }
    }

    let mut cuts =
        vec![0.0_f64; jobs];

    let mut static_h2d_ms = 0.0_f64;
    let mut dynamic_h2d_ms = 0.0_f64;
    let mut kernel_ms = 0.0_f64;
    let mut result_d2h_ms = 0.0_f64;
    let mut wall_ms = 0.0_f64;

    let selected_device =
        device_id.unwrap_or(0);

    let code = unsafe {
        sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark(
            selected_device as c_int,
            snapshot.export_power.as_ptr(),
            snapshot.demand_energy.as_ptr(),
            snapshot.data_length,
            packed_tanks.as_ptr(),
            snapshot.tank_count,
            current_indices.as_ptr(),
            horizons.as_ptr(),
            dynamic_values.as_ptr(),
            energy_cut_levels.as_ptr(),
            step_count,
            cuts.as_mut_ptr(),
            &mut static_h2d_ms,
            &mut dynamic_h2d_ms,
            &mut kernel_ms,
            &mut result_d2h_ms,
            &mut wall_ms,
        )
    };

    if code != 0 {
        return Err(
            cuda_error(
                "sunopt_cuda_run_b14i1_warp_binary_lockstep_benchmark",
                code,
            ),
        );
    }

    let mut checksum =
        1469598103934665603_u64;

    for value in &cuts {
        blackwell_fnv1a_update(
            &mut checksum,
            *value,
        );
    }

    Ok(
        BlackwellB14i1LockstepResult {
            snapshot_path:
                path.display().to_string(),
            device_id:
                selected_device,
            annual_step_count:
                step_count as u32,
            tank_count:
                snapshot.tank_count as u32,
            causal_jobs:
                jobs as u32,
            static_h2d_ms,
            dynamic_h2d_ms,
            kernel_ms,
            result_d2h_ms,
            wall_ms,
            checksum_hex:
                format!("{checksum:016x}"),
        },
    )
}
// SUNOPT_BLACKWELL_B14I1_LOCKSTEP_FN_END
`.trim();

const RUNNER = String.raw`// rust/sunopt-node/scripts/blackwell-b14i1-lockstep.mjs
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");

const outbox = path.join(projectRoot, "updates", "outbox");
fs.mkdirSync(outbox, { recursive: true });

const addon = require(
  path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"),
);

const run = addon.runBlackwellB14i1Lockstep;

if (typeof run !== "function") {
  throw new Error(
    "runBlackwellB14i1Lockstep native export missing",
  );
}

const snapshot =
  process.argv[2]
    ? path.resolve(process.argv[2])
    : path.join(
        nativeRoot,
        "blackwell-b12-sync-blackwell.bin",
      );

console.log(
  "--- B14I1 B14G PRODUCTION-SHAPED LOCKSTEP ---",
);
console.log(\`snapshot=\${snapshot}\`);

const r = run(snapshot, 0);

const expectedChecksum = "bf0e8c9e7b5c879e";

const checksumMatchesB14h =
  r.checksumHex === expectedChecksum;

const report = {
  ...r,
  expectedB14hChecksumHex: expectedChecksum,
  checksumMatchesB14h,
};

const reportPath =
  path.join(
    outbox,
    "blackwell-b14i1-lockstep.json",
  );

fs.writeFileSync(
  reportPath,
  JSON.stringify(report, null, 2) + "\n",
);

console.log(
  \`steps=\${r.annualStepCount} tanks=\${r.tankCount} jobs=\${r.causalJobs}\`,
);
console.log(
  \`staticH2D=\${r.staticH2DMs.toFixed(3)} ms\`,
);
console.log(
  \`dynamicH2D=\${r.dynamicH2DMs.toFixed(3)} ms\`,
);
console.log(
  \`kernel=\${r.kernelMs.toFixed(3)} ms\`,
);
console.log(
  \`resultD2H=\${r.resultD2HMs.toFixed(3)} ms\`,
);
console.log(
  \`wall=\${r.wallMs.toFixed(3)} ms\`,
);
console.log(\`checksum=\${r.checksumHex}\`);
console.log(
  \`checksumMatchesB14H=\${checksumMatchesB14h}\`,
);
console.log(\`report=\${reportPath}\`);
console.log(
  \`B14I1_RESULT=\${
    checksumMatchesB14h
      ? "EXACT_PRODUCTION_SHAPED_LOCKSTEP"
      : "CHECKSUM_MISMATCH"
  }\`,
);
console.log(
  \`SUNOPT_B14I1_JSON=\${JSON.stringify(report)}\`,
);
`;

let h = readText(hPath);
let cu = readText(cuPath);
let rs = readText(rsPath);

h = insertBefore(
  h,
  "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN",
  H_BLOCK,
  "cuda_context.h B14I1",
);

cu = insertBefore(
  cu,
  "// SUNOPT_BLACKWELL_MAX_AFFINE_TREE_BEGIN",
  CU_BLOCK,
  "cuda_context.cu B14I1",
);

rs = insertBefore(
  rs,
  "// SUNOPT_BLACKWELL_DIGITAL_HP_BREAKPOINTS_FFI_BEGIN",
  RS_FFI_BLOCK,
  "cuda_context.rs B14I1 FFI",
);

rs = insertBefore(
  rs,
  "// SUNOPT_BLACKWELL_B14G_WARP_BINARY_FN_BEGIN",
  RS_RESULT_BLOCK + "\n\n" + RS_FN_BLOCK,
  "cuda_context.rs B14I1 function",
);

const files = new Map([
  [hRel, Buffer.from(h, "utf8")],
  [cuRel, Buffer.from(cu, "utf8")],
  [rsRel, Buffer.from(rs, "utf8")],
  [runnerRel, Buffer.from(RUNNER, "utf8")],
]);

const manifest = {
  format: 2,
  parent_sha256: parent,
  files: Object.fromEntries(
    [...files.entries()].map(
      ([name, data]) => [name, sha256Buffer(data)],
    ),
  ),
  delete: [],
};

const manifestBytes = Buffer.from(
  JSON.stringify(manifest, null, 2) + "\n",
  "utf8",
);

const release = sha256Buffer(manifestBytes);

//
// Tiny ZIP writer using STORE mode only.
// This keeps the generator dependency-free.
//
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; k++) {
      c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
    }
  }
  return (c ^ 0xffffffff) >>> 0;
}

function u16(n) {
  const b = Buffer.alloc(2);
  b.writeUInt16LE(n >>> 0);
  return b;
}

function u32(n) {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n >>> 0);
  return b;
}

const zipEntries = [
  ["manifest.json", manifestBytes],
  ["manifest.sha256", Buffer.from(release + "\n", "ascii")],
  ...files.entries(),
];

const localParts = [];
const centralParts = [];
let offset = 0;

for (const [name, data] of zipEntries) {
  const nameBuf = Buffer.from(name, "utf8");
  const crc = crc32(data);

  const local = Buffer.concat([
    u32(0x04034b50),
    u16(20),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(crc),
    u32(data.length),
    u32(data.length),
    u16(nameBuf.length),
    u16(0),
    nameBuf,
    data,
  ]);

  localParts.push(local);

  const central = Buffer.concat([
    u32(0x02014b50),
    u16(20),
    u16(20),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(crc),
    u32(data.length),
    u32(data.length),
    u16(nameBuf.length),
    u16(0),
    u16(0),
    u16(0),
    u16(0),
    u32(0),
    u32(offset),
    nameBuf,
  ]);

  centralParts.push(central);
  offset += local.length;
}

const centralOffset = offset;
const central = Buffer.concat(centralParts);

const end = Buffer.concat([
  u32(0x06054b50),
  u16(0),
  u16(0),
  u16(zipEntries.length),
  u16(zipEntries.length),
  u32(central.length),
  u32(centralOffset),
  u16(0),
]);

const zip = Buffer.concat([
  ...localParts,
  central,
  end,
]);

fs.mkdirSync(inbox, { recursive: true });

const packagePath =
  path.join(
    inbox,
    "20261112-blackwell-b14i1-production-shaped-lockstep.zip",
  );

fs.writeFileSync(packagePath, zip);

console.log(`parent=${parent}`);
console.log(`release=${release}`);
console.log(`package=${packagePath}`);
console.log("B14I1_PACKAGE_RESULT=OK");
