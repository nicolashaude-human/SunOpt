// rust/sunopt-node/scripts/make-blackwell-b14n-precision-diagnostic.mjs
//
// B14N: diagnostics-only extension to B14M.
// - fixes strict-reference header parsing
// - reports first FP32 status mismatch and worst cut-error job with inputs
// - adds a strict diagnostic runner
//
// Run from SunOpt root:
//   node .\rust\sunopt-node\scripts\make-blackwell-b14n-precision-diagnostic.mjs
//
// The normal always-running watcher installs the generated inbox package.
// Then:
//   cd .\rust\sunopt-node
//   npm run build
//   cd ..\..
//   node .\rust\sunopt-node\scripts\blackwell-b14n-precision-diagnostic.mjs
//
// Output:
//   updates/outbox/blackwell-b14n-precision-diagnostic.json
//   updates/outbox/llm.txt

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");
const inbox = path.join(root, "updates", "inbox");
const hashFile = path.join(root, "updates", "history", "hash.txt");

const RS_REL = "rust/sunopt-node/src/cuda_context.rs";
const RUNNER_REL = "rust/sunopt-node/scripts/blackwell-b14n-precision-diagnostic.mjs";

const rsPath = path.join(root, RS_REL);

function sha256(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function replaceExactly(text, oldText, newText, label) {
  const count = text.split(oldText).length - 1;
  if (count !== 1) {
    throw new Error(`${label}: expected exactly 1 occurrence, found ${count}`);
  }
  return text.replace(oldText, newText);
}

if (!fs.existsSync(hashFile)) {
  throw new Error(`Current watcher hash not found: ${hashFile}`);
}
const parent = fs.readFileSync(hashFile, "utf8").trim().toLowerCase();
if (!/^[0-9a-f]{64}$/.test(parent)) {
  throw new Error(`Invalid current watcher hash: ${parent}`);
}

let rs = fs.readFileSync(rsPath, "utf8");

// Fix B14M strict-reference header: "B14MREF1" is 8 bytes + 8-byte count = 16.
rs = replaceExactly(
  rs,
  "            const HEADER: usize = 15;",
  "            const HEADER: usize = 16;",
  "reference HEADER"
);

rs = replaceExactly(
  rs,
  '                &bytes[0..7] == b"B14MREF1"',
  '                &bytes[0..8] == b"B14MREF1"',
  "reference magic"
);

rs = replaceExactly(
  rs,
  "                    &bytes[7..15]",
  "                    &bytes[8..16]",
  "reference count range"
);

// One diagnostics string is enough to avoid expanding the N-API surface heavily.
rs = replaceExactly(
  rs,
  `    pub fp32_cut_mismatches_1e4_vs_fp64: u32,

    pub strict_reference_loaded: bool,`,
  `    pub fp32_cut_mismatches_1e4_vs_fp64: u32,
    pub fp32_diagnostics_json: String,

    pub strict_reference_loaded: bool,`,
  "B14M result struct diagnostics field"
);

rs = replaceExactly(
  rs,
  `    let mut mismatch_1e4 =
        0_u32;

    for job in 0..jobs {`,
  `    let mut mismatch_1e4 =
        0_u32;

    let mut first_status_mismatch_job:
        Option<usize> = None;

    let mut worst_cut_error_job:
        Option<usize> = None;

    for job in 0..jobs {`,
  "B14M diagnostic tracking vars"
);

rs = replaceExactly(
  rs,
  `        if status32[job] != status64[job] {
            status_mismatch_current += 1;
        }

        let error =
            (fp32_value - cuts64[job]).abs();

        max_error_current =
            max_error_current.max(error);

        sum_error_current += error;`,
  `        if status32[job] != status64[job] {
            status_mismatch_current += 1;

            if first_status_mismatch_job.is_none() {
                first_status_mismatch_job =
                    Some(job);
            }
        }

        let error =
            (fp32_value - cuts64[job]).abs();

        if error > max_error_current {
            max_error_current = error;
            worst_cut_error_job = Some(job);
        }

        sum_error_current += error;`,
  "B14M diagnostic tracking loop"
);

const diagBlock = String.raw`
    let describe_job =
        |job: usize| -> String {
            let step =
                job / snapshot.tank_count;

            let tank =
                job - step * snapshot.tank_count;

            let dynamic_base =
                job * 2;

            let target =
                tank_parameters[tank]
                    .target_storage_energy;

            let retention =
                tank_parameters[tank]
                    .heat_loss_retention;

            let loss_intercept =
                tank_parameters[tank]
                    .heat_loss_intercept;

            format!(
                "{{\"job\":{},\"step\":{},\"tank\":{},\"currentIndex\":{},\"horizon\":{},\"statusFp64\":{},\"statusFp32\":{},\"cutFp64\":{:.17},\"cutFp32\":{:.17},\"absCutError\":{:.17},\"current\":{:.17},\"reserve\":{:.17},\"requestedCut\":{:.17},\"target\":{:.17},\"retention\":{:.17},\"lossIntercept\":{:.17}}}",
                job,
                step,
                tank,
                current_indices[step],
                horizons[step],
                status64[job],
                status32[job],
                cuts64[job],
                cuts32[job] as f64,
                ((cuts32[job] as f64) - cuts64[job]).abs(),
                dynamic64[dynamic_base + 0],
                dynamic64[dynamic_base + 1],
                energy_cut64[job],
                target,
                retention,
                loss_intercept,
            )
        };

    let first_status_json =
        first_status_mismatch_job
            .map(&describe_job)
            .unwrap_or_else(
                || "null".to_string()
            );

    let worst_cut_json =
        worst_cut_error_job
            .map(&describe_job)
            .unwrap_or_else(
                || "null".to_string()
            );

    let fp32_diagnostics_json =
        format!(
            "{{\"firstStatusMismatch\":{},\"worstCutError\":{}}}",
            first_status_json,
            worst_cut_json,
        );

`;

rs = replaceExactly(
  rs,
  `    let reference_path =
        strict_reference_path`,
  diagBlock + `    let reference_path =
        strict_reference_path`,
  "B14M diagnostics block insertion"
);

rs = replaceExactly(
  rs,
  `            fp32_cut_mismatches_1e4_vs_fp64:
                mismatch_1e4,

            strict_reference_loaded:`,
  `            fp32_cut_mismatches_1e4_vs_fp64:
                mismatch_1e4,
            fp32_diagnostics_json,

            strict_reference_loaded:`,
  "B14M diagnostics result assignment"
);

const RUNNER = [
  '// rust/sunopt-node/scripts/blackwell-b14n-precision-diagnostic.mjs',
  'import { createRequire } from "node:module";',
  'import fs from "node:fs";',
  'import path from "node:path";',
  'import { fileURLToPath } from "node:url";',
  '',
  'const require = createRequire(import.meta.url);',
  'const here = path.dirname(fileURLToPath(import.meta.url));',
  'const nativeRoot = path.resolve(here, "..");',
  'const projectRoot = path.resolve(nativeRoot, "..", "..");',
  'const outbox = path.join(projectRoot, "updates", "outbox");',
  'const historyHash = path.join(projectRoot, "updates", "history", "hash.txt");',
  'fs.mkdirSync(outbox, { recursive: true });',
  '',
  'const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));',
  'const run = addon.runBlackwellB14MPrecision;',
  'if (typeof run !== "function") throw new Error("runBlackwellB14MPrecision native export missing");',
  '',
  'const snapshot = process.argv[2]',
  '  ? path.resolve(process.argv[2])',
  '  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");',
  'const reference = path.join(outbox, "blackwell-b14m-strict-reference.bin");',
  'const release = fs.existsSync(historyHash) ? fs.readFileSync(historyHash, "utf8").trim() : "unknown";',
  '',
  'const result = run(snapshot, "strict-diagnostic", reference, true, 0);',
  'let diagnostics = null;',
  'try { diagnostics = JSON.parse(result.fp32DiagnosticsJson); } catch { diagnostics = result.fp32DiagnosticsJson; }',
  '',
  'const report = {',
  '  stage: "B14N_PRECISION_DIAGNOSTIC",',
  '  release,',
  '  snapshotPath: snapshot,',
  '  ...result,',
  '  diagnostics,',
  '};',
  '',
  'const reportPath = path.join(outbox, "blackwell-b14n-precision-diagnostic.json");',
  'fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\\n");',
  '',
  'const llmPath = path.join(outbox, "llm.txt");',
  'fs.appendFileSync(llmPath, [',
  '  "",',
  '  `=== ${new Date().toISOString()} B14N_PRECISION_DIAGNOSTIC ===`,',
  '  `release=${release}`,',
  '  `fp64 kernelMs=${result.fp64KernelMs.toFixed(6)} wallMs=${result.fp64WallMs.toFixed(6)} checksum=${result.fp64ChecksumHex}`,',
  '  `fp32 kernelMs=${result.fp32KernelMs.toFixed(6)} wallMs=${result.fp32WallMs.toFixed(6)} checksum=${result.fp32ChecksumHex}`,',
  '  `statusMismatch=${result.fp32StatusMismatchesVsFp64} maxAbsCutError=${result.fp32MaxAbsCutErrorVsFp64}`,',
  '  `strictReferenceLoaded=${result.strictReferenceLoaded}`,',
  '  `diagnostics=${result.fp32DiagnosticsJson}`,',
  '  `report=${reportPath}`,',
  '  "",',
  '].join("\\n"), "utf8");',
  '',
  'console.log(JSON.stringify(report, null, 2));',
  'console.log(`report=${reportPath}`);',
  'console.log(`llm=${llmPath}`);',
  ''
].join("\n");

const files = new Map([
  [RS_REL, Buffer.from(rs, "utf8")],
  [RUNNER_REL, Buffer.from(RUNNER, "utf8")],
]);

const manifest = {
  format: 2,
  parent_sha256: parent,
  files: Object.fromEntries(
    [...files.entries()].map(([name, data]) => [name, sha256(data)]),
  ),
  delete: [],
};

const manifestBytes = Buffer.from(
  JSON.stringify(manifest, null, 2) + "\n",
  "utf8",
);

const release = sha256(manifestBytes);

function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; ++k) {
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

const entries = [
  ["manifest.json", manifestBytes],
  ["manifest.sha256", Buffer.from(release + "\n", "ascii")],
  ...files.entries(),
];

const locals = [];
const centrals = [];
let offset = 0;

for (const [name, data] of entries) {
  const nameBuf = Buffer.from(name, "utf8");
  const crc = crc32(data);

  const local = Buffer.concat([
    u32(0x04034b50),
    u16(20), u16(0), u16(0), u16(0), u16(0),
    u32(crc),
    u32(data.length),
    u32(data.length),
    u16(nameBuf.length),
    u16(0),
    nameBuf,
    data,
  ]);

  locals.push(local);

  const central = Buffer.concat([
    u32(0x02014b50),
    u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
    u32(crc),
    u32(data.length),
    u32(data.length),
    u16(nameBuf.length),
    u16(0), u16(0), u16(0), u16(0),
    u32(0),
    u32(offset),
    nameBuf,
  ]);

  centrals.push(central);
  offset += local.length;
}

const centralOffset = offset;
const central = Buffer.concat(centrals);

const end = Buffer.concat([
  u32(0x06054b50),
  u16(0), u16(0),
  u16(entries.length),
  u16(entries.length),
  u32(central.length),
  u32(centralOffset),
  u16(0),
]);

const zip = Buffer.concat([
  ...locals,
  central,
  end,
]);

fs.mkdirSync(inbox, { recursive: true });

const packagePath = path.join(
  inbox,
  "20261112-blackwell-b14n-precision-diagnostic.zip",
);

fs.writeFileSync(packagePath, zip);

console.log(`parent=${parent}`);
console.log(`release=${release}`);
console.log(`package=${packagePath}`);
console.log("B14N_PACKAGE_RESULT=OK");
