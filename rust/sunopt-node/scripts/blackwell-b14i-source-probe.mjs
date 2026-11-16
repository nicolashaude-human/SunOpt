// rust/sunopt-node/scripts/blackwell-b14i-source-probe.mjs
//
// B14I-0 source probe.
// Reads the CURRENT installed SunOpt sources and prints only the regions
// required to build the production B14G annual integration safely.
//
// Usage from project root:
//   node .\rust\sunopt-node\scripts\blackwell-b14i-source-probe.mjs
//
// No files are modified.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..", "..");

const files = {
  controller: path.join(root, "rust", "sunopt-node", "src", "acceleration_controller.rs"),
  cudaRs: path.join(root, "rust", "sunopt-node", "src", "cuda_context.rs"),
  cudaCu: path.join(root, "rust", "sunopt-node", "cuda", "cuda_context.cu"),
  cudaH: path.join(root, "rust", "sunopt-node", "cuda", "cuda_context.h"),
  libRs: path.join(root, "rust", "sunopt-node", "src", "lib.rs"),
  b14g: path.join(root, "rust", "sunopt-node", "scripts", "blackwell-b14g-warp-binary.mjs"),
};

function read(file) {
  if (!fs.existsSync(file)) {
    return null;
  }
  return fs.readFileSync(file, "utf8");
}

function numbered(text) {
  return text.split(/\r?\n/).map((line, i) => `${String(i + 1).padStart(6)} | ${line}`);
}

function emitMatches(label, file, patterns, before = 35, after = 90) {
  const text = read(file);
  console.log(`\n===== ${label} =====`);
  console.log(`FILE=${file}`);

  if (text == null) {
    console.log("FILE_MISSING");
    console.log(`===== END ${label} =====`);
    return;
  }

  const lines = text.split(/\r?\n/);
  const matched = new Set();

  for (const p of patterns) {
    const re = typeof p === "string" ? new RegExp(p, "i") : p;
    for (let i = 0; i < lines.length; i++) {
      if (re.test(lines[i])) matched.add(i);
    }
  }

  if (matched.size === 0) {
    console.log("NO_MATCH");
    console.log(`===== END ${label} =====`);
    return;
  }

  const ranges = [...matched]
    .sort((a, b) => a - b)
    .map(i => [Math.max(0, i - before), Math.min(lines.length - 1, i + after)]);

  const merged = [];
  for (const r of ranges) {
    const last = merged.at(-1);
    if (!last || r[0] > last[1] + 8) merged.push([...r]);
    else last[1] = Math.max(last[1], r[1]);
  }

  for (const [start, end] of merged) {
    console.log(`--- lines ${start + 1}-${end + 1} ---`);
    for (let i = start; i <= end; i++) {
      console.log(`${String(i + 1).padStart(6)} | ${lines[i]}`);
    }
  }

  console.log(`===== END ${label} =====`);
}

console.log("--- B14I-0 CURRENT SOURCE PROBE ---");
console.log(`projectRoot=${root}`);

emitMatches(
  "ACCELERATION CONTROLLER",
  files.controller,
  [
    /run_causal/i,
    /causal/i,
    /gpu_workers/i,
    /annual/i,
    /simulate/i,
    /peak/i,
  ],
  45,
  120,
);

emitMatches(
  "CUDA RUST B14G + CAUSAL",
  files.cudaRs,
  [
    /B14G/i,
    /WarpBinary/i,
    /runBlackwellB14G/i,
    /run_blackwell_b14g/i,
    /run_causal_dynamic_batch_into/i,
    /run_causal_batch/i,
    /sunopt_cuda_run_causal/i,
  ],
  40,
  120,
);

emitMatches(
  "CUDA HEADER B14G",
  files.cudaH,
  [
    /B14G/i,
    /warp.*binary/i,
    /b14.*warp/i,
  ],
  35,
  100,
);

emitMatches(
  "CUDA IMPLEMENTATION B14G",
  files.cudaCu,
  [
    /B14G/i,
    /warp.*binary/i,
    /b14.*warp/i,
  ],
  45,
  140,
);

emitMatches(
  "LIB EXPORTS / CONTROLLER USE",
  files.libRs,
  [
    /AccelerationController/i,
    /cuda_context/i,
    /causal/i,
    /simulate/i,
  ],
  35,
  100,
);

emitMatches(
  "B14G JS CALL SIGNATURE",
  files.b14g,
  [
    /runBlackwellB14G/i,
    /B14G/i,
    /candidateB14g/i,
  ],
  45,
  130,
);

console.log("\nB14I_SOURCE_PROBE_RESULT=OK");
