// rust/sunopt-node/scripts/blackwell-b13-continuous-piecewise-affine-sync.mjs
//
// B13 research-only synchronized Continuous Piecewise-Affine experiment.
//
// Reuses the already implemented B3 mathematical solver:
//   - continuous/modulating HP:
//       P_hp = min(Pmax, max(0, exportPower - cut))
//   - exact max-affine storage-survival compression where supported
//   - safeguarded piecewise-linear correction
//
// The important change is the DATASET: this runs against the synchronized
// B12 Blackwell capture, not the historical blackwell-benchmark.bin.
//
// Production code and production lockstep are untouched.

import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

if (typeof addon.runBlackwellPiecewiseAffineB3 !== "function") {
  throw new Error(
    "Native addon does not export runBlackwellPiecewiseAffineB3. " +
    "Rebuild the current sunopt-node addon before running B13.",
  );
}

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-b12-sync-blackwell.bin");

const sampleSteps = Number.parseInt(process.argv[3] ?? "192", 10);
const repetitions = Number.parseInt(process.argv[4] ?? "1", 10);
const deviceId = Number.parseInt(process.argv[5] ?? "0", 10);

if (!Number.isInteger(sampleSteps) || sampleSteps < 2) {
  throw new Error("sampleSteps must be an integer >= 2.");
}
if (!Number.isInteger(repetitions) || repetitions < 1) {
  throw new Error("repetitions must be an integer >= 1.");
}
if (!Number.isInteger(deviceId) || deviceId < 0) {
  throw new Error("deviceId must be an integer >= 0.");
}

console.log("--- B13 CONTINUOUS PIECEWISE-AFFINE / SYNCHRONIZED ---");
console.log(`snapshot=${snapshot}`);
console.log(`sampleSteps=${sampleSteps}`);
console.log(`repetitions=${repetitions}`);
console.log(`deviceId=${deviceId}`);
console.log();

const result = addon.runBlackwellPiecewiseAffineB3(
  snapshot,
  sampleSteps,
  repetitions,
  deviceId,
);

const rows = result.iterationLimits.map((iterations, i) => ({
  iterations,
  kernelMs: result.kernelMs[i],
  annualProjectedKernelMs: result.annualProjectedKernelMs[i],
  averageEvaluationsSupported: result.averageEvaluationsSupported[i],
  maxAbsCutError: result.maxAbsCutError[i],
  cutErrorOver1e6: result.cutErrorOver1e6[i],
  cutErrorOver1e5: result.cutErrorOver1e5[i],
  cutErrorOver1e4: result.cutErrorOver1e4[i],
  statusMismatches: result.statusMismatches[i],
  maximumBracketWidth: result.maximumBracketWidth[i],
}));

const exactCandidates = rows.filter(
  (row) =>
    row.statusMismatches === 0 &&
    row.cutErrorOver1e6 === 0,
);

const bestExact = exactCandidates.length > 0
  ? exactCandidates.reduce(
      (best, row) =>
        row.annualProjectedKernelMs < best.annualProjectedKernelMs
          ? row
          : best,
    )
  : null;

const report = {
  experiment: "B13 continuous piecewise-affine synchronized",
  semantics:
    "continuous/modulating HP + max-affine storage compression + safeguarded correction",
  productionModified: false,
  snapshot,
  sampleSteps: result.sampleStepCount,
  sampleJobs: result.sampleJobCount,
  supportedJobs: result.supportedJobs,
  unsupportedJobs: result.unsupportedJobs,
  correctionSearchJobs: result.correctionSearchJobs,
  referenceFullCausalGpuMs: result.referenceFullCausalGpuMs,
  rows,
  bestExact,
};

console.log(`SUNOPT_B13_CONTINUOUS_PWA_JSON=${JSON.stringify(report)}`);
console.log();
console.log(`annualSteps=${result.annualStepCount}`);
console.log(`sampleSteps=${result.sampleStepCount}`);
console.log(`sampleJobs=${result.sampleJobCount}`);
console.log(`supportedJobs=${result.supportedJobs}`);
console.log(`unsupportedJobs=${result.unsupportedJobs}`);
console.log(`correctionSearchJobs=${result.correctionSearchJobs}`);
console.log(
  `referenceFullCausalGpuMs=${result.referenceFullCausalGpuMs.toFixed(6)}`,
);
console.log();

console.log(
  [
    "iters",
    "kernelMs",
    "annualProjectedMs",
    "avgEvals",
    "maxCutErr",
    ">1e-6",
    ">1e-5",
    ">1e-4",
    "statusMis",
    "maxBracket",
  ].join("\t"),
);

for (const row of rows) {
  console.log(
    [
      row.iterations,
      row.kernelMs.toFixed(6),
      row.annualProjectedKernelMs.toFixed(3),
      row.averageEvaluationsSupported.toFixed(3),
      row.maxAbsCutError.toExponential(6),
      row.cutErrorOver1e6,
      row.cutErrorOver1e5,
      row.cutErrorOver1e4,
      row.statusMismatches,
      row.maximumBracketWidth.toExponential(6),
    ].join("\t"),
  );
}

console.log();
if (bestExact === null) {
  console.log("B13_RESULT=no <=1e-6 exact candidate in this run");
} else {
  console.log(
    `B13_RESULT=exact candidate: iterations=${bestExact.iterations} ` +
    `annualProjectedMs=${bestExact.annualProjectedKernelMs.toFixed(3)} ` +
    `avgEvals=${bestExact.averageEvaluationsSupported.toFixed(3)}`,
  );
}

const reportPath = path.join(
  root,
  "blackwell-b13-continuous-piecewise-affine-sync.json",
);
writeFileSync(
  reportPath,
  JSON.stringify(report, null, 2) + "\n",
  "utf8",
);
console.log(`report=${reportPath}`);
