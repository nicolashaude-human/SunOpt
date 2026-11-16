import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");
const sampleSteps = Number.parseInt(process.argv[3] ?? "192", 10);
const repetitions = Number.parseInt(process.argv[4] ?? "20", 10);
const deviceId = Number.parseInt(process.argv[5] ?? "0", 10);

const result = addon.runBlackwellPiecewiseAffineB3(
  snapshot,
  sampleSteps,
  repetitions,
  deviceId,
);

console.log(`SUNOPT_PIECEWISE_AFFINE_B3_JSON=${JSON.stringify(result)}`);
console.log();
console.log("--- BLACKWELL B3 PIECEWISE-AFFINE CUT SOLVER ---");
console.log(`snapshot=${result.snapshotPath}`);
console.log(`sampleSteps=${result.sampleStepCount}`);
console.log(`sampleJobs=${result.sampleJobCount}`);
console.log(`supportedJobs=${result.supportedJobs}/${result.sampleJobCount}`);
console.log(`correctionSearchJobs=${result.correctionSearchJobs}`);
console.log(`referenceFullCausalGpuMs=${result.referenceFullCausalGpuMs.toFixed(4)}`);
console.log();
console.log([
  "iters","kernelMs","annualProjectedMs","avgEvals","maxCutErr",
  ">1e-6",">1e-5",">1e-4","statusMis","maxBracket"
].join("\t"));

for (let i = 0; i < result.iterationLimits.length; i += 1) {
  console.log([
    result.iterationLimits[i],
    result.kernelMs[i].toFixed(6),
    result.annualProjectedKernelMs[i].toFixed(3),
    result.averageEvaluationsSupported[i].toFixed(3),
    result.maxAbsCutError[i].toExponential(6),
    result.cutErrorOver1e6[i],
    result.cutErrorOver1e5[i],
    result.cutErrorOver1e4[i],
    result.statusMismatches[i],
    result.maximumBracketWidth[i].toExponential(6),
  ].join("\t"));
}

const report = path.join(root, "blackwell-piecewise-affine-b3.json");
writeFileSync(report, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log();
console.log(`report=${report}`);
