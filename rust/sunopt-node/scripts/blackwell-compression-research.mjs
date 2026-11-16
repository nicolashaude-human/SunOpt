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
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

if (!Number.isFinite(sampleSteps) || sampleSteps < 2) {
  throw new Error("sampleSteps must be >= 2");
}

const result = addon.runBlackwellPhysicsCompressionResearch(
  snapshot,
  sampleSteps,
  deviceId,
);

console.log(`SUNOPT_PHYSICS_COMPRESSION_JSON=${JSON.stringify(result)}`);
console.log();
console.log("--- PHYSICS COMPRESSION SUMMARY ---");
console.log(`snapshot=${result.snapshotPath}`);
console.log(`sampleSteps=${result.sampleStepCount}`);
console.log(`sampleJobs=${result.sampleJobCount}`);
console.log(`gpuReferenceMs=${result.gpuReferenceMs.toFixed(3)}`);
console.log(`scalarReferenceMs=${result.scalarReferenceMs.toFixed(3)}`);
console.log(`scalarGpuMaxAbsCutError=${result.scalarGpuMaxAbsCutError.toExponential(6)}`);
console.log(`scalarGpuCutErrorOver1e6=${result.scalarGpuCutErrorOver1e6}`);
console.log();
console.log("MAX-AFFINE EXACT-COMPRESSION CANDIDATE");
console.log(`maxAffineMs=${result.maxAffineMs.toFixed(3)}`);
console.log(`treeLevels=${result.maxAffineTreeLevels}`);
console.log(`supportedJobs=${result.maxAffineSupportedJobs}/${result.sampleJobCount}`);
console.log(`maxAbsCutError=${result.maxAffineScalarMaxAbsCutError.toExponential(6)}`);
console.log(`cutErrorOver1e6=${result.maxAffineScalarCutErrorOver1e6}`);
console.log(`normalSurvivalMismatches=${result.maxAffineNormalSurvivalMismatches}`);
console.log(`maxPvSurvivalMismatches=${result.maxAffineMaxpvSurvivalMismatches}`);
console.log();
console.log("FOURIER COMPRESSION");
console.log(
  [
    "harm",
    "coeff",
    "fraction",
    "ms",
    "exportRMSE",
    "demandRMSE",
    "cutMAE",
    "cutRMSE",
    "cutMax",
    ">0.01kW",
    ">0.1kW",
    "normalMis",
    "maxPvMis",
  ].join("\t"),
);

for (let i = 0; i < result.fourierHarmonics.length; i += 1) {
  console.log(
    [
      result.fourierHarmonics[i],
      result.fourierCoefficients[i],
      (100 * result.fourierSignalFraction[i]).toFixed(1) + "%",
      result.fourierMs[i].toFixed(1),
      result.fourierExportRmse[i].toFixed(6),
      result.fourierDemandRmse[i].toFixed(6),
      result.fourierCutMae[i].toFixed(6),
      result.fourierCutRmse[i].toFixed(6),
      result.fourierCutMaxAbsError[i].toFixed(6),
      result.fourierCutErrorOver001[i],
      result.fourierCutErrorOver01[i],
      result.fourierNormalSurvivalMismatches[i],
      result.fourierMaxpvSurvivalMismatches[i],
    ].join("\t"),
  );
}

const outputPath = path.join(root, "blackwell-compression-research.json");
writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log();
console.log(`report=${outputPath}`);