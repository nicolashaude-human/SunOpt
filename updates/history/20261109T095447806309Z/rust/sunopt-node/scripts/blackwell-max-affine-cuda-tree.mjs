import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const addon = require(
  path.join(
    root,
    "sunopt_node.win32-x64-msvc.node",
  ),
);

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(
      root,
      "blackwell-benchmark.bin",
    );

const sampleSteps =
  Number.parseInt(
    process.argv[3] ?? "192",
    10,
  );

const repetitions =
  Number.parseInt(
    process.argv[4] ?? "20",
    10,
  );

const deviceId =
  Number.parseInt(
    process.argv[5] ?? "0",
    10,
  );

if (
  !Number.isFinite(sampleSteps) ||
  sampleSteps < 2
) {
  throw new Error(
    "sampleSteps must be >= 2",
  );
}

if (
  !Number.isFinite(repetitions) ||
  repetitions < 1
) {
  throw new Error(
    "repetitions must be >= 1",
  );
}

const result =
  addon.runBlackwellMaxAffineCudaTree(
    snapshot,
    sampleSteps,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_MAX_AFFINE_CUDA_TREE_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- BLACKWELL MAX-AFFINE CUDA TREE ---",
);

console.log(
  `snapshot=${result.snapshotPath}`,
);

console.log(
  `device=${result.deviceId}`,
);

console.log(
  `sampleSteps=${result.sampleStepCount}`,
);

console.log(
  `sampleJobs=${result.sampleJobCount}`,
);

console.log(
  `survivalChecks=${result.survivalCheckCount}`,
);

console.log(
  `supportedJobs=${result.supportedJobs}/${result.sampleJobCount}`,
);

console.log(
  `unsupportedJobs=${result.unsupportedJobs}`,
);

console.log();
console.log(
  "REFERENCE (existing complete causal kernel)",
);

console.log(
  `referenceFullCausalGpuMs=${result.referenceFullCausalGpuMs.toFixed(4)}`,
);

console.log();
console.log(
  "CANDIDATE (exact max-affine survival tree)",
);

console.log(
  `repetitions=${result.repetitions}`,
);

console.log(
  `h2dMs=${result.candidateH2dMs.toFixed(4)}`,
);

console.log(
  `kernelTotalMs=${result.candidateKernelTotalMs.toFixed(4)}`,
);

console.log(
  `kernelMsPerRun=${result.candidateKernelMs.toFixed(6)}`,
);

console.log(
  `d2hMs=${result.candidateD2hMs.toFixed(4)}`,
);

console.log(
  `wallMs=${result.candidateWallMs.toFixed(4)}`,
);

console.log(
  `nsPerSupportedSurvivalCheck=${result.candidateNsPerSurvivalCheck.toFixed(2)}`,
);

console.log();
console.log(
  "PARITY",
);

console.log(
  `normalSurvivalMismatches=${result.normalSurvivalMismatches}`,
);

console.log(
  `maxPvSurvivalMismatches=${result.maxpvSurvivalMismatches}`,
);

console.log(
  `unexpectedGpuUnsupportedChecks=${result.unexpectedGpuUnsupportedChecks}`,
);

const reportPath =
  path.join(
    root,
    "blackwell-max-affine-cuda-tree.json",
  );

writeFileSync(
  reportPath,
  JSON.stringify(
    result,
    null,
    2,
  ) + "\n",
  "utf8",
);

console.log();
console.log(
  `report=${reportPath}`,
);
