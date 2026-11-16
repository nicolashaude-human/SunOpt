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

const repetitions =
  Number.parseInt(
    process.argv[3] ?? "1",
    10,
  );

const deviceId =
  Number.parseInt(
    process.argv[4] ?? "0",
    10,
  );

const result =
  addon.runBlackwellDigitalLockstepCandidate(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_LOCKSTEP_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- BLACKWELL DIGITAL-HP PRODUCTION-SHAPED LOCKSTEP ---",
);

console.log(
  `snapshot=${result.snapshotPath}`,
);

console.log(
  `steps=${result.annualStepCount}`,
);

console.log(
  `tanks=${result.tankCount}`,
);

console.log(
  `jobs=${result.causalJobs}`,
);

console.log(
  `searchIterations=${result.searchIterations}`,
);

console.log(
  `gpuSupportedJobs=${result.gpuSupportedJobs}`,
);

console.log(
  `fallbackJobs=${result.fallbackJobs}`,
);

console.log();
console.log(
  "TIMING",
);

console.log(
  `staticH2DMs=${result.staticH2DMs.toFixed(3)}`,
);

console.log(
  `directGpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`,
);

console.log(
  `directHostWaitMs=${result.directHostWaitMs.toFixed(3)}`,
);

console.log(
  `directWallMs=${result.directWallMs.toFixed(3)}`,
);

console.log(
  `fallbackCpuMs=${result.fallbackCpuMs.toFixed(3)}`,
);

console.log(
  `candidateCausalMs=${result.candidateCausalMs.toFixed(3)}`,
);

console.log(
  `candidateAverageUsPerJob=${result.candidateAverageUsPerJob.toFixed(3)}`,
);

console.log();
console.log(
  "PARITY SAMPLE",
);

console.log(
  `sampleSteps=${result.paritySampleSteps}`,
);

console.log(
  `statusMismatches=${result.parityStatusMismatches}`,
);

console.log(
  `cutMismatches=${result.parityCutMismatches}`,
);

console.log(
  `maxCutError=${result.parityMaxCutError.toExponential(6)}`,
);

console.log();
console.log(
  "SEARCH WORK",
);

console.log(
  `averageSurvivalEvaluations=${result.averageSurvivalEvaluations.toFixed(3)}`,
);

console.log(
  `maximumSurvivalEvaluations=${result.maximumSurvivalEvaluations}`,
);

console.log(
  `checksum=${result.checksumHex}`,
);

const report =
  path.join(
    root,
    "blackwell-digital-lockstep-candidate.json",
  );

writeFileSync(
  report,
  JSON.stringify(
    result,
    null,
    2,
  ) + "\n",
  "utf8",
);

console.log();
console.log(
  `report=${report}`,
);
