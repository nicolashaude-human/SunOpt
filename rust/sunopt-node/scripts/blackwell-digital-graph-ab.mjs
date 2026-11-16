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
  addon.runBlackwellDigitalGraphAb(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_GRAPH_AB_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- DIGITAL LOCKSTEP: DIRECT vs CUDA GRAPH ---",
);

console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);
console.log(`gpuCompressedJobs=${result.gpuCompressedJobs}`);
console.log(`gpuSerialFallbackJobs=${result.gpuSerialFallbackJobs}`);

console.log();
console.log("DIRECT");
console.log(`staticH2DMs=${result.directStaticH2DMs.toFixed(3)}`);
console.log(`gpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`);
console.log(`hostWaitMs=${result.directHostWaitMs.toFixed(3)}`);
console.log(`wallMs=${result.directWallMs.toFixed(3)}`);

console.log();
console.log("CUDA GRAPH + PINNED STAGING");
console.log(`staticH2DMs=${result.graphStaticH2DMs.toFixed(3)}`);
console.log(`graphBuildMs=${result.graphBuildMs.toFixed(3)}`);
console.log(`gpuTotalMs=${result.graphGpuTotalMs.toFixed(3)}`);
console.log(`hostWaitMs=${result.graphHostWaitMs.toFixed(3)}`);
console.log(`wallMs=${result.graphWallMs.toFixed(3)}`);

console.log();
console.log("DELTA");
console.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);
console.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);
console.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);

console.log();
console.log("DIRECT / GRAPH PARITY");
console.log(`statusMismatches=${result.directGraphStatusMismatches}`);
console.log(`cutMismatches=${result.directGraphCutMismatches}`);
console.log(`maxCutError=${result.directGraphMaxCutError.toExponential(6)}`);

console.log();
console.log("SCALAR PARITY SAMPLE");
console.log(`sampleSteps=${result.paritySampleSteps}`);
console.log(`statusMismatches=${result.parityStatusMismatches}`);
console.log(`cutMismatches=${result.parityCutMismatches}`);
console.log(`maxCutError=${result.parityMaxCutError.toExponential(6)}`);
console.log(`checksum=${result.checksumHex}`);

const report =
  path.join(
    root,
    "blackwell-digital-graph-ab.json",
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
console.log(`report=${report}`);

