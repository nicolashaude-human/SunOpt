import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const addon = require(
  path.join(root, "sunopt_node.win32-x64-msvc.node"),
);

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");

const repetitions =
  Number.parseInt(process.argv[3] ?? "1", 10);

const deviceId =
  Number.parseInt(process.argv[4] ?? "0", 10);

const result =
  addon.runBlackwellDigitalResidentChunk(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_RESIDENT_CHUNK_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log("--- DIGITAL RESIDENT CHUNK REPLAY ---");
console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);

console.log();
console.log("LOCKSTEP REFERENCE");
console.log(`gpuTotalMs=${result.lockstepGpuTotalMs.toFixed(3)}`);
console.log(`wallMs=${result.lockstepWallMs.toFixed(3)}`);
console.log(`checksum=${result.lockstepChecksumHex}`);

console.log();
console.log(
  "chunk launches H2Dms kernelMs D2Hms wallMs speedup reduction statusMis cutMis evalMis maxCutErr",
);

for (const v of result.variants) {
  console.log(
    [
      String(v.chunkSteps),
      String(v.launchesPerYear),
      v.inputH2DMs.toFixed(3),
      v.kernelMs.toFixed(3),
      v.resultD2HMs.toFixed(3),
      v.wallMs.toFixed(3),
      `${v.speedupVsLockstep.toFixed(3)}x`,
      `${v.reductionVsLockstepPercent.toFixed(2)}%`,
      String(v.statusMismatches),
      String(v.cutMismatches),
      String(v.evaluationMismatches),
      v.maximumCutError.toExponential(3),
    ].join("\t"),
  );
}

const report =
  path.join(root, "blackwell-digital-resident-chunk.json");

writeFileSync(
  report,
  JSON.stringify(result, null, 2) + "\n",
  "utf8",
);

console.log();
console.log(`report=${report}`);
