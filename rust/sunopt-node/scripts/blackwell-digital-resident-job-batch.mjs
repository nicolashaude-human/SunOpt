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
  addon.runBlackwellDigitalResidentJobBatch(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_RESIDENT_JOB_BATCH_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- DIGITAL RESIDENT JOB-PARALLEL LOWER BOUND ---",
);

console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);

console.log();
console.log("REFERENCES");
console.log(`lockstepWallMs=${result.lockstepWallMs.toFixed(3)}`);
console.log(`resident28WallMs=${result.chunk28WallMs.toFixed(3)}`);
console.log(`checksum=${result.referenceChecksumHex}`);

console.log();
console.log(
  "jobs/launch launches H2Dms kernelMs D2Hms wallMs vs28 vsLock reduction status cut eval maxErr",
);

for (const v of result.variants) {
  console.log(
    [
      String(v.jobsPerLaunch).padEnd(11),
      String(v.launchesPerYear).padEnd(8),
      v.inputH2DMs.toFixed(3).padEnd(7),
      v.kernelMs.toFixed(3).padEnd(9),
      v.resultD2HMs.toFixed(3).padEnd(7),
      v.wallMs.toFixed(3).padEnd(9),
      `${v.speedupVsChunk28.toFixed(3)}x`.padEnd(7),
      `${v.speedupVsLockstep.toFixed(3)}x`.padEnd(8),
      `${v.reductionVsLockstepPercent.toFixed(2)}%`.padEnd(9),
      String(v.statusMismatches).padEnd(6),
      String(v.cutMismatches).padEnd(4),
      String(v.evaluationMismatches).padEnd(4),
      v.maximumCutError.toExponential(3),
    ].join(" "),
  );
}

const report =
  path.join(
    root,
    "blackwell-digital-resident-job-batch.json",
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
