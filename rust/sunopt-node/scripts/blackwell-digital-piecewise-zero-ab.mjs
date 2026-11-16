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
  addon.runBlackwellDigitalPiecewiseZeroAb(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_PIECEWISE_ZERO_AB_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- DIGITAL PRE-SORTED WARP: SERIAL ZERO FALLBACK vs PIECEWISE ZERO TREE ---",
);

console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);

console.log();
console.log("JOB ROUTING");
console.log(`legacyCompressedJobs=${result.legacyCompressedJobs}`);
console.log(`legacySerialFallbackJobs=${result.legacySerialFallbackJobs}`);
console.log(`piecewiseCompressedJobs=${result.piecewiseCompressedJobs}`);
console.log(`piecewiseSerialFallbackJobs=${result.piecewiseSerialFallbackJobs}`);

console.log();
console.log("LEGACY WARP + SERIAL ZERO FALLBACK");
console.log(`gpuTotalMs=${result.legacyGpuTotalMs.toFixed(3)}`);
console.log(`wallMs=${result.legacyWallMs.toFixed(3)}`);

console.log();
console.log("PIECEWISE ZERO TREE");
console.log(`gpuTotalMs=${result.piecewiseGpuTotalMs.toFixed(3)}`);
console.log(`wallMs=${result.piecewiseWallMs.toFixed(3)}`);

console.log();
console.log("DELTA");
console.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);
console.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);
console.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);

console.log();
console.log("LEGACY / PIECEWISE PARITY");
console.log(`statusMismatches=${result.legacyPiecewiseStatusMismatches}`);
console.log(`cutMismatches=${result.legacyPiecewiseCutMismatches}`);
console.log(`maxCutError=${result.legacyPiecewiseMaxCutError.toExponential(6)}`);
console.log(`evaluationMismatches=${result.evaluationMismatches}`);

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
    "blackwell-digital-piecewise-zero-ab.json",
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

