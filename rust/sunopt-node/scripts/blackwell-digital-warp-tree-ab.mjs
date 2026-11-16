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
  addon.runBlackwellDigitalWarpTreeAb(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_WARP_TREE_AB_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- DIGITAL PRE-SORTED: BLOCK TREE vs WARP TREE ---",
);

console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`searchIterations=${result.searchIterations}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);
console.log(`presortCpuMs=${result.presortCpuMs.toFixed(3)}`);

console.log();
console.log("BLOCK TREE");
console.log(`gpuTotalMs=${result.blockTreeGpuTotalMs.toFixed(3)}`);
console.log(`wallMs=${result.blockTreeWallMs.toFixed(3)}`);

console.log();
console.log("WARP TREE");
console.log(`gpuTotalMs=${result.warpTreeGpuTotalMs.toFixed(3)}`);
console.log(`wallMs=${result.warpTreeWallMs.toFixed(3)}`);

console.log();
console.log("DELTA");
console.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);
console.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);
console.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);

console.log();
console.log("BLOCK / WARP PARITY");
console.log(`statusMismatches=${result.blockWarpStatusMismatches}`);
console.log(`cutMismatches=${result.blockWarpCutMismatches}`);
console.log(`maxCutError=${result.blockWarpMaxCutError.toExponential(6)}`);
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
    "blackwell-digital-warp-tree-ab.json",
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

