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
  addon.runBlackwellDigitalCluster4(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_CLUSTER4_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- DIGITAL RESIDENT 28-TANK: 1 BLOCK vs 4-BLOCK CLUSTER ---",
);

console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);

console.log();
console.log("BINARY RESIDENT REFERENCE");
console.log(`kernelMs=${result.binaryKernelMs.toFixed(3)}`);
console.log(`wallMs=${result.binaryWallMs.toFixed(3)}`);
console.log(`checksum=${result.binaryChecksumHex}`);

console.log();
console.log("CLUSTER4");
console.log(`blocksPerTank=${result.clusterBlocksPerTank}`);
console.log(`threadsPerBlock=${result.clusterThreadsPerBlock}`);
console.log(`blocksPerStep=${result.clusterTotalBlocksPerStep}`);
console.log(`inputH2DMs=${result.clusterInputH2DMs.toFixed(3)}`);
console.log(`kernelMs=${result.clusterKernelMs.toFixed(3)}`);
console.log(`resultD2HMs=${result.clusterResultD2HMs.toFixed(3)}`);
console.log(`wallMs=${result.clusterWallMs.toFixed(3)}`);
console.log(`speedup=${result.speedupVsBinary.toFixed(3)}x`);
console.log(`reduction=${result.reductionVsBinaryPercent.toFixed(2)}%`);

console.log();
console.log("FULL ANNUAL PARITY");
console.log(`statusMismatches=${result.statusMismatches}`);
console.log(`cutMismatches=${result.cutMismatches}`);
console.log(`maxCutError=${result.maximumCutError.toExponential(6)}`);
console.log(`evaluationMismatches=${result.evaluationMismatches}`);
console.log(`checksum=${result.clusterChecksumHex}`);

const report =
  path.join(
    root,
    "blackwell-digital-cluster4.json",
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
