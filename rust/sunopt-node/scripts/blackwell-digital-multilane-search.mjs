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
  addon.runBlackwellDigitalMultilaneSearch(
    snapshot,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_MULTILANE_SEARCH_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- DIGITAL RESIDENT 28-TANK: BINARY vs MULTILANE SEARCH ---",
);

console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);

console.log();
console.log("BINARY REFERENCE");
console.log(`kernelMs=${result.binaryKernelMs.toFixed(3)}`);
console.log(`wallMs=${result.binaryWallMs.toFixed(3)}`);
console.log(`checksum=${result.binaryChecksumHex}`);

console.log();
console.log(
  "lanes threads H2Dms kernelMs D2Hms wallMs speedup reduction status cut maxErr avgEval maxEval checksum",
);

for (const v of result.variants) {
  console.log(
    [
      String(v.laneCount).padEnd(5),
      String(v.threadsPerBlock).padEnd(7),
      v.inputH2DMs.toFixed(3).padEnd(7),
      v.kernelMs.toFixed(3).padEnd(9),
      v.resultD2HMs.toFixed(3).padEnd(7),
      v.wallMs.toFixed(3).padEnd(9),
      `${v.speedupVsBinary.toFixed(3)}x`.padEnd(8),
      `${v.reductionVsBinaryPercent.toFixed(2)}%`.padEnd(9),
      String(v.statusMismatches).padEnd(6),
      String(v.cutMismatches).padEnd(4),
      v.maximumCutError.toExponential(3).padEnd(10),
      v.averageSurvivalEvaluations.toFixed(3).padEnd(7),
      String(v.maximumSurvivalEvaluations).padEnd(7),
      v.checksumHex,
    ].join(" "),
  );
}

const report =
  path.join(
    root,
    "blackwell-digital-multilane-search.json",
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
