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

const result =
  addon.runBlackwellDigitalHpBreakpoints(
    snapshot,
    sampleSteps,
    repetitions,
    deviceId,
  );

console.log(
  `SUNOPT_DIGITAL_HP_BREAKPOINTS_JSON=${JSON.stringify(result)}`,
);

console.log();
console.log(
  "--- BLACKWELL DIGITAL-HP BREAKPOINT SOLVER ---",
);

console.log(
  `snapshot=${result.snapshotPath}`,
);

console.log(
  `annualSteps=${result.annualStepCount}`,
);

console.log(
  `sampleSteps=${result.sampleStepCount}`,
);

console.log(
  `sampleJobs=${result.sampleJobCount}`,
);

console.log(
  `supportedJobs=${result.supportedJobs}/${result.sampleJobCount}`,
);

console.log(
  `scalarReferenceMs=${result.scalarReferenceMs.toFixed(4)}`,
);

console.log();

console.log(
  [
    "iters",
    "kernelMs",
    "annualProjectedMs",
    "avgSurvivalEvals",
    "maxSurvivalEvals",
    "statusMis",
    "cutMis",
    "maxCutErr",
  ].join("\t"),
);

for (
  const variant of
  result.variants
) {
  console.log(
    [
      variant.searchIterations,
      variant.kernelMs.toFixed(6),
      variant.annualProjectedKernelMs.toFixed(3),
      variant.averageSurvivalEvaluations.toFixed(3),
      variant.maximumSurvivalEvaluations,
      variant.statusMismatches,
      variant.cutMismatches,
      variant.maximumCutError.toExponential(6),
    ].join("\t"),
  );
}

const reportPath =
  path.join(
    root,
    "blackwell-digital-hp-breakpoints.json",
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
