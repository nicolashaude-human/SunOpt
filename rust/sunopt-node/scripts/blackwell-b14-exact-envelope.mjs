// rust/sunopt-node/scripts/blackwell-b14-exact-envelope.mjs
// B14B exact continuous piecewise-affine envelope research reporter.
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-b12-sync-blackwell.bin");
const sampleSteps = Number.parseInt(process.argv[3] ?? "192", 10);

if (typeof addon.runBlackwellB14ExactEnvelope !== "function") {
  throw new Error(
    "runBlackwellB14ExactEnvelope missing; rebuild sunopt-node first.",
  );
}

console.log("--- B14B EXACT PIECEWISE-AFFINE ENVELOPE ---");
const r = addon.runBlackwellB14ExactEnvelope(snapshot, sampleSteps);
const exact =
  r.statusMismatches === 0 &&
  r.cutMismatches1E12 === 0 &&
  r.cutMismatches1E9 === 0 &&
  r.cutMismatches1E6 === 0;

console.log(`snapshot=${r.snapshotPath}`);
console.log(
  `sampleSteps=${r.sampleStepCount} jobs=${r.sampleJobCount}` +
  ` supported=${r.supportedJobs} unsupported=${r.unsupportedJobs}` +
  ` correction=${r.correctionSearchJobs}`,
);
console.log(
  `elapsedMs=${r.elapsedMs.toFixed(3)}` +
  ` scalarReferenceMs=${r.scalarReferenceMs.toFixed(3)}` +
  ` envelopeMs=${r.envelopeMs.toFixed(3)}`,
);
console.log(`statusMismatches=${r.statusMismatches}`);
console.log(
  `cutMismatch >1e-12=${r.cutMismatches1E12}` +
  ` >1e-9=${r.cutMismatches1E9}` +
  ` >1e-6=${r.cutMismatches1E6}`,
);
console.log(
  `maxAbsCutError=${r.maxAbsCutError.toExponential(9)}` +
  ` meanAbsCutError=${r.meanAbsCutError.toExponential(9)}`,
);
console.log(
  `productionQuantizationGap max=${r.maxProductionQuantizationGap.toExponential(9)}` +
  ` mean=${r.meanProductionQuantizationGap.toExponential(9)}`,
);
console.log(
  `segments min=${r.minSegments} median=${r.medianSegments}` +
  ` p95=${r.p95Segments} max=${r.maxSegments}` +
  ` mean=${r.meanSegments.toFixed(3)}`,
);
console.log(
  `hpBreakpoints min=${r.minHpBreakpoints} median=${r.medianHpBreakpoints}` +
  ` p95=${r.p95HpBreakpoints} max=${r.maxHpBreakpoints}` +
  ` mean=${r.meanHpBreakpoints.toFixed(3)}`,
);
console.log(
  `capacityIntersections=${r.capacityIntersections}` +
  ` maxPerJob=${r.maxCapacityIntersectionsPerJob}`,
);
console.log(
  `feasibilityBoundUpdates=${r.feasibilityBoundUpdates}` +
  ` maxPerJob=${r.maxFeasibilityBoundUpdatesPerJob}`,
);
console.log(
  `B14B_RESULT=${exact ? "EXACT_PRODUCTION_20STEP_PARITY" : "NOT_EXACT"}`,
);
console.log(
  `B14B_GPU_SHAPE=${exact && r.maxSegments <= 32
    ? "WARP32_CANDIDATE_NEEDS_DENSER_SAMPLE"
    : "SEGMENT_COUNT_EXCEEDS_WARP32_OR_PARITY_FAILED"}`,
);
console.log(`SUNOPT_B14B_JSON=${JSON.stringify(r)}`);
