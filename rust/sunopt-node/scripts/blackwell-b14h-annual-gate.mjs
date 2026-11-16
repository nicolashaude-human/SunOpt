// rust/sunopt-node/scripts/blackwell-b14h-annual-gate.mjs
//
// B14H: full-annual validation gate for the exact B14G warp-binary kernel.
// No production code is changed.
//
// Usage:
//   node .\rust\sunopt-node\scripts\blackwell-b14h-annual-gate.mjs
//   node .\rust\sunopt-node\scripts\blackwell-b14h-annual-gate.mjs <snapshot> <repetitions> <deviceId>
//
// Default:
//   snapshot    = .\rust\sunopt-node\blackwell-b12-sync-blackwell.bin
//   repetitions = 3
//   deviceId    = 0
//
// This runner deliberately reuses the already validated B14G benchmark.
// It forces sampleSteps=35040 so the reported timing is measured over the
// real annual workload rather than projected from a 2048-step sample.

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(here, "..", "..", "..");

const snapshotArg =
  process.argv[2] ??
  path.join(
    projectRoot,
    "rust",
    "sunopt-node",
    "blackwell-b12-sync-blackwell.bin",
  );

const repetitionsRaw = process.argv[3] ?? "3";
const deviceIdRaw = process.argv[4] ?? "0";

const repetitions = Number.parseInt(repetitionsRaw, 10);
const deviceId = Number.parseInt(deviceIdRaw, 10);

if (!Number.isInteger(repetitions) || repetitions < 1) {
  throw new Error(`Invalid repetitions: ${repetitionsRaw}`);
}

if (!Number.isInteger(deviceId) || deviceId < 0) {
  throw new Error(`Invalid deviceId: ${deviceIdRaw}`);
}

const b14g = path.join(
  here,
  "blackwell-b14g-warp-binary.mjs",
);

console.log("--- B14H FULL-ANNUAL B14G GATE ---");
console.log(`snapshot=${path.resolve(snapshotArg)}`);
console.log("annualSteps=35040");
console.log(`repetitions=${repetitions}`);
console.log(`deviceId=${deviceId}`);
console.log();

const run = spawnSync(
  process.execPath,
  [
    b14g,
    snapshotArg,
    "35040",
    String(repetitions),
    String(deviceId),
  ],
  {
    cwd: projectRoot,
    encoding: "utf8",
    stdio: ["inherit", "pipe", "pipe"],
    windowsHide: false,
    maxBuffer: 64 * 1024 * 1024,
  },
);

if (run.stdout) {
  process.stdout.write(run.stdout);
}
if (run.stderr) {
  process.stderr.write(run.stderr);
}

if (run.error) {
  throw run.error;
}

if (run.status !== 0) {
  process.exit(run.status ?? 1);
}

const text = run.stdout ?? "";

const resultMarker = /B14G_RESULT=([^\r\n]+)/.exec(text)?.[1]?.trim();
const jsonText = /SUNOPT_B14G_JSON=(\{[^\r\n]*\})/.exec(text)?.[1];

if (resultMarker !== "EXACT_WARP_BINARY") {
  throw new Error(
    `B14G did not report EXACT_WARP_BINARY (got ${JSON.stringify(resultMarker)}).`,
  );
}

if (!jsonText) {
  throw new Error("SUNOPT_B14G_JSON marker not found.");
}

const data = JSON.parse(jsonText);
const candidate = data.candidateB14g;

if (!candidate) {
  throw new Error("candidateB14g missing from B14G JSON.");
}

if (candidate.annualStepCount !== 35040) {
  throw new Error(
    `Expected annualStepCount=35040, got ${candidate.annualStepCount}.`,
  );
}

if (candidate.sampleStepCount !== 35040) {
  throw new Error(
    `Expected sampleStepCount=35040, got ${candidate.sampleStepCount}.`,
  );
}

const exact =
  candidate.statusMismatches === 0 &&
  candidate.cutMismatches1E12 === 0 &&
  candidate.cutMismatches1E9 === 0 &&
  candidate.cutMismatches1E6 === 0 &&
  candidate.maxAbsCutError === 0 &&
  data.sameChecksum === true;

const wallMs = candidate.wallMs;
const kernelMs = candidate.kernelMs;
const h2dMs = candidate.inputH2DMs;
const d2hMs = candidate.resultD2HMs;

const summary = {
  snapshotPath: candidate.snapshotPath,
  deviceId: candidate.deviceId,
  repetitions: candidate.repetitions,
  annualStepCount: candidate.annualStepCount,
  annualJobCount: candidate.sampleJobCount,
  supportedJobs: candidate.supportedJobs,
  unsupportedJobs: candidate.unsupportedJobs,
  inputH2DMs: h2dMs,
  kernelMs,
  resultD2HMs: d2hMs,
  wallMs,
  exact,
  checksumHex: candidate.checksumHex,
  speedupVsB14F: data.kernelSpeedup,
};

console.log();
console.log("--- B14H ANNUAL RESULT ---");
console.log(`jobs=${summary.annualJobCount}`);
console.log(
  `supported=${summary.supportedJobs} unsupported=${summary.unsupportedJobs}`,
);
console.log(
  `H2D=${Number(h2dMs).toFixed(3)} kernel=${Number(kernelMs).toFixed(3)} D2H=${Number(d2hMs).toFixed(3)} wall=${Number(wallMs).toFixed(3)} ms`,
);
console.log(`speedupVsB14F=${Number(data.kernelSpeedup).toFixed(4)}x`);
console.log(`checksum=${summary.checksumHex}`);
console.log(`exact=${exact}`);

if (!exact) {
  console.log("B14H_RESULT=FAIL_PARITY");
  console.log(`SUNOPT_B14H_JSON=${JSON.stringify(summary)}`);
  process.exit(2);
}

console.log("B14H_RESULT=EXACT_FULL_ANNUAL");
console.log(`SUNOPT_B14H_JSON=${JSON.stringify(summary)}`);
