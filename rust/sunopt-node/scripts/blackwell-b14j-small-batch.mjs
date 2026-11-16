// rust/sunopt-node/scripts/blackwell-b14j-small-batch.mjs
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
fs.mkdirSync(outbox, { recursive: true });

const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));
const run = addon.runBlackwellB14JSmallBatch;

if (typeof run !== "function") {
  throw new Error("runBlackwellB14JSmallBatch native export missing");
}

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");

const threadCounts = [32, 64, 128, 256, 512];
const expectedChecksum = "bf0e8c9e7b5c879e";
const variants = [];

console.log("--- B14J SMALL-BATCH OCCUPANCY SWEEP ---");
console.log(`snapshot=${snapshot}`);

for (const blockThreads of threadCounts) {
  console.log(`running blockThreads=${blockThreads} ...`);
  const r = run(snapshot, blockThreads, 0);
  const exact = r.checksumHex === expectedChecksum;
  const row = {
    ...r,
    exactVsB14H: exact,
    speedupVsB14I1Wall: 5256.6251 / r.wallMs,
    speedupVsB14I1Kernel: 3987.9596444517374 / r.kernelMs,
  };
  variants.push(row);
  console.log(
    `threads=${blockThreads} kernel=${r.kernelMs.toFixed(3)} ms wall=${r.wallMs.toFixed(3)} ms checksum=${r.checksumHex} exact=${exact}`
  );
}

const exactVariants = variants.filter(v => v.exactVsB14H);
const bestExact = exactVariants.length
  ? exactVariants.reduce((a, b) => a.wallMs <= b.wallMs ? a : b)
  : null;

const report = {
  stage: "B14J_SMALL_BATCH_OCCUPANCY",
  snapshotPath: snapshot,
  baselineB14I1: {
    kernelMs: 3987.9596444517374,
    wallMs: 5256.6251,
    checksumHex: expectedChecksum,
  },
  variants,
  bestExact,
};

const reportPath = path.join(outbox, "blackwell-b14j-small-batch.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

console.log("");
console.log(`report=${reportPath}`);
if (bestExact) {
  console.log(`bestExactThreads=${bestExact.blockThreads}`);
  console.log(`bestExactWallMs=${bestExact.wallMs.toFixed(3)}`);
  console.log(`bestExactSpeedup=${bestExact.speedupVsB14I1Wall.toFixed(4)}x`);
  console.log("B14J_RESULT=EXACT_VARIANT_FOUND");
} else {
  console.log("B14J_RESULT=NO_BITWISE_EXACT_VARIANT");
}
