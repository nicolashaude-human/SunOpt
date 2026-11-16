// rust/sunopt-node/scripts/blackwell-b14k-speculative-tree.mjs
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
const run = addon.runBlackwellB14KSpeculativeTree;

if (typeof run !== "function") {
  throw new Error("runBlackwellB14KSpeculativeTree native export missing");
}

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");

const expectedChecksum = "bf0e8c9e7b5c879e";
const variants = [];

console.log("--- B14K SPECULATIVE BINARY-TREE SWEEP ---");
console.log(`snapshot=${snapshot}`);

for (const treeDepth of [1, 2, 3, 4, 5]) {
  console.log(`running treeDepth=${treeDepth} ...`);
  const r = run(snapshot, treeDepth, 0);
  const exact = r.checksumHex === expectedChecksum;
  const row = {
    ...r,
    exactVsB14H: exact,
    speedupVsB14I1Wall: 5256.6251 / r.wallMs,
    speedupVsB14I1Kernel: 3987.9596444517374 / r.kernelMs,
    speedupVsB14J64Wall: 5182.2679 / r.wallMs,
    speedupVsB14J64Kernel: 3376.2135968068615 / r.kernelMs,
  };
  variants.push(row);
  console.log(
    `depth=${treeDepth} warps=${r.warpsPerTank} threads=${r.threadsPerTank} rounds=${r.serialRounds} kernel=${r.kernelMs.toFixed(3)} ms wall=${r.wallMs.toFixed(3)} ms checksum=${r.checksumHex} exact=${exact}`
  );
}

const exactVariants = variants.filter(v => v.exactVsB14H);
const bestExact = exactVariants.length
  ? exactVariants.reduce((a, b) => a.wallMs <= b.wallMs ? a : b)
  : null;

const report = {
  stage: "B14K_SPECULATIVE_BINARY_TREE",
  snapshotPath: snapshot,
  expectedChecksum,
  baselineB14I1: { kernelMs: 3987.9596444517374, wallMs: 5256.6251 },
  baselineB14J64: { kernelMs: 3376.2135968068615, wallMs: 5182.2679 },
  variants,
  bestExact,
};

const reportPath = path.join(outbox, "blackwell-b14k-speculative-tree.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

console.log("");
console.log(`report=${reportPath}`);
if (bestExact) {
  console.log(`bestExactDepth=${bestExact.treeDepth}`);
  console.log(`bestExactWallMs=${bestExact.wallMs.toFixed(3)}`);
  console.log(`bestExactSpeedupVsB14I1=${bestExact.speedupVsB14I1Wall.toFixed(4)}x`);
  console.log("B14K_RESULT=EXACT_VARIANT_FOUND");
} else {
  console.log("B14K_RESULT=NO_BITWISE_EXACT_VARIANT");
}
