// rust/sunopt-node/scripts/blackwell-b14l-packed-transport.mjs
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const historyHash = path.join(projectRoot, "updates", "history", "hash.txt");
fs.mkdirSync(outbox, { recursive: true });

const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));
const run = addon.runBlackwellB14LPackedTransport;

if (typeof run !== "function") {
  throw new Error("runBlackwellB14LPackedTransport native export missing");
}

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");

const expectedChecksum = "bf0e8c9e7b5c879e";
const currentRelease = fs.existsSync(historyHash)
  ? fs.readFileSync(historyHash, "utf8").trim()
  : "unknown";

const variants = [];
console.log("--- B14L PACKED/PINNED TRANSPORT ---");
console.log(`release=${currentRelease}`);
console.log(`snapshot=${snapshot}`);

for (const pinned of [false, true]) {
  console.log(`running pinned=${pinned} ...`);
  const r = run(snapshot, pinned, 0);
  const exact = r.checksumHex === expectedChecksum;
  const row = {
    ...r,
    exactVsB14H: exact,
    speedupVsB14K2Wall: 4547.4845 / r.wallMs,
    speedupVsB14K2Kernel: 2702.7629009000957 / r.kernelMs,
    nonKernelWallMs: r.wallMs - r.kernelMs,
  };
  variants.push(row);
  console.log(`pinned=${pinned} kernel=${r.kernelMs.toFixed(3)} ms wall=${r.wallMs.toFixed(3)} ms exact=${exact}`);
}

const exactVariants = variants.filter(v => v.exactVsB14H);
const bestExact = exactVariants.length
  ? exactVariants.reduce((a, b) => a.wallMs <= b.wallMs ? a : b)
  : null;

const report = {
  stage: "B14L_PACKED_PINNED_TRANSPORT",
  release: currentRelease,
  snapshotPath: snapshot,
  expectedChecksum,
  baselineB14KDepth2: { kernelMs: 2702.7629009000957, wallMs: 4547.4845 },
  variants,
  bestExact,
};

const reportPath = path.join(outbox, "blackwell-b14l-packed-transport.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

const llmPath = path.join(outbox, "llm.txt");
const timestamp = new Date().toISOString();
const llmLines = [
  "",
  `=== ${timestamp} B14L_PACKED_PINNED_TRANSPORT ===`,
  `release=${currentRelease}`,
  `snapshot=${snapshot}`,
  `expectedChecksum=${expectedChecksum}`,
  ...variants.map(v => `variant pinned=${v.pinnedHostMemory} kernelMs=${v.kernelMs.toFixed(6)} wallMs=${v.wallMs.toFixed(6)} checksum=${v.checksumHex} exact=${v.exactVsB14H}`),
  bestExact ? `best pinned=${bestExact.pinnedHostMemory} kernelMs=${bestExact.kernelMs.toFixed(6)} wallMs=${bestExact.wallMs.toFixed(6)} speedupVsB14K2Wall=${bestExact.speedupVsB14K2Wall.toFixed(6)}x` : "best=none",
  `report=${reportPath}`,
  "",
].join("\n");
fs.appendFileSync(llmPath, llmLines, "utf8");

console.log("");
console.log(`report=${reportPath}`);
console.log(`llm=${llmPath}`);
if (bestExact) {
  console.log(`bestPinned=${bestExact.pinnedHostMemory}`);
  console.log(`bestExactWallMs=${bestExact.wallMs.toFixed(3)}`);
  console.log(`bestSpeedupVsB14K2=${bestExact.speedupVsB14K2Wall.toFixed(4)}x`);
  console.log("B14L_RESULT=EXACT_VARIANT_FOUND");
} else {
  console.log("B14L_RESULT=NO_BITWISE_EXACT_VARIANT");
}
