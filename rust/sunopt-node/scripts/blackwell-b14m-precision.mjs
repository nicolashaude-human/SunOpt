// rust/sunopt-node/scripts/blackwell-b14m-precision.mjs
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
const run = addon.runBlackwellB14MPrecision;
if (typeof run !== "function") throw new Error("runBlackwellB14MPrecision native export missing");

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  args.set(process.argv[i], process.argv[i + 1]);
}

const mode = args.get("--mode") ?? "strict";
const snapshot = args.get("--snapshot")
  ? path.resolve(args.get("--snapshot"))
  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");
const reference = args.get("--reference")
  ? path.resolve(args.get("--reference"))
  : path.join(outbox, "blackwell-b14m-strict-reference.bin");
const writeReference = (args.get("--write-reference") ?? "false") === "true";
const currentRelease = fs.existsSync(historyHash) ? fs.readFileSync(historyHash, "utf8").trim() : "unknown";

console.log(`B14M mode=${mode} release=${currentRelease}`);
const result = run(snapshot, mode, reference, writeReference, 0);

const report = {
  stage: "B14M_FP32_FASTMATH",
  release: currentRelease,
  snapshotPath: snapshot,
  strictReferencePath: reference,
  ...result,
  fp32KernelSpeedupVsFp64: result.fp64KernelMs / result.fp32KernelMs,
  fp32WallSpeedupVsFp64: result.fp64WallMs / result.fp32WallMs,
};

const reportPath = path.join(outbox, `blackwell-b14m-${mode}.json`);
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

const llmPath = path.join(outbox, "llm.txt");
const line = [
  "",
  `=== ${new Date().toISOString()} B14M_${mode.toUpperCase()} ===`,
  `release=${currentRelease}`,
  `fp64 kernelMs=${result.fp64KernelMs.toFixed(6)} wallMs=${result.fp64WallMs.toFixed(6)} checksum=${result.fp64ChecksumHex}`,
  `fp32 kernelMs=${result.fp32KernelMs.toFixed(6)} wallMs=${result.fp32WallMs.toFixed(6)} checksum=${result.fp32ChecksumHex}`,
  `fp32 speedupKernel=${report.fp32KernelSpeedupVsFp64.toFixed(6)}x speedupWall=${report.fp32WallSpeedupVsFp64.toFixed(6)}x`,
  `fp32 statusMismatchVsFp64=${result.fp32StatusMismatchesVsFp64} maxAbsCutErrorVsFp64=${result.fp32MaxAbsCutErrorVsFp64}`,
  `strictReferenceLoaded=${result.strictReferenceLoaded} fp64StatusMismatchVsStrict=${result.fp64StatusMismatchesVsStrict} fp64MaxAbsCutErrorVsStrict=${result.fp64MaxAbsCutErrorVsStrict}`,
  `fp32StatusMismatchVsStrict=${result.fp32StatusMismatchesVsStrict} fp32MaxAbsCutErrorVsStrict=${result.fp32MaxAbsCutErrorVsStrict}`,
  `report=${reportPath}`,
  "",
].join("\n");
fs.appendFileSync(llmPath, line, "utf8");

console.log(JSON.stringify(report, null, 2));
console.log(`report=${reportPath}`);
console.log(`llm=${llmPath}`);
