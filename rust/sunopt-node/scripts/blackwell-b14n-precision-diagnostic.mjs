// rust/sunopt-node/scripts/blackwell-b14n-precision-diagnostic.mjs
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

const snapshot = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");
const reference = path.join(outbox, "blackwell-b14m-strict-reference.bin");
const release = fs.existsSync(historyHash) ? fs.readFileSync(historyHash, "utf8").trim() : "unknown";

const result = run(snapshot, "strict-diagnostic", reference, true, 0);
let diagnostics = null;
try { diagnostics = JSON.parse(result.fp32DiagnosticsJson); } catch { diagnostics = result.fp32DiagnosticsJson; }

const report = {
  stage: "B14N_PRECISION_DIAGNOSTIC",
  release,
  snapshotPath: snapshot,
  ...result,
  diagnostics,
};

const reportPath = path.join(outbox, "blackwell-b14n-precision-diagnostic.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

const llmPath = path.join(outbox, "llm.txt");
fs.appendFileSync(llmPath, [
  "",
  `=== ${new Date().toISOString()} B14N_PRECISION_DIAGNOSTIC ===`,
  `release=${release}`,
  `fp64 kernelMs=${result.fp64KernelMs.toFixed(6)} wallMs=${result.fp64WallMs.toFixed(6)} checksum=${result.fp64ChecksumHex}`,
  `fp32 kernelMs=${result.fp32KernelMs.toFixed(6)} wallMs=${result.fp32WallMs.toFixed(6)} checksum=${result.fp32ChecksumHex}`,
  `statusMismatch=${result.fp32StatusMismatchesVsFp64} maxAbsCutError=${result.fp32MaxAbsCutErrorVsFp64}`,
  `strictReferenceLoaded=${result.strictReferenceLoaded}`,
  `diagnostics=${result.fp32DiagnosticsJson}`,
  `report=${reportPath}`,
  "",
].join("\n"), "utf8");

console.log(JSON.stringify(report, null, 2));
console.log(`report=${reportPath}`);
console.log(`llm=${llmPath}`);
