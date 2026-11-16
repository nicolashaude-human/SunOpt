import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshotPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "blackwell-benchmark.bin");

const repetitions = Number.parseInt(process.argv[3] ?? "3", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

if (typeof addon.runBlackwellGraphReplay !== "function") {
  throw new Error(
    "runBlackwellGraphReplay is not exported by sunopt-node. Rebuild the native addon first.",
  );
}

const result = addon.runBlackwellGraphReplay(
  snapshotPath,
  repetitions,
  deviceId,
);

console.log(`SUNOPT_BLACKWELL_GRAPH_JSON=${JSON.stringify(result)}`);
console.log("--- GRAPH REPLAY SUMMARY ---");
console.log(`graphBuildMs=${result.graphBuildMs.toFixed(3)}`);
console.log(`medianWallMs=${result.medianWallMs.toFixed(3)}`);
console.log(`medianGpuMs=${result.medianGpuMs.toFixed(3)}`);
console.log(`resultD2HMs=${result.resultD2HMs.toFixed(3)}`);
console.log(`wallMs=[${result.wallMs.map((value) => value.toFixed(3)).join(", ")}]`);
console.log(`gpuMs=[${result.gpuMs.map((value) => value.toFixed(3)).join(", ")}]`);
console.log(`checksumHex=${result.checksumHex}`);
