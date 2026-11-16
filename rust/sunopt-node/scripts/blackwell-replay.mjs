import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));
const snapshotPath = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, "blackwell-benchmark.bin");
const repetitions = Number.parseInt(process.argv[3] ?? "3", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);
const result = addon.runBlackwellReplay(snapshotPath, repetitions, deviceId);
console.log(`SUNOPT_BLACKWELL_JSON=${JSON.stringify(result)}`);
