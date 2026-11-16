// rust/sunopt-node/scripts/blackwell-b14o-child.mjs
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const addon = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));

const args = new Map();
for (let i = 2; i + 1 < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);

const kind = args.get("--kind") ?? "b14o";
const snapshot = args.get("--snapshot") ?? path.join(nativeRoot, "blackwell-b12-sync-blackwell.bin");
const reference = args.get("--reference") ?? path.join(nativeRoot, "benchmarks", "references", "blackwell-b14m-strict-reference.bin");

function boolArg(name) { return (args.get(name) ?? "false") === "true"; }
function telemetry() {
  try {
    const text = execFileSync("nvidia-smi", [
      "--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw",
      "--format=csv,noheader,nounits"
    ], { encoding: "utf8", windowsHide: true }).trim();
    const [clock, temp, pstate, power] = text.split(",").map(x => x.trim());
    return { clockMHz: Number(clock), temperatureC: Number(temp), pstate, powerW: Number(power) };
  } catch { return null; }
}

const before = telemetry();
let result;

if (kind === "b14l") {
  result = addon.runBlackwellB14LPackedTransport(snapshot, boolArg("--pinned"), 0);
} else if (kind === "reference") {
  fs.mkdirSync(path.dirname(reference), { recursive: true });
  result = addon.runBlackwellB14MPrecision(snapshot, "strict-reference", reference, true, 0);
} else {
  result = addon.runBlackwellB14OFp32Packed(
    snapshot,
    reference,
    boolArg("--pinned-input"),
    Number(args.get("--wait-mode") ?? "0"),
    boolArg("--mapped-output"),
    boolArg("--schedule-spin"),
    0
  );
}

const after = telemetry();
process.stdout.write(JSON.stringify({ kind, before, after, result }) + "\n");
