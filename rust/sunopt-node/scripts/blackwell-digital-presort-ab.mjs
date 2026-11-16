import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const addon = require(path.join(root, "sunopt_node.win32-x64-msvc.node"));

const snapshot = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, "blackwell-benchmark.bin");
const repetitions = Number.parseInt(process.argv[3] ?? "1", 10);
const deviceId = Number.parseInt(process.argv[4] ?? "0", 10);

const result = addon.runBlackwellDigitalPresortAb(snapshot, repetitions, deviceId);
console.log(`SUNOPT_DIGITAL_PRESORT_AB_JSON=${JSON.stringify(result)}`);
console.log("--- DIGITAL LOCKSTEP: BITONIC vs PRE-SORTED ---");
console.log(`steps=${result.annualStepCount}`);
console.log(`tanks=${result.tankCount}`);
console.log(`jobs=${result.causalJobs}`);
console.log(`presortedMiB=${result.presortedMib.toFixed(3)}`);
console.log(`presortCpuMs=${result.presortCpuMs.toFixed(3)}`);
console.log(`presortedWindowH2DMs=${result.presortedWindowH2DMs.toFixed(3)}`);
console.log();
console.log(`directGpuTotalMs=${result.directGpuTotalMs.toFixed(3)}`);
console.log(`directWallMs=${result.directWallMs.toFixed(3)}`);
console.log(`presortedGpuTotalMs=${result.presortedGpuTotalMs.toFixed(3)}`);
console.log(`presortedWallMs=${result.presortedWallMs.toFixed(3)}`);
console.log(`wallSavingMs=${result.wallSavingMs.toFixed(3)}`);
console.log(`wallSpeedup=${result.wallSpeedup.toFixed(3)}x`);
console.log(`wallReductionPercent=${result.wallReductionPercent.toFixed(2)}%`);
console.log();
console.log(`statusMismatches=${result.statusMismatches}`);
console.log(`cutMismatches=${result.cutMismatches}`);
console.log(`maximumCutError=${result.maximumCutError.toExponential(6)}`);
console.log(`checksum=${result.checksumHex}`);

const report = path.join(root, "blackwell-digital-presort-ab.json");
writeFileSync(report, JSON.stringify(result, null, 2) + "\n", "utf8");
console.log(`report=${report}`);
