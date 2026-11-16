// rust/sunopt-node/scripts/blackwell-b14i1-lockstep.mjs
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

const addon = require(
  path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"),
);

const run = addon.runBlackwellB14I1Lockstep;

if (typeof run !== "function") {
  throw new Error(
    "runBlackwellB14I1Lockstep native export missing",
  );
}

const snapshot =
  process.argv[2]
    ? path.resolve(process.argv[2])
    : path.join(
        nativeRoot,
        "blackwell-b12-sync-blackwell.bin",
      );

console.log(
  "--- B14I1 B14G PRODUCTION-SHAPED LOCKSTEP ---",
);
console.log(`snapshot=${snapshot}`);

const r = run(snapshot, 0);

const expectedChecksum = "bf0e8c9e7b5c879e";

const checksumMatchesB14h =
  r.checksumHex === expectedChecksum;

const report = {
  ...r,
  expectedB14hChecksumHex: expectedChecksum,
  checksumMatchesB14h,
};

const reportPath =
  path.join(
    outbox,
    "blackwell-b14i1-lockstep.json",
  );

fs.writeFileSync(
  reportPath,
  JSON.stringify(report, null, 2) + "\n",
);

console.log(
  `steps=${r.annualStepCount} tanks=${r.tankCount} jobs=${r.causalJobs}`,
);
console.log(
  `staticH2D=${r.staticH2DMs.toFixed(3)} ms`,
);
console.log(
  `dynamicH2D=${r.dynamicH2DMs.toFixed(3)} ms`,
);
console.log(
  `kernel=${r.kernelMs.toFixed(3)} ms`,
);
console.log(
  `resultD2H=${r.resultD2HMs.toFixed(3)} ms`,
);
console.log(
  `wall=${r.wallMs.toFixed(3)} ms`,
);
console.log(`checksum=${r.checksumHex}`);
console.log(
  `checksumMatchesB14H=${checksumMatchesB14h}`,
);
console.log(`report=${reportPath}`);
console.log(
  `B14I1_RESULT=${
    checksumMatchesB14h
      ? "EXACT_PRODUCTION_SHAPED_LOCKSTEP"
      : "CHECKSUM_MISMATCH"
  }`,
);
console.log(
  `SUNOPT_B14I1_JSON=${JSON.stringify(report)}`,
);
