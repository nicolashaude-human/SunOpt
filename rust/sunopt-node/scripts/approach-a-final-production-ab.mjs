import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const pairRunner = path.join(here, "approach-a-final-pair.mjs");
const pairReport = path.join(
  outbox,
  "approach-a-final-pair-production-year-parity.json"
);
const hashFile = path.join(
  projectRoot,
  "updates",
  "history",
  "hash.txt"
);
const finalReport = path.join(
  outbox,
  "approach-a-final-production-ab.json"
);
const llmPath = path.join(outbox, "llm.txt");
const failedRun = path.join(outbox, "failed-run.txt");

fs.mkdirSync(outbox, { recursive: true });

function fail(message, extra = "") {
  fs.writeFileSync(
    failedRun,
    [
      "stage=APPROACH_A_FINAL_PRODUCTION_AB",
      `time=${new Date().toISOString()}`,
      `error=${message}`,
      extra,
      "",
    ].join("\n"),
    "utf8"
  );
  throw new Error(message);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function physicsProjection(result) {
  return {
    fake: result.fake,
    year: result.year,
    heatPumpThermalPower: result.heatPumpThermalPower,
    pvPeakPower: result.pvPeakPower,
    gridPriceChfPerKWh: result.gridPriceChfPerKWh,
    feedInPriceChfPerKWh: result.feedInPriceChfPerKWh,
    points: (result.points ?? []).map(point => ({
      storageVolume: point.storageVolume,
      gridImport: point.gridImport,
      feedIn: point.feedIn,
      selfConsumption: point.selfConsumption,
      storageLosses: point.storageLosses,
      unmetThermalDemand: point.unmetThermalDemand,
      gridCost: point.gridCost,
      feedInRevenue: point.feedInRevenue,
      netEnergyCost: point.netEnergyCost,
    })),
  };
}

function runPair(label) {
  try { fs.unlinkSync(pairReport); } catch {}

  const run = spawnSync(
    process.execPath,
    [pairRunner],
    {
      cwd: projectRoot,
      encoding: "utf8",
      windowsHide: true,
      env: {
        ...process.env,
        SUNOPT_CUDA_FAST_MATH: "0",
      },
    }
  );

  const rawLog = path.join(
    outbox,
    `approach-a-final-${label}-runner.txt`
  );

  fs.writeFileSync(
    rawLog,
    [
      `status=${run.status}`,
      `spawnError=${run.error?.stack ?? "none"}`,
      "",
      "=== STDOUT ===",
      run.stdout ?? "",
      "",
      "=== STDERR ===",
      run.stderr ?? "",
      "",
    ].join("\n"),
    "utf8"
  );

  if (run.error || run.status !== 0) {
    fail(
      `Pair runner failed for ${label}`,
      fs.readFileSync(rawLog, "utf8")
    );
  }

  if (!fs.existsSync(pairReport)) {
    fail(`Pair report missing for ${label}: ${pairReport}`);
  }

  const report = JSON.parse(
    fs.readFileSync(pairReport, "utf8")
  );

  const rawAPath = report.raw?.A;
  const rawAGPath = report.raw?.B;

  if (
    !rawAPath ||
    !rawAGPath ||
    !fs.existsSync(rawAPath) ||
    !fs.existsSync(rawAGPath)
  ) {
    fail(`Raw A/AG result files missing for ${label}`);
  }

  const rawA = JSON.parse(
    fs.readFileSync(rawAPath, "utf8")
  );
  const rawAG = JSON.parse(
    fs.readFileSync(rawAGPath, "utf8")
  );

  const exact =
    JSON.stringify(physicsProjection(rawA.result)) ===
    JSON.stringify(physicsProjection(rawAG.result));

  const saved = path.join(
    outbox,
    `approach-a-final-${label}-pair.json`
  );

  fs.writeFileSync(
    saved,
    JSON.stringify({ exact, report }, null, 2) + "\n",
    "utf8"
  );

  return {
    label,
    exact,
    requestWallA: report.performance.requestWallMs.A,
    requestWallAG: report.performance.requestWallMs.B,
    apiA: report.performance.apiTotalDurationMs.A,
    apiAG: report.performance.apiTotalDurationMs.B,
    sweepA: report.performance.sweepDurationMs.A,
    sweepAG: report.performance.sweepDurationMs.B,
    fixedClockA: report.fixedClock?.A,
    fixedClockAG: report.fixedClock?.B,
    saved,
  };
}

if (!fs.existsSync(hashFile)) {
  fail(`Missing release hash: ${hashFile}`);
}

const release = fs.readFileSync(hashFile, "utf8").trim();

runPair("warmup");

const rows = [];
for (let i = 0; i < 5; ++i) {
  rows.push(runPair(`measured-${i}`));
}

const parityExact = rows.every(row => row.exact);

const summary = {
  requestWallMs: {
    A: median(rows.map(row => row.requestWallA)),
    AG: median(rows.map(row => row.requestWallAG)),
  },
  apiTotalDurationMs: {
    A: median(rows.map(row => row.apiA)),
    AG: median(rows.map(row => row.apiAG)),
  },
  sweepDurationMs: {
    A: median(rows.map(row => row.sweepA)),
    AG: median(rows.map(row => row.sweepAG)),
  },
};

for (const metric of Object.values(summary)) {
  metric.deltaPct =
    100 * (metric.AG - metric.A) / metric.A;
  metric.speedupAOverAG =
    metric.A / metric.AG;
}

let decision;
if (!parityExact) {
  decision = "REJECT_AG_PARITY";
} else if (summary.sweepDurationMs.deltaPct <= -10) {
  decision = "AG_READY_FOR_APPROACH_A_DEFAULT";
} else {
  decision = "APPROACH_A_COMPLETE_KEEP_DIRECT_FP64";
}

const report = {
  stage: "APPROACH_A_FINAL_PRODUCTION_AB",
  release,
  variants: {
    A: "exact FP64 direct CUDA launch",
    AG: "exact FP64 persistent single-step CUDA graph",
  },
  measurementPolicy: {
    warmupPairs: 1,
    measuredPairs: 5,
    medianAuthoritative: true,
    noiseThresholdPercent: 10,
    freshNextProcessPerVariant: true,
  },
  parityExact,
  summary,
  decision,
  rows,
};

fs.writeFileSync(
  finalReport,
  JSON.stringify(report, null, 2) + "\n",
  "utf8"
);

fs.appendFileSync(
  llmPath,
  [
    "",
    `=== ${new Date().toISOString()} APPROACH_A_FINAL_PRODUCTION_AB ===`,
    `release=${release}`,
    `parityExact=${parityExact}`,
    `requestWall median A=${summary.requestWallMs.A.toFixed(3)} AG=${summary.requestWallMs.AG.toFixed(3)} deltaPct=${summary.requestWallMs.deltaPct.toFixed(3)}`,
    `apiTotal median A=${summary.apiTotalDurationMs.A.toFixed(3)} AG=${summary.apiTotalDurationMs.AG.toFixed(3)} deltaPct=${summary.apiTotalDurationMs.deltaPct.toFixed(3)}`,
    `sweep median A=${summary.sweepDurationMs.A.toFixed(3)} AG=${summary.sweepDurationMs.AG.toFixed(3)} deltaPct=${summary.sweepDurationMs.deltaPct.toFixed(3)}`,
    `decision=${decision}`,
    `report=${finalReport}`,
    "",
  ].join("\n"),
  "utf8"
);

console.log(JSON.stringify(report, null, 2));
console.log(`report=${finalReport}`);
console.log(`llm=${llmPath}`);
console.log("APPROACH_A_FINAL_PRODUCTION_AB=OK");
