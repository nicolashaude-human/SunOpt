// Approach A5 — ABI hypothesis matrix.
//
// A4 is the reference. All H1-H6 candidates are measured in the SAME process
// against that same reference, but none is automatically installed as the
// normal runtime path.
//
// H1 static physical/storage scalars in native context
// H2 numeric-vector output
// H3 callsite-specialised minimal dynamic ABI
// H4 trusted packed path
// H5 caller-owned reusable output buffer
// H6 direct scalar ABI control

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const sourcePath = path.join(projectRoot, "src", "simulation", "rust-acceleration-scopes.ts");
const hashFile = path.join(projectRoot, "updates", "history", "hash.txt");
const failedBuild = path.join(outbox, "failed-build.txt");
const failedRun = path.join(outbox, "failed-run.txt");

fs.mkdirSync(outbox, { recursive: true });
for (const file of [failedBuild, failedRun]) { try { fs.unlinkSync(file); } catch {} }

function fail(file, stage, error, extra = "") {
  fs.writeFileSync(file, [
    `stage=${stage}`,
    `time=${new Date().toISOString()}`,
    `error=${error?.stack ?? error}`,
    extra,
    "",
  ].join("\n"), "utf8");
}

function die(message, extra = "") {
  fail(failedRun, "approach-a-a5", message, extra);
  throw new Error(message);
}

function median(values) {
  const sorted = [...values].sort((a,b) => a-b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle-1] + sorted[middle]) / 2;
}

function build() {
  const cmd = process.env.ComSpec || process.env.COMSPEC || "C:\\Windows\\System32\\cmd.exe";
  const result = spawnSync(cmd, ["/d","/s","/c","npm run build"], {
    cwd: nativeRoot,
    encoding: "utf8",
    windowsHide: true,
    env: { ...process.env, SUNOPT_CUDA_FAST_MATH: "0" },
  });

  fs.writeFileSync(path.join(outbox, "approach-a-a5-build.txt"),
    `status=${result.status}\n\n=== STDOUT ===\n${result.stdout ?? ""}\n\n=== STDERR ===\n${result.stderr ?? ""}\n`);

  if (result.error || result.status !== 0) {
    fail(failedBuild, "approach-a-a5-build",
      result.error ?? `npm run build exited with status ${result.status}`,
      `${result.stdout ?? ""}\n${result.stderr ?? ""}`);
    process.exit(2);
  }
}

function existing(relative, output) {
  const result = spawnSync(process.execPath, [path.join(projectRoot, relative)], {
    cwd: projectRoot,
    encoding: "utf8",
    windowsHide: true,
  });

  fs.writeFileSync(path.join(outbox, output),
    `status=${result.status}\n\n=== STDOUT ===\n${result.stdout ?? ""}\n\n=== STDERR ===\n${result.stderr ?? ""}\n`);

  if (result.error || result.status !== 0)
    die(`Existing test failed: ${relative}`, `${result.stdout ?? ""}\n${result.stderr ?? ""}`);
}

function timed(fn, repetitions) {
  const start = performance.now();
  let last;
  for (let i=0; i<repetitions; i++) last = fn();
  const totalMs = performance.now() - start;

  return {
    repetitions,
    totalMs,
    perCallUs: totalMs * 1000 / repetitions,
    last,
  };
}

function summarize(rows) {
  const values = rows.map(row => row.perCallUs);

  return {
    perCallUs: values,
    medianPerCallUs: median(values),
    minPerCallUs: Math.min(...values),
    maxPerCallUs: Math.max(...values),
  };
}

function encodeNoSunshine(result) {
  return [
    result.kWhNoSunshine,
    result.bucketDemandEnergy,
    result.bucketCoolingLossEnergy,
    result.preBucketHoldingLossEnergy,
    result.allocatedPvThermalEnergy,
    result.unavoidableThermalDeficit,
    result.requiredStorageEnergyAtStart,
    result.additionalStorageEnergyNeededAtStart,
    result.currentWaterlinePower,
    result.currentScheduledHpElectricalPower,
    result.currentScheduledHpThermalPower,
  ];
}

function encodeCausal(result) {
  return [
    result.cutLevelPower,
    result.normalPlanSurvives ? 1 : 0,
    result.maximumPvPlanSurvives ? 1 : 0,
    result.survivalChecks,
    result.binarySearchIterations,
  ];
}

function assertVector(label, actual, expected) {
  const a = Array.from(actual);
  const b = Array.from(expected);

  if (a.length !== b.length)
    die(`${label}: vector length ${a.length} != ${b.length}`);

  for (let i=0; i<a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i]) || diff > 1e-12)
      die(`${label}[${i}]: actual=${a[i]} expected=${b[i]} diff=${diff}`);
  }
}

function assertObject(label, actual, expected) {
  const a = Object.keys(actual);
  const b = Object.keys(expected);

  if (a.length !== b.length)
    die(`${label}: object shape mismatch`);

  for (const key of b) {
    const av = actual[key];
    const ev = expected[key];

    if (typeof av === "number" && typeof ev === "number") {
      const diff = Math.abs(av-ev);
      if (!Number.isFinite(av) || !Number.isFinite(ev) || diff > 1e-12)
        die(`${label}.${key}: actual=${av} expected=${ev} diff=${diff}`);
    } else if (av !== ev) {
      die(`${label}.${key}: actual=${String(av)} expected=${String(ev)}`);
    }
  }
}

function compareMatrix(variants, repetitions, referenceName = "A4") {
  const names = Object.keys(variants);

  // Warm each variant equally.
  for (const name of names) timed(variants[name], Math.min(1000, repetitions));

  const rows = Object.fromEntries(names.map(name => [name, []]));

  // Five rounds. Rotate starting point each round to reduce order bias.
  for (let round=0; round<5; round++) {
    const shift = (round * 3) % names.length;
    const order = names.slice(shift).concat(names.slice(0, shift));

    // Reverse every second round.
    if (round % 2 === 1) order.reverse();

    for (const name of order)
      rows[name].push(timed(variants[name], repetitions));
  }

  const summaries = Object.fromEntries(
    names.map(name => [name, summarize(rows[name])])
  );

  const reference = summaries[referenceName];
  const comparisons = {};

  for (const name of names) {
    if (name === referenceName) continue;

    const candidate = summaries[name];
    const deltaPct = 100 * (
      candidate.medianPerCallUs -
      reference.medianPerCallUs
    ) / reference.medianPerCallUs;

    comparisons[name] = {
      deltaPct,
      speedup: reference.medianPerCallUs / candidate.medianPerCallUs,
      conclusion: Math.abs(deltaPct) < 10
        ? "INCONCLUSIVE_LT_10_PERCENT"
        : deltaPct < 0
          ? "CANDIDATE_FASTER"
          : "A4_FASTER",
    };
  }

  return {
    summaries,
    comparisons,
  };
}

const NS = 15;
const CAUSAL = 14;

function fillNoSunshineFull(target, p) {
  target[0]=p.currentIndex;
  target[1]=p.horizonSteps;
  target[2]=p.protectedSteps;
  target[3]=p.timeStepHours;
  target[4]=p.householdReservePower;
  target[5]=p.maximumHeatPumpElectricalPower;
  target[6]=p.cop;
  target[7]=p.initialWaterlinePower;
  target[8]=p.targetStorageEnergy;
  target[9]=p.currentStorageEnergy;
  target[10]=p.storageEnergyPerKelvin;
  target[11]=p.storageConductanceWPerKelvin;
  target[12]=p.storageMinTemperature;
  target[13]=p.ambientTemperature;
  target[14]=p.dailyHeatLoss;
  return target;
}

function fillCausalFull(target, p) {
  target[0]=p.currentIndex;
  target[1]=p.horizonSteps;
  target[2]=p.currentStorageEnergy;
  target[3]=p.minimumReserveEnergy;
  target[4]=p.targetStorageEnergy;
  target[5]=p.maximumHeatPumpElectricalPower;
  target[6]=p.cop;
  target[7]=p.energyCutLevel;
  target[8]=p.timeStepHours;
  target[9]=p.storageEnergyPerKelvin;
  target[10]=p.storageConductanceWPerKelvin;
  target[11]=p.storageMinTemperature;
  target[12]=p.ambientTemperature;
  target[13]=p.dailyHeatLoss;
  return target;
}

function fillNoSunshineH1(target, p) {
  target[0]=p.currentIndex;
  target[1]=p.horizonSteps;
  target[2]=p.protectedSteps;
  target[3]=p.timeStepHours;
  target[4]=p.householdReservePower;
  target[5]=p.maximumHeatPumpElectricalPower;
  target[6]=p.cop;
  target[7]=p.initialWaterlinePower;
  target[8]=p.targetStorageEnergy;
  target[9]=p.currentStorageEnergy;
  return target;
}

function fillCausalH1(target, p) {
  target[0]=p.currentIndex;
  target[1]=p.horizonSteps;
  target[2]=p.currentStorageEnergy;
  target[3]=p.minimumReserveEnergy;
  target[4]=p.targetStorageEnergy;
  target[5]=p.maximumHeatPumpElectricalPower;
  target[6]=p.cop;
  target[7]=p.energyCutLevel;
  return target;
}

function fillNoSunshineH3(target, p) {
  target[0]=p.currentIndex;
  target[1]=p.currentStorageEnergy;
  return target;
}

function fillCausalH3(target, p) {
  target[0]=p.currentIndex;
  target[1]=p.currentStorageEnergy;
  target[2]=p.minimumReserveEnergy;
  target[3]=p.energyCutLevel;
  return target;
}

build();
existing("scripts/test-global-acceleration-parity.mjs", "approach-a-a5-parity.txt");
existing("scripts/test-global-acceleration-scopes.mjs", "approach-a-a5-scopes.txt");

const require = createRequire(import.meta.url);
let native;

try {
  native = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));
} catch (error) {
  die("Unable to load rebuilt A5 native binding.", error?.stack ?? String(error));
}

const requiredExports = [
  "registerAccelerationScope",
  "registerAccelerationDataPair",
  "runNoSunshineKernelContextPacked",
  "runPeakShavingCausalKernelContextPacked",
  "a5RegisterNoSunshineH1Context",
  "a5RunNoSunshineH1",
  "a5RegisterCausalH1Context",
  "a5RunCausalH1",
  "a5RunNoSunshineNumericOutput",
  "a5RunCausalNumericOutput",
  "a5RegisterNoSunshineH3Context",
  "a5RunNoSunshineH3",
  "a5RegisterCausalH3Context",
  "a5RunCausalH3",
  "a5RunNoSunshineTrusted",
  "a5RunCausalTrusted",
  "a5RunNoSunshineInto",
  "a5RunCausalInto",
  "a5RunNoSunshineScalars",
  "a5RunCausalScalars",
  "clearAccelerationScopeCache",
];

for (const name of requiredExports) {
  if (typeof native[name] !== "function")
    die(`Missing A5 native export: ${name}`);
}

const source = fs.readFileSync(sourcePath, "utf8");
const length = 35040;
const pv = new Float64Array(length);
const load = new Float64Array(length);

for (let i=0; i<length; i++) {
  const quarter = i % 96;
  const hour = quarter / 4;

  if (hour >= 6 && hour <= 18) {
    const phase = (hour - 6) / 12 * Math.PI;
    pv[i] = 8.5 * Math.max(0, Math.sin(phase));
  }

  if (quarter === 28) load[i] = 0.35;
  else if (quarter === 80) load[i] = 0.28;
}

const exportPower = Float64Array.from(pv, power => Math.max(0, power - 2));

const storage = {
  storageEnergyPerKelvin: 3000 * 4.186 / 3600,
  storageConductanceWPerKelvin: 3.617,
  storageMinTemperature: 60,
  ambientTemperature: 20,
  dailyHeatLoss: -1,
};

const noSunshineReal = {
  currentIndex: 173,
  horizonSteps: 672,
  protectedSteps: 192,
  timeStepHours: 0.25,
  householdReservePower: 2,
  maximumHeatPumpElectricalPower: 5/3,
  cop: 3,
  initialWaterlinePower: 10,
  targetStorageEnergy: 52.325,
  currentStorageEnergy: 10,
  ...storage,
};

const noSunshineMicro = {
  ...noSunshineReal,
  horizonSteps: 1,
  protectedSteps: 1,
};

const causalReal = {
  currentIndex: 173,
  horizonSteps: 192,
  currentStorageEnergy: 10,
  minimumReserveEnergy: 2,
  targetStorageEnergy: 52.325,
  maximumHeatPumpElectricalPower: 5/3,
  cop: 3,
  energyCutLevel: 4,
  timeStepHours: 0.25,
  ...storage,
};

native.clearAccelerationScopeCache();

const nsScope = native.registerAccelerationScope(source, "rust_noSunshine");
const causalScope = native.registerAccelerationScope(source, "rust_peakShavingCausal");
const nsData = native.registerAccelerationDataPair(pv, load);
const causalData = native.registerAccelerationDataPair(exportPower, load);

function buildNoSunshineMatrix(parameters) {
  const initialFull = fillNoSunshineFull(new Float64Array(NS), parameters);

  const h1Context = native.a5RegisterNoSunshineH1Context(initialFull);
  const h3Context = native.a5RegisterNoSunshineH3Context(initialFull);

  const full = new Float64Array(NS);
  const h1 = new Float64Array(10);
  const h3 = new Float64Array(2);
  const out = new Float64Array(11);

  const expected = native.runNoSunshineKernelContextPacked(
    nsScope, nsData, initialFull
  );

  // Parity gates before timing.
  fillNoSunshineH1(h1, parameters);
  assertObject("NS H1", native.a5RunNoSunshineH1(nsScope, nsData, h1Context, h1), expected);

  fillNoSunshineFull(full, parameters);
  assertVector("NS H2", native.a5RunNoSunshineNumericOutput(nsScope, nsData, full), encodeNoSunshine(expected));

  fillNoSunshineH3(h3, parameters);
  assertObject("NS H3", native.a5RunNoSunshineH3(nsScope, nsData, h3Context, h3), expected);

  fillNoSunshineFull(full, parameters);
  assertObject("NS H4", native.a5RunNoSunshineTrusted(nsScope, nsData, full), expected);

  fillNoSunshineFull(full, parameters);
  native.a5RunNoSunshineInto(nsScope, nsData, full, out);
  assertVector("NS H5", out, encodeNoSunshine(expected));

  assertObject(
    "NS H6",
    native.a5RunNoSunshineScalars(
      nsScope, nsData,
      parameters.currentIndex,
      parameters.horizonSteps,
      parameters.protectedSteps,
      parameters.timeStepHours,
      parameters.householdReservePower,
      parameters.maximumHeatPumpElectricalPower,
      parameters.cop,
      parameters.initialWaterlinePower,
      parameters.targetStorageEnergy,
      parameters.currentStorageEnergy,
      parameters.storageEnergyPerKelvin,
      parameters.storageConductanceWPerKelvin,
      parameters.storageMinTemperature,
      parameters.ambientTemperature,
      parameters.dailyHeatLoss,
    ),
    expected,
  );

  return {
    A4: () => {
      fillNoSunshineFull(full, parameters);
      return native.runNoSunshineKernelContextPacked(nsScope, nsData, full);
    },

    H1_staticStorage: () => {
      fillNoSunshineH1(h1, parameters);
      return native.a5RunNoSunshineH1(nsScope, nsData, h1Context, h1);
    },

    H2_numericOutput: () => {
      fillNoSunshineFull(full, parameters);
      return native.a5RunNoSunshineNumericOutput(nsScope, nsData, full);
    },

    H3_minimalDynamic: () => {
      fillNoSunshineH3(h3, parameters);
      return native.a5RunNoSunshineH3(nsScope, nsData, h3Context, h3);
    },

    H4_trusted: () => {
      fillNoSunshineFull(full, parameters);
      return native.a5RunNoSunshineTrusted(nsScope, nsData, full);
    },

    H5_outputBuffer: () => {
      fillNoSunshineFull(full, parameters);
      native.a5RunNoSunshineInto(nsScope, nsData, full, out);
      return out;
    },

    H6_scalars: () =>
      native.a5RunNoSunshineScalars(
        nsScope, nsData,
        parameters.currentIndex,
        parameters.horizonSteps,
        parameters.protectedSteps,
        parameters.timeStepHours,
        parameters.householdReservePower,
        parameters.maximumHeatPumpElectricalPower,
        parameters.cop,
        parameters.initialWaterlinePower,
        parameters.targetStorageEnergy,
        parameters.currentStorageEnergy,
        parameters.storageEnergyPerKelvin,
        parameters.storageConductanceWPerKelvin,
        parameters.storageMinTemperature,
        parameters.ambientTemperature,
        parameters.dailyHeatLoss,
      ),
  };
}

function buildCausalMatrix(parameters) {
  const initialFull = fillCausalFull(new Float64Array(CAUSAL), parameters);

  const h1Context = native.a5RegisterCausalH1Context(initialFull);
  const h3Context = native.a5RegisterCausalH3Context(initialFull);

  const full = new Float64Array(CAUSAL);
  const h1 = new Float64Array(8);
  const h3 = new Float64Array(4);
  const out = new Float64Array(5);

  const expected = native.runPeakShavingCausalKernelContextPacked(
    causalScope, causalData, initialFull
  );

  fillCausalH1(h1, parameters);
  assertObject("Causal H1", native.a5RunCausalH1(causalScope, causalData, h1Context, h1), expected);

  fillCausalFull(full, parameters);
  assertVector("Causal H2", native.a5RunCausalNumericOutput(causalScope, causalData, full), encodeCausal(expected));

  fillCausalH3(h3, parameters);
  assertObject("Causal H3", native.a5RunCausalH3(causalScope, causalData, h3Context, h3), expected);

  fillCausalFull(full, parameters);
  assertObject("Causal H4", native.a5RunCausalTrusted(causalScope, causalData, full), expected);

  fillCausalFull(full, parameters);
  native.a5RunCausalInto(causalScope, causalData, full, out);
  assertVector("Causal H5", out, encodeCausal(expected));

  assertObject(
    "Causal H6",
    native.a5RunCausalScalars(
      causalScope, causalData,
      parameters.currentIndex,
      parameters.horizonSteps,
      parameters.currentStorageEnergy,
      parameters.minimumReserveEnergy,
      parameters.targetStorageEnergy,
      parameters.maximumHeatPumpElectricalPower,
      parameters.cop,
      parameters.energyCutLevel,
      parameters.timeStepHours,
      parameters.storageEnergyPerKelvin,
      parameters.storageConductanceWPerKelvin,
      parameters.storageMinTemperature,
      parameters.ambientTemperature,
      parameters.dailyHeatLoss,
    ),
    expected,
  );

  return {
    A4: () => {
      fillCausalFull(full, parameters);
      return native.runPeakShavingCausalKernelContextPacked(causalScope, causalData, full);
    },

    H1_staticStorage: () => {
      fillCausalH1(h1, parameters);
      return native.a5RunCausalH1(causalScope, causalData, h1Context, h1);
    },

    H2_numericOutput: () => {
      fillCausalFull(full, parameters);
      return native.a5RunCausalNumericOutput(causalScope, causalData, full);
    },

    H3_minimalDynamic: () => {
      fillCausalH3(h3, parameters);
      return native.a5RunCausalH3(causalScope, causalData, h3Context, h3);
    },

    H4_trusted: () => {
      fillCausalFull(full, parameters);
      return native.a5RunCausalTrusted(causalScope, causalData, full);
    },

    H5_outputBuffer: () => {
      fillCausalFull(full, parameters);
      native.a5RunCausalInto(causalScope, causalData, full, out);
      return out;
    },

    H6_scalars: () =>
      native.a5RunCausalScalars(
        causalScope, causalData,
        parameters.currentIndex,
        parameters.horizonSteps,
        parameters.currentStorageEnergy,
        parameters.minimumReserveEnergy,
        parameters.targetStorageEnergy,
        parameters.maximumHeatPumpElectricalPower,
        parameters.cop,
        parameters.energyCutLevel,
        parameters.timeStepHours,
        parameters.storageEnergyPerKelvin,
        parameters.storageConductanceWPerKelvin,
        parameters.storageMinTemperature,
        parameters.ambientTemperature,
        parameters.dailyHeatLoss,
      ),
  };
}

const workloads = {
  microNoSunshine:
    compareMatrix(
      buildNoSunshineMatrix(noSunshineMicro),
      10000,
    ),

  realisticNoSunshine:
    compareMatrix(
      buildNoSunshineMatrix(noSunshineReal),
      5000,
    ),

  realisticCausal:
    compareMatrix(
      buildCausalMatrix(causalReal),
      10000,
    ),
};

const hypothesisNames = [
  "H1_staticStorage",
  "H2_numericOutput",
  "H3_minimalDynamic",
  "H4_trusted",
  "H5_outputBuffer",
  "H6_scalars",
];

const aggregate = {};

for (const name of hypothesisNames) {
  const rows = Object.entries(workloads).map(
    ([workload, result]) => ({
      workload,
      ...result.comparisons[name],
      medianUs: result.summaries[name].medianPerCallUs,
      referenceUs: result.summaries.A4.medianPerCallUs,
    })
  );

  aggregate[name] = {
    rows,
    realisticWins:
      rows
        .filter(row => row.workload !== "microNoSunshine")
        .filter(row => row.conclusion === "CANDIDATE_FASTER")
        .length,
    allWins:
      rows
        .filter(row => row.conclusion === "CANDIDATE_FASTER")
        .length,
  };
}

const winners = hypothesisNames.filter(
  name => aggregate[name].realisticWins > 0
);

const release = fs.existsSync(hashFile)
  ? fs.readFileSync(hashFile, "utf8").trim()
  : "unknown";

const report = {
  stage: "APPROACH_A_A5_ABI_HYPOTHESIS_MATRIX",
  release,
  reference: "A4 packed single-call ABI",
  hypotheses: {
    H1_staticStorage:
      "Register only physical/storage constants in native context.",
    H2_numericOutput:
      "Return compact numeric vector instead of JS result object.",
    H3_minimalDynamic:
      "Register all callsite-stable values; transmit only truly dynamic values.",
    H4_trusted:
      "Remove repeated packed length/integer validation on a trusted hot path.",
    H5_outputBuffer:
      "Write result into caller-owned reusable Float64Array.",
    H6_scalars:
      "Direct scalar N-API arguments as control against packed input.",
  },
  measurementPolicy: {
    measuredRoundsPerVariant: 5,
    medianAuthoritative: true,
    noiseThresholdPercent: 10,
    sameProcess: true,
    rotatingOrder: true,
  },
  parity: {
    existingTests: "PASS",
    everyHypothesis: "EXACT",
  },
  workloads,
  aggregate,
  winners,
  decision:
    winners.length
      ? "COMBINE_ONLY_WINNING_FACTORS_IN_NEXT_TEST"
      : "KEEP_A4_AS_REFERENCE_NO_A5_FACTOR_ACCEPTED",
  nextCandidate:
    "Build A6 only from A5 factors that win a realistic workload by >=10%; then test the combined path against A4 before production integration.",
};

const reportPath = path.join(outbox, "approach-a-a5-abi-hypothesis-matrix.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

const llm = path.join(outbox, "llm.txt");

function oneLine(workload, result) {
  const chunks = [`${workload} A4Us=${result.summaries.A4.medianPerCallUs.toFixed(6)}`];

  for (const name of hypothesisNames) {
    const summary = result.summaries[name];
    const cmp = result.comparisons[name];
    chunks.push(
      `${name}=${summary.medianPerCallUs.toFixed(6)}us/${cmp.speedup.toFixed(4)}x/${cmp.conclusion}`
    );
  }

  return chunks.join(" ");
}

fs.appendFileSync(llm, [
  "",
  `=== ${new Date().toISOString()} APPROACH_A_A5_ABI_HYPOTHESIS_MATRIX ===`,
  `release=${release}`,
  "parity existing=PASS everyHypothesis=EXACT",
  oneLine("microNoSunshine", workloads.microNoSunshine),
  oneLine("realisticNoSunshine", workloads.realisticNoSunshine),
  oneLine("realisticCausal", workloads.realisticCausal),
  `winners=${winners.join(",") || "NONE"}`,
  `decision=${report.decision}`,
  `nextCandidate=${report.nextCandidate}`,
  `report=${reportPath}`,
  "",
].join("\n"));

console.log(JSON.stringify(report, null, 2));
console.log(`report=${reportPath}`);
console.log(`llm=${llm}`);
console.log("APPROACH_A_A5_ABI_HYPOTHESIS_MATRIX=OK");
