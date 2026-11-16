// Approach A3b — packed dynamic batch ABI.
// Experimental only: independent calls; no feedback-dependent timestep batching.

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
for (const f of [failedBuild, failedRun]) { try { fs.unlinkSync(f); } catch {} }

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
  fail(failedRun, "approach-a-a3b", message, extra);
  throw new Error(message);
}

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function build() {
  const cmd = process.env.ComSpec || process.env.COMSPEC || "C:\\Windows\\System32\\cmd.exe";
  const r = spawnSync(cmd, ["/d", "/s", "/c", "npm run build"], {
    cwd: nativeRoot,
    encoding: "utf8",
    windowsHide: true,
    env: { ...process.env, SUNOPT_CUDA_FAST_MATH: "0" },
  });
  fs.writeFileSync(path.join(outbox, "approach-a-a3b-build.txt"),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout ?? ""}\n\n=== STDERR ===\n${r.stderr ?? ""}\n`);
  if (r.error || r.status !== 0) {
    fail(failedBuild, "approach-a-a3b-build", r.error ?? `build status ${r.status}`, `${r.stdout}\n${r.stderr}`);
    process.exit(2);
  }
}

function existing(relative, output) {
  const r = spawnSync(process.execPath, [path.join(projectRoot, relative)], {
    cwd: projectRoot, encoding: "utf8", windowsHide: true,
  });
  fs.writeFileSync(path.join(outbox, output),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout ?? ""}\n\n=== STDERR ===\n${r.stderr ?? ""}\n`);
  if (r.error || r.status !== 0) die(`Existing test failed: ${relative}`, `${r.stdout}\n${r.stderr}`);
}

function timed(fn, repetitions, logicalPerInvocation) {
  const t0 = performance.now();
  let last;
  for (let i = 0; i < repetitions; i++) last = fn();
  const totalMs = performance.now() - t0;
  return {
    repetitions,
    logicalCalls: repetitions * logicalPerInvocation,
    totalMs,
    perLogicalCallUs: totalMs * 1000 / (repetitions * logicalPerInvocation),
    last,
  };
}

function summary(rows) {
  const v = rows.map(x => x.perLogicalCallUs);
  return {
    perLogicalCallUs: v,
    medianPerLogicalCallUs: median(v),
    minPerLogicalCallUs: Math.min(...v),
    maxPerLogicalCallUs: Math.max(...v),
  };
}

function assertObject(label, a, b) {
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) die(`${label}: result shape differs`);
  for (const key of keys) {
    const av = a[key], bv = b[key];
    if (typeof av === "number" && typeof bv === "number") {
      const d = Math.abs(av - bv);
      if (!Number.isFinite(av) || !Number.isFinite(bv) || d > 1e-12)
        die(`${label}.${key}: A2=${av} packed=${bv} diff=${d}`);
    } else if (av !== bv) {
      die(`${label}.${key}: A2=${String(av)} packed=${String(bv)}`);
    }
  }
}

function assertArray(label, a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length)
    die(`${label}: result array differs`);
  for (let i = 0; i < a.length; i++) assertObject(`${label}[${i}]`, a[i], b[i]);
}

function compare(batchSize, sequential, prepacked, packAndCall, targetLogicalCalls) {
  const repetitions = Math.max(1, Math.ceil(targetLogicalCalls / batchSize));

  const seq = sequential();
  assertArray(`prepacked-${batchSize}`, seq, prepacked());
  assertArray(`packAndCall-${batchSize}`, seq, packAndCall());

  const warm = Math.max(1, Math.min(100, Math.ceil(1000 / batchSize)));
  timed(sequential, warm, batchSize);
  timed(prepacked, warm, batchSize);
  timed(packAndCall, warm, batchSize);

  const rows = { a2: [], prepacked: [], packAndCall: [] };
  const order = [
    "a2","prepacked","packAndCall",
    "packAndCall","prepacked","a2",
    "prepacked","a2","packAndCall",
    "a2","packAndCall","prepacked",
    "packAndCall","a2","prepacked",
  ];

  for (const variant of order) {
    const fn = variant === "a2" ? sequential : variant === "prepacked" ? prepacked : packAndCall;
    rows[variant].push(timed(fn, repetitions, batchSize));
  }

  const a2 = summary(rows.a2);
  const pre = summary(rows.prepacked);
  const live = summary(rows.packAndCall);

  function againstA2(candidate) {
    const deltaPct = 100 * (candidate.medianPerLogicalCallUs - a2.medianPerLogicalCallUs) /
      a2.medianPerLogicalCallUs;
    return {
      deltaPct,
      speedup: a2.medianPerLogicalCallUs / candidate.medianPerLogicalCallUs,
      conclusion: Math.abs(deltaPct) < 10
        ? "INCONCLUSIVE_LT_10_PERCENT"
        : deltaPct < 0 ? "PACKED_FASTER" : "A2_FASTER",
    };
  }

  return {
    batchSize,
    repetitionsPerRound: repetitions,
    logicalCallsPerRound: repetitions * batchSize,
    a2,
    prepacked: pre,
    packAndCall: live,
    prepackedVsA2: againstA2(pre),
    packAndCallVsA2: againstA2(live),
    exactParity: true,
  };
}

const NS = 15, CAUSAL = 14;

function fillNS(target, params) {
  for (let i = 0; i < params.length; i++) {
    const p = params[i], b = i * NS;
    target[b] = p.currentIndex;
    target[b+1] = p.horizonSteps;
    target[b+2] = p.protectedSteps;
    target[b+3] = p.timeStepHours;
    target[b+4] = p.householdReservePower;
    target[b+5] = p.maximumHeatPumpElectricalPower;
    target[b+6] = p.cop;
    target[b+7] = p.initialWaterlinePower;
    target[b+8] = p.targetStorageEnergy;
    target[b+9] = p.currentStorageEnergy;
    target[b+10] = p.storageEnergyPerKelvin;
    target[b+11] = p.storageConductanceWPerKelvin;
    target[b+12] = p.storageMinTemperature;
    target[b+13] = p.ambientTemperature;
    target[b+14] = p.dailyHeatLoss;
  }
  return target;
}

function fillCausal(target, params) {
  for (let i = 0; i < params.length; i++) {
    const p = params[i], b = i * CAUSAL;
    target[b] = p.currentIndex;
    target[b+1] = p.horizonSteps;
    target[b+2] = p.currentStorageEnergy;
    target[b+3] = p.minimumReserveEnergy;
    target[b+4] = p.targetStorageEnergy;
    target[b+5] = p.maximumHeatPumpElectricalPower;
    target[b+6] = p.cop;
    target[b+7] = p.energyCutLevel;
    target[b+8] = p.timeStepHours;
    target[b+9] = p.storageEnergyPerKelvin;
    target[b+10] = p.storageConductanceWPerKelvin;
    target[b+11] = p.storageMinTemperature;
    target[b+12] = p.ambientTemperature;
    target[b+13] = p.dailyHeatLoss;
  }
  return target;
}

function best(rows, key) {
  const c = rows.filter(r => r[key].conclusion === "PACKED_FASTER");
  return c.length ? c.reduce((a,b) => b[key].speedup > a[key].speedup ? b : a) : null;
}

build();
existing("scripts/test-global-acceleration-parity.mjs", "approach-a-a3b-parity.txt");
existing("scripts/test-global-acceleration-scopes.mjs", "approach-a-a3b-scopes.txt");

const require = createRequire(import.meta.url);
let native;
try {
  native = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));
} catch (e) {
  die("Unable to load rebuilt A3b native binding.", e?.stack ?? String(e));
}

for (const n of [
  "registerAccelerationScope",
  "registerAccelerationDataPair",
  "runNoSunshineKernelContext",
  "runPeakShavingCausalKernelContext",
  "runNoSunshineKernelContextBatchPacked",
  "runPeakShavingCausalKernelContextBatchPacked",
  "clearAccelerationScopeCache",
]) if (typeof native[n] !== "function") die(`Missing A3b export: ${n}`);

const source = fs.readFileSync(sourcePath, "utf8");
const length = 35040;
const pv = new Float64Array(length);
const load = new Float64Array(length);

for (let i = 0; i < length; i++) {
  const q = i % 96, hour = q / 4;
  if (hour >= 6 && hour <= 18) {
    const phase = (hour - 6) / 12 * Math.PI;
    pv[i] = 8.5 * Math.max(0, Math.sin(phase));
  }
  if (q === 28) load[i] = 0.35;
  else if (q === 80) load[i] = 0.28;
}

const exp = Float64Array.from(pv, p => Math.max(0, p - 2));
const storage = {
  storageEnergyPerKelvin: 3000 * 4.186 / 3600,
  storageConductanceWPerKelvin: 3.617,
  storageMinTemperature: 60,
  ambientTemperature: 20,
  dailyHeatLoss: -1,
};
const nsReal = {
  currentIndex: 173, horizonSteps: 672, protectedSteps: 192,
  timeStepHours: 0.25, householdReservePower: 2,
  maximumHeatPumpElectricalPower: 5/3, cop: 3,
  initialWaterlinePower: 10, targetStorageEnergy: 52.325,
  currentStorageEnergy: 10, ...storage,
};
const nsMicro = { ...nsReal, horizonSteps: 1, protectedSteps: 1 };
const causalReal = {
  currentIndex: 173, horizonSteps: 192, currentStorageEnergy: 10,
  minimumReserveEnergy: 2, targetStorageEnergy: 52.325,
  maximumHeatPumpElectricalPower: 5/3, cop: 3, energyCutLevel: 4,
  timeStepHours: 0.25, ...storage,
};

function makeNS(base, n) {
  return Array.from({length:n}, (_,i) => ({
    ...base,
    currentIndex: (base.currentIndex + i*37) % length,
    currentStorageEnergy: base.currentStorageEnergy + (i%7)*0.125,
  }));
}
function makeC(base, n) {
  return Array.from({length:n}, (_,i) => ({
    ...base,
    currentIndex: (base.currentIndex + i*29) % length,
    currentStorageEnergy: base.currentStorageEnergy + (i%5)*0.1,
    energyCutLevel: base.energyCutLevel + (i%3)*0.05,
  }));
}

native.clearAccelerationScopeCache();
const nsScope = native.registerAccelerationScope(source, "rust_noSunshine");
const cScope = native.registerAccelerationScope(source, "rust_peakShavingCausal");
const nsData = native.registerAccelerationDataPair(pv, load);
const cData = native.registerAccelerationDataPair(exp, load);

const batchSizes = [2,4,8,16,32,64];
const workloads = { microNoSunshine: [], realisticNoSunshine: [], realisticCausal: [] };

function runNS(base, n, target) {
  const params = makeNS(base, n);
  const pre = fillNS(new Float64Array(n*NS), params);
  const live = new Float64Array(n*NS);
  return compare(
    n,
    () => params.map(p => native.runNoSunshineKernelContext(nsScope, nsData, p)),
    () => native.runNoSunshineKernelContextBatchPacked(nsScope, nsData, pre),
    () => {
      fillNS(live, params);
      return native.runNoSunshineKernelContextBatchPacked(nsScope, nsData, live);
    },
    target,
  );
}

function runC(n) {
  const params = makeC(causalReal, n);
  const pre = fillCausal(new Float64Array(n*CAUSAL), params);
  const live = new Float64Array(n*CAUSAL);
  return compare(
    n,
    () => params.map(p => native.runPeakShavingCausalKernelContext(cScope, cData, p)),
    () => native.runPeakShavingCausalKernelContextBatchPacked(cScope, cData, pre),
    () => {
      fillCausal(live, params);
      return native.runPeakShavingCausalKernelContextBatchPacked(cScope, cData, live);
    },
    10000,
  );
}

for (const n of batchSizes) {
  workloads.microNoSunshine.push(runNS(nsMicro, n, 20000));
  workloads.realisticNoSunshine.push(runNS(nsReal, n, 5000));
  workloads.realisticCausal.push(runC(n));
}

const bestResults = {
  prepacked: {
    microNoSunshine: best(workloads.microNoSunshine, "prepackedVsA2"),
    realisticNoSunshine: best(workloads.realisticNoSunshine, "prepackedVsA2"),
    realisticCausal: best(workloads.realisticCausal, "prepackedVsA2"),
  },
  packAndCall: {
    microNoSunshine: best(workloads.microNoSunshine, "packAndCallVsA2"),
    realisticNoSunshine: best(workloads.realisticNoSunshine, "packAndCallVsA2"),
    realisticCausal: best(workloads.realisticCausal, "packAndCallVsA2"),
  },
};

const practical = bestResults.packAndCall.realisticNoSunshine || bestResults.packAndCall.realisticCausal;
const preOnly = !practical && (bestResults.prepacked.realisticNoSunshine || bestResults.prepacked.realisticCausal);

const decision = practical
  ? "KEEP_PACKED_BATCH_FOR_INDEPENDENT_HOT_PATHS"
  : preOnly
    ? "PACKED_NATIVE_ABI_WORKS_BUT_JS_PACKING_ERASES_PRACTICAL_GAIN"
    : "DO_NOT_KEEP_A3B";

const release = fs.existsSync(hashFile) ? fs.readFileSync(hashFile, "utf8").trim() : "unknown";
const report = {
  stage: "APPROACH_A_A3B_PACKED_BATCH_ABI",
  release,
  change: "Independent dynamic jobs use one flat reusable Float64Array instead of Vec<JS object>.",
  semanticLimit: "Only independent calls may be batched. Feedback-dependent sequential timesteps are not automatically batchable.",
  measurementPolicy: {
    batchSizes, measuredRoundsPerVariant: 5, medianAuthoritative: true,
    noiseThresholdPercent: 10,
    variants: ["A2 sequential context calls","A3b prepacked reusable buffer","A3b refill reusable buffer + call"],
  },
  parity: { existingTests: "PASS", everyPackedComparison: "EXACT" },
  packedStrides: { noSunshine: NS, causal: CAUSAL },
  workloads,
  bestMeaningful: bestResults,
  decision,
  nextCandidate: practical
    ? "A4: investigate detection of independent call groups before any FP32 experiment."
    : "A4: skip generic batching for now and profile the remaining A2 single-call ABI.",
};

const reportPath = path.join(outbox, "approach-a-a3b-packed-batch-abi.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

function line(prefix, row, key, candidateKey) {
  if (!row) return `${prefix}=NONE`;
  return `${prefix} batch=${row.batchSize} A2Us=${row.a2.medianPerLogicalCallUs.toFixed(6)} candidateUs=${row[candidateKey].medianPerLogicalCallUs.toFixed(6)} speedup=${row[key].speedup.toFixed(4)}x deltaPct=${row[key].deltaPct.toFixed(3)}`;
}

const llm = path.join(outbox, "llm.txt");
fs.appendFileSync(llm, [
  "",
  `=== ${new Date().toISOString()} APPROACH_A_A3B_PACKED_BATCH_ABI ===`,
  `release=${release}`,
  "parity existing=PASS everyPackedComparison=EXACT",
  line("prepacked.micro", bestResults.prepacked.microNoSunshine, "prepackedVsA2", "prepacked"),
  line("prepacked.noSunshine", bestResults.prepacked.realisticNoSunshine, "prepackedVsA2", "prepacked"),
  line("prepacked.causal", bestResults.prepacked.realisticCausal, "prepackedVsA2", "prepacked"),
  line("packAndCall.micro", bestResults.packAndCall.microNoSunshine, "packAndCallVsA2", "packAndCall"),
  line("packAndCall.noSunshine", bestResults.packAndCall.realisticNoSunshine, "packAndCallVsA2", "packAndCall"),
  line("packAndCall.causal", bestResults.packAndCall.realisticCausal, "packAndCallVsA2", "packAndCall"),
  `decision=${decision}`,
  `semanticLimit=${report.semanticLimit}`,
  `nextCandidate=${report.nextCandidate}`,
  `report=${reportPath}`,
  "",
].join("\n"));

console.log(JSON.stringify(report, null, 2));
console.log(`report=${reportPath}`);
console.log(`llm=${llm}`);
console.log("APPROACH_A_A3B_PACKED_BATCH_ABI=OK");
