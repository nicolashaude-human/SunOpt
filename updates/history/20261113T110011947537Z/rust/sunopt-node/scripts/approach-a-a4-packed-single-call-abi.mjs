// Approach A4 — packed single-call ABI.
//
// Question:
//   Does the flat ABI from A3b help even when batch size is exactly 1?
//
// Variants:
//   A2 object       : existing context export with one JS parameter object
//   A4 prepacked    : reusable Float64Array already contains dynamic values
//   A4 pack+call    : reusable Float64Array is filled immediately before call
//
// This is directly relevant to sequential feedback-dependent controller steps.

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

function writeFailure(file, stage, error, extra = "") {
  fs.writeFileSync(file, [
    `stage=${stage}`,
    `time=${new Date().toISOString()}`,
    `error=${error?.stack ?? error}`,
    extra,
    "",
  ].join("\n"), "utf8");
}

function die(message, extra = "") {
  writeFailure(failedRun, "approach-a-a4", message, extra);
  throw new Error(message);
}

function median(values) {
  const s = [...values].sort((a,b) => a-b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2;
}

function build() {
  const cmd = process.env.ComSpec || process.env.COMSPEC || "C:\\Windows\\System32\\cmd.exe";
  const r = spawnSync(cmd, ["/d","/s","/c","npm run build"], {
    cwd: nativeRoot,
    encoding: "utf8",
    windowsHide: true,
    env: { ...process.env, SUNOPT_CUDA_FAST_MATH: "0" },
  });

  fs.writeFileSync(path.join(outbox, "approach-a-a4-build.txt"),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout ?? ""}\n\n=== STDERR ===\n${r.stderr ?? ""}\n`);

  if (r.error || r.status !== 0) {
    writeFailure(failedBuild, "approach-a-a4-build",
      r.error ?? `npm run build exited with status ${r.status}`,
      `${r.stdout ?? ""}\n${r.stderr ?? ""}`);
    process.exit(2);
  }
}

function existing(relative, output) {
  const r = spawnSync(process.execPath, [path.join(projectRoot, relative)], {
    cwd: projectRoot,
    encoding: "utf8",
    windowsHide: true,
  });

  fs.writeFileSync(path.join(outbox, output),
    `status=${r.status}\n\n=== STDOUT ===\n${r.stdout ?? ""}\n\n=== STDERR ===\n${r.stderr ?? ""}\n`);

  if (r.error || r.status !== 0)
    die(`Existing test failed: ${relative}`, `${r.stdout ?? ""}\n${r.stderr ?? ""}`);
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
  const v = rows.map(x => x.perCallUs);
  return {
    perCallUs: v,
    medianPerCallUs: median(v),
    minPerCallUs: Math.min(...v),
    maxPerCallUs: Math.max(...v),
  };
}

function assertSame(label, a, b) {
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length)
    die(`${label}: result shape differs`);

  for (const key of keys) {
    const av = a[key], bv = b[key];
    if (typeof av === "number" && typeof bv === "number") {
      const diff = Math.abs(av-bv);
      if (!Number.isFinite(av) || !Number.isFinite(bv) || diff > 1e-12)
        die(`${label}.${key}: object=${av} packed=${bv} diff=${diff}`);
    } else if (av !== bv) {
      die(`${label}.${key}: object=${String(av)} packed=${String(bv)}`);
    }
  }
}

function compareThree(objectCall, prepackedCall, packAndCall, repetitions) {
  const reference = objectCall();
  assertSame("prepacked", reference, prepackedCall());
  assertSame("packAndCall", reference, packAndCall());

  for (const fn of [objectCall, prepackedCall, packAndCall]) timed(fn, 1000);

  const rows = { object: [], prepacked: [], packAndCall: [] };
  const order = [
    "object","prepacked","packAndCall",
    "packAndCall","prepacked","object",
    "prepacked","object","packAndCall",
    "object","packAndCall","prepacked",
    "packAndCall","object","prepacked",
  ];

  for (const variant of order) {
    const fn = variant === "object" ? objectCall
      : variant === "prepacked" ? prepackedCall
      : packAndCall;
    rows[variant].push(timed(fn, repetitions));
  }

  const object = summarize(rows.object);
  const prepacked = summarize(rows.prepacked);
  const live = summarize(rows.packAndCall);

  function cmp(candidate) {
    const deltaPct = 100 * (candidate.medianPerCallUs - object.medianPerCallUs) /
      object.medianPerCallUs;

    return {
      deltaPct,
      speedup: object.medianPerCallUs / candidate.medianPerCallUs,
      conclusion: Math.abs(deltaPct) < 10
        ? "INCONCLUSIVE_LT_10_PERCENT"
        : deltaPct < 0
          ? "PACKED_FASTER"
          : "OBJECT_FASTER",
    };
  }

  return {
    object,
    prepacked,
    packAndCall: live,
    prepackedVsObject: cmp(prepacked),
    packAndCallVsObject: cmp(live),
    exactParity: true,
  };
}

const NS = 15;
const CAUSAL = 14;

function fillNS(target, p) {
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

function fillCausal(target, p) {
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

build();
existing("scripts/test-global-acceleration-parity.mjs", "approach-a-a4-parity.txt");
existing("scripts/test-global-acceleration-scopes.mjs", "approach-a-a4-scopes.txt");

const require = createRequire(import.meta.url);
let native;
try {
  native = require(path.join(nativeRoot, "sunopt_node.win32-x64-msvc.node"));
} catch (e) {
  die("Unable to load rebuilt A4 native binding.", e?.stack ?? String(e));
}

for (const name of [
  "registerAccelerationScope",
  "registerAccelerationDataPair",
  "runNoSunshineKernelContext",
  "runPeakShavingCausalKernelContext",
  "runNoSunshineKernelContextPacked",
  "runPeakShavingCausalKernelContextPacked",
  "clearAccelerationScopeCache",
]) {
  if (typeof native[name] !== "function")
    die(`Missing A4 native export: ${name}`);
}

const source = fs.readFileSync(sourcePath, "utf8");
const length = 35040;
const pv = new Float64Array(length);
const load = new Float64Array(length);

for (let i=0; i<length; i++) {
  const q = i % 96;
  const hour = q / 4;

  if (hour >= 6 && hour <= 18) {
    const phase = (hour - 6) / 12 * Math.PI;
    pv[i] = 8.5 * Math.max(0, Math.sin(phase));
  }

  if (q === 28) load[i] = 0.35;
  else if (q === 80) load[i] = 0.28;
}

const exportPower = Float64Array.from(pv, p => Math.max(0, p-2));

const storage = {
  storageEnergyPerKelvin: 3000 * 4.186 / 3600,
  storageConductanceWPerKelvin: 3.617,
  storageMinTemperature: 60,
  ambientTemperature: 20,
  dailyHeatLoss: -1,
};

const nsReal = {
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

const nsMicro = {
  ...nsReal,
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

function runNS(parameters, repetitions) {
  const pre = fillNS(new Float64Array(NS), parameters);
  const live = new Float64Array(NS);

  return compareThree(
    () => native.runNoSunshineKernelContext(nsScope, nsData, parameters),
    () => native.runNoSunshineKernelContextPacked(nsScope, nsData, pre),
    () => {
      fillNS(live, parameters);
      return native.runNoSunshineKernelContextPacked(nsScope, nsData, live);
    },
    repetitions,
  );
}

function runCausal(parameters, repetitions) {
  const pre = fillCausal(new Float64Array(CAUSAL), parameters);
  const live = new Float64Array(CAUSAL);

  return compareThree(
    () => native.runPeakShavingCausalKernelContext(causalScope, causalData, parameters),
    () => native.runPeakShavingCausalKernelContextPacked(causalScope, causalData, pre),
    () => {
      fillCausal(live, parameters);
      return native.runPeakShavingCausalKernelContextPacked(causalScope, causalData, live);
    },
    repetitions,
  );
}

const comparisons = {
  microNoSunshine: runNS(nsMicro, 10000),
  realisticNoSunshine: runNS(nsReal, 5000),
  realisticCausal: runCausal(causalReal, 10000),
};

const practical = Object.values(comparisons).filter(
  x => x.packAndCallVsObject.conclusion === "PACKED_FASTER"
);

const realisticPractical = [
  comparisons.realisticNoSunshine,
  comparisons.realisticCausal,
].filter(
  x => x.packAndCallVsObject.conclusion === "PACKED_FASTER"
);

const decision =
  realisticPractical.length === 2
    ? "KEEP_A4_PACKED_SINGLE_CALL"
    : realisticPractical.length === 1
      ? "KEEP_A4_FOR_RELEVANT_HOT_PATHS_AND_REVIEW"
      : "DO_NOT_KEEP_A4";

const release = fs.existsSync(hashFile)
  ? fs.readFileSync(hashFile, "utf8").trim()
  : "unknown";

const report = {
  stage: "APPROACH_A_A4_PACKED_SINGLE_CALL_ABI",
  release,
  change: "Single sequential context call uses a fixed-stride Float64Array instead of a JS parameter object.",
  measurementPolicy: {
    measuredRoundsPerVariant: 5,
    medianAuthoritative: true,
    noiseThresholdPercent: 10,
    variants: [
      "A2 JS object",
      "A4 prepacked reusable Float64Array",
      "A4 refill reusable Float64Array + call",
    ],
  },
  parity: {
    existingTests: "PASS",
    everyComparison: "EXACT",
  },
  packedStrides: {
    noSunshine: NS,
    causal: CAUSAL,
  },
  comparisons,
  decision,
  nextCandidate:
    decision === "KEEP_A4_PACKED_SINGLE_CALL"
      ? "Integrate A4 into the real TypeScript runtime/adapters, then measure production-shaped annual call totals before FP32."
      : "Do not integrate A4 globally; proceed to the next isolated runtime/compiler lesson.",
};

const reportPath = path.join(outbox, "approach-a-a4-packed-single-call-abi.json");
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

function line(name, x) {
  return `${name} objectUs=${x.object.medianPerCallUs.toFixed(6)} prepackedUs=${x.prepacked.medianPerCallUs.toFixed(6)} packAndCallUs=${x.packAndCall.medianPerCallUs.toFixed(6)} packSpeedup=${x.packAndCallVsObject.speedup.toFixed(4)}x packDeltaPct=${x.packAndCallVsObject.deltaPct.toFixed(3)} conclusion=${x.packAndCallVsObject.conclusion}`;
}

const llm = path.join(outbox, "llm.txt");
fs.appendFileSync(llm, [
  "",
  `=== ${new Date().toISOString()} APPROACH_A_A4_PACKED_SINGLE_CALL_ABI ===`,
  `release=${release}`,
  "parity existing=PASS everyComparison=EXACT",
  line("microNoSunshine", comparisons.microNoSunshine),
  line("realisticNoSunshine", comparisons.realisticNoSunshine),
  line("realisticCausal", comparisons.realisticCausal),
  `decision=${decision}`,
  `nextCandidate=${report.nextCandidate}`,
  `report=${reportPath}`,
  "",
].join("\n"));

console.log(JSON.stringify(report, null, 2));
console.log(`report=${reportPath}`);
console.log(`llm=${llm}`);
console.log("APPROACH_A_A4_PACKED_SINGLE_CALL_ABI=OK");
