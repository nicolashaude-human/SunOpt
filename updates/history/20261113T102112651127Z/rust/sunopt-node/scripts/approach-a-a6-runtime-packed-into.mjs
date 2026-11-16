// Approach A6.1 — repaired A6 runner with complete diagnostics.
//
// No optimisation semantics changed.
// This runner:
//   - captures every uncaught failure in failed-run.txt
//   - verifies both direct .node and package-loader exports
//   - directly smoke-tests packed-into exports
//   - then tests the TypeScript runtime wrapper
//   - benchmarks only after all parity gates pass

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL, fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const here = path.dirname(fileURLToPath(import.meta.url));
const nativeRoot = path.resolve(here, "..");
const projectRoot = path.resolve(nativeRoot, "..", "..");
const outbox = path.join(projectRoot, "updates", "outbox");
const runtimePath = path.join(projectRoot, "src", "simulation", "rust-acceleration-runtime.ts");
const sourcePath = path.join(projectRoot, "src", "simulation", "rust-acceleration-scopes.ts");
const hashFile = path.join(projectRoot, "updates", "history", "hash.txt");
const failedBuild = path.join(outbox, "failed-build.txt");
const failedRun = path.join(outbox, "failed-run.txt");

fs.mkdirSync(outbox, { recursive: true });

for (const file of [failedBuild, failedRun]) {
  try { fs.unlinkSync(file); } catch {}
}

function errorText(error) {
  return error instanceof Error
    ? (error.stack ?? error.message)
    : String(error);
}

function writeFailure(stage, error, extra = "") {
  fs.writeFileSync(
    failedRun,
    [
      `stage=${stage}`,
      `time=${new Date().toISOString()}`,
      `error=${errorText(error)}`,
      extra,
      "",
    ].join("\n"),
    "utf8",
  );
}

function die(stage, message, extra = "") {
  const error = new Error(message);
  writeFailure(stage, error, extra);
  throw error;
}

process.on("uncaughtException", error => {
  if (!fs.existsSync(failedRun)) writeFailure("uncaughtException", error);
  console.error(error);
  process.exit(1);
});

process.on("unhandledRejection", error => {
  if (!fs.existsSync(failedRun)) writeFailure("unhandledRejection", error);
  console.error(error);
  process.exit(1);
});

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

  fs.writeFileSync(
    path.join(outbox, "approach-a-a6-build.txt"),
    [
      `status=${result.status}`,
      `spawnError=${result.error?.stack ?? "none"}`,
      "",
      "=== STDOUT ===",
      result.stdout ?? "",
      "",
      "=== STDERR ===",
      result.stderr ?? "",
      "",
    ].join("\n"),
    "utf8",
  );

  if (result.error || result.status !== 0) {
    fs.writeFileSync(
      failedBuild,
      [
        "stage=approach-a-a6-build",
        `time=${new Date().toISOString()}`,
        `error=${errorText(result.error ?? `build status ${result.status}`)}`,
        result.stdout ?? "",
        result.stderr ?? "",
        "",
      ].join("\n"),
      "utf8",
    );
    process.exit(2);
  }
}

function existing(relative, outputName) {
  const result = spawnSync(process.execPath, [path.join(projectRoot, relative)], {
    cwd: projectRoot,
    encoding: "utf8",
    windowsHide: true,
  });

  fs.writeFileSync(
    path.join(outbox, outputName),
    [
      `status=${result.status}`,
      `spawnError=${result.error?.stack ?? "none"}`,
      "",
      "=== STDOUT ===",
      result.stdout ?? "",
      "",
      "=== STDERR ===",
      result.stderr ?? "",
      "",
    ].join("\n"),
    "utf8",
  );

  if (result.error || result.status !== 0) {
    die(
      `existing-test:${relative}`,
      `Existing test failed: ${relative}`,
      `${result.stdout ?? ""}\n${result.stderr ?? ""}`,
    );
  }
}

function timed(fn, repetitions) {
  const start = performance.now();
  let last;
  for (let i=0; i<repetitions; i++) last = fn();
  const totalMs = performance.now() - start;
  return {
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

function assertSame(label, reference, candidate) {
  if (reference === null || reference === undefined)
    die(`parity:${label}`, `${label}: reference returned ${String(reference)}`);

  if (candidate === null || candidate === undefined)
    die(`parity:${label}`, `${label}: candidate returned ${String(candidate)}`);

  const a = Object.keys(reference);
  const b = Object.keys(candidate);

  if (a.length !== b.length)
    die(`parity:${label}`, `${label}: result shape differs`, `reference=${JSON.stringify(reference)}\ncandidate=${JSON.stringify(candidate)}`);

  for (const key of a) {
    const av = reference[key];
    const bv = candidate[key];

    if (typeof av === "number" && typeof bv === "number") {
      const diff = Math.abs(av - bv);
      if (!Number.isFinite(av) || !Number.isFinite(bv) || diff > 1e-12)
        die(`parity:${label}`, `${label}.${key}: ref=${av} candidate=${bv} diff=${diff}`);
    } else if (av !== bv) {
      die(`parity:${label}`, `${label}.${key}: ref=${String(av)} candidate=${String(bv)}`);
    }
  }
}

function compare(label, referenceFn, candidateFn, repetitions, statusFn) {
  const reference = referenceFn();
  const candidate = candidateFn();

  assertSame(label, reference, candidate);

  timed(referenceFn, Math.min(1000, repetitions));
  timed(candidateFn, Math.min(1000, repetitions));

  const rows = { reference: [], candidate: [] };
  const order = [
    "reference","candidate",
    "candidate","reference",
    "reference","candidate",
    "candidate","reference",
    "reference","candidate",
  ];

  for (const variant of order) {
    rows[variant].push(
      timed(
        variant === "reference" ? referenceFn : candidateFn,
        repetitions,
      ),
    );
  }

  const ref = summarize(rows.reference);
  const cand = summarize(rows.candidate);
  const deltaPct = 100 * (cand.medianPerCallUs - ref.medianPerCallUs) / ref.medianPerCallUs;

  return {
    reference: ref,
    candidate: cand,
    deltaPct,
    speedup: ref.medianPerCallUs / cand.medianPerCallUs,
    conclusion:
      Math.abs(deltaPct) < 10
        ? "INCONCLUSIVE_LT_10_PERCENT"
        : deltaPct < 0
          ? "A6_FASTER"
          : "REFERENCE_FASTER",
    exactParity: true,
    status: statusFn(),
  };
}

async function main() {
  build();

  existing(
    "scripts/test-global-acceleration-parity.mjs",
    "approach-a-a6-parity.txt",
  );

  existing(
    "scripts/test-global-acceleration-scopes.mjs",
    "approach-a-a6-scopes.txt",
  );

  let runtime;
  try {
    runtime = await import(
      pathToFileURL(runtimePath).href +
      `?a61=${Date.now()}`
    );
  } catch (error) {
    die("runtime-import", "Failed to import rust-acceleration-runtime.ts", errorText(error));
  }

  const require = createRequire(import.meta.url);

  let directBinding;
  try {
    directBinding = require(
      path.join(
        nativeRoot,
        "sunopt_node.win32-x64-msvc.node",
      ),
    );
  } catch (error) {
    die("direct-binding-load", "Failed to load direct native .node binding", errorText(error));
  }

  let packageBinding;
  try {
    packageBinding = runtime.getSunOptNativeBinding();
  } catch (error) {
    die("package-binding-load", "getSunOptNativeBinding() threw", errorText(error));
  }

  if (packageBinding === null)
    die("package-binding-load", "getSunOptNativeBinding() returned null");

  const required = [
    "registerAccelerationScope",
    "registerAccelerationDataPair",
    "runNoSunshineKernelContext",
    "runPeakShavingCausalKernelContext",
    "runNoSunshineKernelContextPackedInto",
    "runPeakShavingCausalKernelContextPackedInto",
    "clearAccelerationScopeCache",
  ];

  const exportDiagnostic = {};
  for (const name of required) {
    exportDiagnostic[name] = {
      direct: typeof directBinding[name],
      package: typeof packageBinding[name],
    };

    if (typeof directBinding[name] !== "function")
      die("direct-export-check", `Direct binding missing export ${name}`, JSON.stringify(exportDiagnostic, null, 2));

    if (typeof packageBinding[name] !== "function")
      die("package-export-check", `Package binding missing export ${name}`, JSON.stringify(exportDiagnostic, null, 2));
  }

  fs.writeFileSync(
    path.join(outbox, "approach-a-a6-export-diagnostic.json"),
    JSON.stringify(exportDiagnostic, null, 2) + "\n",
    "utf8",
  );

  const source = fs.readFileSync(sourcePath, "utf8");

  const length = 35_040;
  const pv = new Float64Array(length);
  const load = new Float64Array(length);

  for (let i=0; i<length; i++) {
    const q = i % 96;
    const hour = q / 4;

    if (hour >= 6 && hour <= 18) {
      pv[i] = 8.5 * Math.max(0, Math.sin((hour - 6) / 12 * Math.PI));
    }

    if (q === 28) load[i] = 0.35;
    else if (q === 80) load[i] = 0.28;
  }

  const exportPower = Float64Array.from(
    pv,
    power => Math.max(0, power - 2),
  );

  const storage = {
    storageEnergyPerKelvin: 3000 * 4.186 / 3600,
    storageConductanceWPerKelvin: 3.617,
    storageMinTemperature: 60,
    ambientTemperature: 20,
    dailyHeatLoss: -1,
  };

  const ns = {
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

  const causal = {
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

  // Direct package-loader smoke test first.
  packageBinding.clearAccelerationScopeCache();

  const nsScope = packageBinding.registerAccelerationScope(
    source,
    "rust_noSunshine",
  );
  const causalScope = packageBinding.registerAccelerationScope(
    source,
    "rust_peakShavingCausal",
  );

  const nsData = packageBinding.registerAccelerationDataPair(
    pv,
    load,
  );
  const causalData = packageBinding.registerAccelerationDataPair(
    exportPower,
    load,
  );

  const nsPacked = new Float64Array([
    ns.currentIndex,
    ns.horizonSteps,
    ns.protectedSteps,
    ns.timeStepHours,
    ns.householdReservePower,
    ns.maximumHeatPumpElectricalPower,
    ns.cop,
    ns.initialWaterlinePower,
    ns.targetStorageEnergy,
    ns.currentStorageEnergy,
    ns.storageEnergyPerKelvin,
    ns.storageConductanceWPerKelvin,
    ns.storageMinTemperature,
    ns.ambientTemperature,
    ns.dailyHeatLoss,
  ]);
  const nsOut = new Float64Array(11);

  const nsDirectReference = packageBinding.runNoSunshineKernelContext(
    nsScope,
    nsData,
    ns,
  );

  packageBinding.runNoSunshineKernelContextPackedInto(
    nsScope,
    nsData,
    nsPacked,
    nsOut,
  );

  const decodeNS = output => ({
    kWhNoSunshine: output[0],
    bucketDemandEnergy: output[1],
    bucketCoolingLossEnergy: output[2],
    preBucketHoldingLossEnergy: output[3],
    allocatedPvThermalEnergy: output[4],
    unavoidableThermalDeficit: output[5],
    requiredStorageEnergyAtStart: output[6],
    additionalStorageEnergyNeededAtStart: output[7],
    currentWaterlinePower: output[8],
    currentScheduledHpElectricalPower: output[9],
    currentScheduledHpThermalPower: output[10],
  });

  assertSame(
    "direct-package-noSunshine",
    nsDirectReference,
    decodeNS(nsOut),
  );

  const causalPacked = new Float64Array([
    causal.currentIndex,
    causal.horizonSteps,
    causal.currentStorageEnergy,
    causal.minimumReserveEnergy,
    causal.targetStorageEnergy,
    causal.maximumHeatPumpElectricalPower,
    causal.cop,
    causal.energyCutLevel,
    causal.timeStepHours,
    causal.storageEnergyPerKelvin,
    causal.storageConductanceWPerKelvin,
    causal.storageMinTemperature,
    causal.ambientTemperature,
    causal.dailyHeatLoss,
  ]);
  const causalOut = new Float64Array(5);

  const causalDirectReference =
    packageBinding.runPeakShavingCausalKernelContext(
      causalScope,
      causalData,
      causal,
    );

  packageBinding.runPeakShavingCausalKernelContextPackedInto(
    causalScope,
    causalData,
    causalPacked,
    causalOut,
  );

  const decodeCausal = output => ({
    cutLevelPower: output[0],
    normalPlanSurvives: output[1] !== 0,
    maximumPvPlanSurvives: output[2] !== 0,
    survivalChecks: output[3],
    binarySearchIterations: output[4],
  });

  assertSame(
    "direct-package-causal",
    causalDirectReference,
    decodeCausal(causalOut),
  );

  // Now the TypeScript runtime wrapper.
  runtime.clearNativeAccelerationCache();

  const runtimeNsPacked = new Float64Array(15);
  const runtimeNsOut = new Float64Array(11);

  function fillNS() {
    runtimeNsPacked.set(nsPacked);
  }

  function nsReference() {
    const result = runtime.tryRunNativeScope({
      scopeId: "rust_noSunshine",
      exportName: "runNoSunshineKernel",
      args: [pv, load, ns],
    });

    if (result === null)
      die(
        "runtime-reference-noSunshine",
        "Reference tryRunNativeScope returned null",
        JSON.stringify(runtime.getNativeScopeStatus("rust_noSunshine"), null, 2),
      );

    return result;
  }

  function nsCandidate() {
    fillNS();

    const result = runtime.tryRunNativeScopePackedInto({
      scopeId: "rust_noSunshine",
      firstStaticArray: pv,
      secondStaticArray: load,
      packedInput: runtimeNsPacked,
      outputBuffer: runtimeNsOut,
      decodeResult: decodeNS,
    });

    if (result === null)
      die(
        "runtime-a6-noSunshine",
        "A6 packed-into runtime returned null",
        JSON.stringify(runtime.getNativeScopeStatus("rust_noSunshine"), null, 2),
      );

    return result;
  }

  const runtimeCPacked = new Float64Array(14);
  const runtimeCOut = new Float64Array(5);

  function fillCausal() {
    runtimeCPacked.set(causalPacked);
  }

  function causalReference() {
    const result = runtime.tryRunNativeScope({
      scopeId: "rust_peakShavingCausal",
      exportName: "runPeakShavingCausalKernel",
      args: [exportPower, load, causal],
    });

    if (result === null)
      die(
        "runtime-reference-causal",
        "Reference tryRunNativeScope returned null",
        JSON.stringify(runtime.getNativeScopeStatus("rust_peakShavingCausal"), null, 2),
      );

    return result;
  }

  function causalCandidate() {
    fillCausal();

    const result = runtime.tryRunNativeScopePackedInto({
      scopeId: "rust_peakShavingCausal",
      firstStaticArray: exportPower,
      secondStaticArray: load,
      packedInput: runtimeCPacked,
      outputBuffer: runtimeCOut,
      decodeResult: decodeCausal,
    });

    if (result === null)
      die(
        "runtime-a6-causal",
        "A6 packed-into runtime returned null",
        JSON.stringify(runtime.getNativeScopeStatus("rust_peakShavingCausal"), null, 2),
      );

    return result;
  }

  const comparisons = {
    realisticNoSunshine:
      compare(
        "runtime-noSunshine",
        nsReference,
        nsCandidate,
        5000,
        () => runtime.getNativeScopeStatus("rust_noSunshine"),
      ),

    realisticCausal:
      compare(
        "runtime-causal",
        causalReference,
        causalCandidate,
        10000,
        () => runtime.getNativeScopeStatus("rust_peakShavingCausal"),
      ),
  };

  const accepted =
    Object.values(comparisons).every(
      result => result.conclusion === "A6_FASTER",
    );

  const release = fs.existsSync(hashFile)
    ? fs.readFileSync(hashFile, "utf8").trim()
    : "unknown";

  const report = {
    stage: "APPROACH_A_A6_RUNTIME_PACKED_INTO",
    runnerRevision: "A6.1_DIAGNOSTIC",
    release,
    integration: "A4 packed input + H5 caller-owned output + safe JS decode",
    adapterGate: "SUNOPT_A6_PACKED_INTO=1",
    exportDiagnostic,
    parity: {
      existingTests: "PASS",
      directPackageBinding: "EXACT",
      runtimeComparisons: "EXACT",
    },
    measurementPolicy: {
      measuredRounds: 5,
      medianAuthoritative: true,
      noiseThresholdPercent: 10,
    },
    comparisons,
    decision:
      accepted
        ? "ENABLE_A6_DEFAULT_CANDIDATE"
        : "KEEP_A6_EXPERIMENTAL",
    nextCandidate:
      accepted
        ? "Run full production annual/end-to-end A/B with SUNOPT_A6_PACKED_INTO=0 vs 1 before making A6 default."
        : "Do not default-enable A6; inspect per-workload result.",
  };

  const reportPath = path.join(
    outbox,
    "approach-a-a6-runtime-packed-into.json",
  );

  fs.writeFileSync(
    reportPath,
    JSON.stringify(report, null, 2) + "\n",
    "utf8",
  );

  const llm = path.join(outbox, "llm.txt");

  fs.appendFileSync(
    llm,
    [
      "",
      `=== ${new Date().toISOString()} APPROACH_A_A6_RUNTIME_PACKED_INTO_A6_1 ===`,
      `release=${release}`,
      "parity existing=PASS directPackageBinding=EXACT runtimeComparisons=EXACT",
      ...Object.entries(comparisons).map(
        ([name, value]) =>
          `${name} refUs=${value.reference.medianPerCallUs.toFixed(6)} A6Us=${value.candidate.medianPerCallUs.toFixed(6)} speedup=${value.speedup.toFixed(4)}x deltaPct=${value.deltaPct.toFixed(3)} conclusion=${value.conclusion}`
      ),
      `decision=${report.decision}`,
      `nextCandidate=${report.nextCandidate}`,
      `report=${reportPath}`,
      "",
    ].join("\n"),
    "utf8",
  );

  console.log(JSON.stringify(report, null, 2));
  console.log(`report=${reportPath}`);
  console.log("APPROACH_A_A6_RUNTIME_PACKED_INTO_A6_1=OK");
}

main().catch(error => {
  if (!fs.existsSync(failedRun))
    writeFailure("main", error);

  console.error(error);
  process.exit(1);
});
