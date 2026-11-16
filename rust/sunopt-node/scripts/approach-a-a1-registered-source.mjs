// rust/sunopt-node/scripts/approach-a-a1-registered-source.mjs
//
// Approach A — A1 registered-source test.
//
// One change only:
//   source + scopeId is registered once and native hot-path execution uses a
//   small u32 handle.
//
// Automatically:
//   1. rebuild sunopt-node
//   2. run existing parity + global-scope tests
//   3. compare old source-based path vs registered-handle path
//      - exact result parity
//      - 5 measured rounds
//      - ABBA-style alternating order
//      - median authoritative
//      - <10% = inconclusive
//   4. exercise the real TypeScript runtime and prove it uses the registered
//      exports successfully.
//
// Output:
//   updates/outbox/approach-a-a1-registered-source.json
//   updates/outbox/approach-a-a1-build.txt
//   updates/outbox/approach-a-a1-parity.txt
//   updates/outbox/approach-a-a1-scopes.txt
//   updates/outbox/llm.txt
//
// Failure:
//   updates/outbox/failed-build.txt
//   updates/outbox/failed-run.txt

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL, fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const here =
  path.dirname(
    fileURLToPath(
      import.meta.url
    )
  );

const nativeRoot =
  path.resolve(
    here,
    ".."
  );

const projectRoot =
  path.resolve(
    nativeRoot,
    "..",
    ".."
  );

const outbox =
  path.join(
    projectRoot,
    "updates",
    "outbox"
  );

const sourcePath =
  path.join(
    projectRoot,
    "src",
    "simulation",
    "rust-acceleration-scopes.ts"
  );

const runtimePath =
  path.join(
    projectRoot,
    "src",
    "simulation",
    "rust-acceleration-runtime.ts"
  );

const hashFile =
  path.join(
    projectRoot,
    "updates",
    "history",
    "hash.txt"
  );

const failedBuild =
  path.join(
    outbox,
    "failed-build.txt"
  );

const failedRun =
  path.join(
    outbox,
    "failed-run.txt"
  );

fs.mkdirSync(
  outbox,
  {
    recursive: true,
  }
);

for (
  const file
  of [
    failedBuild,
    failedRun,
  ]
) {
  try {
    fs.unlinkSync(file);
  } catch {}
}

function writeFailure(
  file,
  stage,
  error,
  extra = ""
) {
  fs.writeFileSync(
    file,
    [
      `stage=${stage}`,
      `time=${new Date().toISOString()}`,
      `error=${error?.stack ?? error}`,
      extra,
      "",
    ].join("\n"),
    "utf8"
  );
}

function failRun(
  message,
  extra = ""
) {
  writeFailure(
    failedRun,
    "approach-a-a1",
    message,
    extra
  );

  throw new Error(
    message
  );
}

function median(values) {
  const sorted =
    [...values]
      .sort(
        (a, b) =>
          a - b
      );

  const middle =
    Math.floor(
      sorted.length / 2
    );

  return sorted.length % 2
    ? sorted[middle]
    : (
        sorted[middle - 1] +
        sorted[middle]
      ) / 2;
}

function buildNative() {
  const comspec =
    process.env.ComSpec ||
    process.env.COMSPEC ||
    "C:\\Windows\\System32\\cmd.exe";

  const result =
    spawnSync(
      comspec,
      [
        "/d",
        "/s",
        "/c",
        "npm run build",
      ],
      {
        cwd:
          nativeRoot,

        encoding:
          "utf8",

        windowsHide:
          true,

        env: {
          ...process.env,
          SUNOPT_CUDA_FAST_MATH:
            "0",
        },
      }
    );

  const log =
    path.join(
      outbox,
      "approach-a-a1-build.txt"
    );

  fs.writeFileSync(
    log,
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
    "utf8"
  );

  if (
    result.error ||
    result.status !== 0
  ) {
    writeFailure(
      failedBuild,
      "approach-a-a1-build",
      result.error ??
        `npm run build exited with status ${result.status}`,
      [
        result.stdout ?? "",
        result.stderr ?? "",
      ].join("\n")
    );

    process.exit(2);
  }
}

function runExistingTest(
  relative,
  outputName
) {
  const result =
    spawnSync(
      process.execPath,
      [
        path.join(
          projectRoot,
          relative
        ),
      ],
      {
        cwd:
          projectRoot,

        encoding:
          "utf8",

        windowsHide:
          true,
      }
    );

  fs.writeFileSync(
    path.join(
      outbox,
      outputName
    ),
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
    "utf8"
  );

  if (
    result.error ||
    result.status !== 0
  ) {
    failRun(
      `Existing test failed after A1: ${relative}`,
      [
        `status=${result.status}`,
        result.stdout ?? "",
        result.stderr ?? "",
      ].join("\n")
    );
  }
}

function timedLoop(
  fn,
  repetitions
) {
  const started =
    performance.now();

  let last;

  for (
    let index = 0;
    index < repetitions;
    index += 1
  ) {
    last =
      fn();
  }

  const totalMs =
    performance.now() -
    started;

  return {
    repetitions,
    totalMs,
    perCallUs:
      totalMs *
      1000 /
      repetitions,
    last,
  };
}

function summarize(
  rows
) {
  const values =
    rows.map(
      row =>
        row.perCallUs
    );

  return {
    repetitionsPerRound:
      rows[0]
        ?.repetitions ??
      0,

    perCallUs:
      values,

    medianPerCallUs:
      median(
        values
      ),

    minPerCallUs:
      Math.min(
        ...values
      ),

    maxPerCallUs:
      Math.max(
        ...values
      ),
  };
}

function assertSame(
  label,
  a,
  b
) {
  const keys =
    Object.keys(a);

  if (
    keys.length !==
    Object.keys(b).length
  ) {
    failRun(
      `${label}: result shape differs`
    );
  }

  for (
    const key
    of keys
  ) {
    const av =
      a[key];

    const bv =
      b[key];

    if (
      typeof av ===
        "number" &&
      typeof bv ===
        "number"
    ) {
      const diff =
        Math.abs(
          av - bv
        );

      if (
        !Number.isFinite(av) ||
        !Number.isFinite(bv) ||
        diff > 1e-12
      ) {
        failRun(
          `${label}.${key}: legacy=${av} registered=${bv} diff=${diff}`
        );
      }
    } else if (
      av !== bv
    ) {
      failRun(
        `${label}.${key}: legacy=${String(av)} registered=${String(bv)}`
      );
    }
  }
}

function compareVariants(
  legacy,
  registered,
  repetitions
) {
  /*
   * Both variants are already compiled and registered.
   * Warm each immediately before the measured alternating sequence.
   */
  timedLoop(
    legacy,
    Math.min(
      1000,
      repetitions
    )
  );

  timedLoop(
    registered,
    Math.min(
      1000,
      repetitions
    )
  );

  const rows = {
    legacy: [],
    registered: [],
  };

  const order = [
    "legacy",
    "registered",
    "registered",
    "legacy",
    "legacy",
    "registered",
    "registered",
    "legacy",
    "legacy",
    "registered",
  ];

  for (
    const variant
    of order
  ) {
    rows[
      variant
    ].push(
      timedLoop(
        variant ===
          "legacy"
          ? legacy
          : registered,
        repetitions
      )
    );
  }

  const legacySummary =
    summarize(
      rows.legacy
    );

  const registeredSummary =
    summarize(
      rows.registered
    );

  const deltaPct =
    100 *
    (
      registeredSummary
        .medianPerCallUs -
      legacySummary
        .medianPerCallUs
    ) /
    legacySummary
      .medianPerCallUs;

  const speedup =
    legacySummary
      .medianPerCallUs /
    registeredSummary
      .medianPerCallUs;

  const conclusion =
    Math.abs(
      deltaPct
    ) < 10
      ? "INCONCLUSIVE_LT_10_PERCENT"
      : (
          deltaPct < 0
            ? "REGISTERED_FASTER"
            : "LEGACY_FASTER"
        );

  return {
    legacy:
      legacySummary,

    registered:
      registeredSummary,

    deltaPct,
    speedup,
    conclusion,
  };
}

buildNative();

runExistingTest(
  "scripts/test-global-acceleration-parity.mjs",
  "approach-a-a1-parity.txt"
);

runExistingTest(
  "scripts/test-global-acceleration-scopes.mjs",
  "approach-a-a1-scopes.txt"
);

const require =
  createRequire(
    import.meta.url
  );

let native;

try {
  native =
    require(
      path.join(
        nativeRoot,
        "sunopt_node.win32-x64-msvc.node"
      )
    );
} catch (error) {
  failRun(
    "Unable to load rebuilt sunopt-node binding.",
    error?.stack ?? String(error)
  );
}

for (
  const exportName
  of [
    "registerAccelerationScope",
    "runNoSunshineKernelRegistered",
    "runPeakShavingCausalKernelRegistered",
    "runNoSunshineKernel",
    "runPeakShavingCausalKernel",
    "clearAccelerationScopeCache",
  ]
) {
  if (
    typeof native[
      exportName
    ] !==
    "function"
  ) {
    failRun(
      `Missing A1 native export after build: ${exportName}`
    );
  }
}

const source =
  fs.readFileSync(
    sourcePath,
    "utf8"
  );

const length =
  35_040;

const pvPower =
  new Float64Array(
    length
  );

const loadEnergy =
  new Float64Array(
    length
  );

for (
  let index = 0;
  index < length;
  index += 1
) {
  const quarter =
    index % 96;

  const hour =
    quarter / 4;

  if (
    hour >= 6 &&
    hour <= 18
  ) {
    const phase =
      (
        hour - 6
      ) /
      12 *
      Math.PI;

    pvPower[index] =
      8.5 *
      Math.max(
        0,
        Math.sin(
          phase
        )
      );
  }

  if (
    quarter === 28
  ) {
    loadEnergy[index] =
      0.35;
  } else if (
    quarter === 80
  ) {
    loadEnergy[index] =
      0.28;
  }
}

const exportPower =
  Float64Array.from(
    pvPower,
    power =>
      Math.max(
        0,
        power - 2
      )
  );

const commonStorage = {
  storageEnergyPerKelvin:
    3000 *
    4.186 /
    3600,

  storageConductanceWPerKelvin:
    3.617,

  storageMinTemperature:
    60,

  ambientTemperature:
    20,

  dailyHeatLoss:
    -1,
};

const noSunshineReal = {
  currentIndex:
    173,
  horizonSteps:
    672,
  protectedSteps:
    192,
  timeStepHours:
    0.25,
  householdReservePower:
    2,
  maximumHeatPumpElectricalPower:
    5 / 3,
  cop:
    3,
  initialWaterlinePower:
    10,
  targetStorageEnergy:
    52.325,
  currentStorageEnergy:
    10,
  ...commonStorage,
};

const noSunshineMicro = {
  ...noSunshineReal,
  horizonSteps:
    1,
  protectedSteps:
    1,
};

const causalReal = {
  currentIndex:
    173,
  horizonSteps:
    192,
  currentStorageEnergy:
    10,
  minimumReserveEnergy:
    2,
  targetStorageEnergy:
    52.325,
  maximumHeatPumpElectricalPower:
    5 / 3,
  cop:
    3,
  energyCutLevel:
    4,
  timeStepHours:
    0.25,
  ...commonStorage,
};

native
  .clearAccelerationScopeCache();

const noSunshineHandle =
  native
    .registerAccelerationScope(
      source,
      "rust_noSunshine"
    );

const causalHandle =
  native
    .registerAccelerationScope(
      source,
      "rust_peakShavingCausal"
    );

if (
  !Number.isInteger(
    noSunshineHandle
  ) ||
  !Number.isInteger(
    causalHandle
  )
) {
  failRun(
    `Invalid A1 handles: noSunshine=${String(noSunshineHandle)} causal=${String(causalHandle)}`
  );
}

const legacyMicro =
  native
    .runNoSunshineKernel(
      source,
      "rust_noSunshine",
      pvPower,
      loadEnergy,
      noSunshineMicro
    );

const registeredMicro =
  native
    .runNoSunshineKernelRegistered(
      noSunshineHandle,
      pvPower,
      loadEnergy,
      noSunshineMicro
    );

assertSame(
  "noSunshine.micro",
  legacyMicro,
  registeredMicro
);

const legacyNoSunshine =
  native
    .runNoSunshineKernel(
      source,
      "rust_noSunshine",
      pvPower,
      loadEnergy,
      noSunshineReal
    );

const registeredNoSunshine =
  native
    .runNoSunshineKernelRegistered(
      noSunshineHandle,
      pvPower,
      loadEnergy,
      noSunshineReal
    );

assertSame(
  "noSunshine.real",
  legacyNoSunshine,
  registeredNoSunshine
);

const legacyCausal =
  native
    .runPeakShavingCausalKernel(
      source,
      "rust_peakShavingCausal",
      exportPower,
      loadEnergy,
      causalReal
    );

const registeredCausal =
  native
    .runPeakShavingCausalKernelRegistered(
      causalHandle,
      exportPower,
      loadEnergy,
      causalReal
    );

assertSame(
  "causal.real",
  legacyCausal,
  registeredCausal
);

const microComparison =
  compareVariants(
    () =>
      native
        .runNoSunshineKernel(
          source,
          "rust_noSunshine",
          pvPower,
          loadEnergy,
          noSunshineMicro
        ),

    () =>
      native
        .runNoSunshineKernelRegistered(
          noSunshineHandle,
          pvPower,
          loadEnergy,
          noSunshineMicro
        ),

    10_000
  );

const noSunshineComparison =
  compareVariants(
    () =>
      native
        .runNoSunshineKernel(
          source,
          "rust_noSunshine",
          pvPower,
          loadEnergy,
          noSunshineReal
        ),

    () =>
      native
        .runNoSunshineKernelRegistered(
          noSunshineHandle,
          pvPower,
          loadEnergy,
          noSunshineReal
        ),

    1000
  );

const causalComparison =
  compareVariants(
    () =>
      native
        .runPeakShavingCausalKernel(
          source,
          "rust_peakShavingCausal",
          exportPower,
          loadEnergy,
          causalReal
        ),

    () =>
      native
        .runPeakShavingCausalKernelRegistered(
          causalHandle,
          exportPower,
          loadEnergy,
          causalReal
        ),

    1000
  );

/*
 * Exercise the actual TypeScript runtime. It should now choose the registered
 * hot path automatically.
 */
let runtimeSmoke = {
  ok: false,
  result:
    null,
};

try {
  const runtime =
    await import(
      pathToFileURL(
        runtimePath
      ).href +
      `?a1=${Date.now()}`
    );

  runtime
    .clearNativeAccelerationCache();

  const value =
    runtime
      .tryRunNativeScope({
        scopeId:
          "rust_noSunshine",

        exportName:
          "runNoSunshineKernel",

        args: [
          pvPower,
          loadEnergy,
          noSunshineReal,
        ],
      });

  if (
    value ===
    null
  ) {
    failRun(
      "A1 TypeScript runtime smoke test fell back instead of executing native registered path."
    );
  }

  runtimeSmoke = {
    ok: true,
    result:
      value,

    status:
      runtime
        .getNativeScopeStatus(
          "rust_noSunshine"
        ),
  };
} catch (error) {
  failRun(
    "A1 TypeScript runtime smoke test failed.",
    error?.stack ?? String(error)
  );
}

const release =
  fs.existsSync(
    hashFile
  )
    ? fs
        .readFileSync(
          hashFile,
          "utf8"
        )
        .trim()
    : "unknown";

const allMeaningfullyFaster =
  [
    microComparison,
    noSunshineComparison,
    causalComparison,
  ].every(
    result =>
      result.conclusion ===
      "REGISTERED_FASTER"
  );

const report = {
  stage:
    "APPROACH_A_A1_REGISTERED_SOURCE",

  release,

  change:
    "Register TypeScript source + scope once; execute hot-path calls by u32 native handle.",

  measurementPolicy: {
    measuredRounds:
      5,
    medianAuthoritative:
      true,
    noiseThresholdPercent:
      10,
    order:
      "ABBA alternating",
  },

  parity: {
    existingTests:
      "PASS",
    directRegisteredVsLegacy:
      "EXACT",
    runtimeSmoke:
      runtimeSmoke.ok,
  },

  handles: {
    noSunshine:
      noSunshineHandle,
    causal:
      causalHandle,
  },

  comparisons: {
    microNoSunshine:
      microComparison,
    realisticNoSunshine:
      noSunshineComparison,
    realisticCausal:
      causalComparison,
  },

  decision:
    allMeaningfullyFaster
      ? "KEEP_A1"
      : (
          [
            microComparison,
            noSunshineComparison,
            causalComparison,
          ].some(
            result =>
              result.conclusion ===
              "REGISTERED_FASTER"
          )
            ? "KEEP_FOR_RELEVANT_HOT_PATHS_AND_REVIEW"
            : "DO_NOT_KEEP_A1"
        ),

  nextCandidate:
    "Only after A1 decision: A2 persistent large numeric inputs / execution context. Do not combine with A1 measurement.",
};

const reportPath =
  path.join(
    outbox,
    "approach-a-a1-registered-source.json"
  );

fs.writeFileSync(
  reportPath,
  JSON.stringify(
    report,
    null,
    2
  ) + "\n",
  "utf8"
);

const llmPath =
  path.join(
    outbox,
    "llm.txt"
  );

fs.appendFileSync(
  llmPath,
  [
    "",
    `=== ${new Date().toISOString()} APPROACH_A_A1_REGISTERED_SOURCE ===`,
    `release=${release}`,
    `parity existing=PASS direct=EXACT runtimeSmoke=${runtimeSmoke.ok}`,
    `micro legacyUs=${microComparison.legacy.medianPerCallUs.toFixed(6)} registeredUs=${microComparison.registered.medianPerCallUs.toFixed(6)} speedup=${microComparison.speedup.toFixed(4)}x deltaPct=${microComparison.deltaPct.toFixed(3)} conclusion=${microComparison.conclusion}`,
    `noSunshine legacyUs=${noSunshineComparison.legacy.medianPerCallUs.toFixed(6)} registeredUs=${noSunshineComparison.registered.medianPerCallUs.toFixed(6)} speedup=${noSunshineComparison.speedup.toFixed(4)}x deltaPct=${noSunshineComparison.deltaPct.toFixed(3)} conclusion=${noSunshineComparison.conclusion}`,
    `causal legacyUs=${causalComparison.legacy.medianPerCallUs.toFixed(6)} registeredUs=${causalComparison.registered.medianPerCallUs.toFixed(6)} speedup=${causalComparison.speedup.toFixed(4)}x deltaPct=${causalComparison.deltaPct.toFixed(3)} conclusion=${causalComparison.conclusion}`,
    `decision=${report.decision}`,
    `nextCandidate=${report.nextCandidate}`,
    `report=${reportPath}`,
    "",
  ].join("\n"),
  "utf8"
);

console.log(
  JSON.stringify(
    report,
    null,
    2
  )
);

console.log(
  `report=${reportPath}`
);

console.log(
  `llm=${llmPath}`
);

console.log(
  "APPROACH_A_A1_REGISTERED_SOURCE=OK"
);
