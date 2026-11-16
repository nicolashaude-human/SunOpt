// rust/sunopt-node/scripts/approach-a-a2-persistent-input-context.mjs
//
// Approach A — A2 persistent static numeric inputs.
//
// Only new idea relative to A1:
//   the two invariant Float64Array inputs are copied once to native-owned
//   storage and referenced by a small data handle.
//
// Dynamic parameters still cross N-API on every call.
//
// Measures A1 registered-source path against A2 context path:
//   - exact output parity
//   - micro NoSunshine
//   - realistic NoSunshine
//   - realistic Causal
//   - 5 rounds, alternating order, median authoritative
//   - <10% = inconclusive
//
// Also smoke-tests the real TypeScript runtime, which now prefers A2 when the
// first two numerical args are stable object identities.

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
    "approach-a-a2",
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

  fs.writeFileSync(
    path.join(
      outbox,
      "approach-a-a2-build.txt"
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
    writeFailure(
      failedBuild,
      "approach-a-a2-build",
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
      `Existing test failed after A2: ${relative}`,
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
  const start =
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
    start;

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
      rows[0]?.repetitions ?? 0,
    perCallUs:
      values,
    medianPerCallUs:
      median(values),
    minPerCallUs:
      Math.min(...values),
    maxPerCallUs:
      Math.max(...values),
  };
}

function compareVariants(
  a1,
  a2,
  repetitions
) {
  timedLoop(
    a1,
    Math.min(
      repetitions,
      1000
    )
  );

  timedLoop(
    a2,
    Math.min(
      repetitions,
      1000
    )
  );

  const rows = {
    a1: [],
    a2: [],
  };

  const order = [
    "a1",
    "a2",
    "a2",
    "a1",
    "a1",
    "a2",
    "a2",
    "a1",
    "a1",
    "a2",
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
          "a1"
          ? a1
          : a2,
        repetitions
      )
    );
  }

  const a1Summary =
    summarize(
      rows.a1
    );

  const a2Summary =
    summarize(
      rows.a2
    );

  const deltaPct =
    100 *
    (
      a2Summary
        .medianPerCallUs -
      a1Summary
        .medianPerCallUs
    ) /
    a1Summary
      .medianPerCallUs;

  return {
    a1:
      a1Summary,

    a2:
      a2Summary,

    deltaPct,

    speedup:
      a1Summary
        .medianPerCallUs /
      a2Summary
        .medianPerCallUs,

    conclusion:
      Math.abs(
        deltaPct
      ) < 10
        ? "INCONCLUSIVE_LT_10_PERCENT"
        : (
            deltaPct < 0
              ? "A2_FASTER"
              : "A1_FASTER"
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
          `${label}.${key}: A1=${av} A2=${bv} diff=${diff}`
        );
      }
    } else if (
      av !== bv
    ) {
      failRun(
        `${label}.${key}: A1=${String(av)} A2=${String(bv)}`
      );
    }
  }
}

buildNative();

runExistingTest(
  "scripts/test-global-acceleration-parity.mjs",
  "approach-a-a2-parity.txt"
);

runExistingTest(
  "scripts/test-global-acceleration-scopes.mjs",
  "approach-a-a2-scopes.txt"
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
    "Unable to load rebuilt A2 native binding.",
    error?.stack ?? String(error)
  );
}

for (
  const exportName
  of [
    "registerAccelerationScope",
    "registerAccelerationDataPair",
    "runNoSunshineKernelRegistered",
    "runNoSunshineKernelContext",
    "runPeakShavingCausalKernelRegistered",
    "runPeakShavingCausalKernelContext",
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
      `Missing A2 native export: ${exportName}`
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

const noSunshineScopeHandle =
  native
    .registerAccelerationScope(
      source,
      "rust_noSunshine"
    );

const causalScopeHandle =
  native
    .registerAccelerationScope(
      source,
      "rust_peakShavingCausal"
    );

const noSunshineDataHandle =
  native
    .registerAccelerationDataPair(
      pvPower,
      loadEnergy
    );

const causalDataHandle =
  native
    .registerAccelerationDataPair(
      exportPower,
      loadEnergy
    );

const a1Micro =
  native
    .runNoSunshineKernelRegistered(
      noSunshineScopeHandle,
      pvPower,
      loadEnergy,
      noSunshineMicro
    );

const a2Micro =
  native
    .runNoSunshineKernelContext(
      noSunshineScopeHandle,
      noSunshineDataHandle,
      noSunshineMicro
    );

assertSame(
  "microNoSunshine",
  a1Micro,
  a2Micro
);

const a1NoSunshine =
  native
    .runNoSunshineKernelRegistered(
      noSunshineScopeHandle,
      pvPower,
      loadEnergy,
      noSunshineReal
    );

const a2NoSunshine =
  native
    .runNoSunshineKernelContext(
      noSunshineScopeHandle,
      noSunshineDataHandle,
      noSunshineReal
    );

assertSame(
  "realisticNoSunshine",
  a1NoSunshine,
  a2NoSunshine
);

const a1Causal =
  native
    .runPeakShavingCausalKernelRegistered(
      causalScopeHandle,
      exportPower,
      loadEnergy,
      causalReal
    );

const a2Causal =
  native
    .runPeakShavingCausalKernelContext(
      causalScopeHandle,
      causalDataHandle,
      causalReal
    );

assertSame(
  "realisticCausal",
  a1Causal,
  a2Causal
);

const comparisons = {
  microNoSunshine:
    compareVariants(
      () =>
        native
          .runNoSunshineKernelRegistered(
            noSunshineScopeHandle,
            pvPower,
            loadEnergy,
            noSunshineMicro
          ),

      () =>
        native
          .runNoSunshineKernelContext(
            noSunshineScopeHandle,
            noSunshineDataHandle,
            noSunshineMicro
          ),

      10_000
    ),

  realisticNoSunshine:
    compareVariants(
      () =>
        native
          .runNoSunshineKernelRegistered(
            noSunshineScopeHandle,
            pvPower,
            loadEnergy,
            noSunshineReal
          ),

      () =>
        native
          .runNoSunshineKernelContext(
            noSunshineScopeHandle,
            noSunshineDataHandle,
            noSunshineReal
          ),

      1000
    ),

  realisticCausal:
    compareVariants(
      () =>
        native
          .runPeakShavingCausalKernelRegistered(
            causalScopeHandle,
            exportPower,
            loadEnergy,
            causalReal
          ),

      () =>
        native
          .runPeakShavingCausalKernelContext(
            causalScopeHandle,
            causalDataHandle,
            causalReal
          ),

      1000
    ),
};

let runtimeSmoke = {
  ok: false,
};

try {
  const runtime =
    await import(
      pathToFileURL(
        runtimePath
      ).href +
      `?a2=${Date.now()}`
    );

  runtime
    .clearNativeAccelerationCache();

  const first =
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

  const second =
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
    first ===
      null ||
    second ===
      null
  ) {
    failRun(
      "A2 runtime smoke test fell back."
    );
  }

  runtimeSmoke = {
    ok: true,
    exact:
      JSON.stringify(first) ===
      JSON.stringify(second),
    status:
      runtime
        .getNativeScopeStatus(
          "rust_noSunshine"
        ),
  };
} catch (error) {
  failRun(
    "A2 TypeScript runtime smoke test failed.",
    error?.stack ?? String(error)
  );
}

const outcomes =
  Object.values(
    comparisons
  );

const decision =
  outcomes.every(
    value =>
      value.conclusion ===
      "A2_FASTER"
  )
    ? "KEEP_A2"
    : (
        outcomes.some(
          value =>
            value.conclusion ===
            "A2_FASTER"
        )
          ? "KEEP_FOR_RELEVANT_HOT_PATHS_AND_REVIEW"
          : "DO_NOT_KEEP_A2"
      );

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

const report = {
  stage:
    "APPROACH_A_A2_PERSISTENT_INPUT_CONTEXT",

  release,

  change:
    "Static numeric array pair registered once in native-owned memory; hot path uses scope handle + data handle + dynamic parameters.",

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
    directA1VsA2:
      "EXACT",
    runtimeSmoke:
      runtimeSmoke,
  },

  handles: {
    noSunshineScopeHandle,
    causalScopeHandle,
    noSunshineDataHandle,
    causalDataHandle,
  },

  comparisons,

  decision,

  nextCandidate:
    "Only after A2 decision: A3 batch interface / multiple dynamic calls per native boundary.",
};

const reportPath =
  path.join(
    outbox,
    "approach-a-a2-persistent-input-context.json"
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
    `=== ${new Date().toISOString()} APPROACH_A_A2_PERSISTENT_INPUT_CONTEXT ===`,
    `release=${release}`,
    `parity existing=PASS direct=EXACT runtimeSmoke=${runtimeSmoke.ok}`,
    ...Object.entries(
      comparisons
    ).map(
      ([name, value]) =>
        `${name} A1Us=${value.a1.medianPerCallUs.toFixed(6)} A2Us=${value.a2.medianPerCallUs.toFixed(6)} speedup=${value.speedup.toFixed(4)}x deltaPct=${value.deltaPct.toFixed(3)} conclusion=${value.conclusion}`
    ),
    `decision=${decision}`,
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
  "APPROACH_A_A2_PERSISTENT_INPUT_CONTEXT=OK"
);
