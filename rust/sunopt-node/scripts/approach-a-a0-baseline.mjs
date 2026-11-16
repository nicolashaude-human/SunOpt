// rust/sunopt-node/scripts/approach-a-a0-baseline.mjs
//
// Approach A — A0 baseline only.
// NO production source is modified.
//
// Measures:
//   - existing parity tests
//   - first-call/JIT cost (5 fresh-cache measurements per scope)
//   - cached realistic call time (5 measured rounds)
//   - source/N-API/hash overhead probe:
//       canonical 15k+ TypeScript source vs. a minimal source that passes
//       the current shape validator and therefore generates the same
//       hard-coded Cranelift kernel.
//     Both variants are warmed and cached before timing.
//   - native cache sizes
//
// Output:
//   updates/outbox/approach-a-a0-baseline.json
//   updates/outbox/approach-a-a0-parity.txt
//   updates/outbox/approach-a-a0-scopes.txt
//   updates/outbox/llm.txt
//
// Failure:
//   updates/outbox/failed-run.txt

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

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

const hashFile =
  path.join(
    projectRoot,
    "updates",
    "history",
    "hash.txt"
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

try {
  fs.unlinkSync(
    failedRun
  );
} catch {}

function fail(
  message,
  extra = ""
) {
  fs.writeFileSync(
    failedRun,
    [
      "stage=approach-a-a0",
      `time=${new Date().toISOString()}`,
      `error=${message}`,
      extra,
      "",
    ].join("\n"),
    "utf8"
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

function runExistingTest(
  scriptRelative,
  outputName
) {
  const result =
    spawnSync(
      process.execPath,
      [
        path.join(
          projectRoot,
          scriptRelative
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
    fail(
      `Existing A test failed: ${scriptRelative}`,
      [
        `status=${result.status}`,
        `spawnError=${result.error?.stack ?? ""}`,
        result.stdout ?? "",
        result.stderr ?? "",
      ].join("\n")
    );
  }

  return {
    script:
      scriptRelative,

    status:
      result.status,

    stdout:
      result.stdout ?? "",

    stderr:
      result.stderr ?? "",
  };
}

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
} catch (directError) {
  try {
    native =
      require(
        "sunopt-node"
      );
  } catch (packageError) {
    fail(
      "Unable to load sunopt-node native binding.",
      [
        "=== direct ===",
        directError?.stack ?? String(directError),
        "=== package ===",
        packageError?.stack ?? String(packageError),
      ].join("\n")
    );
  }
}

for (
  const exportName
  of [
    "runNoSunshineKernel",
    "runPeakShavingCausalKernel",
    "clearAccelerationScopeCache",
    "getAccelerationScopeCacheSize",
  ]
) {
  if (
    typeof native[
      exportName
    ] !==
    "function"
  ) {
    fail(
      `Missing native export: ${exportName}`
    );
  }
}

if (
  !fs.existsSync(
    sourcePath
  )
) {
  fail(
    `Canonical A source is missing: ${sourcePath}`
  );
}

const source =
  fs.readFileSync(
    sourcePath,
    "utf8"
  );

/*
 * IMPORTANT DIAGNOSTIC:
 *
 * Current sunopt-core only validates the scope shape, then invokes a
 * hard-coded Cranelift builder for the scope ID. Therefore this tiny source
 * is expected to compile to the SAME current native kernels.
 *
 * This is not an optimization and is never used by production.
 * It exists only to measure the repeated source-string transfer/hash cost.
 */
const minimalSource =
  [
    "export function a(){",
    "rust_noSunshine:{for(let i=0;i<1;i++){} }",
    "rust_peakShavingCausal:{let x=0;}",
    "}",
  ].join("");

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

const noSunshineFullParameters = {
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

const noSunshineMicroParameters = {
  ...noSunshineFullParameters,

  horizonSteps:
    1,

  protectedSteps:
    1,
};

const causalFullParameters = {
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

function runNoSunshine(
  sourceText,
  parameters =
    noSunshineFullParameters
) {
  return native
    .runNoSunshineKernel(
      sourceText,

      "rust_noSunshine",

      pvPower,

      loadEnergy,

      parameters
    );
}

function runCausal(
  sourceText,
  parameters =
    causalFullParameters
) {
  return native
    .runPeakShavingCausalKernel(
      sourceText,

      "rust_peakShavingCausal",

      exportPower,

      loadEnergy,

      parameters
    );
}

function timedOne(
  fn
) {
  const start =
    performance.now();

  const value =
    fn();

  return {
    ms:
      performance.now() -
      start,

    value,
  };
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

function assertResultsEqual(
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
    fail(
      `${label}: result key-count mismatch`
    );
  }

  for (
    const key
    of keys
  ) {
    if (
      typeof a[key] ===
      "number" &&
      typeof b[key] ===
      "number"
    ) {
      const diff =
        Math.abs(
          a[key] -
          b[key]
        );

      if (
        !Number.isFinite(
          a[key]
        ) ||
        !Number.isFinite(
          b[key]
        ) ||
        diff >
          1e-12
      ) {
        fail(
          `${label}: numeric mismatch at ${key}: A=${a[key]} B=${b[key]} diff=${diff}`
        );
      }
    } else if (
      a[key] !==
      b[key]
    ) {
      fail(
        `${label}: mismatch at ${key}: A=${a[key]} B=${b[key]}`
      );
    }
  }
}

function summarizeRounds(
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

function measureFive(
  fn,
  repetitions
) {
  /*
   * One explicit unmeasured warmup for this variant.
   */
  timedLoop(
    fn,
    Math.max(
      1,
      Math.min(
        250,
        Math.floor(
          repetitions / 10
        )
      )
    )
  );

  const rows = [];

  for (
    let index = 0;
    index < 5;
    index += 1
  ) {
    rows.push(
      timedLoop(
        fn,
        repetitions
      )
    );
  }

  return {
    rows,
    summary:
      summarizeRounds(
        rows
      ),
  };
}

function measureFirstCalls() {
  const noSunshine = [];
  const causal = [];

  /*
   * Alternate compile order to avoid systematically assigning the
   * first cold CPU state to the same scope.
   */
  const order = [
    "noSunshine",
    "causal",
    "causal",
    "noSunshine",
    "noSunshine",
    "causal",
    "causal",
    "noSunshine",
    "noSunshine",
    "causal",
  ];

  for (
    const name
    of order
  ) {
    native
      .clearAccelerationScopeCache();

    if (
      name ===
      "noSunshine"
    ) {
      noSunshine.push(
        timedOne(
          () =>
            runNoSunshine(
              source
            )
        ).ms
      );
    } else {
      causal.push(
        timedOne(
          () =>
            runCausal(
              source
            )
        ).ms
      );
    }
  }

  return {
    noSunshineMs:
      noSunshine,

    noSunshineMedianMs:
      median(
        noSunshine
      ),

    causalMs:
      causal,

    causalMedianMs:
      median(
        causal
      ),
  };
}

/*
 * Existing correctness gates first.
 */
const existingTests = {
  parity:
    runExistingTest(
      "scripts/test-global-acceleration-parity.mjs",
      "approach-a-a0-parity.txt"
    ),

  scopes:
    runExistingTest(
      "scripts/test-global-acceleration-scopes.mjs",
      "approach-a-a0-scopes.txt"
    ),
};

const firstCall =
  measureFirstCalls();

/*
 * Warm canonical source for both scopes.
 */
native
  .clearAccelerationScopeCache();

runNoSunshine(
  source
);

runCausal(
  source
);

const canonicalCacheSize =
  native
    .getAccelerationScopeCacheSize();

if (
  canonicalCacheSize !==
  2
) {
  fail(
    `Expected canonical cache size 2, got ${canonicalCacheSize}.`
  );
}

/*
 * Realistic cached-call baseline.
 */
const cachedRealistic = {
  noSunshine:
    measureFive(
      () =>
        runNoSunshine(
          source
        ),
      1000
    ),

  causal:
    measureFive(
      () =>
        runCausal(
          source
        ),
      1000
    ),
};

/*
 * Source transfer/hash diagnostic.
 *
 * Warm the minimal noSunshine source into a third cache entry.
 * From here on BOTH canonical and minimal calls are cache hits.
 */
const minimalResult =
  runNoSunshine(
    minimalSource,
    noSunshineMicroParameters
  );

const canonicalResult =
  runNoSunshine(
    source,
    noSunshineMicroParameters
  );

assertResultsEqual(
  "canonical-vs-minimal cached kernel",
  canonicalResult,
  minimalResult
);

const cacheSizeWithProbe =
  native
    .getAccelerationScopeCacheSize();

if (
  cacheSizeWithProbe !==
  3
) {
  fail(
    `Expected cache size 3 after minimal-source probe warmup, got ${cacheSizeWithProbe}.`
  );
}

/*
 * Warm both source variants immediately before measured ABBA sequence.
 */
timedLoop(
  () =>
    runNoSunshine(
      source,
      noSunshineMicroParameters
    ),
  1000
);

timedLoop(
  () =>
    runNoSunshine(
      minimalSource,
      noSunshineMicroParameters
    ),
  1000
);

const sourceProbe = {
  canonical:
    [],
  minimal:
    [],
};

const sourceOrder = [
  "canonical",
  "minimal",
  "minimal",
  "canonical",
  "canonical",
  "minimal",
  "minimal",
  "canonical",
  "canonical",
  "minimal",
];

for (
  const variant
  of sourceOrder
) {
  const measured =
    variant ===
    "canonical"
      ? timedLoop(
          () =>
            runNoSunshine(
              source,
              noSunshineMicroParameters
            ),
          10_000
        )
      : timedLoop(
          () =>
            runNoSunshine(
              minimalSource,
              noSunshineMicroParameters
            ),
          10_000
        );

  sourceProbe[
    variant
  ].push(
    measured
  );
}

const canonicalSummary =
  summarizeRounds(
    sourceProbe.canonical
  );

const minimalSummary =
  summarizeRounds(
    sourceProbe.minimal
  );

const sourceOverheadDeltaPct =
  100 *
  (
    canonicalSummary
      .medianPerCallUs -
    minimalSummary
      .medianPerCallUs
  ) /
  canonicalSummary
    .medianPerCallUs;

const sourceProbeConclusion =
  Math.abs(
    sourceOverheadDeltaPct
  ) < 10
    ? "INCONCLUSIVE_LT_10_PERCENT"
    : (
        sourceOverheadDeltaPct >
        0
          ? "CANONICAL_SOURCE_PATH_MEASURABLY_SLOWER"
          : "MINIMAL_SOURCE_PATH_SLOWER"
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
    "APPROACH_A_A0_BASELINE",

  release,

  measurementPolicy: {
    measuredRounds:
      5,

    medianAuthoritative:
      true,

    noiseThresholdPercent:
      10,

    sourceProbeOrder:
      "ABBA alternating",

    sourceProbeBothVariantsCached:
      true,
  },

  source: {
    path:
      sourcePath,

    canonicalBytes:
      Buffer.byteLength(
        source,
        "utf8"
      ),

    minimalProbeBytes:
      Buffer.byteLength(
        minimalSource,
        "utf8"
      ),

    note:
      "Minimal source is diagnostic only. Current compiler validates scope shape then calls the same hard-coded Cranelift builder.",
  },

  existingTests: {
    parityStatus:
      existingTests
        .parity
        .status,

    scopesStatus:
      existingTests
        .scopes
        .status,
  },

  firstCall,

  cachedRealistic: {
    noSunshine:
      cachedRealistic
        .noSunshine
        .summary,

    causal:
      cachedRealistic
        .causal
        .summary,
  },

  sourceTransferHashProbe: {
    cacheSize:
      cacheSizeWithProbe,

    canonical:
      canonicalSummary,

    minimal:
      minimalSummary,

    canonicalMinusMinimalPct:
      sourceOverheadDeltaPct,

    conclusion:
      sourceProbeConclusion,

    exactResultParity:
      true,
  },

  canonicalCacheSize,

  nextDecision:
    sourceProbeConclusion ===
      "CANONICAL_SOURCE_PATH_MEASURABLY_SLOWER"
      ? "A1 candidate: register/compile source once and execute by stable native handle/cache key without resending/re-hashing the full TypeScript source."
      : "Do not optimize source registration yet; inspect next runtime cost centre.",
};

const reportPath =
  path.join(
    outbox,
    "approach-a-a0-baseline.json"
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
    `=== ${new Date().toISOString()} APPROACH_A_A0_BASELINE ===`,
    `release=${release}`,
    `canonicalSourceBytes=${report.source.canonicalBytes} minimalProbeBytes=${report.source.minimalProbeBytes}`,
    `existingParity=${report.existingTests.parityStatus} existingScopes=${report.existingTests.scopesStatus}`,
    `firstCall noSunshineMedianMs=${firstCall.noSunshineMedianMs.toFixed(6)} causalMedianMs=${firstCall.causalMedianMs.toFixed(6)}`,
    `cachedRealistic noSunshineMedianUs=${report.cachedRealistic.noSunshine.medianPerCallUs.toFixed(6)} causalMedianUs=${report.cachedRealistic.causal.medianPerCallUs.toFixed(6)}`,
    `sourceProbe canonicalMedianUs=${canonicalSummary.medianPerCallUs.toFixed(6)} minimalMedianUs=${minimalSummary.medianPerCallUs.toFixed(6)} canonicalMinusMinimalPct=${sourceOverheadDeltaPct.toFixed(3)} conclusion=${sourceProbeConclusion}`,
    `nextDecision=${report.nextDecision}`,
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
  "APPROACH_A_A0_BASELINE=OK"
);
