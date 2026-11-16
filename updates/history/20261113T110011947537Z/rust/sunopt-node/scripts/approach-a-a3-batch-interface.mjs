// rust/sunopt-node/scripts/approach-a-a3-batch-interface.mjs
//
// Approach A — A3 independent-call batch interface.
//
// Baseline:
//   A2 = scope handle + persistent static input context + one N-API call
//        per logical dynamic invocation.
//
// Candidate:
//   A3 = same compiled kernel and same persistent input context, but multiple
//        independent parameter objects are evaluated inside one N-API call.
//
// IMPORTANT:
//   This does NOT claim that sequential feedback-dependent timesteps can be
//   batched. It measures the reusable compiler/runtime primitive for groups of
//   calls whose independence has already been established.
//
// Batch sizes:
//   2, 4, 8, 16, 32, 64
//
// Measurements:
//   micro NoSunshine
//   realistic NoSunshine
//   realistic Causal
//
// Policy:
//   exact result parity
//   5 measured rounds / variant / batch size
//   ABBA-style alternating order
//   median authoritative
//   <10% = inconclusive
//
// Output:
//   updates/outbox/approach-a-a3-batch-interface.json
//   updates/outbox/approach-a-a3-build.txt
//   updates/outbox/approach-a-a3-parity.txt
//   updates/outbox/approach-a-a3-scopes.txt
//   updates/outbox/llm.txt

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
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
    fs.unlinkSync(
      file
    );
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
    "approach-a-a3",
    message,
    extra
  );

  throw new Error(
    message
  );
}

function median(
  values
) {
  const sorted =
    [...values]
      .sort(
        (
          a,
          b,
        ) =>
          a - b
      );

  const middle =
    Math.floor(
      sorted.length / 2
    );

  return sorted.length % 2
    ? sorted[
        middle
      ]
    : (
        sorted[
          middle - 1
        ] +
        sorted[
          middle
        ]
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
      "approach-a-a3-build.txt"
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
    result.status !==
      0
  ) {
    writeFailure(
      failedBuild,
      "approach-a-a3-build",
      result.error ??
        `npm run build exited with status ${result.status}`,
      [
        result.stdout ?? "",
        result.stderr ?? "",
      ].join("\n")
    );

    process.exit(
      2
    );
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
    result.status !==
      0
  ) {
    failRun(
      `Existing test failed after A3: ${relative}`,
      [
        `status=${result.status}`,
        result.stdout ?? "",
        result.stderr ?? "",
      ].join("\n")
    );
  }
}

function timedLogicalCalls(
  fn,
  repetitions,
  logicalCallsPerInvocation
) {
  const start =
    performance.now();

  let last;

  for (
    let index = 0;
    index <
      repetitions;
    index +=
      1
  ) {
    last =
      fn();
  }

  const totalMs =
    performance.now() -
    start;

  const logicalCalls =
    repetitions *
    logicalCallsPerInvocation;

  return {
    repetitions,
    logicalCalls,
    totalMs,

    perLogicalCallUs:
      totalMs *
      1000 /
      logicalCalls,

    last,
  };
}

function summarize(
  rows
) {
  const values =
    rows.map(
      row =>
        row
          .perLogicalCallUs
    );

  return {
    perLogicalCallUs:
      values,

    medianPerLogicalCallUs:
      median(
        values
      ),

    minPerLogicalCallUs:
      Math.min(
        ...values
      ),

    maxPerLogicalCallUs:
      Math.max(
        ...values
      ),
  };
}

function assertSameObject(
  label,
  a,
  b
) {
  const keys =
    Object.keys(
      a
    );

  if (
    keys.length !==
    Object.keys(
      b
    ).length
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
      a[
        key
      ];

    const bv =
      b[
        key
      ];

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
        !Number.isFinite(
          av
        ) ||
        !Number.isFinite(
          bv
        ) ||
        diff >
          1e-12
      ) {
        failRun(
          `${label}.${key}: A2=${av} A3=${bv} diff=${diff}`
        );
      }
    } else if (
      av !==
      bv
    ) {
      failRun(
        `${label}.${key}: A2=${String(av)} A3=${String(bv)}`
      );
    }
  }
}

function assertSameArray(
  label,
  a,
  b
) {
  if (
    !Array.isArray(
      a
    ) ||
    !Array.isArray(
      b
    ) ||
    a.length !==
      b.length
  ) {
    failRun(
      `${label}: batch result length/shape differs`
    );
  }

  for (
    let index = 0;
    index <
      a.length;
    index +=
      1
  ) {
    assertSameObject(
      `${label}[${index}]`,
      a[
        index
      ],
      b[
        index
      ]
    );
  }
}

function compareBatch(
  parameters,
  sequentialOne,
  batchOne,
  targetLogicalCalls
) {
  const batchSize =
    parameters.length;

  const repetitions =
    Math.max(
      1,
      Math.ceil(
        targetLogicalCalls /
        batchSize
      )
    );

  const sequentialInvocation =
    () => {
      const values =
        new Array(
          batchSize
        );

      for (
        let index = 0;
        index <
          batchSize;
        index +=
          1
      ) {
        values[
          index
        ] =
          sequentialOne(
            parameters[
              index
            ]
          );
      }

      return values;
    };

  const batchInvocation =
    () =>
      batchOne(
        parameters
      );

  /*
   * Exact parity before timing.
   */
  assertSameArray(
    `batchSize=${batchSize}`,
    sequentialInvocation(),
    batchInvocation()
  );

  /*
   * Warm both variants.
   */
  const warmupRepetitions =
    Math.max(
      1,
      Math.min(
        100,
        Math.ceil(
          1000 /
          batchSize
        )
      )
    );

  timedLogicalCalls(
    sequentialInvocation,
    warmupRepetitions,
    batchSize
  );

  timedLogicalCalls(
    batchInvocation,
    warmupRepetitions,
    batchSize
  );

  const rows = {
    a2: [],
    a3: [],
  };

  const order = [
    "a2",
    "a3",
    "a3",
    "a2",
    "a2",
    "a3",
    "a3",
    "a2",
    "a2",
    "a3",
  ];

  for (
    const variant
    of order
  ) {
    rows[
      variant
    ].push(
      timedLogicalCalls(
        variant ===
          "a2"
          ? sequentialInvocation
          : batchInvocation,
        repetitions,
        batchSize
      )
    );
  }

  const a2 =
    summarize(
      rows.a2
    );

  const a3 =
    summarize(
      rows.a3
    );

  const deltaPct =
    100 *
    (
      a3
        .medianPerLogicalCallUs -
      a2
        .medianPerLogicalCallUs
    ) /
    a2
      .medianPerLogicalCallUs;

  const conclusion =
    Math.abs(
      deltaPct
    ) < 10
      ? "INCONCLUSIVE_LT_10_PERCENT"
      : (
          deltaPct < 0
            ? "A3_FASTER"
            : "A2_FASTER"
        );

  return {
    batchSize,
    repetitionsPerRound:
      repetitions,

    logicalCallsPerRound:
      repetitions *
      batchSize,

    a2,
    a3,

    deltaPct,

    speedup:
      a2
        .medianPerLogicalCallUs /
      a3
        .medianPerLogicalCallUs,

    conclusion,

    exactParity:
      true,
  };
}

function makeNoSunshineParameters(
  base,
  count
) {
  return Array.from(
    {
      length:
        count,
    },
    (
      _,
      index,
    ) => ({
      ...base,

      currentIndex:
        (
          base.currentIndex +
          index *
          37
        ) %
        length,

      currentStorageEnergy:
        base
          .currentStorageEnergy +
        (
          index %
          7
        ) *
        0.125,
    })
  );
}

function makeCausalParameters(
  base,
  count
) {
  return Array.from(
    {
      length:
        count,
    },
    (
      _,
      index,
    ) => ({
      ...base,

      currentIndex:
        (
          base.currentIndex +
          index *
          29
        ) %
        length,

      currentStorageEnergy:
        base
          .currentStorageEnergy +
        (
          index %
          5
        ) *
        0.1,

      energyCutLevel:
        base
          .energyCutLevel +
        (
          index %
          3
        ) *
        0.05,
    })
  );
}

function bestMeaningful(
  rows
) {
  const candidates =
    rows.filter(
      row =>
        row.conclusion ===
        "A3_FASTER"
    );

  if (
    candidates.length ===
    0
  ) {
    return null;
  }

  return candidates.reduce(
    (
      best,
      current,
    ) =>
      current.speedup >
        best.speedup
        ? current
        : best
  );
}

buildNative();

runExistingTest(
  "scripts/test-global-acceleration-parity.mjs",
  "approach-a-a3-parity.txt"
);

runExistingTest(
  "scripts/test-global-acceleration-scopes.mjs",
  "approach-a-a3-scopes.txt"
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
    "Unable to load rebuilt A3 native binding.",
    error?.stack ??
      String(
        error
      )
  );
}

for (
  const exportName
  of [
    "registerAccelerationScope",
    "registerAccelerationDataPair",
    "runNoSunshineKernelContext",
    "runPeakShavingCausalKernelContext",
    "runNoSunshineKernelContextBatch",
    "runPeakShavingCausalKernelContextBatch",
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
      `Missing A3 native export: ${exportName}`
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
  index <
    length;
  index +=
    1
) {
  const quarter =
    index %
    96;

  const hour =
    quarter /
    4;

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

    pvPower[
      index
    ] =
      8.5 *
      Math.max(
        0,
        Math.sin(
          phase
        )
      );
  }

  if (
    quarter ===
    28
  ) {
    loadEnergy[
      index
    ] =
      0.35;
  } else if (
    quarter ===
    80
  ) {
    loadEnergy[
      index
    ] =
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
    5 /
    3,

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
    5 /
    3,

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

const batchSizes = [
  2,
  4,
  8,
  16,
  32,
  64,
];

const workloads = {
  microNoSunshine:
    [],

  realisticNoSunshine:
    [],

  realisticCausal:
    [],
};

for (
  const batchSize
  of batchSizes
) {
  const microParameters =
    makeNoSunshineParameters(
      noSunshineMicro,
      batchSize
    );

  workloads
    .microNoSunshine
    .push(
      compareBatch(
        microParameters,

        parameters =>
          native
            .runNoSunshineKernelContext(
              noSunshineScopeHandle,
              noSunshineDataHandle,
              parameters
            ),

        parameters =>
          native
            .runNoSunshineKernelContextBatch(
              noSunshineScopeHandle,
              noSunshineDataHandle,
              parameters
            ),

        20_000
      )
    );


  const realisticNoSunshineParameters =
    makeNoSunshineParameters(
      noSunshineReal,
      batchSize
    );

  workloads
    .realisticNoSunshine
    .push(
      compareBatch(
        realisticNoSunshineParameters,

        parameters =>
          native
            .runNoSunshineKernelContext(
              noSunshineScopeHandle,
              noSunshineDataHandle,
              parameters
            ),

        parameters =>
          native
            .runNoSunshineKernelContextBatch(
              noSunshineScopeHandle,
              noSunshineDataHandle,
              parameters
            ),

        5_000
      )
    );


  const causalParameters =
    makeCausalParameters(
      causalReal,
      batchSize
    );

  workloads
    .realisticCausal
    .push(
      compareBatch(
        causalParameters,

        parameters =>
          native
            .runPeakShavingCausalKernelContext(
              causalScopeHandle,
              causalDataHandle,
              parameters
            ),

        parameters =>
          native
            .runPeakShavingCausalKernelContextBatch(
              causalScopeHandle,
              causalDataHandle,
              parameters
            ),

        10_000
      )
    );
}

const best = {
  microNoSunshine:
    bestMeaningful(
      workloads
        .microNoSunshine
    ),

  realisticNoSunshine:
    bestMeaningful(
      workloads
        .realisticNoSunshine
    ),

  realisticCausal:
    bestMeaningful(
      workloads
        .realisticCausal
    ),
};

const realisticHasMeaningfulGain =
  best
    .realisticNoSunshine !==
      null ||
  best
    .realisticCausal !==
      null;

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
    "APPROACH_A_A3_BATCH_INTERFACE",

  release,

  change:
    "Multiple independent dynamic parameter sets are evaluated within one N-API call using the existing A2 persistent input context.",

  semanticLimit:
    "A3 is valid only when calls are independent. Feedback-dependent sequential timesteps are not automatically batchable.",

  measurementPolicy: {
    batchSizes,

    measuredRoundsPerVariant:
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

    everyBatchComparison:
      "EXACT",
  },

  workloads,

  bestMeaningful,

  decision:
    realisticHasMeaningfulGain
      ? "KEEP_A3_FOR_INDEPENDENT_BATCHES"
      : "DO_NOT_KEEP_A3_YET",

  nextCandidate:
    realisticHasMeaningfulGain
      ? "A4: test packed dynamic batch ABI separately from object-array batching, without changing A3 semantics."
      : "A3b: packed dynamic batch ABI may be tested separately; do not infer that batching itself is useless from object-marshalling overhead.",
};

const reportPath =
  path.join(
    outbox,
    "approach-a-a3-batch-interface.json"
  );

fs.writeFileSync(
  reportPath,
  JSON.stringify(
    report,
    null,
    2
  ) +
    "\n",
  "utf8"
);

const llmPath =
  path.join(
    outbox,
    "llm.txt"
  );

function bestLine(
  name,
  value
) {
  if (
    value ===
    null
  ) {
    return `${name} best=NONE`;
  }

  return [
    `${name}`,
    `bestBatch=${value.batchSize}`,
    `A2Us=${value.a2.medianPerLogicalCallUs.toFixed(6)}`,
    `A3Us=${value.a3.medianPerLogicalCallUs.toFixed(6)}`,
    `speedup=${value.speedup.toFixed(4)}x`,
    `deltaPct=${value.deltaPct.toFixed(3)}`,
  ].join(" ");
}

fs.appendFileSync(
  llmPath,
  [
    "",
    `=== ${new Date().toISOString()} APPROACH_A_A3_BATCH_INTERFACE ===`,
    `release=${release}`,
    "parity existing=PASS everyBatchComparison=EXACT",
    bestLine(
      "microNoSunshine",
      best.microNoSunshine
    ),
    bestLine(
      "realisticNoSunshine",
      best.realisticNoSunshine
    ),
    bestLine(
      "realisticCausal",
      best.realisticCausal
    ),
    `decision=${report.decision}`,
    `semanticLimit=${report.semanticLimit}`,
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
  "APPROACH_A_A3_BATCH_INTERFACE=OK"
);
