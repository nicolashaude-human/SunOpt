// rust/sunopt-node/scripts/blackwell-b14p-fixed-clock-production-b.mjs
import fs from "node:fs";
import path from "node:path";
import {
  spawn,
  spawnSync,
} from "node:child_process";
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

const child =
  path.join(
    nativeRoot,
    "scripts",
    "blackwell-b14p-child.mjs"
  );

const guard =
  path.join(
    nativeRoot,
    "scripts",
    "nvapi-blackwell-benchmark-guard.py"
  );

const snapshot =
  path.join(
    nativeRoot,
    "blackwell-b12-sync-blackwell.bin"
  );

const reference =
  path.join(
    nativeRoot,
    "benchmarks",
    "references",
    "blackwell-b14m-strict-reference.bin"
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

function sleepMs(ms) {
  const wait =
    new Int32Array(
      new SharedArrayBuffer(4)
    );

  Atomics.wait(
    wait,
    0,
    0,
    ms
  );
}

function fail(message, extra = "") {
  fs.writeFileSync(
    failedRun,
    [
      "stage=run",
      `time=${new Date().toISOString()}`,
      `error=${message}`,
      extra,
      "",
    ].join("\n"),
    "utf8"
  );

  throw new Error(message);
}

function runChild(extra) {
  const p =
    spawnSync(
      process.execPath,
      [
        child,
        "--snapshot",
        snapshot,
        "--reference",
        reference,
        ...extra,
      ],
      {
        cwd:
          projectRoot,

        encoding:
          "utf8",

        windowsHide:
          true,

        env: {
          ...process.env,
          SUNOPT_CUDA_CAUSAL_PATH:
            "B",
        },
      }
    );

  if (
    p.error ||
    p.status !== 0
  ) {
    fail(
      `B14P child failed: ${extra.join(" ")}`,
      [
        `status=${p.status}`,
        `spawnError=${p.error?.stack ?? ""}`,
        "=== STDOUT ===",
        p.stdout ?? "",
        "=== STDERR ===",
        p.stderr ?? "",
      ].join("\n")
    );
  }

  const lines =
    (p.stdout ?? "")
      .trim()
      .split(/\r?\n/);

  return JSON.parse(
    lines[
      lines.length - 1
    ]
  );
}

function median(values) {
  const a =
    [...values]
      .sort(
        (x, y) =>
          x - y
      );

  const m =
    Math.floor(
      a.length / 2
    );

  return a.length % 2
    ? a[m]
    : (
        a[m - 1] +
        a[m]
      ) / 2;
}

function summarize(label, rows, wallKey) {
  const wall =
    rows.map(
      row =>
        row.result[wallKey]
    );

  const kernel =
    rows
      .map(
        row =>
          row.result.kernelMs
      )
      .filter(
        value =>
          Number.isFinite(
            value
          )
      );

  return {
    label,
    count:
      rows.length,
    wallMs:
      wall,
    medianWallMs:
      median(wall),
    minWallMs:
      Math.min(...wall),
    maxWallMs:
      Math.max(...wall),
    medianKernelMs:
      kernel.length
        ? median(kernel)
        : null,
    telemetry:
      rows.map(
        row => ({
          before:
            row.before,
          after:
            row.after,
        })
      ),
    firstResult:
      rows[0]?.result ??
      null,
  };
}

function parseTelemetryFile(file) {
  if (!fs.existsSync(file)) {
    return [];
  }

  const rows = [];

  for (
    const rawLine
    of fs
      .readFileSync(
        file,
        "utf8"
      )
      .split(/\r?\n/)
  ) {
    const line =
      rawLine.trim();

    if (
      !line ||
      line.startsWith(
        "clocks"
      )
    ) {
      continue;
    }

    const parts =
      line
        .split(",")
        .map(
          value =>
            value
              .trim()
              .replace(
                /[^0-9A-Za-z.+-]/g,
                ""
              )
        );

    if (
      parts.length < 5
    ) {
      continue;
    }

    const [
      clock,
      temp,
      pstate,
      power,
      util,
    ] = parts;

    rows.push({
      clockMHz:
        Number(clock),
      temperatureC:
        Number(temp),
      pstate,
      powerW:
        Number(power),
      utilizationPercent:
        Number(util),
    });
  }

  return rows.filter(
    row =>
      Number.isFinite(
        row.clockMHz
      ) &&
      Number.isFinite(
        row.utilizationPercent
      )
  );
}

function runWithClockTelemetry(
  label,
  extra
) {
  const telemetryFile =
    path.join(
      outbox,
      `b14p-clock-${label}-${Date.now()}.csv`
    );

  const sampler =
    spawn(
      "nvidia-smi",
      [
        "--query-gpu=clocks.sm,temperature.gpu,pstate,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
        "-lms",
        "200",
        "-f",
        telemetryFile,
      ],
      {
        cwd:
          projectRoot,

        stdio:
          "ignore",

        windowsHide:
          true,
      }
    );

  sleepMs(300);

  let row;

  try {
    row =
      runChild(extra);
  } finally {
    try {
      sampler.kill();
    } catch {}

    sleepMs(250);
  }

  const telemetry =
    parseTelemetryFile(
      telemetryFile
    );

  const active =
    telemetry.filter(
      sample =>
        sample
          .utilizationPercent >=
        20
    );

  if (active.length === 0) {
    fail(
      `CLOCK_LOCK_FAILED: no active telemetry samples for ${label}`,
      JSON.stringify(
        telemetry,
        null,
        2
      )
    );
  }

  const clocks =
    active.map(
      sample =>
        sample.clockMHz
    );

  const minimum =
    Math.min(...clocks);

  const maximum =
    Math.max(...clocks);

  const clockMedian =
    median(clocks);

  const clockSpan =
    maximum -
    minimum;

  const inBand =
    clocks.filter(
      value =>
        value >= 3450 &&
        value <= 3510
    ).length;

  const inBandRatio =
    inBand /
    clocks.length;

  /*
   * +600 previously measured about 3457 MHz under active load.
   * +625 should land near 3.48 GHz. Gate on median/stability so one
   * short V/F transition sample does not invalidate a valid fixed run.
   */
  if (
    clockMedian < 3450 ||
    clockMedian > 3510 ||
    clockSpan > 35 ||
    inBandRatio < 0.8
  ) {
    fail(
      `CLOCK_LOCK_FAILED: active clock is not stably locked near the +625 MHz point for ${label}`,
      JSON.stringify(
        {
          minimum,
          maximum,
          clockMedian,
          clockSpan,
          inBandRatio,
          active,
        },
        null,
        2
      )
    );
  }

  return {
    ...row,
    fixedClock: {
      activeSamples:
        active.length,
      minMHz:
        minimum,
      maxMHz:
        maximum,
      medianMHz:
        clockMedian,
      spanMHz:
        clockSpan,
      inBandRatio,
      temperatureMaxC:
        Math.max(
          ...active.map(
            sample =>
              sample
                .temperatureC
          )
        ),
    },
  };
}

const release =
  fs.existsSync(hashFile)
    ? fs
        .readFileSync(
          hashFile,
          "utf8"
        )
        .trim()
    : "unknown";

if (!fs.existsSync(guard)) {
  fail(
    `NVAPI guard missing: ${guard}`
  );
}

const stateDir =
  path.join(
    nativeRoot,
    "blackwell-runtime-control",
    `b14p-${process.pid}`
  );

fs.mkdirSync(
  stateDir,
  {
    recursive: true,
  }
);

const readyFile =
  path.join(
    stateDir,
    "ready.json"
  );

const stopFile =
  path.join(
    stateDir,
    "stop.signal"
  );

const doneFile =
  path.join(
    stateDir,
    "done.json"
  );

const guardLog =
  path.join(
    outbox,
    "b14p-nvapi-guard.txt"
  );

for (
  const file
  of [
    readyFile,
    stopFile,
    doneFile,
    guardLog,
  ]
) {
  try {
    fs.unlinkSync(file);
  } catch {}
}

const python =
  process.env
    .SUNOPT_PYTHON ??
  "python";

const guardArgs = [
  guard,
  "--gpu",
  "0",
  "--target-voltage-mv",
  "1050",
  "--native-vf-frequency-mhz",
  "2865",
  "--curve-offset-mhz",
  "625",
  "--cap-band-points",
  "8",
  "--parent-pid",
  String(
    process.pid
  ),
  "--ready-file",
  readyFile,
  "--stop-file",
  stopFile,
  "--done-file",
  doneFile,
  "--log-file",
  guardLog,
  "--fan-percent",
  "100",
];

const guardProcess =
  spawn(
    python,
    guardArgs,
    {
      cwd:
        projectRoot,

      stdio:
        "ignore",

      windowsHide:
        true,
    }
  );

let guardStopped =
  false;

function stopGuard() {
  if (guardStopped) {
    return;
  }

  guardStopped =
    true;

  try {
    fs.writeFileSync(
      stopFile,
      "stop\n",
      "utf8"
    );
  } catch {}

  const deadline =
    Date.now() +
    10_000;

  while (
    Date.now() <
    deadline
  ) {
    if (
      fs.existsSync(
        doneFile
      )
    ) {
      return;
    }

    sleepMs(200);
  }

  try {
    guardProcess.kill();
  } catch {}
}

process.on(
  "exit",
  stopGuard
);

try {
  const readyDeadline =
    Date.now() +
    30_000;

  while (
    Date.now() <
    readyDeadline &&
    !fs.existsSync(
      readyFile
    )
  ) {
    sleepMs(200);
  }

  if (
    !fs.existsSync(
      readyFile
    )
  ) {
    fail(
      "CLOCK_LOCK_FAILED: NVAPI guard did not reach READY",
      fs.existsSync(guardLog)
        ? fs.readFileSync(
            guardLog,
            "utf8"
          )
        : ""
    );
  }

  const ready =
    JSON.parse(
      fs.readFileSync(
        readyFile,
        "utf8"
      )
    );

  if (
    ready.status !==
    "READY"
  ) {
    fail(
      `CLOCK_LOCK_FAILED: unexpected NVAPI guard state ${JSON.stringify(ready)}`,
      fs.existsSync(guardLog)
        ? fs.readFileSync(
            guardLog,
            "utf8"
          )
        : ""
    );
  }

  // Reference is persistent, not outbox.
  if (
    !fs.existsSync(
      reference
    )
  ) {
    runChild(
      [
        "--kind",
        "reference",
      ]
    );
  }

  /*
   * Clock gate warmup.
   * This is not counted.  If +625 is not actually active,
   * the workflow ends before any A/B result is accepted.
   */
  const clockWarmup =
    runWithClockTelemetry(
      "clock-warmup",
      [
        "--kind",
        "b14l",
        "--pinned",
        "false",
      ]
    );

  // One additional warmup per pinned variant.
  runWithClockTelemetry(
    "pageable-warmup",
    [
      "--kind",
      "b14l",
      "--pinned",
      "false",
    ]
  );

  runWithClockTelemetry(
    "pinned-warmup",
    [
      "--kind",
      "b14l",
      "--pinned",
      "true",
    ]
  );

  const pinnedRows = {
    pageable: [],
    pinned: [],
  };

  const abba = [
    "pageable",
    "pinned",
    "pinned",
    "pageable",
    "pageable",
    "pinned",
    "pinned",
    "pageable",
    "pageable",
    "pinned",
  ];

  let measuredIndex = 0;

  for (const name of abba) {
    const row =
      runWithClockTelemetry(
        `pinned-${measuredIndex}-${name}`,
        [
          "--kind",
          "b14l",
          "--pinned",
          String(
            name ===
            "pinned"
          ),
        ]
      );

    pinnedRows[name]
      .push(row);

    measuredIndex += 1;
  }

  const pageable =
    summarize(
      "pageable",
      pinnedRows.pageable,
      "wallMs"
    );

  const pinned =
    summarize(
      "pinned",
      pinnedRows.pinned,
      "wallMs"
    );

  const pinnedDeltaPct =
    100 *
    (
      pinned
        .medianWallMs -
      pageable
        .medianWallMs
    ) /
    pageable
      .medianWallMs;

  const pinnedConclusion =
    Math.abs(
      pinnedDeltaPct
    ) < 10
      ? "INCONCLUSIVE_LT_10_PERCENT"
      : (
          pinnedDeltaPct < 0
            ? "PINNED_FASTER"
            : "PAGEABLE_FASTER"
        );

  // Production-B warmup.
  runWithClockTelemetry(
    "production-b-warmup",
    [
      "--kind",
      "production-b",
    ]
  );

  const productionRows = [];

  for (
    let i = 0;
    i < 5;
    ++i
  ) {
    productionRows.push(
      runWithClockTelemetry(
        `production-b-${i}`,
        [
          "--kind",
          "production-b",
        ]
      )
    );
  }

  const productionB =
    summarize(
      "production-b",
      productionRows,
      "wallMs"
    );

  productionB
    .statusMismatchesVsStrict =
      productionRows
        .map(
          row =>
            row
              .result
              .statusMismatchesVsStrict
        );

  productionB
    .cutMismatches1E4VsStrict =
      productionRows
        .map(
          row =>
            row
              .result
              .cutMismatches1E4VsStrict
        );

  productionB
    .maxAbsCutErrorVsStrict =
      Math.max(
        ...productionRows
          .map(
            row =>
              row
                .result
                .maxAbsCutErrorVsStrict
          )
      );

  productionB
    .unsupportedJobs =
      Math.max(
        ...productionRows
          .map(
            row =>
              row
                .result
                .unsupportedJobs
          )
      );

  const report = {
    stage:
      "B14P_FIXED_CLOCK_PRODUCTION_B",

    release,

    productionPath: {
      default:
        "B14P_FP32",

      forceLegacy:
        "SUNOPT_CUDA_CAUSAL_PATH=A",

      fallback:
        "FP64 for unsupported/error",
    },

    fixedClock: {
      method:
        "NVAPI V/F guard",

      voltageMv:
        1050,

      nativeFrequencyMHz:
        2865,

      coreOffsetMHz:
        625,

      expectedActiveClockMedianMHz:
        [
          3450,
          3510,
        ],

      maximumActiveClockSpanMHz:
        35,

      minimumInBandRatio:
        0.8,

      fanPercent:
        100,

      warmup:
        clockWarmup
          .fixedClock,
    },

    measurementPolicy: {
      warmupPerVariant:
        1,

      measuredRunsPerVariant:
        5,

      medianAuthoritative:
        true,

      noiseThresholdPercent:
        10,

      pinnedOrder:
        "ABBA alternating",
    },

    pinnedAB: {
      pageable,
      pinned,
      deltaPct:
        pinnedDeltaPct,
      conclusion:
        pinnedConclusion,
    },

    productionB,

    persistentStrictReference:
      reference,
  };

  const reportPath =
    path.join(
      outbox,
      "blackwell-b14p-fixed-clock-production-b.json"
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
      `=== ${new Date().toISOString()} B14P_FIXED_CLOCK_PRODUCTION_B ===`,
      `release=${release}`,
      `clock target=1050mV native=2865 offset=+625 medianRange=3450..3510 maxSpan=35 inBand>=0.8 warmupMedian=${clockWarmup.fixedClock.medianMHz}`,
      `pinned pageableMedianWallMs=${pageable.medianWallMs.toFixed(6)} pinnedMedianWallMs=${pinned.medianWallMs.toFixed(6)} deltaPct=${pinnedDeltaPct.toFixed(3)} conclusion=${pinnedConclusion}`,
      `productionB medianWallMs=${productionB.medianWallMs.toFixed(6)} min=${productionB.minWallMs.toFixed(6)} max=${productionB.maxWallMs.toFixed(6)}`,
      `productionB statusMismatchRuns=${productionB.statusMismatchesVsStrict.join(",")} cutMismatch1e4Runs=${productionB.cutMismatches1E4VsStrict.join(",")} maxAbsCutError=${productionB.maxAbsCutErrorVsStrict} unsupportedJobs=${productionB.unsupportedJobs}`,
      `persistentStrictReference=${reference}`,
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
    "B14P_FIXED_CLOCK_PRODUCTION_B_RESULT=OK"
  );
} finally {
  stopGuard();
}
