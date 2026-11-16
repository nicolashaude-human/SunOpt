// rust/sunopt-node/scripts/approach-a-final-pair-production-year-parity.mjs
//
// APPROACH_A_FINAL_PAIR = real SunOpt end-to-end annual A/B parity gate.
//
// This runner deliberately uses the current Next.js production API route:
//   POST /api/tank-comparison
//
// It starts two completely separate Next processes:
//   A: SUNOPT_CUDA_CAUSAL_PATH=A  (legacy FP64 causal path)
//   B: SUNOPT_CUDA_CAUSAL_PATH=B  (B14P FP32 production path)
//
// The child process pool inherits the selected environment, so A and B cannot
// share a CudaCausalWorker.
//
// FAIL-CLOSED:
// Before running, inspect current project source and verify that the current
// tank-comparison production path actually references the lockstep CUDA API.
// If the route still uses only the older per-tank cudaWorker.run() path,
// write failed-prompt.txt instead of producing a false parity result.
//
// GPU clock:
// APPROACH_A_FINAL_PAIR explicitly starts the same NVAPI benchmark guard proven by B14P:
// 1050 mV / native 2865 MHz / +625 MHz V/F offset / 100% fan.
// It waits for READY, keeps that guard alive across both independent A/B
// annual runs, samples nvidia-smi during each run, and restores state after B.
//
// No build is required: Next dev compiles the current source on demand.
//
// Output:
//   updates/outbox/approach-a-final-pair-production-A.json
//   updates/outbox/approach-a-final-pair-production-B.json
//   updates/outbox/approach-a-final-pair-production-year-parity.json
//   updates/outbox/approach-a-final-pair-server-A.txt
//   updates/outbox/approach-a-final-pair-server-B.txt
//   updates/outbox/llm.txt
//
// Failure:
//   updates/outbox/failed-prompt.txt
//   updates/outbox/failed-run.txt

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

const telemetryHelper =
  path.join(
    nativeRoot,
    "scripts",
    "blackwell-b14p-telemetry.mjs"
  );

const guard =
  path.join(
    nativeRoot,
    "scripts",
    "nvapi-blackwell-benchmark-guard.py"
  );

const routePath =
  path.join(
    projectRoot,
    "src",
    "app",
    "api",
    "tank-comparison",
    "route.ts"
  );

const srcRoot =
  path.join(
    projectRoot,
    "src"
  );

const hashFile =
  path.join(
    projectRoot,
    "updates",
    "history",
    "hash.txt"
  );

const failedPrompt =
  path.join(
    outbox,
    "failed-prompt.txt"
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
    failedPrompt,
    failedRun,
  ]
) {
  try {
    fs.unlinkSync(file);
  } catch {}
}

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

function failPrompt(
  message,
  extra = ""
) {
  writeFailure(
    failedPrompt,
    "prompt/source-gate",
    message,
    extra
  );

  throw new Error(message);
}

function failRun(
  message,
  extra = ""
) {
  writeFailure(
    failedRun,
    "run",
    message,
    extra
  );

  throw new Error(message);
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

function walkSource(
  directory,
  result = []
) {
  for (
    const entry
    of fs.readdirSync(
      directory,
      {
        withFileTypes: true,
      }
    )
  ) {
    if (
      entry.name ===
        "node_modules" ||
      entry.name ===
        ".next" ||
      entry.name ===
        "updates"
    ) {
      continue;
    }

    const full =
      path.join(
        directory,
        entry.name
      );

    if (
      entry.isDirectory()
    ) {
      walkSource(
        full,
        result
      );

      continue;
    }

    if (
      entry.isFile() &&
      /\.(?:ts|tsx|js|mjs|cjs)$/.test(
        entry.name
      )
    ) {
      result.push(full);
    }
  }

  return result;
}

function sourceGate() {
  if (
    !fs.existsSync(
      routePath
    )
  ) {
    failPrompt(
      `Current production route is missing: ${routePath}`
    );
  }

  const route =
    fs.readFileSync(
      routePath,
      "utf8"
    );

  const files =
    walkSource(
      srcRoot
    );

  const lockstepHits = [];

  for (
    const file
    of files
  ) {
    let text;

    try {
      text =
        fs.readFileSync(
          file,
          "utf8"
        );
    } catch {
      continue;
    }

    if (
      text.includes(
        "configureLockstepPacked"
      ) &&
      (
        text.includes(
          "runLockstepDynamicInto"
        ) ||
        text.includes(
          "runLockstepDynamic"
        )
      )
    ) {
      lockstepHits.push(
        path.relative(
          projectRoot,
          file
        )
      );
    }
  }

  const routeLooksGpuLockstep =
    /tank-comparison-gpu-lockstep/i
      .test(route) ||
    /runTankComparisonGpuLockstep/i
      .test(route) ||
    /runTankComparison.*Lockstep/i
      .test(route);

  const routeLooksOldPool =
    /runTankComparisonProcessPool/i
      .test(route);

  if (
    !routeLooksGpuLockstep ||
    lockstepHits.length === 0
  ) {
    failPrompt(
      "APPROACH_A_FINAL_PAIR refused to run: the current /api/tank-comparison source does not prove that the real production route uses the lockstep CUDA interface.",
      JSON.stringify(
        {
          route:
            path.relative(
              projectRoot,
              routePath
            ),

          routeLooksGpuLockstep,

          routeLooksOldPool,

          lockstepHits,

          required:
            "Route must use the GPU lockstep tank-comparison path, and current src must contain configureLockstepPacked + runLockstepDynamic/Into.",
        },
        null,
        2
      )
    );
  }

  return {
    route:
      path.relative(
        projectRoot,
        routePath
      ),

    routeLooksGpuLockstep,

    routeLooksOldPool,

    lockstepHits,
  };
}

function parseTelemetryFile(
  file
) {
  if (
    !fs.existsSync(
      file
    )
  ) {
    return [];
  }

  const rows = [];

  for (
    const raw
    of fs
      .readFileSync(
        file,
        "utf8"
      )
      .split(/\r?\n/)
  ) {
    const line =
      raw.trim();

    if (!line) {
      continue;
    }

    const parts =
      line
        .split(",")
        .map(
          value =>
            value.trim()
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

    const row = {
      clockMHz:
        Number(clock),
      temperatureC:
        Number(temp),
      pstate,
      powerW:
        Number(power),
      utilizationPercent:
        Number(util),
    };

    if (
      Number.isFinite(
        row.clockMHz
      ) &&
      Number.isFinite(
        row.utilizationPercent
      )
    ) {
      rows.push(row);
    }
  }

  return rows;
}

function validateFixedClock(
  label,
  telemetry
) {
  const active =
    telemetry.filter(
      row =>
        row
          .utilizationPercent >=
        20
    );

  if (
    active.length === 0
  ) {
    failRun(
      `CLOCK_LOCK_FAILED: no active GPU telemetry samples for production ${label}.`,
      JSON.stringify(
        telemetry,
        null,
        2
      )
    );
  }

  const clocks =
    active.map(
      row =>
        row.clockMHz
    );

  const minMHz =
    Math.min(...clocks);

  const maxMHz =
    Math.max(...clocks);

  const medianMHz =
    median(clocks);

  const spanMHz =
    maxMHz -
    minMHz;

  const inBand =
    clocks.filter(
      value =>
        value >= 3450 &&
        value <= 3510
    ).length;

  const inBandRatio =
    inBand /
    clocks.length;

  if (
    medianMHz < 3450 ||
    medianMHz > 3510 ||
    spanMHz > 35 ||
    inBandRatio < 0.8
  ) {
    failRun(
      `CLOCK_LOCK_FAILED: production ${label} was not stably locked near the +625 MHz / ~3480 MHz point.`,
      JSON.stringify(
        {
          minMHz,
          maxMHz,
          medianMHz,
          spanMHz,
          inBandRatio,
          active,
        },
        null,
        2
      )
    );
  }

  return {
    activeSamples:
      active.length,

    minMHz,
    maxMHz,
    medianMHz,
    spanMHz,
    inBandRatio,

    temperatureMinC:
      Math.min(
        ...active.map(
          row =>
            row
              .temperatureC
        )
      ),

    temperatureMaxC:
      Math.max(
        ...active.map(
          row =>
            row
              .temperatureC
        )
      ),
  };
}

function killProcessTree(
  child
) {
  if (
    !child ||
    child.pid == null
  ) {
    return;
  }

  if (
    process.platform ===
    "win32"
  ) {
    spawnSync(
      "taskkill",
      [
        "/PID",
        String(
          child.pid
        ),
        "/T",
        "/F",
      ],
      {
        stdio:
          "ignore",
        windowsHide:
          true,
      }
    );

    return;
  }

  try {
    child.kill(
      "SIGTERM"
    );
  } catch {}
}

async function waitForServer(
  origin,
  server,
  logFile
) {
  const deadline =
    Date.now() +
    120_000;

  while (
    Date.now() <
    deadline
  ) {
    if (
      server.exitCode !==
      null
    ) {
      failRun(
        `Next server exited before becoming ready: ${origin}`,
        fs.existsSync(logFile)
          ? fs.readFileSync(
              logFile,
              "utf8"
            )
          : ""
      );
    }

    try {
      const response =
        await fetch(
          `${origin}/api/weather`,
          {
            cache:
              "no-store",
          }
        );

      if (
        response.status >=
        100
      ) {
        return;
      }
    } catch {}

    await new Promise(
      resolve =>
        setTimeout(
          resolve,
          500
        )
    );
  }

  failRun(
    `Timed out waiting for Next server: ${origin}`,
    fs.existsSync(logFile)
      ? fs.readFileSync(
          logFile,
          "utf8"
        )
      : ""
  );
}

function startTelemetry(
  label
) {
  if (
    !fs.existsSync(
      telemetryHelper
    )
  ) {
    failPrompt(
      `B14P telemetry helper is missing: ${telemetryHelper}`
    );
  }

  const telemetryFile =
    path.join(
      outbox,
      `approach-a-final-pair-clock-${label}.csv`
    );

  const stopFile =
    telemetryFile +
    ".stop";

  const errorFile =
    telemetryFile +
    ".errors.txt";

  for (
    const file
    of [
      telemetryFile,
      stopFile,
      errorFile,
    ]
  ) {
    try {
      fs.unlinkSync(file);
    } catch {}
  }

  const child =
    spawn(
      process.execPath,
      [
        telemetryHelper,
        "--output",
        telemetryFile,
        "--stop-file",
        stopFile,
        "--parent-pid",
        String(
          process.pid
        ),
        "--interval-ms",
        "200",
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

  sleepMs(350);

  return {
    child,
    telemetryFile,
    stopFile,
    errorFile,
  };
}

function stopTelemetry(
  sampler
) {
  try {
    fs.writeFileSync(
      sampler.stopFile,
      "stop\n",
      "utf8"
    );
  } catch {}

  const deadline =
    Date.now() +
    1500;

  while (
    Date.now() <
    deadline
  ) {
    if (
      sampler
        .child
        .exitCode !==
      null
    ) {
      break;
    }

    sleepMs(50);
  }

  if (
    sampler
      .child
      .exitCode ===
    null
  ) {
    try {
      sampler
        .child
        .kill();
    } catch {}
  }

  return parseTelemetryFile(
    sampler.telemetryFile
  );
}

async function runProduction(
  label,
  causalPath,
  port,
  requestBody
) {
  const logFile =
    path.join(
      outbox,
      `approach-a-final-pair-server-${label}.txt`
    );

  try {
    fs.unlinkSync(
      logFile
    );
  } catch {}

  const log =
    fs.openSync(
      logFile,
      "a"
    );

  const comspec =
    process.env.ComSpec ||
    process.env.COMSPEC ||
    "C:\\Windows\\System32\\cmd.exe";

  const server =
    spawn(
      comspec,
      [
        "/d",
        "/s",
        "/c",
        `npm run dev -- --hostname 127.0.0.1 --port ${port}`,
      ],
      {
        cwd:
          projectRoot,

        env: {
          ...process.env,

          SUNOPT_CUDA_CAUSAL_PATH:
            "A",

          SUNOPT_APPROACH_A_STEP_GRAPH:
            causalPath === "AG"
              ? "1"
              : "0",

          /*
           * Do not allow an earlier shell value to accidentally switch
           * the project to FastMath during the parity test.
           */
          SUNOPT_CUDA_FAST_MATH:
            "0",
        },

        stdio: [
          "ignore",
          log,
          log,
        ],

        windowsHide:
          true,
      }
    );

  const origin =
    `http://127.0.0.1:${port}`;

  try {
    await waitForServer(
      origin,
      server,
      logFile
    );

    const sampler =
      startTelemetry(
        label
      );

    const requestStart =
      performance.now();

    let response;
    let text;

    try {
      response =
        await fetch(
          `${origin}/api/tank-comparison`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                requestBody
              ),

            cache:
              "no-store",
          }
        );

      text =
        await response.text();
    } finally {
      const telemetry =
        stopTelemetry(
          sampler
        );

      const fixedClock =
        validateFixedClock(
          label,
          telemetry
        );

      fs.writeFileSync(
        path.join(
          outbox,
          `approach-a-final-pair-clock-${label}-summary.json`
        ),
        JSON.stringify(
          fixedClock,
          null,
          2
        ) + "\n",
        "utf8"
      );
    }

    const requestWallMs =
      performance.now() -
      requestStart;

    if (
      !response.ok
    ) {
      failRun(
        `Production ${label} /api/tank-comparison returned HTTP ${response.status}.`,
        [
          text,
          "",
          fs.existsSync(logFile)
            ? fs.readFileSync(
                logFile,
                "utf8"
              )
            : "",
        ].join("\n")
      );
    }

    let result;

    try {
      result =
        JSON.parse(text);
    } catch (error) {
      failRun(
        `Production ${label} returned invalid JSON: ${error}`,
        text
      );
    }

    if (
      !result ||
      !Array.isArray(
        result.points
      ) ||
      result.points.length ===
        0
    ) {
      failRun(
        `Production ${label} returned no tank-comparison points.`,
        text
      );
    }

    const fixedClock =
      JSON.parse(
        fs.readFileSync(
          path.join(
            outbox,
            `approach-a-final-pair-clock-${label}-summary.json`
          ),
          "utf8"
        )
      );

    const rawPath =
      path.join(
        outbox,
        `approach-a-final-pair-production-${label}.json`
      );

    fs.writeFileSync(
      rawPath,
      JSON.stringify(
        {
          causalPath,
          requestWallMs,
          fixedClock,
          result,
        },
        null,
        2
      ) + "\n",
      "utf8"
    );

    return {
      causalPath,
      requestWallMs,
      fixedClock,
      result,
      rawPath,
    };
  } finally {
    killProcessTree(
      server
    );

    try {
      fs.closeSync(log);
    } catch {}

  }
}

function pointMap(
  result
) {
  const map =
    new Map();

  for (
    const point
    of result.points
  ) {
    map.set(
      Number(
        point.storageVolume
      ),
      point
    );
  }

  return map;
}

function numericDifference(
  a,
  b
) {
  const av =
    Number(a);

  const bv =
    Number(b);

  if (
    !Number.isFinite(av) ||
    !Number.isFinite(bv)
  ) {
    return {
      a: av,
      b: bv,
      abs:
        Number.POSITIVE_INFINITY,
      rel:
        Number.POSITIVE_INFINITY,
    };
  }

  const abs =
    Math.abs(
      bv -
      av
    );

  const scale =
    Math.max(
      Math.abs(av),
      Math.abs(bv),
      1e-12
    );

  return {
    a: av,
    b: bv,
    abs,
    rel:
      abs /
      scale,
  };
}

function bestBy(
  points,
  key,
  direction
) {
  const sorted =
    [...points]
      .filter(
        point =>
          Number.isFinite(
            Number(
              point[key]
            )
          )
      )
      .sort(
        (a, b) =>
          direction === "max"
            ? Number(
                b[key]
              ) -
              Number(
                a[key]
              )
            : Number(
                a[key]
              ) -
              Number(
                b[key]
              )
      );

  return sorted[0] ??
    null;
}

function compareResults(
  a,
  b
) {
  const aMap =
    pointMap(a);

  const bMap =
    pointMap(b);

  const volumes =
    [...new Set([
      ...aMap.keys(),
      ...bMap.keys(),
    ])]
      .sort(
        (x, y) =>
          x - y
      );

  const keys = [
    "gridImport",
    "feedIn",
    "selfConsumption",
    "storageLosses",
    "unmetThermalDemand",
    "gridCost",
    "feedInRevenue",
    "netEnergyCost",
  ];

  const maxima =
    Object.fromEntries(
      keys.map(
        key => [
          key,
          {
            abs: 0,
            rel: 0,
            storageVolume:
              null,
          },
        ]
      )
    );

  const perTank = [];
  const missing = [];

  for (
    const volume
    of volumes
  ) {
    const ap =
      aMap.get(
        volume
      );

    const bp =
      bMap.get(
        volume
      );

    if (!ap || !bp) {
      missing.push({
        storageVolume:
          volume,

        inA:
          Boolean(ap),

        inB:
          Boolean(bp),
      });

      continue;
    }

    const differences =
      {};

    for (
      const key
      of keys
    ) {
      const diff =
        numericDifference(
          ap[key],
          bp[key]
        );

      differences[key] =
        diff;

      if (
        diff.abs >
        maxima[key].abs
      ) {
        maxima[key] = {
          abs:
            diff.abs,

          rel:
            diff.rel,

          storageVolume:
            volume,

          a:
            diff.a,

          b:
            diff.b,
        };
      }
    }

    perTank.push({
      storageVolume:
        volume,

      A:
        Object.fromEntries(
          keys.map(
            key => [
              key,
              ap[key],
            ]
          )
        ),

      B:
        Object.fromEntries(
          keys.map(
            key => [
              key,
              bp[key],
            ]
          )
        ),

      differences,
    });
  }

  const optimum = {
    netEnergyCost: {
      A:
        bestBy(
          a.points,
          "netEnergyCost",
          "min"
        ),

      B:
        bestBy(
          b.points,
          "netEnergyCost",
          "min"
        ),
    },

    gridImport: {
      A:
        bestBy(
          a.points,
          "gridImport",
          "min"
        ),

      B:
        bestBy(
          b.points,
          "gridImport",
          "min"
        ),
    },

    selfConsumption: {
      A:
        bestBy(
          a.points,
          "selfConsumption",
          "max"
        ),

      B:
        bestBy(
          b.points,
          "selfConsumption",
          "max"
        ),
    },
  };

  const optimumVolumesMatch =
    Object.values(
      optimum
    ).every(
      pair =>
        pair.A &&
        pair.B &&
        Number(
          pair.A
            .storageVolume
        ) ===
        Number(
          pair.B
            .storageVolume
        )
    );

  const energyKeys = [
    "gridImport",
    "feedIn",
    "selfConsumption",
    "storageLosses",
    "unmetThermalDemand",
  ];

  const energyWithin10Wh =
    energyKeys.every(
      key =>
        maxima[key].abs <=
        0.01
    );

  const costsWithin1Rp =
    [
      "gridCost",
      "feedInRevenue",
      "netEnergyCost",
    ].every(
      key =>
        maxima[key].abs <=
        0.01
    );

  let classification =
    "MATERIAL_DIFFERENCE";

  const exact =
    keys.every(
      key =>
        maxima[key].abs ===
        0
    ) &&
    missing.length ===
      0;

  if (
    exact &&
    optimumVolumesMatch
  ) {
    classification =
      "EXACT";
  } else if (
    missing.length === 0 &&
    optimumVolumesMatch &&
    energyWithin10Wh &&
    costsWithin1Rp
  ) {
    classification =
      "PRACTICALLY_IDENTICAL";
  }

  return {
    keys,
    maxima,
    missing,
    perTank,
    optimum,
    optimumVolumesMatch,
    thresholds: {
      annualEnergyAbsKWh:
        0.01,
      annualCostAbsChf:
        0.01,
    },
    energyWithin10Wh,
    costsWithin1Rp,
    classification,
  };
}


function startNvapiGuard() {
  if (
    !fs.existsSync(
      guard
    )
  ) {
    failPrompt(
      `NVAPI benchmark guard is missing: ${guard}`
    );
  }

  const stateDir =
    path.join(
      nativeRoot,
      "blackwell-runtime-control",
      `approach-a-final-pair-${process.pid}`
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
      "approach-a-final-pair-nvapi-guard.txt"
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

  const guardProcess =
    spawn(
      python,
      [
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

  const deadline =
    Date.now() +
    30_000;

  while (
    Date.now() <
      deadline &&
    !fs.existsSync(
      readyFile
    )
  ) {
    if (
      guardProcess.exitCode !==
      null
    ) {
      failRun(
        "CLOCK_LOCK_FAILED: NVAPI guard exited before READY.",
        fs.existsSync(
          guardLog
        )
          ? fs.readFileSync(
              guardLog,
              "utf8"
            )
          : ""
      );
    }

    sleepMs(200);
  }

  if (
    !fs.existsSync(
      readyFile
    )
  ) {
    failRun(
      "CLOCK_LOCK_FAILED: NVAPI guard did not reach READY.",
      fs.existsSync(
        guardLog
      )
        ? fs.readFileSync(
            guardLog,
            "utf8"
          )
        : ""
    );
  }

  let ready;

  try {
    ready =
      JSON.parse(
        fs.readFileSync(
          readyFile,
          "utf8"
        )
      );
  } catch (error) {
    failRun(
      `CLOCK_LOCK_FAILED: invalid NVAPI READY file: ${error}`,
      fs.existsSync(
        guardLog
      )
        ? fs.readFileSync(
            guardLog,
            "utf8"
          )
        : ""
    );
  }

  if (
    ready.status !==
    "READY"
  ) {
    failRun(
      `CLOCK_LOCK_FAILED: unexpected NVAPI guard state ${JSON.stringify(ready)}`,
      fs.existsSync(
        guardLog
      )
        ? fs.readFileSync(
            guardLog,
            "utf8"
          )
        : ""
    );
  }

  function stop() {
    try {
      fs.writeFileSync(
        stopFile,
        "stop\n",
        "utf8"
      );
    } catch {}

    const stopDeadline =
      Date.now() +
      10_000;

    while (
      Date.now() <
      stopDeadline
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

  return {
    ready,
    guardLog,
    stop,
  };
}

async function main() {
  const source =
    sourceGate();

  const clockGuard =
    startNvapiGuard();

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

  /*
   * Benchmark economics only.
   * They do not affect controller physics; they only convert the returned
   * annual energy balance to CHF and give us a deterministic cost optimum.
   */
  try {
    const requestBody = {
      heatPumpThermalPower:
          5,

      pvPeakPower:
        10,

      gridPriceChfPerKWh:
        0.30,

      feedInPriceChfPerKWh:
        0.08,
    };

    const A =
    await runProduction(
      "A",
      "A",
      31471,
      requestBody
    );

  const B =
    await runProduction(
      "AG",
      "AG",
      31472,
      requestBody
    );

  const comparison =
    compareResults(
      A.result,
      B.result
    );

  const performance = {
    requestWallMs: {
      A:
        A.requestWallMs,
      B:
        B.requestWallMs,
      speedupAOverB:
        A.requestWallMs /
        B.requestWallMs,
    },

    apiTotalDurationMs: {
      A:
        Number(
          A.result
            .totalDurationMs
        ),
      B:
        Number(
          B.result
            .totalDurationMs
        ),
      speedupAOverB:
        Number(
          A.result
            .totalDurationMs
        ) /
        Number(
          B.result
            .totalDurationMs
        ),
    },

    sweepDurationMs: {
      A:
        Number(
          A.result
            .sweepDurationMs
        ),
      B:
        Number(
          B.result
            .sweepDurationMs
        ),
      speedupAOverB:
        Number(
          A.result
            .sweepDurationMs
        ) /
        Number(
          B.result
            .sweepDurationMs
        ),
    },

    averageSimulationDurationMs: {
      A:
        Number(
          A.result
            .averageSimulationDurationMs
        ),
      B:
        Number(
          B.result
            .averageSimulationDurationMs
        ),
      speedupAOverB:
        Number(
          A.result
            .averageSimulationDurationMs
        ) /
        Number(
          B.result
            .averageSimulationDurationMs
        ),
    },
  };

  const report = {
    comparisonSlots: {
      A:
        "Approach A direct exact FP64",
      B:
        "AG persistent single-step CUDA graph exact FP64",
    },

    stage:
      "APPROACH_A_FINAL_PAIR_PRODUCTION_YEAR_PARITY",

    release,

    sourceGate:
      source,

    requestBody,

    note:
      "CHF prices are benchmark conversion inputs only; controller physics is unchanged. The physical parity decision uses annual energy metrics and selected optima.",

    fixedClock: {
      guardReady:
        clockGuard.ready,

      A:
        A.fixedClock,

      B:
        B.fixedClock,
    },

    performance,

    comparison,

    raw: {
      A:
        A.rawPath,
      B:
        B.rawPath,
    },
  };

  const reportPath =
    path.join(
      outbox,
      "approach-a-final-pair-production-year-parity.json"
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

  const optCostA =
    comparison
      .optimum
      .netEnergyCost
      .A
      ?.storageVolume ??
    null;

  const optCostB =
    comparison
      .optimum
      .netEnergyCost
      .B
      ?.storageVolume ??
    null;

  const optGridA =
    comparison
      .optimum
      .gridImport
      .A
      ?.storageVolume ??
    null;

  const optGridB =
    comparison
      .optimum
      .gridImport
      .B
      ?.storageVolume ??
    null;

  const llmPath =
    path.join(
      outbox,
      "llm.txt"
    );

  fs.appendFileSync(
    llmPath,
    [
      "",
      `=== ${new Date().toISOString()} APPROACH_A_FINAL_PAIR_PRODUCTION_YEAR_PARITY ===`,
      `release=${release}`,
      `sourceGate=${JSON.stringify(source)}`,
      `clockA median=${A.fixedClock.medianMHz} span=${A.fixedClock.spanMHz} samples=${A.fixedClock.activeSamples}`,
      `clockB median=${B.fixedClock.medianMHz} span=${B.fixedClock.spanMHz} samples=${B.fixedClock.activeSamples}`,
      `performance requestWall A=${A.requestWallMs.toFixed(3)} B=${B.requestWallMs.toFixed(3)} speedup=${performance.requestWallMs.speedupAOverB.toFixed(4)}x`,
      `performance sweep A=${performance.sweepDurationMs.A.toFixed(3)} B=${performance.sweepDurationMs.B.toFixed(3)} speedup=${performance.sweepDurationMs.speedupAOverB.toFixed(4)}x`,
      `parity classification=${comparison.classification} optimumVolumesMatch=${comparison.optimumVolumesMatch}`,
      `optimum netEnergyCost A=${optCostA}L B=${optCostB}L`,
      `optimum gridImport A=${optGridA}L B=${optGridB}L`,
      `maxDiffs=${JSON.stringify(comparison.maxima)}`,
      `rawA=${A.rawPath}`,
      `rawB=${B.rawPath}`,
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
    "APPROACH_A_FINAL_PAIR_PRODUCTION_YEAR_PARITY=OK"
  );
  } finally {
    clockGuard.stop();
  }
}

main()
  .catch(
    error => {
      if (
        !fs.existsSync(
          failedPrompt
        ) &&
        !fs.existsSync(
          failedRun
        )
      ) {
        writeFailure(
          failedRun,
          "run",
          error
        );
      }

      console.error(error);
      process.exitCode = 1;
    }
  );
