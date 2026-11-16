// Approach A7 — full production annual A/B for A6.
//
// Fresh-process balanced A/B:
//
//   R C C R R C C R R C
//
// Each entry starts one fresh Next dev server, performs one excluded warmup,
// performs exactly one measured annual request, and then stops the server
// before the next entry starts.
//
// This avoids races in npm predev/native:install and avoids two Next dev
// instances sharing the same .next/dev lock/build directory.
//
// We compare request wall, API totalDurationMs, sweepDurationMs and every
// annual tank metric. A6 changes only ABI representation, so production
// physics should remain exact.
//
// No default flag is changed by this script.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import net from "node:net";
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

const reportPath =
  path.join(
    outbox,
    "approach-a-a7-production-year-ab.json"
  );

const referenceRawPath =
  path.join(
    outbox,
    "approach-a-a7-reference-runs.json"
  );

const candidateRawPath =
  path.join(
    outbox,
    "approach-a-a7-candidate-runs.json"
  );

const referenceLogPath =
  path.join(
    outbox,
    "approach-a-a7-reference-server.log"
  );

const candidateLogPath =
  path.join(
    outbox,
    "approach-a-a7-candidate-server.log"
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


function writeFailure(
  stage,
  error,
  extra = ""
) {

  fs.writeFileSync(
    failedRun,
    [
      `stage=${stage}`,
      `time=${new Date().toISOString()}`,
      `error=${error?.stack ?? error}`,
      extra,
      "",
    ].join(
      "\n"
    ),
    "utf8"
  );
}


function die(
  stage,
  message,
  extra = ""
) {

  writeFailure(
    stage,
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
      sorted.length /
      2
    );

  return sorted.length %
    2
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
      ) /
      2;
}


function summarize(
  values
) {

  return {
    values,
    median:
      median(
        values
      ),
    min:
      Math.min(
        ...values
      ),
    max:
      Math.max(
        ...values
      ),
  };
}


function compareTiming(
  reference,
  candidate
) {

  const ref =
    summarize(
      reference
    );

  const cand =
    summarize(
      candidate
    );

  const deltaPct =
    100 *
    (
      cand.median -
      ref.median
    ) /
    ref.median;

  return {
    reference:
      ref,

    candidate:
      cand,

    deltaPct,

    speedup:
      ref.median /
      cand.median,

    conclusion:
      Math.abs(
        deltaPct
      ) < 10
        ? "INCONCLUSIVE_LT_10_PERCENT"
        : (
            deltaPct < 0
              ? "A6_FASTER"
              : "REFERENCE_FASTER"
          ),
  };
}


function commandForNext(
  port
) {

  const comspec =
    process.env.ComSpec ||
    process.env.COMSPEC ||
    "C:\\Windows\\System32\\cmd.exe";

  return {
    command:
      comspec,

    args: [
      "/d",
      "/s",
      "/c",
      `npm run dev -- --hostname 127.0.0.1 --port ${port}`,
    ],
  };
}


function startServer(
  label,
  port,
  enabled,
  logPath
) {

  const {
    command,
    args,
  } =
    commandForNext(
      port
    );

  const log =
    fs.createWriteStream(
      logPath,
      {
        flags:
          "w",
      }
    );

  const child =
    spawn(
      command,
      args,
      {
        cwd:
          projectRoot,

        windowsHide:
          true,

        env: {
          ...process.env,

          SUNOPT_A6_PACKED_INTO:
            enabled
              ? "1"
              : "0",

          SUNOPT_CUDA_FAST_MATH:
            "0",

          PORT:
            String(
              port
            ),
        },

        stdio: [
          "ignore",
          "pipe",
          "pipe",
        ],
      }
    );


  child
    .stdout
    ?.pipe(
      log
    );

  child
    .stderr
    ?.pipe(
      log
    );


  return {
    label,
    port,
    enabled,
    child,
    log,
    logPath,
  };
}


async function stopServer(
  server
) {

  if (
    server.child.exitCode !==
    null
  ) {

    server.log.end();

    return;
  }


  if (
    process.platform ===
    "win32"
  ) {

    await new Promise(
      resolve => {

        const killer =
          spawn(
            "taskkill",
            [
              "/PID",
              String(
                server.child.pid
              ),
              "/T",
              "/F",
            ],
            {
              windowsHide:
                true,

              stdio:
                "ignore",
            }
          );


        killer.on(
          "exit",
          () =>
            resolve()
        );

        killer.on(
          "error",
          () =>
            resolve()
        );
      }
    );

  } else {

    try {
      server.child.kill(
        "SIGTERM"
      );
    } catch {}
  }


  server.log.end();
}


function probeTcpPort(
  port,
  timeoutMs =
    1000
) {

  return new Promise(
    resolve => {

      const socket =
        net.createConnection({
          host:
            "127.0.0.1",

          port,
        });


      let settled =
        false;


      function finish(
        value
      ) {

        if (
          settled
        ) {

          return;
        }


        settled =
          true;


        socket.destroy();


        resolve(
          value
        );
      }


      socket.setTimeout(
        timeoutMs
      );


      socket.once(
        "connect",
        () =>
          finish(
            true
          )
      );


      socket.once(
        "timeout",
        () =>
          finish(
            false
          )
      );


      socket.once(
        "error",
        () =>
          finish(
            false
          )
      );
    }
  );
}


async function waitForServer(
  server,
  timeoutMs =
    120_000
) {

  const start =
    performance.now();


  while (
    performance.now() -
      start <
    timeoutMs
  ) {

    if (
      server.child.exitCode !==
      null
    ) {

      die(
        "server-start",
        `${server.label} server exited before becoming ready with code ${server.child.exitCode}.`,
        `log=${server.logPath}`
      );
    }


    if (
      await probeTcpPort(
        server.port
      )
    ) {

      /*
       * Do NOT GET "/" here. The first explicit warmup request should own
       * route compilation. This avoids compiling the app/CSS graph merely
       * for readiness and avoids touching Visual Studio's locked .vs files.
       */
      await new Promise(
        resolve =>
          setTimeout(
            resolve,
            500
          )
      );


      return;
    }


    await new Promise(
      resolve =>
        setTimeout(
          resolve,
          250
        )
    );
  }


  die(
    "server-start-timeout",
    `${server.label} server was not listening on port ${server.port} within ${timeoutMs} ms.`,
    `log=${server.logPath}`
  );
}

const requestBody = {
  heatPumpThermalPower:
    5,

  pvPeakPower:
    10,

  gridPriceChfPerKWh:
    0.3,

  feedInPriceChfPerKWh:
    0.08,
};


async function runRequest(
  server,
  kind
) {

  const start =
    performance.now();


  const response =
    await fetch(
      `http://127.0.0.1:${server.port}/api/tank-comparison`,
      {
        method:
          "POST",

        headers: {
          "content-type":
            "application/json",
        },

        body:
          JSON.stringify(
            requestBody
          ),

        signal:
          AbortSignal.timeout(
            180_000
          ),
      }
    );


  const requestWallMs =
    performance.now() -
    start;


  const text =
    await response.text();


  if (
    !response.ok
  ) {

    die(
      "http-request",
      `${server.label} ${kind} request failed with HTTP ${response.status}.`,
      text.slice(
        0,
        5000
      )
    );
  }


  let body;

  try {

    body =
      JSON.parse(
        text
      );

  } catch (error) {

    die(
      "json-response",
      `${server.label} returned invalid JSON.`,
      [
        error?.stack ??
          String(
            error
          ),
        text.slice(
          0,
          5000
        ),
      ].join(
        "\n"
      )
    );
  }


  if (
    !Array.isArray(
      body.points
    ) ||
    body.points.length !==
      28
  ) {

    die(
      "response-shape",
      `${server.label} response does not contain 28 tank points.`,
      JSON.stringify(
        body,
        null,
        2
      ).slice(
        0,
        10_000
      )
    );
  }


  return {
    label:
      server.label,

    enabled:
      server.enabled,

    kind,

    requestWallMs,

    apiTotalDurationMs:
      Number(
        body.totalDurationMs
      ),

    sweepDurationMs:
      Number(
        body.sweepDurationMs
      ),

    preparationDurationMs:
      Number(
        body.preparationDurationMs
      ),

    response:
      body,
  };
}


const metricKeys = [
  "gridImport",
  "feedIn",
  "selfConsumption",
  "storageLosses",
  "unmetThermalDemand",
  "gridCost",
  "feedInRevenue",
  "netEnergyCost",
];


function compareResponses(
  reference,
  candidate
) {

  const a =
    reference.response;

  const b =
    candidate.response;


  const scalarKeys = [
    "fake",
    "year",
    "heatPumpThermalPower",
    "pvPeakPower",
    "gridPriceChfPerKWh",
    "feedInPriceChfPerKWh",
  ];


  for (
    const key
    of scalarKeys
  ) {

    if (
      a[
        key
      ] !==
      b[
        key
      ]
    ) {

      die(
        "production-parity",
        `Response scalar mismatch at ${key}: ${String(a[key])} vs ${String(b[key])}`
      );
    }
  }


  const maxima =
    Object.fromEntries(
      metricKeys.map(
        key => [
          key,
          {
            abs:
              0,

            storageVolume:
              null,

            reference:
              null,

            candidate:
              null,
          },
        ]
      )
    );


  for (
    let index = 0;
    index <
      a.points.length;
    index +=
      1
  ) {

    const pa =
      a.points[
        index
      ];

    const pb =
      b.points[
        index
      ];


    if (
      pa.storageVolume !==
      pb.storageVolume
    ) {

      die(
        "production-parity",
        `Tank order/volume mismatch at index ${index}: ${pa.storageVolume} vs ${pb.storageVolume}`
      );
    }


    for (
      const key
      of metricKeys
    ) {

      const av =
        Number(
          pa[
            key
          ]
        );

      const bv =
        Number(
          pb[
            key
          ]
        );

      const diff =
        Math.abs(
          av -
          bv
        );


      if (
        !Number.isFinite(
          av
        ) ||
        !Number.isFinite(
          bv
        )
      ) {

        die(
          "production-parity",
          `Non-finite ${key} at ${pa.storageVolume} L.`
        );
      }


      if (
        diff >
        maxima[
          key
        ].abs
      ) {

        maxima[
          key
        ] = {
          abs:
            diff,

          storageVolume:
            pa.storageVolume,

          reference:
            av,

          candidate:
            bv,
        };
      }
    }
  }


  return {
    exact:
      Object.values(
        maxima
      ).every(
        item =>
          item.abs ===
          0
      ),

    maxima,
  };
}


function optimum(
  response,
  key,
  direction
) {

  let best =
    null;


  for (
    const point
    of response.points
  ) {

    if (
      best ===
      null
    ) {

      best =
        point;

      continue;
    }


    if (
      direction ===
      "min"
        ? point[
            key
          ] <
          best[
            key
          ]
        : point[
            key
          ] >
          best[
            key
          ]
    ) {

      best =
        point;
    }
  }


  return best?.storageVolume ??
    null;
}


async function runFreshMeasuredProcess(
  label,
  enabled,
  sequenceIndex
) {

  const port =
    3111;

  const logPath =
    path.join(
      outbox,
      `approach-a-a7-${String(sequenceIndex + 1).padStart(2, "0")}-${label}-server.log`
    );


  const server =
    startServer(
      label,
      port,
      enabled,
      logPath
    );


  try {

    await waitForServer(
      server
    );


    /*
     * Excluded warmup:
     * route compilation + native/runtime cache population happen here.
     */
    await runRequest(
      server,
      `warmup-${sequenceIndex + 1}`
    );


    const measured =
      await runRequest(
        server,
        `measured-${sequenceIndex + 1}`
      );


    console.log(
      [
        `A7.1 ${label}`,
        `sequence=${sequenceIndex + 1}`,
        `wall=${measured.requestWallMs.toFixed(3)}ms`,
        `api=${measured.apiTotalDurationMs.toFixed(3)}ms`,
        `sweep=${measured.sweepDurationMs.toFixed(3)}ms`,
      ].join(
        " "
      )
    );


    return measured;

  } finally {

    await stopServer(
      server
    );


    /*
     * Give Windows a short moment to release the native .node file and
     * Next's .next/dev lock before npm predev runs for the next process.
     */
    await new Promise(
      resolve =>
        setTimeout(
          resolve,
          750
        )
    );
  }
}


async function main() {

  const sequence = [
    {
      label:
        "reference",

      enabled:
        false,
    },

    {
      label:
        "candidate",

      enabled:
        true,
    },

    {
      label:
        "candidate",

      enabled:
        true,
    },

    {
      label:
        "reference",

      enabled:
        false,
    },

    {
      label:
        "reference",

      enabled:
        false,
    },

    {
      label:
        "candidate",

      enabled:
        true,
    },

    {
      label:
        "candidate",

      enabled:
        true,
    },

    {
      label:
        "reference",

      enabled:
        false,
    },

    {
      label:
        "reference",

      enabled:
        false,
    },

    {
      label:
        "candidate",

      enabled:
        true,
    },
  ];


  const measured = {
    reference: [],
    candidate: [],
  };


  for (
    let index = 0;
    index <
      sequence.length;
    index +=
      1
  ) {

    const entry =
      sequence[
        index
      ];


    const result =
      await runFreshMeasuredProcess(
        entry.label,
        entry.enabled,
        index
      );


    measured[
      entry.label
    ].push(
      result
    );
  }


  if (
    measured.reference.length !==
      5 ||
    measured.candidate.length !==
      5
  ) {

    die(
      "measurement-count",
      `Expected 5 measured runs per variant, got reference=${measured.reference.length} candidate=${measured.candidate.length}.`
    );
  }


  /*
   * Every run must be exactly physically identical to the first reference
   * run. Timing fields are intentionally excluded from comparison.
   */
  const canonical =
    measured.reference[
      0
    ];

  const parityRuns =
    [];


  for (
    const run
    of measured.reference
  ) {

    parityRuns.push({
      variant:
        "reference",

      ...compareResponses(
        canonical,
        run
      ),
    });
  }


  for (
    const run
    of measured.candidate
  ) {

    parityRuns.push({
      variant:
        "candidate",

      ...compareResponses(
        canonical,
        run
      ),
    });
  }


  const exactParity =
    parityRuns.every(
      run =>
        run.exact
    );


  const performance = {
    requestWallMs:
      compareTiming(
        measured.reference.map(
          run =>
            run.requestWallMs
        ),

        measured.candidate.map(
          run =>
            run.requestWallMs
        ),
      ),

    apiTotalDurationMs:
      compareTiming(
        measured.reference.map(
          run =>
            run.apiTotalDurationMs
        ),

        measured.candidate.map(
          run =>
            run.apiTotalDurationMs
        ),
      ),

    sweepDurationMs:
      compareTiming(
        measured.reference.map(
          run =>
            run.sweepDurationMs
        ),

        measured.candidate.map(
          run =>
            run.sweepDurationMs
        ),
      ),
  };


  const referenceOptima = {
    netEnergyCost:
      optimum(
        canonical.response,
        "netEnergyCost",
        "min"
      ),

    gridImport:
      optimum(
        canonical.response,
        "gridImport",
        "min"
      ),

    selfConsumption:
      optimum(
        canonical.response,
        "selfConsumption",
        "max"
      ),
  };


  const candidateCanonical =
    measured.candidate[
      0
    ];


  const candidateOptima = {
    netEnergyCost:
      optimum(
        candidateCanonical.response,
        "netEnergyCost",
        "min"
      ),

    gridImport:
      optimum(
        candidateCanonical.response,
        "gridImport",
        "min"
      ),

    selfConsumption:
      optimum(
        candidateCanonical.response,
        "selfConsumption",
        "max"
      ),
  };


  const optimaMatch =
    JSON.stringify(
      referenceOptima
    ) ===
    JSON.stringify(
      candidateOptima
    );


  const sweepConclusion =
    performance
      .sweepDurationMs
      .conclusion;


  const decision =
    !exactParity ||
    !optimaMatch
      ? "REJECT_A6_PRODUCTION_INTEGRATION"
      : (
          sweepConclusion ===
            "A6_FASTER"
            ? "A6_DEFAULT_READY"
            : (
                sweepConclusion ===
                  "REFERENCE_FASTER"
                  ? "KEEP_A6_EXPERIMENTAL_REGRESSION"
                  : "A6_END_TO_END_GAIN_INCONCLUSIVE_LT_10_PERCENT"
              )
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
      "APPROACH_A_A7_1_PRODUCTION_YEAR_AB_FRESH_PROCESS",

    release,

    requestBody,

    variants: {
      reference: {
        SUNOPT_A6_PACKED_INTO:
          "0",
      },

      candidate: {
        SUNOPT_A6_PACKED_INTO:
          "1",
      },
    },

    measurementPolicy: {
      warmupPerFreshProcess:
        1,

      measuredRunsPerVariant:
        5,

      measuredRequestsPerFreshProcess:
        1,

      medianAuthoritative:
        true,

      noiseThresholdPercent:
        10,

      order:
        "R,C,C,R,R,C,C,R,R,C",

      freshNextProcessPerMeasuredRun:
        true,

      simultaneousNextProcesses:
        false,

      readinessProbe:
        "TCP_ONLY_NO_GET_ROOT",

      nativeLoader:
        "PROCESS_GET_BUILTIN_MODULE_CREATE_REQUIRE",

      rationale:
        "Avoid npm predev/native:install races, shared Next .next/dev locking, page/CSS compilation during readiness checks, and Turbopack static resolution of the native addon path.",
    },

    parity: {
      exact:
        exactParity,

      optimaMatch,

      referenceOptima,

      candidateOptima,

      runs:
        parityRuns,
    },

    performance,

    decision,

    nextCandidate:
      decision ===
        "A6_DEFAULT_READY"
        ? "Enable A6 packed-into as the default runtime path while retaining the old object path as fallback/reference."
        : (
            decision ===
              "A6_END_TO_END_GAIN_INCONCLUSIVE_LT_10_PERCENT"
              ? "Keep A6 behind the flag; isolated ABI gain is proven but full production gain is below the 10% confidence threshold."
              : "Do not default-enable A6; investigate the production result first."
          ),
  };


  fs.writeFileSync(
    referenceRawPath,
    JSON.stringify(
      measured.reference,
      null,
      2
    ) +
      "\n",
    "utf8"
  );


  fs.writeFileSync(
    candidateRawPath,
    JSON.stringify(
      measured.candidate,
      null,
      2
    ) +
      "\n",
    "utf8"
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


  const llm =
    path.join(
      outbox,
      "llm.txt"
    );


  fs.appendFileSync(
    llm,
    [
      "",
      `=== ${new Date().toISOString()} APPROACH_A_A7_1_PRODUCTION_YEAR_AB_FRESH_PROCESS ===`,
      `release=${release}`,
      `parity exact=${exactParity} optimaMatch=${optimaMatch}`,
      `requestWall ref=${performance.requestWallMs.reference.median.toFixed(3)}ms a6=${performance.requestWallMs.candidate.median.toFixed(3)}ms speedup=${performance.requestWallMs.speedup.toFixed(4)}x conclusion=${performance.requestWallMs.conclusion}`,
      `apiTotal ref=${performance.apiTotalDurationMs.reference.median.toFixed(3)}ms a6=${performance.apiTotalDurationMs.candidate.median.toFixed(3)}ms speedup=${performance.apiTotalDurationMs.speedup.toFixed(4)}x conclusion=${performance.apiTotalDurationMs.conclusion}`,
      `sweep ref=${performance.sweepDurationMs.reference.median.toFixed(3)}ms a6=${performance.sweepDurationMs.candidate.median.toFixed(3)}ms speedup=${performance.sweepDurationMs.speedup.toFixed(4)}x conclusion=${performance.sweepDurationMs.conclusion}`,
      `decision=${decision}`,
      `report=${reportPath}`,
      "",
    ].join(
      "\n"
    ),
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
    "APPROACH_A_A7_1_PRODUCTION_YEAR_AB_FRESH_PROCESS=OK"
  );
}

main()
  .catch(
    error => {

      if (
        !fs.existsSync(
          failedRun
        )
      ) {

        writeFailure(
          "unhandled",
          error
        );
      }


      console.error(
        error
      );

      process.exitCode =
        1;
    }
  );
