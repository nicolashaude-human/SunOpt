// Approach A7 — full production annual A/B for A6.
//
// Two separate Next dev servers:
//   reference: SUNOPT_A6_PACKED_INTO=0
//   candidate: SUNOPT_A6_PACKED_INTO=1
//
// Each server gets one excluded warmup request.
// Then five measured requests per variant in rotating order.
// We compare request wall, API totalDurationMs, sweepDurationMs and every
// annual tank metric. A6 changes only ABI representation, so production
// physics should remain exact.
//
// No default flag is changed by this script.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
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


async function waitForServer(
  server,
  timeoutMs =
    120_000
) {

  const start =
    performance.now();

  let lastError =
    null;


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


    try {

      const response =
        await fetch(
          `http://127.0.0.1:${server.port}/`,
          {
            signal:
              AbortSignal.timeout(
                5000
              ),
          }
        );


      if (
        response.status >=
          200 &&
        response.status <
          600
      ) {

        return;
      }

    } catch (error) {

      lastError =
        error;
    }


    await new Promise(
      resolve =>
        setTimeout(
          resolve,
          500
        )
    );
  }


  die(
    "server-start-timeout",
    `${server.label} server was not ready within ${timeoutMs} ms.`,
    [
      `lastError=${lastError?.stack ?? lastError}`,
      `log=${server.logPath}`,
    ].join(
      "\n"
    )
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


async function main() {

  const reference =
    startServer(
      "reference",
      3111,
      false,
      referenceLogPath
    );

  const candidate =
    startServer(
      "candidate",
      3112,
      true,
      candidateLogPath
    );


  try {

    await Promise.all([
      waitForServer(
        reference
      ),

      waitForServer(
        candidate
      ),
    ]);


    /*
     * Excluded warmups compile the route and populate caches.
     */
    await runRequest(
      reference,
      "warmup"
    );

    await runRequest(
      candidate,
      "warmup"
    );


    const measured = {
      reference: [],
      candidate: [],
    };


    const order = [
      reference,
      candidate,
      candidate,
      reference,
      reference,
      candidate,
      candidate,
      reference,
      reference,
      candidate,
    ];


    for (
      let index = 0;
      index <
        order.length;
      index +=
        1
    ) {

      const server =
        order[
          index
        ];

      const result =
        await runRequest(
          server,
          `measured-${index + 1}`
        );


      measured[
        server.label
      ].push(
        result
      );


      console.log(
        [
          `A7 ${server.label}`,
          `wall=${result.requestWallMs.toFixed(3)}ms`,
          `api=${result.apiTotalDurationMs.toFixed(3)}ms`,
          `sweep=${result.sweepDurationMs.toFixed(3)}ms`,
        ].join(
          " "
        )
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
     * Every candidate run must be exactly physically identical to the first
     * reference run. We intentionally ignore timing fields.
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
        "APPROACH_A_A7_PRODUCTION_YEAR_AB",

      release,

      requestBody,

      variants: {
        reference: {
          SUNOPT_A6_PACKED_INTO:
            "0",

          port:
            reference.port,
        },

        candidate: {
          SUNOPT_A6_PACKED_INTO:
            "1",

          port:
            candidate.port,
        },
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

        order:
          "R,C,C,R,R,C,C,R,R,C",

        separateNextProcesses:
          true,
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
          ? "Enable SUNOPT_A6_PACKED_INTO behavior as the default path while retaining the old object path as fallback/reference."
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
        `=== ${new Date().toISOString()} APPROACH_A_A7_PRODUCTION_YEAR_AB ===`,
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
      "APPROACH_A_A7_PRODUCTION_YEAR_AB=OK"
    );

  } finally {

    await Promise.all([
      stopServer(
        reference
      ),

      stopServer(
        candidate
      ),
    ]);
  }
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
