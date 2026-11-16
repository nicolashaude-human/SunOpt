// src/simulation/gpu-tuning-runtime.ts

import {
  spawn,

  type ChildProcess,
} from "node:child_process";

import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";

import {
  join,
} from "node:path";


export interface GpuTuningRuntimeConfig {
  enabled:
    boolean;

  deviceId:
    number;

  voltageMv:
    number;

  nativeFrequencyMHz:
    number;

  coreOffsetMHz:
    number;

  capBandPoints:
    number;

  showeffekt:
    boolean;

  showeffektFanPercent:
    number;
}


interface GpuTuningRuntimeState {
  started:
    boolean;

  child:
    ChildProcess | null;

  stopFile:
    string | null;

  logFile:
    string | null;
}


const globalKey =
  "__sunoptGpuTuningRuntimeState";


function globalState():
  GpuTuningRuntimeState {

  const holder =
    globalThis as typeof globalThis & {
      [globalKey]?:
        GpuTuningRuntimeState;
    };


  holder[globalKey] ??= {
    started:
      false,

    child:
      null,

    stopFile:
      null,

    logFile:
      null,
  };


  return holder[globalKey];
}


function sleepSync(
  milliseconds:
    number,
): void {

  const waitArray =
    new Int32Array(
      new SharedArrayBuffer(
        Int32Array.BYTES_PER_ELEMENT,
      ),
    );


  Atomics.wait(
    waitArray,
    0,
    0,
    milliseconds,
  );
}


function readTextIfPresent(
  path:
    string,
): string {

  if (
    !existsSync(
      path,
    )
  ) {
    return "";
  }


  try {

    return readFileSync(
      path,
      "utf8",
    );

  } catch {

    return "";
  }
}


export function ensureGpuTuningRuntime(
  config:
    GpuTuningRuntimeConfig,
): void {

  if (
    !config.enabled
  ) {
    return;
  }


  const state =
    globalState();


  if (
    state.started
  ) {
    return;
  }


  if (
    process.platform !==
    "win32"
  ) {

    console.warn(
      "[SunOpt] GPU tuning skipped: NVAPI runtime control is Windows-only.",
    );

    state.started =
      true;

    return;
  }


  const projectRoot =
    process.cwd();


  const guardScript =
    join(
      projectRoot,
      "rust",
      "sunopt-node",
      "scripts",
      "nvapi-blackwell-benchmark-guard.py",
    );


  if (
    !existsSync(
      guardScript,
    )
  ) {

    throw new Error(
      `SunOpt NVAPI runtime guard not found: ${guardScript}`,
    );
  }


  const stateDir =
    join(
      projectRoot,
      "rust",
      "sunopt-node",
      "blackwell-runtime-control",
      String(
        process.pid,
      ),
    );


  mkdirSync(
    stateDir,
    {
      recursive:
        true,
    },
  );


  const readyFile =
    join(
      stateDir,
      "ready.json",
    );

  const stopFile =
    join(
      stateDir,
      "stop.signal",
    );

  const doneFile =
    join(
      stateDir,
      "done.json",
    );

  const logFile =
    join(
      stateDir,
      "nvapi-runtime.log",
    );


  for (
    const path of [
      readyFile,
      stopFile,
      doneFile,
      logFile,
    ]
  ) {

    rmSync(
      path,
      {
        force:
          true,
      },
    );
  }


  const args = [
    guardScript,

    "--gpu",
    String(
      config.deviceId,
    ),

    "--target-voltage-mv",
    String(
      config.voltageMv,
    ),

    "--native-vf-frequency-mhz",
    String(
      config.nativeFrequencyMHz,
    ),

    "--curve-offset-mhz",
    String(
      config.coreOffsetMHz,
    ),

    "--cap-band-points",
    String(
      config.capBandPoints,
    ),

    "--parent-pid",
    String(
      process.pid,
    ),

    "--ready-file",
    readyFile,

    "--stop-file",
    stopFile,

    "--done-file",
    doneFile,

    "--log-file",
    logFile,
  ];


  if (
    config.showeffekt
  ) {

    args.push(
      "--fan-percent",
      String(
        config
          .showeffektFanPercent,
      ),
    );

  } else {

    args.push(
      "--fan-auto",
    );
  }


  const python =
    process.env
      .SUNOPT_PYTHON ??
    "python";


  const child =
    spawn(
      python,
      args,
      {
        cwd:
          projectRoot,

        stdio:
          "ignore",

        windowsHide:
          true,
      },
    );


  state.child =
    child;

  state.stopFile =
    stopFile;

  state.logFile =
    logFile;


  const deadline =
    Date.now() +
    30_000;


  while (
    Date.now() <
      deadline
  ) {

    if (
      existsSync(
        readyFile,
      )
    ) {

      const ready =
        JSON.parse(
          readFileSync(
            readyFile,
            "utf8",
          ),
        ) as {
          status?: string;

          reportedFanPercent?:
            number | null;

          idleClockMHz?:
            number | null;

          temperatureC?:
            number | null;

          fanMode?:
            string;
        };


      if (
        ready.status !==
        "READY"
      ) {

        throw new Error(
          `Unexpected NVAPI runtime state: ${JSON.stringify(ready)}`,
        );
      }


      state.started =
        true;


      console.log(
        "[SunOpt] RTX 5060 runtime tuning active.",
        {
          voltageMv:
            config.voltageMv,

          coreOffsetMHz:
            config.coreOffsetMHz,

          showeffekt:
            config.showeffekt,

          fanMode:
            ready.fanMode ??
            (
              config.showeffekt
                ? "manual"
                : "automatic"
            ),

          fanPercent:
            ready
              .reportedFanPercent ??
            null,

          idleClockMHz:
            ready.idleClockMHz ??
            null,

          temperatureC:
            ready.temperatureC ??
            null,
        },
      );


      process.once(
        "exit",
        () => {

          try {

            writeFileSync(
              stopFile,
              "",
            );

          } catch {
          }
        },
      );


      return;
    }


    if (
      child.exitCode !==
      null
    ) {

      const log =
        readTextIfPresent(
          logFile,
        );


      throw new Error(
        "SunOpt NVAPI runtime control exited before READY." +
        (
          log
            ? `\n${log}`
            : ""
        ),
      );
    }


    sleepSync(
      100,
    );
  }


  const log =
    readTextIfPresent(
      logFile,
    );


  throw new Error(
    "Timed out waiting for SunOpt NVAPI runtime control." +
    (
      log
        ? `\n${log}`
        : ""
    ),
  );
}
