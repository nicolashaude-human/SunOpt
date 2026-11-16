#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path


ROOT = Path.cwd()
INBOX = ROOT / "updates" / "inbox"
HASH_FILE = ROOT / "updates" / "history" / "hash.txt"
RELEASE_FILE = ROOT / "updates" / "installed-release.json"

CONFIG_FILE = ROOT / "src" / "config" / "config.ts"
ACCELERATOR_FILE = ROOT / "src" / "simulation" / "peak-shaving-causal-accelerator.ts"
GUARD_FILE = (
    ROOT
    / "rust"
    / "sunopt-node"
    / "scripts"
    / "nvapi-blackwell-benchmark-guard.py"
)

CONFIG_PATH = "src/config/config.ts"
ACCELERATOR_PATH = "src/simulation/peak-shaving-causal-accelerator.ts"
RUNTIME_PATH = "src/simulation/gpu-tuning-runtime.ts"
GUARD_PATH = "rust/sunopt-node/scripts/nvapi-blackwell-benchmark-guard.py"

GENESIS = "0" * 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

CONFIG_BEGIN = "// SUNOPT_GPU_TUNING_CONFIG_BEGIN"
CONFIG_END = "// SUNOPT_GPU_TUNING_CONFIG_END"

CONFIG_BLOCK = r'''
// SUNOPT_GPU_TUNING_CONFIG_BEGIN
/*
 * RTX 5060 runtime tuning.
 *
 * `showeffekt` affects ONLY the fan mode:
 *
 *   true  -> manual 100% fan speed for visible/audible show effect
 *   false -> leave NVIDIA automatic fan control untouched
 *
 * The core V/F tuning remains enabled independently of `showeffekt`.
 *
 * Proven stable region:
 *   +600 MHz -> stable
 *   +650 MHz -> stable
 *   +700 MHz -> system/GPU hang during warm-up
 *
 * Production choice:
 *   +625 MHz at the native 1050 mV / 2865 MHz V/F point.
 */
export const GPU_TUNING_CONFIG = {
  enabled:
    true,

  deviceId:
    0,

  voltageMv:
    1050,

  nativeFrequencyMHz:
    2865,

  coreOffsetMHz:
    625,

  capBandPoints:
    8,

  showeffekt:
    true,

  showeffektFanPercent:
    100,
} as const;
// SUNOPT_GPU_TUNING_CONFIG_END
'''.strip()


RUNTIME_TS = r'''// src/simulation/gpu-tuning-runtime.ts

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
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_parent_hash() -> str:
    if HASH_FILE.is_file():
        value = HASH_FILE.read_text(encoding="utf-8").strip().lower()
        if HASH_RE.fullmatch(value):
            return value

    if RELEASE_FILE.is_file():
        state = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
        value = str(state.get("sha256", "")).strip().lower()
        if HASH_RE.fullmatch(value):
            return value

    return GENESIS


def replace_between_markers(
    source: str,
    begin: str,
    end: str,
    replacement: str,
) -> str:
    if begin in source or end in source:
        if begin not in source or end not in source:
            raise RuntimeError(
                f"Only one marker found: {begin!r} / {end!r}"
            )

        start = source.index(begin)
        finish = source.index(end, start) + len(end)

        return (
            source[:start]
            + replacement
            + source[finish:]
        )

    separator = "" if source.endswith("\n") else "\n"

    return (
        source
        + separator
        + "\n"
        + replacement
        + "\n"
    )


def patch_config(source: str) -> str:
    return replace_between_markers(
        source,
        CONFIG_BEGIN,
        CONFIG_END,
        CONFIG_BLOCK,
    )


def patch_accelerator(source: str) -> str:
    if "// SUNOPT_GPU_TUNING_IMPORTS_BEGIN" not in source:
        config_import = re.search(
            r'import\s*\{[^}]*\bCONFIG\b[^}]*\}\s*'
            r'from\s*"@/config/config";',
            source,
            flags=re.DOTALL,
        )

        if not config_import:
            raise RuntimeError(
                "Could not find the existing CONFIG import in "
                "peak-shaving-causal-accelerator.ts"
            )

        addition = r'''

// SUNOPT_GPU_TUNING_IMPORTS_BEGIN
import {
  GPU_TUNING_CONFIG,
} from "@/config/config";

import {
  ensureGpuTuningRuntime,
} from "./gpu-tuning-runtime";
// SUNOPT_GPU_TUNING_IMPORTS_END
'''

        insert_at = config_import.end()

        source = (
            source[:insert_at]
            + addition
            + source[insert_at:]
        )

    if "// SUNOPT_GPU_TUNING_CALL_BEGIN" not in source:
        assignment = re.search(
            r'(?P<indent>[ \t]*)cudaWorker\s*=\s*\n?'
            r'[ \t]*new\s+CudaCausalWorker\s*\(',
            source,
        )

        if not assignment:
            raise RuntimeError(
                "Could not find CudaCausalWorker construction point."
            )

        indent = assignment.group("indent")

        call = (
            f'{indent}// SUNOPT_GPU_TUNING_CALL_BEGIN\n'
            f'{indent}try {{\n\n'
            f'{indent}  ensureGpuTuningRuntime(\n'
            f'{indent}    GPU_TUNING_CONFIG,\n'
            f'{indent}  );\n\n'
            f'{indent}}} catch (error) {{\n\n'
            f'{indent}  console.warn(\n'
            f'{indent}    "[SunOpt] GPU tuning unavailable; CUDA continues with driver defaults.",\n'
            f'{indent}    error,\n'
            f'{indent}  );\n'
            f'{indent}}}\n'
            f'{indent}// SUNOPT_GPU_TUNING_CALL_END\n\n\n'
        )

        source = (
            source[:assignment.start()]
            + call
            + source[assignment.start():]
        )

    return source


def patch_guard(source: str) -> str:
    if '--fan-auto' not in source:
        old = (
            '    ap.add_argument("--fan-percent", type=int, default=70)\n'
            '    ap.add_argument("--fan-tolerance", type=int, default=3)\n'
        )

        new = (
            '    ap.add_argument("--fan-percent", type=int, default=70)\n'
            '    ap.add_argument("--fan-auto", action="store_true")\n'
            '    ap.add_argument("--fan-tolerance", type=int, default=3)\n'
        )

        if old not in source:
            raise RuntimeError(
                "Could not patch NVAPI guard CLI for --fan-auto."
            )

        source = source.replace(
            old,
            new,
            1,
        )

    old_log = (
        '        log.line(f"fanPercent={args.fan_percent}")\n'
        '        log.line(f"curveOffsetMHz={args.curve_offset_mhz}")\n'
    )

    new_log = (
        '        log.line(f"fanPercent={args.fan_percent}")\n'
        '        log.line(\n'
        '            "fanMode=" + ("automatic" if args.fan_auto else "manual")\n'
        '        )\n'
        '        log.line(f"curveOffsetMHz={args.curve_offset_mhz}")\n'
    )

    if "fanMode=" not in source:
        if old_log not in source:
            raise RuntimeError(
                "Could not patch NVAPI guard fan-mode logging."
            )

        source = source.replace(
            old_log,
            new_log,
            1,
        )

    old_fan = '''        fan_candidate = make_fan_candidate(original_fan, args.fan_percent)
        status = nv.set_raw(handle, ID_GPU_SET_FAN_CONTROL, fan_candidate)
        log.line(f"fanControl.set={nv.status_text(status)}")
        if status != 0:
            raise RuntimeError("fan-control write failed")
        fan_written = True

        status, fan_rb = nv.get_raw(
            handle,
            ID_GPU_GET_FAN_CONTROL,
            FAN_CONTROL_CLAIMED_SIZE,
            1,
            alloc_size=FAN_CONTROL_ALLOC_SIZE,
        )
        log.line(f"fanControl.readback={nv.status_text(status)}")
        if status != 0 or fan_rb is None:
            raise RuntimeError("fan-control readback failed")
        _, _, fan_entries = decode_fan_control(fan_rb)
        for e in fan_entries:
            log.line(
                f"fanControl.entry={e['entry']} index={e['index']} "
                f"level={e['level']} mode={e['mode']}"
            )
        if not fan_entries or any(
            e["level"] != args.fan_percent or e["mode"] != 1
            for e in fan_entries
        ):
            raise RuntimeError("fan-control readback mismatch")
'''

    new_fan = '''        if args.fan_auto:
            log.line("fanControl.set=SKIPPED_AUTOMATIC")
        else:
            fan_candidate = make_fan_candidate(
                original_fan,
                args.fan_percent,
            )
            status = nv.set_raw(
                handle,
                ID_GPU_SET_FAN_CONTROL,
                fan_candidate,
            )
            log.line(f"fanControl.set={nv.status_text(status)}")
            if status != 0:
                raise RuntimeError("fan-control write failed")
            fan_written = True

            status, fan_rb = nv.get_raw(
                handle,
                ID_GPU_GET_FAN_CONTROL,
                FAN_CONTROL_CLAIMED_SIZE,
                1,
                alloc_size=FAN_CONTROL_ALLOC_SIZE,
            )
            log.line(f"fanControl.readback={nv.status_text(status)}")
            if status != 0 or fan_rb is None:
                raise RuntimeError("fan-control readback failed")
            _, _, fan_entries = decode_fan_control(fan_rb)
            for e in fan_entries:
                log.line(
                    f"fanControl.entry={e['entry']} index={e['index']} "
                    f"level={e['level']} mode={e['mode']}"
                )
            if not fan_entries or any(
                e["level"] != args.fan_percent or e["mode"] != 1
                for e in fan_entries
            ):
                raise RuntimeError("fan-control readback mismatch")
'''

    if "fanControl.set=SKIPPED_AUTOMATIC" not in source:
        if old_fan not in source:
            raise RuntimeError(
                "Could not patch NVAPI guard manual fan-control block."
            )

        source = source.replace(
            old_fan,
            new_fan,
            1,
        )

    old_ramp = '''        # Let the physical fans ramp to the requested setting.
        stable = 0
        last_smi = None
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            last_smi = read_nvidia_smi(args.gpu)
            if last_smi is not None:
                log.line(
                    "fanRamp "
                    f"fan={last_smi['fanPercent']:.0f}% "
                    f"temp={last_smi['temperatureC']:.0f}C "
                    f"clock={last_smi['clockMHz']:.0f}MHz"
                )
                if abs(last_smi["fanPercent"] - args.fan_percent) <= args.fan_tolerance:
                    stable += 1
                else:
                    stable = 0
                if stable >= 3:
                    break
            time.sleep(0.5)

        if stable < 3:
            raise RuntimeError(
                f"fan did not stabilize at {args.fan_percent} +/- "
                f"{args.fan_tolerance}%"
            )
'''

    new_ramp = '''        # Manual show-effect mode waits for the physical fans.
        # Automatic mode deliberately leaves NVIDIA fan control untouched.
        stable = 0
        last_smi = None

        if args.fan_auto:
            last_smi = read_nvidia_smi(args.gpu)
            if last_smi is not None:
                log.line(
                    "fanAutomatic "
                    f"fan={last_smi['fanPercent']:.0f}% "
                    f"temp={last_smi['temperatureC']:.0f}C "
                    f"clock={last_smi['clockMHz']:.0f}MHz"
                )
        else:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                last_smi = read_nvidia_smi(args.gpu)
                if last_smi is not None:
                    log.line(
                        "fanRamp "
                        f"fan={last_smi['fanPercent']:.0f}% "
                        f"temp={last_smi['temperatureC']:.0f}C "
                        f"clock={last_smi['clockMHz']:.0f}MHz"
                    )
                    if (
                        abs(
                            last_smi["fanPercent"] -
                            args.fan_percent
                        )
                        <= args.fan_tolerance
                    ):
                        stable += 1
                    else:
                        stable = 0
                    if stable >= 3:
                        break
                time.sleep(0.5)

            if stable < 3:
                raise RuntimeError(
                    f"fan did not stabilize at {args.fan_percent} +/- "
                    f"{args.fan_tolerance}%"
                )
'''

    if "fanAutomatic " not in source:
        if old_ramp not in source:
            raise RuntimeError(
                "Could not patch NVAPI guard fan ramp block."
            )

        source = source.replace(
            old_ramp,
            new_ramp,
            1,
        )

    old_ready = '''            "fanPercent": args.fan_percent,
            "curveOffsetMHz": args.curve_offset_mhz,
'''

    new_ready = '''            "fanPercent": None if args.fan_auto else args.fan_percent,
            "fanMode": "automatic" if args.fan_auto else "manual",
            "curveOffsetMHz": args.curve_offset_mhz,
'''

    if '"fanMode": "automatic"' not in source:
        if old_ready not in source:
            raise RuntimeError(
                "Could not patch NVAPI guard READY payload."
            )

        source = source.replace(
            old_ready,
            new_ready,
            1,
        )

    return source


def build_package() -> tuple[Path, str]:
    required = [
        CONFIG_FILE,
        ACCELERATOR_FILE,
        GUARD_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Required current project files are missing:\n- "
            + "\n- ".join(missing)
        )

    parent = read_parent_hash()

    current_config = CONFIG_FILE.read_text(encoding="utf-8")
    current_accelerator = ACCELERATOR_FILE.read_text(encoding="utf-8")
    current_guard = GUARD_FILE.read_text(encoding="utf-8")

    patched_config = patch_config(current_config)
    patched_accelerator = patch_accelerator(current_accelerator)
    patched_guard = patch_guard(current_guard)

    payload = {
        CONFIG_PATH:
            patched_config.encode("utf-8"),

        ACCELERATOR_PATH:
            patched_accelerator.encode("utf-8"),

        RUNTIME_PATH:
            RUNTIME_TS.encode("utf-8"),

        GUARD_PATH:
            patched_guard.encode("utf-8"),
    }

    manifest = {
        "format":
            2,

        "parent_sha256":
            parent,

        "files": {
            path:
                sha256_bytes(data)

            for path, data
            in payload.items()
        },

        "delete":
            [],
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    release = sha256_bytes(
        manifest_bytes,
    )

    INBOX.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        INBOX
        / "20261106-gpu-runtime-625-showeffekt.zip"
    )

    with zipfile.ZipFile(
        out,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        archive.writestr(
            "manifest.json",
            manifest_bytes,
        )

        archive.writestr(
            "manifest.sha256",
            release + "\n",
        )

        for path, data in payload.items():

            archive.writestr(
                path,
                data,
            )

    with zipfile.ZipFile(
        out,
        "r",
    ) as archive:

        names = archive.namelist()

        expected_names = (
            set(payload)
            | {
                "manifest.json",
                "manifest.sha256",
            }
        )

        if set(names) != expected_names:
            raise RuntimeError(
                "ZIP membership verification failed."
            )

        if len(names) != len(set(names)):
            raise RuntimeError(
                "ZIP contains duplicate members."
            )

        stored_manifest = archive.read(
            "manifest.json",
        )

        stored_release = archive.read(
            "manifest.sha256",
        ).decode("ascii").strip()

        if sha256_bytes(stored_manifest) != stored_release:
            raise RuntimeError(
                "manifest.sha256 verification failed."
            )

        for path, expected_hash in manifest["files"].items():

            actual = sha256_bytes(
                archive.read(
                    path,
                ),
            )

            if actual != expected_hash:
                raise RuntimeError(
                    f"Payload hash mismatch: {path}"
                )

    return out, release


def main() -> int:
    try:
        out, release = build_package()

        print("SunOpt GPU runtime tuning package created.")
        print(f"Package: {out}")
        print(f"Release: {release}")
        print()
        print("Configured:")
        print("  core offset: +625 MHz")
        print("  voltage point: 1050 mV / 2865 MHz")
        print("  showeffekt=true  -> fans manual 100%")
        print("  showeffekt=false -> NVIDIA automatic fan mode")
        print()
        print(
            "The update watcher may apply the package automatically "
            "if --watch is already running."
        )

        return 0

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
