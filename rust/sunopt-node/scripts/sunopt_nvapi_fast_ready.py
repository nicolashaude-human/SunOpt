#!/usr/bin/env python3
"""Runtime-only production variant of the existing B14P/Q NVAPI guard.

The benchmark guard remains untouched. For normal productive A/B runs, the
manual 70% fan CONTROL readback is sufficient to report READY. Waiting for
physical RPM ramp is deliberately omitted; RPM is telemetry, not the command.
Exact-match patch or FAIL CLOSED; no V/F, fan, clock or restore code changed.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path

OLD_RAMP = '        # Let the physical fans ramp to the requested setting.\n        stable = 0\n        last_smi = None\n        deadline = time.monotonic() + 20.0\n        while time.monotonic() < deadline:\n            last_smi = read_nvidia_smi(args.gpu)\n            if last_smi is not None:\n                log.line(\n                    "fanRamp "\n                    f"fan={last_smi[\'fanPercent\']:.0f}% "\n                    f"temp={last_smi[\'temperatureC\']:.0f}C "\n                    f"clock={last_smi[\'clockMHz\']:.0f}MHz"\n                )\n                if abs(last_smi["fanPercent"] - args.fan_percent) <= args.fan_tolerance:\n                    stable += 1\n                else:\n                    stable = 0\n                if stable >= 3:\n                    break\n            time.sleep(0.5)\n\n        if stable < 3:\n            raise RuntimeError(\n                f"fan did not stabilize at {args.fan_percent} +/- "\n                f"{args.fan_tolerance}%"\n            )\n'
NEW_RAMP = '        # Manual show-effect mode waits for the physical fans.\n        # Automatic mode deliberately leaves NVIDIA fan control untouched.\n        stable = 0\n        last_smi = None\n\n        if args.fan_auto:\n            last_smi = read_nvidia_smi(args.gpu)\n            if last_smi is not None:\n                log.line(\n                    "fanAutomatic "\n                    f"fan={last_smi[\'fanPercent\']:.0f}% "\n                    f"temp={last_smi[\'temperatureC\']:.0f}C "\n                    f"clock={last_smi[\'clockMHz\']:.0f}MHz"\n                )\n        else:\n            deadline = time.monotonic() + 20.0\n            while time.monotonic() < deadline:\n                last_smi = read_nvidia_smi(args.gpu)\n                if last_smi is not None:\n                    log.line(\n                        "fanRamp "\n                        f"fan={last_smi[\'fanPercent\']:.0f}% "\n                        f"temp={last_smi[\'temperatureC\']:.0f}C "\n                        f"clock={last_smi[\'clockMHz\']:.0f}MHz"\n                    )\n                    if (\n                        abs(\n                            last_smi["fanPercent"] -\n                            args.fan_percent\n                        )\n                        <= args.fan_tolerance\n                    ):\n                        stable += 1\n                    else:\n                        stable = 0\n                    if stable >= 3:\n                        break\n                time.sleep(0.5)\n\n            if stable < 3:\n                raise RuntimeError(\n                    f"fan did not stabilize at {args.fan_percent} +/- "\n                    f"{args.fan_tolerance}%"\n                )\n'

FAST_READY = '        # NVAPI has already verified the manual fan command.\n        # The production adapter samples live telemetry through NVML.\n        last_smi = None\n'

def prepare_fast_nvapi_script(source_file: Path, destination: Path) -> Path:
    source = source_file.read_text(encoding='utf-8')
    # Match a single, known block from the tested B14 NVAPI implementation.
    found = [block for block in (OLD_RAMP, NEW_RAMP) if source.count(block) == 1]
    if len(found) != 1:
        raise RuntimeError(
            'Cannot safely enable B14 NVAPI fast-ready: unsupported original fan-ramp block. '
            'Original guard not changed. Capture the original source for review.'
        )
    for critical in ('fanControl.readback=', 'curveControl.readback=', 'clockLock.readback=',
                     'curveControl.restore.', 'fanControl.restore.'):
        if critical not in source:
            raise RuntimeError('NVAPI source missing safety marker ' + critical)
    result = source.replace(found[0], FAST_READY, 1)
    ast.parse(result, filename=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(result, encoding='utf-8')
    os.replace(temp, destination)
    return destination
