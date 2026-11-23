#!/usr/bin/env python3
"""Standalone capability check: sets both GPU controls and immediately restores them.
No GPU sweep is run; this does not prove the loaded SM clock is 3480 MHz.
"""
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('sunopt_gpu_profile_guard', Path(__file__).with_name('sunopt-gpu-profile-guard.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
outbox = root / 'updates' / 'outbox'
outbox.mkdir(parents=True, exist_ok=True)
result = {
    'probeTime': datetime.now(timezone.utc).isoformat(),
    'requested': {'smClockMHz': 3480, 'fanPercent': 70},
    'status': 'NOT_RUN', 'error': None, 'note': 'Clock 3480 MHz under load must still be verified by real A/B run',
}
locked = False
fans = []
dev = None
file_lock = None
try:
    file_lock = mod.lock_guard_file(outbox / 'sunopt-gpu-profile.lock')
    dev = mod.NvidiaNvml()
    dev.lock_clock()
    locked = True
    for fan in dev.fans():
        dev.set_fan(fan)
        fans.append(fan)
    result['status'] = 'NVML_CONTROLS_ACCEPTED'
    result['fanCount'] = len(fans)
    result['currentReading'] = dev.sample(fans)
except Exception as exc:
    result['status'] = 'FAILED'
    result['error'] = str(exc)
finally:
    if dev:
        restore = []
        for fan in fans:
            try:
                dev.reset_fan(fan)
            except Exception as exc:
                restore.append(str(exc))
        if locked:
            try:
                dev.reset_clock()
            except Exception as exc:
                restore.append(str(exc))
        try:
            dev.close()
        except Exception as exc:
            restore.append(str(exc))
        result['restoreErrors'] = restore
        if restore:
            result['status'] = 'RESTORE_FAILED'
    if file_lock:
        file_lock.close()

name = 'sunopt-gpu-profile-probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
(outbox / name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
print('Report:', 'updates/outbox/' + name)
sys.exit(0 if result['status'] == 'NVML_CONTROLS_ACCEPTED' else 1)
