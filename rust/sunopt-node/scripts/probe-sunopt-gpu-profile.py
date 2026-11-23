#!/usr/bin/env python3
"""A short NVAPI control/readback/restore probe without a CUDA annual sweep."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import os

root = Path(__file__).resolve().parents[3]
outbox = root / 'updates' / 'outbox'
outbox.mkdir(parents=True, exist_ok=True)
guard = Path(__file__).with_name('sunopt-gpu-profile-guard.py')
name = 'sunopt-gpu-profile-probe-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'
report = {'probeTime': datetime.now(timezone.utc).isoformat(),
          'requested': {'smClockMHz': 3480, 'fanPercent': 70, 'voltageMv': 1050,
                        'nativeVfFrequencyMHz': 2865, 'curveOffsetMHz': 625},
          'status': 'NOT_RUN', 'error': None,
          'note': 'Checks NVAPI V/F, fan setpoint readback and restoration; active 3480 MHz requires real A/B load.'}
try:
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    run = subprocess.run([sys.executable, str(guard), '--parent-pid', str(os.getpid()),
                          '--context', 'B', '--outbox', str(outbox), '--probe'],
                         capture_output=True, text=True, timeout=75, creationflags=flags)
    events = []
    for line in run.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    final = next((x for x in reversed(events) if x.get('status') == 'done'), None)
    if final:
        report['guardReport'] = final.get('reportPath')
        report['guardLog'] = final.get('guardLog')
        report['controlsAccepted'] = final.get('targetAcceptedByDriver')
        report['restored'] = final.get('restoreConfirmed')
        report['status'] = ('NVAPI_CONTROLS_ACCEPTED' if run.returncode == 0 and final.get('probePassed')
                            else 'FAILED')
        if report['status'] == 'FAILED':
            report['error'] = final.get('error') or final.get('restoreError') or final.get('endReason')
    else:
        report['status'] = 'FAILED'
        report['error'] = next((e.get('error') for e in events if e.get('status') == 'error'), None) or (
            'Guard returned without DONE; exit=' + str(run.returncode))
    if run.stderr.strip():
        report['stderrTail'] = run.stderr[-1500:]
except Exception as exc:
    report['status'] = 'FAILED'
    report['error'] = str(exc)
(outbox / name).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False))
print('Report: updates/outbox/' + name)
sys.exit(0 if report['status'] == 'NVAPI_CONTROLS_ACCEPTED' else 1)
