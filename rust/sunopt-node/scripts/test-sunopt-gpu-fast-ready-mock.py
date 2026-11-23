#!/usr/bin/env python3
# Mock end-to-end child start/STOP/restore with an original-version fan ramp.
# Does NOT call NVAPI or change clocks/fans.
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import runpy
import struct
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).parent))
from sunopt_nvapi_fast_ready import NEW_RAMP
spec = importlib.util.spec_from_file_location('gpu_guard',Path(__file__).with_name('sunopt-gpu-profile-guard.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
with tempfile.TemporaryDirectory() as td:
    root=Path(td); out=root/'outbox';out.mkdir();runtime=root/'runtime';runtime.mkdir()
    original=root/'nvapi-blackwell-benchmark-guard.py'
    # Only a mock for runtime protocol. The NVAPI markers are deliberately comments.
    script='''# fanControl.readback=\n# curveControl.readback=\n# clockLock.readback=\n# curveControl.restore.\n# fanControl.restore.\nimport argparse,json,time\nfrom pathlib import Path\ndef unused_original_fan_block():\n'''+NEW_RAMP+'''        pass\np=argparse.ArgumentParser()\nfor a in ('gpu','target-voltage-mv','native-vf-frequency-mhz','curve-offset-mhz','cap-band-points','fan-percent','parent-pid','ready-file','stop-file','done-file','log-file'):p.add_argument('--'+a)\np.add_argument('--allow-native-vf-bin-drift',action='store_true')\np.add_argument('--fast-control-readback',action='store_true')\na=p.parse_args()\nassert a.fan_percent=='70' and a.curve_offset_mhz=='625' and a.allow_native_vf_bin_drift and a.fast_control_readback\nPath(a.ready_file).write_text(json.dumps({'status':'READY'}))\nwhile not Path(a.stop_file).exists(): time.sleep(0.03)\nPath(a.log_file).write_text('''+repr('''curveControl.readback=0 (NVAPI_OK)
clockLock.readback=0 (NVAPI_OK)
fanControl.readback=0 (NVAPI_OK)
fanControl.entry=0 index=1 level=70 mode=1
fanControl.entry=1 index=2 level=70 mode=1
clockLock.restore.set=0 (NVAPI_OK)
curveControl.restore.exactMatch=True
fanControl.restore.read=0 (NVAPI_OK)
restoreOk=True
''')+''')\nPath(a.done_file).write_text(json.dumps({'status':'DONE','restoreOk':True}))\n'''
    original.write_text(script,encoding='utf-8')
    client=mod.LegacyNvapiGuard(out,runtime,os.getpid(),'B',script=original)
    client.start()
    result=client.finish()
    assert result['restoreOk'] and result['curveRestoredExactly'] and result['fanControlReadback'] and result['clockLockReadback'],result
    assert original.read_text(encoding='utf-8')==script
    print('MOCK_FAST_READY_PASS: startup, STOP, NVAPI readback validation, restore, original untouched')
    busy=[{'gpuUtilization':95,'clockMHz':3480,'temperatureC':42,'fanPercent':[v,v]} for v in (0,20,40,52,62,68)]
    accepted=mod.make_report(busy,mod.time.monotonic()-1,'requested_stop','B',True,result)
    assert accepted['compliant'] is True and accepted['fanTelemetryVerified'] is False
    wrong=[dict(x,clockMHz=2880) for x in busy]
    rejected=mod.make_report(wrong,mod.time.monotonic()-1,'requested_stop','B',True,result)
    assert rejected['compliant'] is False
    print('MOCK_TELEMETRY_PASS: 3480 accepted, 2880 rejected, fan RPM ramp not treated as command failure')

    def check_clocks(clocks):
        samples = [dict(busy[0], clockMHz=clock) for clock in clocks]
        return mod.make_report(samples, mod.time.monotonic() - 1, 'requested_stop', 'B', True, result)

    brief_dips = check_clocks([3360] * 5 + [3480] * 23)
    assert brief_dips['compliant'] is True
    assert brief_dips['underLoadSamplesInTargetRange'] == 23
    assert brief_dips['underLoadSmClockSpanMHz'] == 120
    assert brief_dips['underLoadSmClockSpanInTargetRangeMHz'] == 0
    assert check_clocks([3360] * 6 + [3480] * 22)['compliant'] is False
    assert check_clocks([3450, 3480, 3510] * 4)['compliant'] is False
    assert check_clocks([3480, 3480])['compliant'] is False
    print('MOCK_CLOCK_BAND_PASS: brief dips accepted, sustained or unstable clock rejected')


benchmark = runpy.run_path(str(Path(__file__).with_name('nvapi-blackwell-benchmark-guard.py')))
resolve_point = benchmark['resolve_production_point']


def vf_table(*points):
    status = bytearray(benchmark['VF_STATUS_SIZE'])
    mask = bytearray(32)
    for slot, voltage, frequency in points:
        mask[slot // 8] |= 1 << (slot % 8)
        base = benchmark['VF_STATUS_BASE'] + slot * benchmark['VF_STATUS_STRIDE']
        struct.pack_into('<I', status, base + benchmark['VF_STATUS_FREQ_OFF'], frequency)
        struct.pack_into('<I', status, base + benchmark['VF_STATUS_VOLT_OFF'], voltage)
    return status, mask


for native, offset in [(2_850_000, 625), (2_865_000, 625), (2_880_000, 610), (2_872_500, 617)]:
    table = vf_table((96, 1_050_000, native), (97, 1_060_000, 2_895_000))
    assert resolve_point(*table, 1_050_000, 2_865_000, 625) == (96, native, offset)
    assert native + offset * 1000 <= 3_490_000

for invalid_points in [
    [(96, 1_045_000, 2_865_000)],
    [(96, 1_050_000, 2_865_000), (97, 1_050_000, 2_880_000)],
    [(96, 1_050_000, 2_835_000)],
]:
    try:
        resolve_point(*vf_table(*invalid_points), 1_050_000, 2_865_000, 625)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f'unverified V/F point accepted: {invalid_points}')
print('MOCK_NATIVE_VF_BIN_PASS: bounded drift, unchanged voltage and maximum curve frequency')

read_restored_control = benchmark['read_restored_control']
clock = 0.0


def sleep(seconds):
    global clock
    clock += seconds


fake_time = SimpleNamespace(monotonic=lambda: clock, sleep=Mock(side_effect=sleep))
with patch.dict(read_restored_control.__globals__, time=fake_time):
    matches_original = lambda value: value == b'original'
    immediate = Mock(return_value=(0, b'original'))
    assert read_restored_control(immediate, matches_original, True) == (0, b'original', True)
    immediate.assert_called_once()
    fake_time.sleep.assert_not_called()

    delayed = Mock(side_effect=[(0, b'old'), (-1, None), (0, b'original')])
    assert read_restored_control(delayed, matches_original, True) == (0, b'original', True)
    assert delayed.call_count == 3
    assert fake_time.sleep.call_count == 2

    stale = Mock(return_value=(0, b'old'))
    assert read_restored_control(stale, matches_original, True) == (0, b'old', False)
    assert 0.23 < clock < 0.25
print('MOCK_RESTORE_READBACK_PASS: immediate match, delayed readback and bounded timeout')
