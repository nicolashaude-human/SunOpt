#!/usr/bin/env python3
"""SunOpt Windows GPU run guard: NVML clock/fan request, live telemetry, automatic reset.

One dedicated process owns both settings during one productive A/B sweep.
If a setting is unsupported, it reports ERROR before the sweep begins.
No silent fallback to a variable-clock run.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

TARGET_SM_MHZ = 3480
TARGET_FAN_PERCENT = 70
MAX_GPU_TEMP_C = 80
TIMEOUT_SECONDS = 240
SAMPLE_SECONDS = 0.1


def send(payload):
    print(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), flush=True)


class NvidiaNvml:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('GPU-Guard requires Windows NVML (RTX 5060 / WDDM).')
        try:
            self.lib = ctypes.WinDLL('nvml.dll')
        except OSError as exc:
            raise RuntimeError('NVML nvml.dll not found: ' + str(exc)) from exc
        self._bind('nvmlInit_v2', [], ctypes.c_int)
        self._bind('nvmlShutdown', [], ctypes.c_int)
        self._bind('nvmlDeviceGetHandleByIndex_v2', [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int)
        self._bind('nvmlDeviceSetGpuLockedClocks', [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint], ctypes.c_int)
        self._bind('nvmlDeviceResetGpuLockedClocks', [ctypes.c_void_p], ctypes.c_int)
        self._bind('nvmlDeviceGetNumFans', [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)], ctypes.c_int)
        self._bind('nvmlDeviceSetFanSpeed_v2', [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint], ctypes.c_int)
        self._bind('nvmlDeviceSetDefaultFanSpeed_v2', [ctypes.c_void_p, ctypes.c_uint], ctypes.c_int)
        self._bind('nvmlDeviceGetClockInfo', [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)], ctypes.c_int)
        self._bind('nvmlDeviceGetTemperature', [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)], ctypes.c_int)
        self.get_fan_speed = self._bind('nvmlDeviceGetFanSpeed_v2', [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)], ctypes.c_int, optional=True)
        class Util(ctypes.Structure):
            _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]
        self.Util = Util
        self.get_util = self._bind('nvmlDeviceGetUtilizationRates', [ctypes.c_void_p, ctypes.POINTER(Util)], ctypes.c_int, optional=True)
        self._check(self.lib.nvmlInit_v2(), 'nvmlInit_v2')
        self.handle = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)), 'nvmlDeviceGetHandleByIndex_v2(0)')

    def _bind(self, name, args, ret, optional=False):
        function = getattr(self.lib, name, None)
        if function is None:
            if optional:
                return None
            raise RuntimeError('NVML function unsupported: ' + name)
        function.argtypes = args
        function.restype = ret
        return function

    @staticmethod
    def _check(rc, operation):
        if rc != 0:
            raise RuntimeError(f'{operation}: NVML error {rc}; check driver support/administrator rights')

    def lock_clock(self):
        self._check(self.lib.nvmlDeviceSetGpuLockedClocks(self.handle, TARGET_SM_MHZ, TARGET_SM_MHZ), 'lock GPU SM clock to 3480 MHz')

    def reset_clock(self):
        self._check(self.lib.nvmlDeviceResetGpuLockedClocks(self.handle), 'reset GPU clock lock')

    def fans(self):
        count = ctypes.c_uint(0)
        self._check(self.lib.nvmlDeviceGetNumFans(self.handle, ctypes.byref(count)), 'read fan count')
        if not 1 <= count.value <= 8:
            raise RuntimeError('Unsupported GPU fan count: ' + str(count.value))
        return list(range(count.value))

    def set_fan(self, index):
        self._check(self.lib.nvmlDeviceSetFanSpeed_v2(self.handle, index, TARGET_FAN_PERCENT), f'set GPU fan {index} to 70%')

    def reset_fan(self, index):
        self._check(self.lib.nvmlDeviceSetDefaultFanSpeed_v2(self.handle, index), f'reset fan {index} to automatic control')

    def sample(self, fan_indices):
        clock = ctypes.c_uint()
        temp = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetClockInfo(self.handle, 1, ctypes.byref(clock)), 'read SM clock')
        self._check(self.lib.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp)), 'read GPU temperature')
        util = None
        if self.get_util:
            u = self.Util()
            if self.get_util(self.handle, ctypes.byref(u)) == 0:
                util = int(u.gpu)
        fans = []
        if self.get_fan_speed:
            for i in fan_indices:
                v = ctypes.c_uint()
                fans.append(int(v.value) if self.get_fan_speed(self.handle, i, ctypes.byref(v)) == 0 else None)
        return {'clockMHz': int(clock.value), 'temperatureC': int(temp.value), 'gpuUtilization': util, 'fanPercent': fans}

    def close(self):
        self.lib.nvmlShutdown()


def parent_is_alive(pid):
    if os.name != 'nt':
        return True
    # SYNCHRONIZE process handles allow checking parent death without psutil.
    win = ctypes.windll.kernel32
    win.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
    win.OpenProcess.restype = ctypes.c_void_p
    win.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    win.WaitForSingleObject.restype = ctypes.c_uint
    win.CloseHandle.argtypes = [ctypes.c_void_p]
    win.CloseHandle.restype = ctypes.c_int
    handle = win.OpenProcess(0x00100000, 0, pid)
    if not handle:
        return False
    try:
        return win.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        win.CloseHandle(ctypes.c_void_p(handle))


def lock_guard_file(path):
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    f.seek(0)
    if path.stat().st_size == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        f.close()
        raise RuntimeError('Another SunOpt GPU sweep already owns the GPU profile lock') from exc
    return f


def make_report(samples, started, reason, request, succeeded, restored, error=None):
    busy = [p for p in samples if p.get('gpuUtilization') is not None and p['gpuUtilization'] >= 40]
    # When NVML utilization is not exposed, the clock cannot be verified under load.
    clocks = [p['clockMHz'] for p in busy]
    max_temp = max((p['temperatureC'] for p in samples), default=None)
    good = [c for c in clocks if abs(c - TARGET_SM_MHZ) <= 30]
    clock_verified = bool(len(clocks) >= 3 and len(good) / len(clocks) >= 0.95
                          and abs(statistics.median(clocks) - TARGET_SM_MHZ) <= 30)
    fan_readings = [x for p in samples for x in p.get('fanPercent', []) if x is not None]
    fan_verified = bool(fan_readings and all(abs(x - TARGET_FAN_PERCENT) <= 2 for x in fan_readings))
    # Setting both clocks and fans is mandatory; positive NVML returns are required.
    # A non-reporting fan tachometer is acceptable only if the NVML set call succeeded.
    compliant = bool(succeeded and reason == 'requested_stop' and clock_verified
                     and max_temp is not None and max_temp < MAX_GPU_TEMP_C
                     and not error and not restored)
    # "restored" signifies a restore error, not a successful restoration.
    return {
        'context': request, 'target': {'smClockMHz': TARGET_SM_MHZ, 'fanPercent': TARGET_FAN_PERCENT},
        'targetAcceptedByDriver': succeeded, 'verifiedClockUnderLoad': clock_verified,
        'fanTelemetryVerified': fan_verified, 'fanControlSetAccepted': succeeded,
        'compliant': compliant, 'endReason': reason, 'elapsedSeconds': round(time.monotonic() - started, 3),
        'sampleCount': len(samples), 'underLoadSamples': len(clocks),
        'underLoadSmClockMinMHz': min(clocks) if clocks else None,
        'underLoadSmClockMaxMHz': max(clocks) if clocks else None,
        'underLoadSmClockMedianMHz': statistics.median(clocks) if clocks else None,
        'underLoadSamplesInTargetRange': len(good),
        'maximumTemperatureC': max_temp, 'fanPercentMinObserved': min(fan_readings) if fan_readings else None,
        'fanPercentMaxObserved': max(fan_readings) if fan_readings else None,
        'restoreError': restored, 'error': error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--context', choices=['A', 'B'], required=True)
    parser.add_argument('--outbox', required=True)
    args = parser.parse_args()
    outbox = Path(args.outbox).resolve()
    started = time.monotonic()
    session_name = f'gpu-profile-{args.context}-{os.getpid()}-{time.time_ns()}.json'
    samples = []
    dev = None
    locked = False
    configured_fans = []
    guard_lock = None
    good = False
    end_reason = 'startup_error'
    error_text = None
    restore_error = None
    stop = threading.Event()
    stop_confirmed = threading.Event()
    try:
        guard_lock = lock_guard_file(outbox / 'sunopt-gpu-profile.lock')
        dev = NvidiaNvml()
        dev.lock_clock()
        locked = True
        for fan in dev.fans():
            dev.set_fan(fan)
            configured_fans.append(fan)
        good = True
        send({'status': 'ready', 'targetClockMHz': TARGET_SM_MHZ,
              'targetFanPercent': TARGET_FAN_PERCENT, 'fanCount': len(configured_fans)})

        def stdin_watcher():
            for line in sys.stdin:
                if line.strip() == 'STOP':
                    stop_confirmed.set()
                    stop.set()
                    return
            # Parent closed pipe / process ended.
            stop.set()
        threading.Thread(target=stdin_watcher, daemon=True).start()
        end_reason = 'requested_stop'
        while not stop.is_set():
            if not parent_is_alive(args.parent_pid):
                end_reason = 'parent_disappeared'
                break
            if time.monotonic() - started > TIMEOUT_SECONDS:
                end_reason = 'timeout'
                break
            sample = dev.sample(configured_fans)
            samples.append(sample)
            if sample['temperatureC'] >= MAX_GPU_TEMP_C:
                end_reason = 'temperature_limit_80C'
                break
            stop.wait(SAMPLE_SECONDS)
        if end_reason == 'requested_stop' and not stop_confirmed.is_set():
            end_reason = 'stdin_closed_without_stop'
    except Exception as exc:
        error_text = str(exc)
        if not good:
            send({'status': 'error', 'error': error_text})
        else:
            end_reason = 'runtime_error'
    finally:
        # Failure to restore is reported and makes every test noncompliant.
        problems = []
        if dev:
            for fan in configured_fans:
                try:
                    dev.reset_fan(fan)
                except Exception as exc:
                    problems.append(str(exc))
            if locked:
                try:
                    dev.reset_clock()
                except Exception as exc:
                    problems.append(str(exc))
            try:
                dev.close()
            except Exception as exc:
                problems.append(str(exc))
        if guard_lock:
            guard_lock.close()
        restore_error = '; '.join(problems) or None
        try:
            report = make_report(samples, started, end_reason, args.context, good, restore_error, error_text)
            outbox.mkdir(parents=True, exist_ok=True)
            (outbox / session_name).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            if not report['compliant']:
                (outbox / 'failed-run.txt').write_text(
                    f'GPU profile A/B failed: {report["endReason"]}. Details: {session_name}\n', encoding='utf-8')
            if good:
                send({'status': 'done', **report, 'reportPath': str(outbox / session_name)})
        except Exception as exc:
            if good:
                send({'status': 'error', 'error': 'Unable to save GPU telemetry: ' + str(exc)})
    return 0 if good and report['compliant'] else 1


if __name__ == '__main__':
    sys.exit(main())
