#!/usr/bin/env python3
"""SunOpt productive A/B guard: reuse the existing, tested NVAPI V/F benchmark guard.

The legacy NVAPI implementation owns all V/F-curve and fan control / restoration.
This adapter owns a concurrent lock, starts that implementation, reads NVML
TELEMETRY ONLY, verifies actual clocks under load and reports one sweep to Node.
It intentionally never calls NVML GPU-clock/fan SET functions.

Requires native Windows, administrator rights where required, and the existing
rust/sunopt-node/scripts/nvapi-blackwell-benchmark-guard.py from B14P/Q.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

TARGET_SM_MHZ = 3480
TARGET_FAN_PERCENT = 70
TARGET_VOLTAGE_MV = 1050
NATIVE_VF_FREQUENCY_MHZ = 2865
CURVE_OFFSET_MHZ = 625
CAP_BAND_POINTS = 8
MAX_GPU_TEMP_C = 80
TIMEOUT_SECONDS = 240
SAMPLE_SECONDS = 0.1
READY_TIMEOUT_SECONDS = 40
DONE_TIMEOUT_SECONDS = 20
CLOCK_LOW_MHZ, CLOCK_HIGH_MHZ = 3450, 3510


def send(payload):
    print(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), flush=True)


def parent_is_alive(pid):
    if os.name != 'nt':
        return True
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
    file = path.open('a+b')
    file.seek(0)
    if path.stat().st_size == 0:
        file.write(b'0')
        file.flush()
    file.seek(0)
    try:
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        file.close()
        raise RuntimeError('Another SunOpt A/B sweep already owns the GPU profile lock') from exc
    return file


class NvidiaTelemetry:
    """Read-only NVML. Percent readings are observed fan speeds, NOT PWM commands."""
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('NVAPI productive guard requires Windows.')
        self.lib = ctypes.WinDLL('nvml.dll')
        self._bind('nvmlInit_v2', [])
        self._bind('nvmlShutdown', [])
        self._bind('nvmlDeviceGetHandleByIndex_v2', [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])
        self._bind('nvmlDeviceGetClockInfo', [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)])
        self._bind('nvmlDeviceGetTemperature', [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)])
        self.get_fan = self._bind('nvmlDeviceGetFanSpeed_v2',
                                  [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)], optional=True)
        class Util(ctypes.Structure):
            _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]
        self.Util = Util
        self.get_util = self._bind('nvmlDeviceGetUtilizationRates',
                                   [ctypes.c_void_p, ctypes.POINTER(Util)], optional=True)
        self.get_fan_count = self._bind('nvmlDeviceGetNumFans',
                                        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)], optional=True)
        self._check(self.lib.nvmlInit_v2(), 'NVML init')
        self.handle = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)), 'GPU index 0')
        self.fans = []
        if self.get_fan_count:
            count = ctypes.c_uint()
            if self.get_fan_count(self.handle, ctypes.byref(count)) == 0:
                self.fans = list(range(min(int(count.value), 8)))

    def _bind(self, name, args, optional=False):
        fn = getattr(self.lib, name, None)
        if fn is None:
            if optional:
                return None
            raise RuntimeError('Required NVML telemetry API unavailable: ' + name)
        fn.argtypes, fn.restype = args, ctypes.c_int
        return fn

    @staticmethod
    def _check(rc, op):
        if rc != 0:
            raise RuntimeError(f'{op}: NVML error {rc}')

    def sample(self):
        clock, temp = ctypes.c_uint(), ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetClockInfo(self.handle, 1, ctypes.byref(clock)), 'read SM clock')
        self._check(self.lib.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(temp)), 'read temperature')
        utilization = None
        if self.get_util:
            util = self.Util()
            if self.get_util(self.handle, ctypes.byref(util)) == 0:
                utilization = int(util.gpu)
        observed_fans = []
        if self.get_fan:
            for i in self.fans:
                v = ctypes.c_uint()
                observed_fans.append(int(v.value) if self.get_fan(self.handle, i, ctypes.byref(v)) == 0 else None)
        return {'clockMHz': int(clock.value), 'temperatureC': int(temp.value),
                'gpuUtilization': utilization, 'fanPercent': observed_fans,
                'atMs': int(time.time() * 1000)}

    def close(self):
        self.lib.nvmlShutdown()


class LegacyNvapiGuard:
    def __init__(self, outbox, runtime_root, parent_pid, context, script=None):
        self.script = script or Path(__file__).with_name('nvapi-blackwell-benchmark-guard.py')
        if not self.script.is_file():
            raise RuntimeError('Original B14P/Q NVAPI guard missing: ' + str(self.script))
        unique = f'productive-{context}-{os.getpid()}-{time.time_ns()}'
        self.state = runtime_root / unique
        self.state.mkdir(parents=True, exist_ok=False)
        self.ready = self.state / 'ready.json'
        self.stop_file = self.state / 'stop.signal'
        self.done = self.state / 'done.json'
        self.log = outbox / ('sunopt-nvapi-guard-' + unique + '.txt')
        self.child = None
        self.stopped = False
        self.args = [sys.executable, str(self.script),
                     '--gpu', '0', '--target-voltage-mv', str(TARGET_VOLTAGE_MV),
                     '--native-vf-frequency-mhz', str(NATIVE_VF_FREQUENCY_MHZ),
                     '--curve-offset-mhz', str(CURVE_OFFSET_MHZ),
                     '--cap-band-points', str(CAP_BAND_POINTS),
                     '--fan-percent', str(TARGET_FAN_PERCENT),
                     '--parent-pid', str(parent_pid),
                     '--ready-file', str(self.ready),
                     '--stop-file', str(self.stop_file),
                     '--done-file', str(self.done),
                     '--log-file', str(self.log)]

    def start(self):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.child = subprocess.Popen(self.args, stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                      creationflags=flags)
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.ready.exists():
                try:
                    ready = json.loads(self.ready.read_text(encoding='utf-8'))
                except (json.JSONDecodeError, OSError):
                    time.sleep(.1)
                    continue
                if ready.get('status') != 'READY':
                    raise RuntimeError('NVAPI guard not READY: ' + str(ready))
                return ready
            if self.child.poll() is not None:
                log_tail = self.log.read_text(encoding='utf-8', errors='replace')[-4000:] if self.log.exists() else ''
                raise RuntimeError(f'Original NVAPI guard exited {self.child.returncode} before READY: {log_tail}')
            time.sleep(.1)
        raise RuntimeError('Original NVAPI guard did not reach READY in 40 s; see ' + str(self.log))

    def request_stop(self):
        if not self.stopped:
            self.stopped = True
            self.stop_file.write_text('stop\n', encoding='utf-8')

    def finish(self):
        if self.child is None:
            return {'restoreOk': True, 'guardStarted': False, 'guardLog': str(self.log)}
        self.request_stop()
        deadline = time.monotonic() + DONE_TIMEOUT_SECONDS
        done_content = None
        while time.monotonic() < deadline:
            if self.done.exists():
                try:
                    done_content = json.loads(self.done.read_text(encoding='utf-8'))
                    break
                except (ValueError, OSError):
                    pass
            if self.child.poll() is not None:
                # Give the done-file flush a short opportunity.
                time.sleep(.2)
                if not self.done.exists():
                    break
            time.sleep(.1)
        if self.child.poll() is None:
            try:
                self.child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass   # Do NOT kill process: old guard must retain chance to restore GPU.
        log = self.log.read_text(encoding='utf-8', errors='replace') if self.log.exists() else ''
        restore_ok = bool(re.search(r'(?m)^restoreOk=True\s*$', log))
        curve_restored = bool(re.search(r'(?m)^curveControl\.restore\.exactMatch=True\s*$', log))
        fan_control_ok = 'fanControl.readback=0 (NVAPI_OK)' in log
        curve_control_ok = 'curveControl.readback=0 (NVAPI_OK)' in log
        clock_lock_ok = 'clockLock.readback=0 (NVAPI_OK)' in log
        fan_levels = re.findall(r'(?m)^fanControl\.entry=\d+ index=\d+ level=(\d+) mode=(\d+)\s*$', log)
        pwm_ok = len(fan_levels) >= 2 and all((int(level) == TARGET_FAN_PERCENT and int(mode) == 1)
                                               for level, mode in fan_levels)
        result = {'guardStarted': True, 'guardLog': str(self.log), 'legacyDone': done_content,
                  'restoreOk': restore_ok, 'curveRestoredExactly': curve_restored,
                  'fanControlReadback': fan_control_ok and pwm_ok,
                  'curveControlReadback': curve_control_ok, 'clockLockReadback': clock_lock_ok,
                  'fanCommandPercent': TARGET_FAN_PERCENT,
                  'legacyExited': self.child.poll() is not None}
        # Preserve files when a restore operation failed for diagnosis; otherwise remove only runtime files.
        if restore_ok and curve_restored:
            shutil.rmtree(self.state, ignore_errors=True)
        return result


def make_report(samples, started, end_reason, context, configured, finish, error=None, probe=False):
    busy = [p for p in samples if isinstance(p.get('gpuUtilization'), int) and p['gpuUtilization'] >= 20]
    clocks = [p['clockMHz'] for p in busy]
    in_band = [c for c in clocks if CLOCK_LOW_MHZ <= c <= CLOCK_HIGH_MHZ]
    median = statistics.median(clocks) if clocks else None
    span = (max(clocks) - min(clocks)) if clocks else None
    verified = bool(len(clocks) >= 3 and median is not None
                    and CLOCK_LOW_MHZ <= median <= CLOCK_HIGH_MHZ
                    and span <= 35 and len(in_band) / len(clocks) >= .8)
    fan_obs = [v for p in samples for v in p.get('fanPercent', []) if v is not None]
    max_temp = max((p['temperatureC'] for p in samples), default=None)
    controls_ok = (bool(configured) and bool(finish.get('fanControlReadback'))
                   and bool(finish.get('curveControlReadback'))
                   and bool(finish.get('clockLockReadback')))
    restore_ok = (bool(finish.get('restoreOk')) and bool(finish.get('curveRestoredExactly'))
                  and bool(finish.get('legacyExited')))
    compliant = bool(controls_ok and restore_ok and end_reason == 'requested_stop'
                     and max_temp is not None and max_temp < MAX_GPU_TEMP_C and verified and not error)
    probe_pass = bool(probe and controls_ok and restore_ok and not error)
    return {
        'context': context,
        'method': 'NVAPI_B14PQ_VF_CURVE_GUARD',
        'target': {'smClockMHz': TARGET_SM_MHZ, 'fanPercent': TARGET_FAN_PERCENT,
                   'voltageMv': TARGET_VOLTAGE_MV, 'nativeVfFrequencyMHz': NATIVE_VF_FREQUENCY_MHZ,
                   'curveOffsetMHz': CURVE_OFFSET_MHZ, 'capBandPoints': CAP_BAND_POINTS},
        'targetAcceptedByDriver': controls_ok, 'verifiedClockUnderLoad': verified,
        'fanTelemetryVerified': False,  # Speed is deliberately NOT a PWM-setpoint check.
        'fanControlSetAccepted': bool(finish.get('fanControlReadback')),
        'fanControlVerification': 'NVAPI_MODE1_70_PERCENT_READBACK',
        'fanObservedSpeedsPercent': {'min': min(fan_obs) if fan_obs else None,
                                    'max': max(fan_obs) if fan_obs else None},
        'compliant': compliant, 'probePassed': probe_pass,
        'endReason': end_reason, 'elapsedSeconds': round(time.monotonic() - started, 3),
        'sampleCount': len(samples), 'underLoadSamples': len(clocks),
        'underLoadSmClockMinMHz': min(clocks) if clocks else None,
        'underLoadSmClockMaxMHz': max(clocks) if clocks else None,
        'underLoadSmClockMedianMHz': median, 'underLoadSmClockSpanMHz': span,
        'underLoadSamplesInTargetRange': len(in_band),
        'maximumTemperatureC': max_temp,
        'restoreError': None if restore_ok else 'Legacy NVAPI guard could not confirm exact V/F/clock/fan restoration',
        'restoreConfirmed': restore_ok,
        'error': error, 'guardLog': finish.get('guardLog'), 'legacyDone': finish.get('legacyDone'),
        'hardwareSafetyNote': 'Temperature monitor stops NVAPI guard and resets clocks on limit; synchronous CUDA sweep cannot be interrupted by this Python sidecar.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', required=True, type=int)
    parser.add_argument('--context', choices=['A', 'B'], default='B')
    parser.add_argument('--outbox', required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        send({'status': 'error', 'error': 'Windows NVAPI guard is required; refusing production GPU run'})
        return 1
    started = time.monotonic()
    outbox = Path(args.outbox).resolve()
    outbox.mkdir(parents=True, exist_ok=True)
    session_name = f'gpu-profile-{args.context}-{os.getpid()}-{time.time_ns()}.json'
    project_root = Path(__file__).resolve().parents[3]
    runtime_root = project_root / 'rust' / 'sunopt-node' / 'blackwell-runtime-control'
    samples, dev, guard, lock = [], None, None, None
    configured = False
    end_reason = 'startup_error'
    error_text = None
    stop_event = threading.Event()
    confirmed_stop = threading.Event()
    try:
        lock = lock_guard_file(outbox / 'sunopt-gpu-profile.lock')
        dev = NvidiaTelemetry()
        guard = LegacyNvapiGuard(outbox, runtime_root, os.getpid(), args.context)
        guard.start()  # Old guard writes READY only after NVAPI V/F and fan accepted.
        configured = True
        send({'status': 'ready', 'method': 'NVAPI_B14PQ_VF_CURVE_GUARD',
              'targetClockMHz': TARGET_SM_MHZ, 'targetFanPercent': TARGET_FAN_PERCENT,
              'curveOffsetMHz': CURVE_OFFSET_MHZ})

        def stdin_watcher():
            for line in sys.stdin:
                if line.strip() == 'STOP':
                    confirmed_stop.set()
                    stop_event.set()
                    return
            stop_event.set()
        if not args.probe:
            threading.Thread(target=stdin_watcher, daemon=True).start()
        end_reason = 'probe_stop' if args.probe else 'requested_stop'
        if args.probe:
            # A control-only probe is not supposed to verify clocks at 0% GPU utilization.
            samples.append(dev.sample())
        else:
            while not stop_event.is_set():
                if not parent_is_alive(args.parent_pid):
                    end_reason = 'parent_disappeared'
                    break
                if time.monotonic() - started > TIMEOUT_SECONDS:
                    end_reason = 'timeout'
                    break
                if guard.child and guard.child.poll() is not None:
                    end_reason = 'legacy_guard_exited_early'
                    break
                sample = dev.sample()
                samples.append(sample)
                if sample['temperatureC'] >= MAX_GPU_TEMP_C:
                    end_reason = 'temperature_limit_80C'
                    break
                stop_event.wait(SAMPLE_SECONDS)
            if end_reason == 'requested_stop' and not confirmed_stop.is_set():
                end_reason = 'stdin_closed_without_stop'
    except Exception as exc:
        error_text = str(exc)
        end_reason = 'runtime_error' if configured else 'startup_error'
        if not configured:
            send({'status': 'error', 'error': error_text})
    finally:
        finish = {'restoreOk': not configured, 'curveRestoredExactly': not configured}
        if guard:
            try:
                finish = guard.finish()
            except Exception as exc:
                finish = {'restoreOk': False, 'curveRestoredExactly': False,
                          'guardLog': str(guard.log), 'error': str(exc)}
                error_text = (error_text or '') + ' Restore: ' + str(exc)
        if dev:
            try:
                dev.close()
            except Exception as exc:
                error_text = (error_text or '') + ' NVML close: ' + str(exc)
        if lock:
            lock.close()
        report = make_report(samples, started, end_reason, args.context, configured, finish, error_text, args.probe)
        report_path = outbox / session_name
        try:
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            if not report['compliant'] and not report['probePassed']:
                (outbox / 'failed-run.txt').write_text(
                    f'GPU NVAPI profile A/B failed: {report["endReason"]}; '
                    f'error={report["error"]}; details={session_name}\n', encoding='utf-8')
            if configured:
                send({'status': 'done', **report, 'reportPath': str(report_path)})
        except Exception as exc:
            send({'status': 'error', 'error': 'Cannot write GPU telemetry: ' + str(exc)})
    return 0 if report['compliant'] or report['probePassed'] else 1


if __name__ == '__main__':
    sys.exit(main())
