#!/usr/bin/env python3
"""
SunOpt controlled NVAPI Blackwell write/readback/restore test.

Purpose:
- Use only the GPU's EXISTING 1050 mV -> 2865 MHz V/F point.
- Set the private NVAPI boost lock to that voltage point.
- Set all detected fans to 70% manual.
- Read both controls back.
- Optionally run one existing SunOpt replay while 250 ms nvidia-smi telemetry is recorded.
- ALWAYS restore the exact pre-test clock-lock and fan-control buffers in finally.

No V/F curve offsets, power-limit changes, memory OC, or voltage-boost changes are performed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

NVAPI_OK = 0
MAX_PHYSICAL_GPUS = 64
SHORT_STRING_MAX = 64

ID_INITIALIZE = 0x0150E828
ID_UNLOAD = 0xD22BDD7E
ID_GET_ERROR_MESSAGE = 0x6C2D048C
ID_ENUM_PHYSICAL_GPUS = 0xE5AC921F
ID_GPU_GET_FULL_NAME = 0xCEEE8E9F

ID_GPU_GET_VF_POINTS_INFO = 0x507B4B59
ID_GPU_GET_VF_CURVE_STATUS = 0x21537AD4

ID_GPU_GET_CLOCK_BOOST_LOCK = 0xE440B867
ID_GPU_SET_CLOCK_BOOST_LOCK = 0x39442CFB

ID_GPU_GET_FAN_STATUS = 0x35AED5E8
ID_GPU_GET_FAN_CONTROL = 0x814B209F
ID_GPU_SET_FAN_CONTROL = 0xA58971A5

VF_POINTS_INFO_SIZE = 6188
VF_STATUS_SIZE = 7208

CLOCK_LOCK_SIZE = 0x030C
CLOCK_LOCK_VERSION = 2

FAN_STATUS_SIZE = 1704
FAN_CONTROL_CLAIMED_SIZE = 1452
FAN_CONTROL_ALLOC_SIZE = 4096

TARGET_VOLTAGE_UV = 1_050_000
TARGET_FREQUENCY_KHZ = 2_865_000
TARGET_FAN_PERCENT = 70

VF_STATUS_BASE = 0x40
VF_STATUS_STRIDE = 28
VF_STATUS_TYPE_OFF = 0x04
VF_STATUS_FREQ_OFF = 0x08
VF_STATUS_VOLT_OFF = 0x0C


class Reporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8", newline="\n")

    def line(self, s: str = "") -> None:
        print(s)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def make_version(size: int, version: int) -> int:
    return size | (version << 16)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NvApi:
    def __init__(self) -> None:
        self.dll = ctypes.CDLL("nvapi64.dll")
        self.query = self.dll.nvapi_QueryInterface
        self.query.argtypes = [ctypes.c_uint32]
        self.query.restype = ctypes.c_void_p

    def address(self, fid: int) -> int:
        return int(self.query(fid) or 0)

    def delegate(self, fid: int, proto):
        addr = self.address(fid)
        return proto(addr) if addr else None

    def status_text(self, status: int) -> str:
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
        fn = self.delegate(ID_GET_ERROR_MESSAGE, proto)
        if fn is None:
            return str(status)
        buf = ctypes.create_string_buffer(SHORT_STRING_MAX)
        try:
            rc = int(fn(status, ctypes.cast(buf, ctypes.c_char_p)))
            if rc == 0:
                return f"{status} ({buf.value.decode('ascii', errors='replace')})"
        except Exception:
            pass
        return str(status)

    def initialize(self) -> int:
        proto = ctypes.CFUNCTYPE(ctypes.c_int)
        fn = self.delegate(ID_INITIALIZE, proto)
        if fn is None:
            raise RuntimeError("NvAPI_Initialize unavailable")
        return int(fn())

    def unload(self) -> int:
        proto = ctypes.CFUNCTYPE(ctypes.c_int)
        fn = self.delegate(ID_UNLOAD, proto)
        return int(fn()) if fn else 0

    def enum_physical_gpus(self):
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        )
        fn = self.delegate(ID_ENUM_PHYSICAL_GPUS, proto)
        if fn is None:
            raise RuntimeError("NvAPI_EnumPhysicalGPUs unavailable")
        handles = (ctypes.c_void_p * MAX_PHYSICAL_GPUS)()
        count = ctypes.c_int()
        status = int(fn(handles, ctypes.byref(count)))
        return status, [int(handles[i] or 0) for i in range(count.value)]

    def gpu_name(self, handle: int):
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)
        fn = self.delegate(ID_GPU_GET_FULL_NAME, proto)
        if fn is None:
            return -3, None
        buf = ctypes.create_string_buffer(SHORT_STRING_MAX)
        status = int(fn(ctypes.c_void_p(handle), ctypes.cast(buf, ctypes.c_char_p)))
        return status, (buf.value.decode("ascii", errors="replace") if status == 0 else None)

    def get_raw(
        self,
        handle: int,
        fid: int,
        claimed_size: int,
        version: int,
        alloc_size: Optional[int] = None,
        request_mask: bytes = b"",
    ):
        alloc_size = alloc_size or claimed_size
        buf = ctypes.create_string_buffer(alloc_size)
        struct.pack_into("<I", buf, 0, make_version(claimed_size, version))
        if request_mask:
            ctypes.memmove(ctypes.addressof(buf) + 4, request_mask, len(request_mask))
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        fn = self.delegate(fid, proto)
        if fn is None:
            return -3, None
        status = int(fn(ctypes.c_void_p(handle), ctypes.byref(buf)))
        return status, (bytes(buf.raw) if status == 0 else None)

    def set_raw(self, handle: int, fid: int, data: bytes):
        buf = ctypes.create_string_buffer(len(data))
        ctypes.memmove(buf, data, len(data))
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        fn = self.delegate(fid, proto)
        if fn is None:
            return -3
        return int(fn(ctypes.c_void_p(handle), ctypes.byref(buf)))


def mask_slots(mask: bytes):
    out = []
    for i in range(len(mask) * 8):
        if mask[i // 8] & (1 << (i % 8)):
            out.append(i)
    return out


def find_native_curve_point(vf_status: bytes, mask: bytes):
    matches = []
    for slot in mask_slots(mask):
        off = VF_STATUS_BASE + slot * VF_STATUS_STRIDE
        if off + VF_STATUS_VOLT_OFF + 4 > len(vf_status):
            continue
        typ = struct.unpack_from("<I", vf_status, off + VF_STATUS_TYPE_OFF)[0]
        freq = struct.unpack_from("<I", vf_status, off + VF_STATUS_FREQ_OFF)[0]
        volt = struct.unpack_from("<I", vf_status, off + VF_STATUS_VOLT_OFF)[0]
        if typ == 0 and volt == TARGET_VOLTAGE_UV and freq == TARGET_FREQUENCY_KHZ:
            matches.append(slot)
    return matches


def decode_clock_lock(data: bytes):
    version, flags, count = struct.unpack_from("<III", data, 0)
    entries = []
    for i in range(min(count, 32)):
        off = 12 + i * 24
        vals = struct.unpack_from("<IIIIII", data, off)
        entries.append({
            "entry": i,
            "index": vals[0],
            "unknown1": vals[1],
            "mode": vals[2],
            "unknown2": vals[3],
            "voltageUV": vals[4],
            "unknown3": vals[5],
            "offset": off,
        })
    return version, flags, count, entries


def make_clock_candidate(original: bytes):
    version, flags, count, entries = decode_clock_lock(original)
    if version != make_version(CLOCK_LOCK_SIZE, CLOCK_LOCK_VERSION):
        raise RuntimeError(f"unexpected clock lock version 0x{version:08X}")
    targets = [e for e in entries if e["index"] == 6]
    if len(targets) != 1:
        raise RuntimeError(f"expected one clock-lock index=6, found {len(targets)}")
    target = targets[0]
    buf = bytearray(original)
    struct.pack_into("<I", buf, target["offset"] + 8, 3)
    struct.pack_into("<I", buf, target["offset"] + 16, TARGET_VOLTAGE_UV)
    struct.pack_into("<I", buf, target["offset"] + 20, 0)
    return bytes(buf)


def read_lock_entry6(data: bytes):
    _, _, _, entries = decode_clock_lock(data)
    matches = [e for e in entries if e["index"] == 6]
    if len(matches) != 1:
        raise RuntimeError(f"readback clock-lock index=6 count={len(matches)}")
    return matches[0]


def decode_fan_control(data: bytes):
    version = struct.unpack_from("<I", data, 0)[0]
    count = struct.unpack_from("<I", data, 8)[0]
    entries = []
    for i in range(min(count, 32)):
        off = 44 + i * 44
        index, level, mode = struct.unpack_from("<III", data, off)
        entries.append({
            "entry": i,
            "index": index,
            "level": level,
            "mode": mode,
            "offset": off,
        })
    return version, count, entries


def make_fan_candidate(original: bytes):
    version, count, entries = decode_fan_control(original)
    if version != make_version(FAN_CONTROL_CLAIMED_SIZE, 1):
        raise RuntimeError(f"unexpected fan control version 0x{version:08X}")
    if count < 1 or count > 8:
        raise RuntimeError(f"unexpected fan count {count}")
    buf = bytearray(original)
    for e in entries:
        struct.pack_into("<I", buf, e["offset"] + 4, TARGET_FAN_PERCENT)
        struct.pack_into("<I", buf, e["offset"] + 8, 1)
    return bytes(buf)


def parse_fan_status(data: bytes):
    version, count = struct.unpack_from("<II", data, 0)
    entries = []
    for i in range(min(count, 32)):
        off = 40 + i * 52
        cooler_id, rpm, min_level, max_level, level = struct.unpack_from("<IIIII", data, off)
        entries.append({
            "entry": i,
            "coolerId": cooler_id,
            "rpm": rpm,
            "minLevel": min_level,
            "maxLevel": max_level,
            "level": level,
        })
    return version, count, entries


def telemetry_worker(stop_event: threading.Event, rows: list[str], device_id: int):
    query = (
        "timestamp,pstate,clocks.gr,clocks.mem,temperature.gpu,"
        "fan.speed,power.draw,power.limit,utilization.gpu"
    )
    while not stop_event.is_set():
        try:
            cp = subprocess.run(
                [
                    "nvidia-smi",
                    "-i", str(device_id),
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if cp.returncode == 0 and cp.stdout.strip():
                rows.append(cp.stdout.strip().splitlines()[0])
        except Exception as exc:
            rows.append(f"TELEMETRY_ERROR,{type(exc).__name__},{exc}")
        stop_event.wait(0.250)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "llm.txt")
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    snapshot = root / "blackwell-benchmark.bin"
    replay = root / "scripts" / "blackwell-replay.mjs"

    report = Reporter(args.output.resolve())
    nv = None
    handle = None

    original_lock = None
    original_fan = None
    clock_write_succeeded = False
    fan_write_succeeded = False

    try:
        report.line("")
        report.line("================================================================")
        report.line("SUNOPT NVAPI BLACKWELL CONTROLLED WRITE/READBACK/RESTORE TEST")
        report.line("================================================================")
        report.line("generated=" + datetime.now(timezone.utc).astimezone().isoformat())
        report.line(f"targetVoltage_mV={TARGET_VOLTAGE_UV/1000:.3f}")
        report.line(f"targetFrequency_MHz={TARGET_FREQUENCY_KHZ/1000:.3f}")
        report.line(f"targetFanPercent={TARGET_FAN_PERCENT}")
        report.line("vfOffsetChanges=0")
        report.line("powerLimitChanges=0")
        report.line("memoryClockChanges=0")
        report.line("restorePolicy=ALWAYS_IN_FINALLY")
        report.line("")

        if sys.platform != "win32":
            raise RuntimeError("Windows required")

        nv = NvApi()
        init = nv.initialize()
        report.line(f"NvAPI_Initialize={nv.status_text(init)}")
        if init != 0:
            raise RuntimeError("NvAPI initialize failed")

        status, handles = nv.enum_physical_gpus()
        report.line(f"NvAPI_EnumPhysicalGPUs={nv.status_text(status)}")
        if status != 0 or not handles:
            raise RuntimeError("GPU enumeration failed")
        if args.gpu < 0 or args.gpu >= len(handles):
            raise RuntimeError("GPU index out of range")

        handle = handles[args.gpu]
        name_status, name = nv.gpu_name(handle)
        report.line(f"gpuName={name}")
        if name != "NVIDIA GeForce RTX 5060":
            raise RuntimeError(f"unexpected GPU: {name}")

        # Safety precondition: exact native curve point must be present now.
        s, info = nv.get_raw(handle, ID_GPU_GET_VF_POINTS_INFO, VF_POINTS_INFO_SIZE, 1)
        report.line(f"vfPointsInfo={nv.status_text(s)}")
        if s != 0 or info is None:
            raise RuntimeError("cannot read VF point mask")
        mask = info[4:36]

        s, vf = nv.get_raw(
            handle,
            ID_GPU_GET_VF_CURVE_STATUS,
            VF_STATUS_SIZE,
            1,
            request_mask=mask,
        )
        report.line(f"vfStatus={nv.status_text(s)}")
        if s != 0 or vf is None:
            raise RuntimeError("cannot read VF curve")
        matching_slots = find_native_curve_point(vf, mask)
        report.line("nativeTargetSlots=" + ",".join(map(str, matching_slots)))
        if len(matching_slots) != 1:
            raise RuntimeError(
                "exact native 1050 mV / 2865 MHz curve point not uniquely present"
            )

        # Snapshot exact current controls.
        s, original_lock = nv.get_raw(
            handle,
            ID_GPU_GET_CLOCK_BOOST_LOCK,
            CLOCK_LOCK_SIZE,
            CLOCK_LOCK_VERSION,
        )
        report.line(f"clockLock.original.read={nv.status_text(s)}")
        if s != 0 or original_lock is None:
            raise RuntimeError("cannot read original clock lock")

        s, original_fan = nv.get_raw(
            handle,
            ID_GPU_GET_FAN_CONTROL,
            FAN_CONTROL_CLAIMED_SIZE,
            1,
            alloc_size=FAN_CONTROL_ALLOC_SIZE,
        )
        report.line(f"fanControl.original.read={nv.status_text(s)}")
        if s != 0 or original_fan is None:
            raise RuntimeError("cannot read original fan control")

        report.line(f"clockLock.original.sha256={sha256(original_lock)}")
        report.line(f"fanControl.original.sha256={sha256(original_fan)}")

        clock_candidate = make_clock_candidate(original_lock)
        fan_candidate = make_fan_candidate(original_fan)

        # Fan first: easier to observe and provides cooling before the clock lock.
        report.line("")
        report.line("--- APPLY FAN 70% ---")
        s = nv.set_raw(handle, ID_GPU_SET_FAN_CONTROL, fan_candidate)
        report.line(f"fanControl.set={nv.status_text(s)}")
        if s != 0:
            raise RuntimeError("fan-control write failed")
        fan_write_succeeded = True

        time.sleep(0.5)

        s, fan_rb = nv.get_raw(
            handle,
            ID_GPU_GET_FAN_CONTROL,
            FAN_CONTROL_CLAIMED_SIZE,
            1,
            alloc_size=FAN_CONTROL_ALLOC_SIZE,
        )
        report.line(f"fanControl.readback={nv.status_text(s)}")
        if s != 0 or fan_rb is None:
            raise RuntimeError("fan-control readback failed")
        _, fan_count, fan_entries = decode_fan_control(fan_rb)
        report.line(f"fanControl.readback.count={fan_count}")
        for e in fan_entries:
            report.line(
                f"fanControl.readback.entry={e['entry']} index={e['index']} "
                f"level={e['level']} mode={e['mode']}"
            )
        if not fan_entries or any(e["level"] != 70 or e["mode"] != 1 for e in fan_entries):
            raise RuntimeError("fan-control readback did not confirm 70% manual")

        # Now apply the native 1050 mV point lock.
        report.line("")
        report.line("--- APPLY CLOCK LOCK 1050mV / NATIVE 2865MHz POINT ---")
        s = nv.set_raw(handle, ID_GPU_SET_CLOCK_BOOST_LOCK, clock_candidate)
        report.line(f"clockLock.set={nv.status_text(s)}")
        if s != 0:
            raise RuntimeError("clock-lock write failed")
        clock_write_succeeded = True

        time.sleep(0.5)

        s, lock_rb = nv.get_raw(
            handle,
            ID_GPU_GET_CLOCK_BOOST_LOCK,
            CLOCK_LOCK_SIZE,
            CLOCK_LOCK_VERSION,
        )
        report.line(f"clockLock.readback={nv.status_text(s)}")
        if s != 0 or lock_rb is None:
            raise RuntimeError("clock-lock readback failed")
        e6 = read_lock_entry6(lock_rb)
        report.line(
            f"clockLock.readback.index={e6['index']} "
            f"mode={e6['mode']} voltage_mV={e6['voltageUV']/1000:.3f}"
        )
        if e6["mode"] != 3 or e6["voltageUV"] != TARGET_VOLTAGE_UV:
            raise RuntimeError("clock-lock readback did not confirm requested lock")

        # Observe under a real existing SunOpt workload if available.
        report.line("")
        report.line("--- LOCKED TELEMETRY / OPTIONAL SUNOPT REPLAY ---")
        rows: list[str] = []
        stop = threading.Event()
        thread = threading.Thread(
            target=telemetry_worker,
            args=(stop, rows, args.gpu),
            daemon=True,
        )
        thread.start()

        replay_returncode = None
        replay_stdout = ""
        try:
            time.sleep(0.5)
            if not args.skip_replay and snapshot.exists() and replay.exists():
                cp = subprocess.run(
                    [
                        "node",
                        str(replay),
                        str(snapshot),
                        "1",
                        str(args.gpu),
                    ],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                replay_returncode = cp.returncode
                replay_stdout = (cp.stdout or "") + (cp.stderr or "")
                report.line(f"sunoptReplay.returncode={replay_returncode}")
                for line in replay_stdout.splitlines():
                    if "SUNOPT_BLACKWELL_JSON=" in line:
                        report.line(line)
            else:
                report.line("sunoptReplay=SKIPPED_OR_UNAVAILABLE")
                time.sleep(max(0.0, args.hold_seconds))
        finally:
            time.sleep(0.5)
            stop.set()
            thread.join(timeout=5)

        report.line("telemetry.header=timestamp,pstate,clocks.gr,clocks.mem,temp,fan,power,powerLimit,util")
        for row in rows:
            report.line("telemetry=" + row)

        # Fan status after load.
        s, fan_status = nv.get_raw(
            handle,
            ID_GPU_GET_FAN_STATUS,
            FAN_STATUS_SIZE,
            1,
        )
        report.line(f"fanStatus.locked.read={nv.status_text(s)}")
        if s == 0 and fan_status:
            _, cnt, entries = parse_fan_status(fan_status)
            report.line(f"fanStatus.locked.count={cnt}")
            for e in entries:
                report.line(
                    f"fanStatus.locked.entry={e['entry']} coolerId={e['coolerId']} "
                    f"rpm={e['rpm']} min={e['minLevel']} max={e['maxLevel']} "
                    f"level={e['level']}"
                )

        report.line("")
        report.line("CONTROLLED_WRITE_TEST=PASS_BEFORE_RESTORE")
        return 0

    except Exception as exc:
        report.line("")
        report.line("CONTROLLED_WRITE_TEST=FAIL")
        report.line(f"errorType={type(exc).__name__}")
        report.line(f"errorMessage={exc}")
        return 1

    finally:
        report.line("")
        report.line("--- RESTORE ORIGINAL GPU CONTROLS ---")

        if nv is not None and handle is not None:
            if original_lock is not None and clock_write_succeeded:
                try:
                    s = nv.set_raw(handle, ID_GPU_SET_CLOCK_BOOST_LOCK, original_lock)
                    report.line(f"clockLock.restore.set={nv.status_text(s)}")
                    time.sleep(0.25)
                    sr, rb = nv.get_raw(
                        handle,
                        ID_GPU_GET_CLOCK_BOOST_LOCK,
                        CLOCK_LOCK_SIZE,
                        CLOCK_LOCK_VERSION,
                    )
                    report.line(f"clockLock.restore.read={nv.status_text(sr)}")
                    if sr == 0 and rb:
                        e6 = read_lock_entry6(rb)
                        report.line(
                            f"clockLock.restore.index={e6['index']} "
                            f"mode={e6['mode']} voltage_mV={e6['voltageUV']/1000:.3f}"
                        )
                except Exception as exc:
                    report.line(f"clockLock.restore.error={type(exc).__name__}: {exc}")
            else:
                report.line("clockLock.restore=NOT_NEEDED")

            if original_fan is not None and fan_write_succeeded:
                try:
                    s = nv.set_raw(handle, ID_GPU_SET_FAN_CONTROL, original_fan)
                    report.line(f"fanControl.restore.set={nv.status_text(s)}")
                    time.sleep(0.25)
                    sr, rb = nv.get_raw(
                        handle,
                        ID_GPU_GET_FAN_CONTROL,
                        FAN_CONTROL_CLAIMED_SIZE,
                        1,
                        alloc_size=FAN_CONTROL_ALLOC_SIZE,
                    )
                    report.line(f"fanControl.restore.read={nv.status_text(sr)}")
                    if sr == 0 and rb:
                        _, count, entries = decode_fan_control(rb)
                        report.line(f"fanControl.restore.count={count}")
                        for e in entries:
                            report.line(
                                f"fanControl.restore.entry={e['entry']} index={e['index']} "
                                f"level={e['level']} mode={e['mode']}"
                            )
                except Exception as exc:
                    report.line(f"fanControl.restore.error={type(exc).__name__}: {exc}")
            else:
                report.line("fanControl.restore=NOT_NEEDED")

            try:
                status = nv.unload()
                report.line(f"NvAPI_Unload={nv.status_text(status)}")
            except Exception as exc:
                report.line(f"NvAPI_Unload.error={exc}")

        report.line("================================================================")
        report.line("END CONTROLLED WRITE TEST")
        report.line("================================================================")
        report.close()


if __name__ == "__main__":
    raise SystemExit(main())
