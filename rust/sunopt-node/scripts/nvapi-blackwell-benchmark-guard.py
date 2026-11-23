#!/usr/bin/env python3
"""
SunOpt Blackwell NVAPI benchmark guard.

This process stays alive for the complete compiler autotune:
- verifies the exact native 1050 mV -> 2865 MHz V/F point
- snapshots the current clock-lock and fan-control buffers
- sets both detected fans to 70% manual
- locks NVAPI boost control to the native 1050 mV point
- waits for the physical fan to ramp close to 70%
- remains alive while the benchmark runs
- restores the exact original clock/fan buffers when:
    * the stop file appears, or
    * the parent benchmark PowerShell process exits
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
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
ID_GPU_GET_VF_CURVE_CONTROL = 0x23F1B133
ID_GPU_SET_VF_CURVE_CONTROL = 0x0733E009
ID_GPU_GET_FAN_CONTROL = 0x814B209F
ID_GPU_SET_FAN_CONTROL = 0xA58971A5

VF_POINTS_INFO_SIZE = 6188
VF_STATUS_SIZE = 7208
VF_CONTROL_SIZE = 9248
VF_CTRL_ENTRY_BASE = 0x64
VF_CTRL_ENTRY_STRIDE = 36
VF_CTRL_DELTA_OFFSET = 0x18
CLOCK_LOCK_SIZE = 0x030C
CLOCK_LOCK_VERSION = 2
FAN_CONTROL_CLAIMED_SIZE = 1452
FAN_CONTROL_ALLOC_SIZE = 4096

VF_STATUS_BASE = 0x40
VF_STATUS_STRIDE = 28
VF_STATUS_TYPE_OFF = 0x04
VF_STATUS_FREQ_OFF = 0x08
VF_STATUS_VOLT_OFF = 0x0C

SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 0x00000102


def make_version(size: int, version: int) -> int:
    return size | (version << 16)


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8", newline="\n")

    def line(self, s: str = ""):
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class NvApi:
    def __init__(self):
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
        if not fn:
            return str(status)
        buf = ctypes.create_string_buffer(SHORT_STRING_MAX)
        try:
            rc = int(fn(status, ctypes.cast(buf, ctypes.c_char_p)))
            if rc == 0:
                return f"{status} ({buf.value.decode('ascii', errors='replace')})"
        except Exception:
            pass
        return str(status)

    def initialize(self):
        proto = ctypes.CFUNCTYPE(ctypes.c_int)
        fn = self.delegate(ID_INITIALIZE, proto)
        if not fn:
            raise RuntimeError("NvAPI_Initialize unavailable")
        return int(fn())

    def unload(self):
        proto = ctypes.CFUNCTYPE(ctypes.c_int)
        fn = self.delegate(ID_UNLOAD, proto)
        return int(fn()) if fn else 0

    def enum_gpus(self):
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        )
        fn = self.delegate(ID_ENUM_PHYSICAL_GPUS, proto)
        if not fn:
            raise RuntimeError("NvAPI_EnumPhysicalGPUs unavailable")
        handles = (ctypes.c_void_p * MAX_PHYSICAL_GPUS)()
        count = ctypes.c_int()
        status = int(fn(handles, ctypes.byref(count)))
        return status, [int(handles[i] or 0) for i in range(count.value)]

    def gpu_name(self, handle: int):
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)
        fn = self.delegate(ID_GPU_GET_FULL_NAME, proto)
        if not fn:
            return -3, None
        buf = ctypes.create_string_buffer(SHORT_STRING_MAX)
        status = int(fn(ctypes.c_void_p(handle), ctypes.cast(buf, ctypes.c_char_p)))
        return status, buf.value.decode("ascii", errors="replace") if status == 0 else None

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
        if not fn:
            return -3, None
        status = int(fn(ctypes.c_void_p(handle), ctypes.byref(buf)))
        return status, bytes(buf.raw) if status == 0 else None

    def set_raw(self, handle: int, fid: int, data: bytes):
        buf = ctypes.create_string_buffer(len(data))
        ctypes.memmove(buf, data, len(data))
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        fn = self.delegate(fid, proto)
        if not fn:
            return -3
        return int(fn(ctypes.c_void_p(handle), ctypes.byref(buf)))


def mask_slots(mask: bytes):
    out = []
    for i in range(len(mask) * 8):
        if mask[i // 8] & (1 << (i % 8)):
            out.append(i)
    return out


def find_native_point(vf_status: bytes, mask: bytes, target_uv: int, target_khz: int):
    matches = []
    for slot in mask_slots(mask):
        off = VF_STATUS_BASE + slot * VF_STATUS_STRIDE
        if off + VF_STATUS_VOLT_OFF + 4 > len(vf_status):
            continue
        typ = struct.unpack_from("<I", vf_status, off + VF_STATUS_TYPE_OFF)[0]
        freq = struct.unpack_from("<I", vf_status, off + VF_STATUS_FREQ_OFF)[0]
        volt = struct.unpack_from("<I", vf_status, off + VF_STATUS_VOLT_OFF)[0]
        if typ == 0 and volt == target_uv and freq == target_khz:
            matches.append(slot)
    return matches



def vf_ctrl_delta_pos(anchor: int) -> int:
    if anchor < 1:
        raise ValueError("V/F anchor 0 has no control entry")
    return (
        VF_CTRL_ENTRY_BASE
        + (anchor - 1) * VF_CTRL_ENTRY_STRIDE
        + VF_CTRL_DELTA_OFFSET
    )


def get_curve_delta_khz(control: bytes, anchor: int) -> int:
    pos = vf_ctrl_delta_pos(anchor)
    if pos + 4 > len(control):
        raise ValueError(f"V/F control anchor {anchor} is out of range")
    return struct.unpack_from("<i", control, pos)[0]


def decode_core_curve(vf_status: bytes, mask: bytes):
    points = []
    for slot in mask_slots(mask):
        off = VF_STATUS_BASE + slot * VF_STATUS_STRIDE
        if off + VF_STATUS_VOLT_OFF + 4 > len(vf_status):
            continue
        typ = struct.unpack_from("<I", vf_status, off + VF_STATUS_TYPE_OFF)[0]
        freq = struct.unpack_from("<I", vf_status, off + VF_STATUS_FREQ_OFF)[0]
        volt = struct.unpack_from("<I", vf_status, off + VF_STATUS_VOLT_OFF)[0]
        if typ != 0:
            continue
        if volt < 300_000 or volt > 1_300_000:
            continue

        # Blackwell status exposes non-core clock rows in the same table.
        # This RTX 5060 reports GDDR rows around 13.8/14.0 GHz at slots
        # 130/131. A core curve point must be in the physically plausible
        # graphics-clock range. Never let memory rows enter a core OC edit.
        if freq < 100_000 or freq > 5_000_000:
            continue

        points.append((slot, volt, freq))
    return points


def resolve_production_point(vf_status, mask, target_uv, target_khz, offset_mhz):
    points = [(slot, freq) for slot, volt, freq in decode_core_curve(vf_status, mask)
              if volt == target_uv]
    available = ", ".join(f"slot {slot}: {freq / 1000:.3f} MHz" for slot, freq in points)
    if len(points) != 1:
        raise RuntimeError(
            f"expected one core V/F point at {target_uv / 1000:g} mV; "
            f"found {available or 'none'}"
        )
    slot, frequency_khz = points[0]
    if abs(frequency_khz - target_khz) > 15_000:
        raise RuntimeError(
            f"core V/F point at {target_uv / 1000:g} mV is {available}; "
            f"outside ±15 MHz of {target_khz / 1000:g} MHz"
        )
    # Tolerate a single boost bin while preserving the previous voltage,
    # upper frequency cap and maximum positive offset.
    curve_cap_khz = target_khz + offset_mhz * 1000
    applied_offset_mhz = min(offset_mhz, (curve_cap_khz - frequency_khz) // 1000)
    if applied_offset_mhz <= 0:
        raise RuntimeError("production V/F target has no positive offset within the allowed cap")
    return slot, frequency_khz, applied_offset_mhz


def make_curve_oc_candidate(
    original_control: bytes,
    vf_status: bytes,
    mask: bytes,
    target_slot: int,
    offset_mhz: int,
    cap_band_points: int,
):
    if offset_mhz < 0:
        raise ValueError("this guard only supports nonnegative core offsets")
    if cap_band_points < 1:
        raise ValueError("cap_band_points must be >= 1")

    core = decode_core_curve(vf_status, mask)
    by_slot = {slot: (volt, freq) for slot, volt, freq in core}
    if target_slot not in by_slot:
        raise RuntimeError("target V/F slot missing from decoded core curve")

    # Refuse to stack on top of an existing per-point curve tune in the
    # region we are about to change. This makes restoration unambiguous.
    highest_core_slot = max(by_slot)
    if highest_core_slot > 126:
        raise RuntimeError(
            f"unexpected highest core V/F slot {highest_core_slot}; "
            "refusing to risk non-core clock domains"
        )

    start = max(1, target_slot - cap_band_points + 1)

    existing_nonzero = []
    for anchor in range(start, highest_core_slot + 1):
        try:
            delta = get_curve_delta_khz(original_control, anchor)
        except ValueError:
            continue
        if delta != 0:
            existing_nonzero.append((anchor, delta))

    if existing_nonzero:
        preview = ", ".join(
            f"{a}:{d/1000.0:.1f}MHz" for a, d in existing_nonzero[:8]
        )
        raise RuntimeError(
            "existing V/F curve deltas are nonzero in the tuning region; "
            f"refusing to stack tuning ({preview})"
        )

    target_native_khz = by_slot[target_slot][1]
    desired_khz = target_native_khz + offset_mhz * 1000

    buf = bytearray(original_control)

    # A small band below/at the cap gets the same positive offset so the
    # boost algorithm cannot fall one or two anchors below the cap and lose
    # the OC. Above the cap, flatten the curve at the desired cap frequency.
    changed = []
    for anchor in range(start, highest_core_slot + 1):
        if anchor not in by_slot:
            continue

        native_khz = by_slot[anchor][1]
        if anchor <= target_slot:
            delta_khz = offset_mhz * 1000
        else:
            delta_khz = desired_khz - native_khz

        pos = vf_ctrl_delta_pos(anchor)
        if pos + 4 > len(buf):
            raise RuntimeError(f"V/F control table overflow at anchor {anchor}")

        struct.pack_into("<i", buf, pos, int(delta_khz))
        changed.append((anchor, native_khz, delta_khz))

    # Preserve the driver's own version/mask and opaque fields; only deltas
    # are edited in the read-modify-write buffer.
    return bytes(buf), changed, desired_khz


def decode_clock_lock(data: bytes):
    version, flags, count = struct.unpack_from("<III", data, 0)
    entries = []
    for i in range(min(count, 32)):
        off = 12 + i * 24
        vals = struct.unpack_from("<IIIIII", data, off)
        entries.append({
            "entry": i,
            "index": vals[0],
            "mode": vals[2],
            "voltageUV": vals[4],
            "offset": off,
        })
    return version, flags, count, entries


def make_clock_candidate(original: bytes, target_uv: int):
    version, _, _, entries = decode_clock_lock(original)
    expected = make_version(CLOCK_LOCK_SIZE, CLOCK_LOCK_VERSION)
    if version != expected:
        raise RuntimeError(f"unexpected clock-lock version 0x{version:08X}")
    targets = [e for e in entries if e["index"] == 6]
    if len(targets) != 1:
        raise RuntimeError(f"expected exactly one clock-lock index=6; got {len(targets)}")
    e = targets[0]
    buf = bytearray(original)
    struct.pack_into("<I", buf, e["offset"] + 8, 3)
    struct.pack_into("<I", buf, e["offset"] + 16, target_uv)
    struct.pack_into("<I", buf, e["offset"] + 20, 0)
    return bytes(buf)


def read_clock_entry6(data: bytes):
    _, _, _, entries = decode_clock_lock(data)
    hits = [e for e in entries if e["index"] == 6]
    if len(hits) != 1:
        raise RuntimeError("clock-lock index=6 readback ambiguous")
    return hits[0]


def read_restored_control(read, matches, fast):
    if not fast:
        time.sleep(0.2)
        status, value = read()
        return status, value, status == 0 and value is not None and matches(value)

    deadline = time.monotonic() + 0.2
    while True:
        status, value = read()
        matched = status == 0 and value is not None and matches(value)
        remaining = deadline - time.monotonic()
        if matched or remaining <= 0:
            return status, value, matched
        time.sleep(min(0.02, remaining))


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


def fan_settings(data: bytes):
    _, _, entries = decode_fan_control(data)
    return {e["index"]: (e["mode"], e["level"] if e["mode"] == 1 else None) for e in entries}


def make_fan_candidate(original: bytes, target: int):
    version, count, entries = decode_fan_control(original)
    expected = make_version(FAN_CONTROL_CLAIMED_SIZE, 1)
    if version != expected:
        raise RuntimeError(f"unexpected fan-control version 0x{version:08X}")
    if count < 1 or count > 8:
        raise RuntimeError(f"unexpected fan-controller count={count}")
    buf = bytearray(original)
    for e in entries:
        struct.pack_into("<I", buf, e["offset"] + 4, target)
        struct.pack_into("<I", buf, e["offset"] + 8, 1)
    return bytes(buf)


def read_nvidia_smi(device: int):
    query = (
        "pstate,clocks.gr,clocks.mem,temperature.gpu,"
        "fan.speed,power.draw,power.limit,utilization.gpu"
    )
    cp = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(device),
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if cp.returncode != 0 or not cp.stdout.strip():
        return None
    parts = [p.strip() for p in cp.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 8:
        return None
    return {
        "pstate": parts[0],
        "clockMHz": float(parts[1]),
        "memoryClockMHz": float(parts[2]),
        "temperatureC": float(parts[3]),
        "fanPercent": float(parts[4]),
        "powerW": float(parts[5]),
        "powerLimitW": float(parts[6]),
        "utilizationPercent": float(parts[7]),
    }


def parent_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        result = kernel32.WaitForSingleObject(handle, 0)
        return result == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def atomic_json(path: Path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--target-voltage-mv", type=int, default=1050)
    ap.add_argument("--native-vf-frequency-mhz", type=int, default=2865)
    ap.add_argument("--allow-native-vf-bin-drift", action="store_true")
    ap.add_argument("--fast-control-readback", action="store_true")
    ap.add_argument("--fan-percent", type=int, default=70)
    ap.add_argument("--fan-auto", action="store_true")
    ap.add_argument("--fan-tolerance", type=int, default=3)
    ap.add_argument("--curve-offset-mhz", type=int, default=0)
    ap.add_argument("--cap-band-points", type=int, default=8)
    ap.add_argument("--parent-pid", type=int, required=True)
    ap.add_argument("--ready-file", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--done-file", type=Path, required=True)
    ap.add_argument("--log-file", type=Path, required=True)
    args = ap.parse_args()

    target_uv = args.target_voltage_mv * 1000
    target_khz = args.native_vf_frequency_mhz * 1000
    log = Log(args.log_file)

    nv = None
    handle = None
    original_lock = None
    original_fan = None
    original_curve_control = None
    lock_written = False
    fan_written = False
    curve_written = False
    exit_code = 1

    try:
        for p in (args.ready_file, args.stop_file, args.done_file):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

        log.line("SUNOPT NVAPI BLACKWELL BENCHMARK GUARD")
        log.line("generated=" + datetime.now(timezone.utc).astimezone().isoformat())
        log.line(f"gpu={args.gpu}")
        log.line(f"parentPid={args.parent_pid}")
        log.line(f"targetVoltageMv={args.target_voltage_mv}")
        log.line(f"nativeVfFrequencyMHz={args.native_vf_frequency_mhz}")
        log.line(f"fanPercent={args.fan_percent}")
        log.line(
            "fanMode=" + ("automatic" if args.fan_auto else "manual")
        )
        log.line(f"curveOffsetMHz={args.curve_offset_mhz}")
        log.line(f"capBandPoints={args.cap_band_points}")

        nv = NvApi()
        status = nv.initialize()
        log.line(f"NvAPI_Initialize={nv.status_text(status)}")
        if status != 0:
            raise RuntimeError("NvAPI initialize failed")

        status, handles = nv.enum_gpus()
        log.line(f"NvAPI_EnumPhysicalGPUs={nv.status_text(status)}")
        if status != 0 or not handles:
            raise RuntimeError("GPU enumeration failed")
        if args.gpu < 0 or args.gpu >= len(handles):
            raise RuntimeError("GPU index out of range")
        handle = handles[args.gpu]

        ns, name = nv.gpu_name(handle)
        log.line(f"gpuName={name}")
        if name != "NVIDIA GeForce RTX 5060":
            raise RuntimeError(f"unexpected GPU: {name}")

        status, info = nv.get_raw(handle, ID_GPU_GET_VF_POINTS_INFO, VF_POINTS_INFO_SIZE, 1)
        log.line(f"vfPointsInfo={nv.status_text(status)}")
        if status != 0 or info is None:
            raise RuntimeError("cannot read V/F point mask")

        mask = info[4:36]
        status, vf = nv.get_raw(
            handle,
            ID_GPU_GET_VF_CURVE_STATUS,
            VF_STATUS_SIZE,
            1,
            request_mask=mask,
        )
        log.line(f"vfStatus={nv.status_text(status)}")
        if status != 0 or vf is None:
            raise RuntimeError("cannot read V/F status")

        native_frequency_khz = target_khz
        curve_offset_mhz = args.curve_offset_mhz
        if args.allow_native_vf_bin_drift:
            target_slot, native_frequency_khz, curve_offset_mhz = resolve_production_point(
                vf, mask, target_uv, target_khz, curve_offset_mhz,
            )
        else:
            slots = find_native_point(vf, mask, target_uv, target_khz)
            if len(slots) != 1:
                raise RuntimeError(
                    f"native {args.target_voltage_mv} mV / "
                    f"{args.native_vf_frequency_mhz} MHz point not unique"
                )
            target_slot = slots[0]

        log.line(f"nativeTargetSlots={target_slot}")
        log.line(f"nativeVfFrequency.actualMHz={native_frequency_khz / 1000:.3f}")
        log.line(f"curveOffset.appliedMHz={curve_offset_mhz}")

        status, original_lock = nv.get_raw(
            handle,
            ID_GPU_GET_CLOCK_BOOST_LOCK,
            CLOCK_LOCK_SIZE,
            CLOCK_LOCK_VERSION,
        )
        log.line(f"clockLock.original.read={nv.status_text(status)}")
        if status != 0 or original_lock is None:
            raise RuntimeError("cannot read original clock lock")

        status, original_fan = nv.get_raw(
            handle,
            ID_GPU_GET_FAN_CONTROL,
            FAN_CONTROL_CLAIMED_SIZE,
            1,
            alloc_size=FAN_CONTROL_ALLOC_SIZE,
        )
        log.line(f"fanControl.original.read={nv.status_text(status)}")
        if status != 0 or original_fan is None:
            raise RuntimeError("cannot read original fan control")

        status, original_curve_control = nv.get_raw(
            handle,
            ID_GPU_GET_VF_CURVE_CONTROL,
            VF_CONTROL_SIZE,
            1,
            request_mask=mask,
        )
        log.line(f"curveControl.original.read={nv.status_text(status)}")
        if status != 0 or original_curve_control is None:
            raise RuntimeError("cannot read original V/F curve control")

        if curve_offset_mhz != 0:
            curve_candidate, changed, desired_khz = make_curve_oc_candidate(
                original_curve_control,
                vf,
                mask,
                target_slot,
                curve_offset_mhz,
                args.cap_band_points,
            )

            log.line(
                f"curveTarget.nativeFrequencyMHz="
                f"{native_frequency_khz/1000.0:.3f}"
            )
            log.line(
                f"curveTarget.desiredFrequencyMHz={desired_khz/1000.0:.3f}"
            )
            log.line(f"curveChangedAnchors={len(changed)}")
            for anchor, native_khz, delta_khz in changed:
                log.line(
                    f"curve.anchor={anchor} nativeMHz={native_khz/1000.0:.3f} "
                    f"deltaMHz={delta_khz/1000.0:.3f}"
                )

            status = nv.set_raw(
                handle,
                ID_GPU_SET_VF_CURVE_CONTROL,
                curve_candidate,
            )
            log.line(f"curveControl.set={nv.status_text(status)}")
            if status != 0:
                raise RuntimeError("V/F curve-control write failed")
            curve_written = True

            status, curve_rb = nv.get_raw(
                handle,
                ID_GPU_GET_VF_CURVE_CONTROL,
                VF_CONTROL_SIZE,
                1,
                request_mask=mask,
            )
            log.line(f"curveControl.readback={nv.status_text(status)}")
            if status != 0 or curve_rb is None:
                raise RuntimeError("V/F curve-control readback failed")

            # Verify the cap-band target anchor's raw delta.
            target_delta = get_curve_delta_khz(curve_rb, target_slot)
            log.line(
                f"curveControl.targetAnchor={target_slot} "
                f"deltaMHz={target_delta/1000.0:.3f}"
            )
            if target_delta != curve_offset_mhz * 1000:
                raise RuntimeError("V/F curve-control target delta mismatch")

            if not args.fast_control_readback:
                # The benchmark logs an additional settled V/F status sample.
                time.sleep(0.5)
                status, vf_after = nv.get_raw(
                    handle,
                    ID_GPU_GET_VF_CURVE_STATUS,
                    VF_STATUS_SIZE,
                    1,
                    request_mask=mask,
                )
                log.line(f"vfStatus.afterCurve={nv.status_text(status)}")
                if status == 0 and vf_after is not None:
                    core_after = {
                        slot: (volt, freq)
                        for slot, volt, freq in decode_core_curve(vf_after, mask)
                    }
                    if target_slot in core_after:
                        effective = core_after[target_slot][1] / 1000.0
                        log.line(
                            f"curveTarget.effectiveFrequencyMHz={effective:.3f}"
                        )

        if args.fan_auto:
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

        clock_candidate = make_clock_candidate(original_lock, target_uv)
        status = nv.set_raw(handle, ID_GPU_SET_CLOCK_BOOST_LOCK, clock_candidate)
        log.line(f"clockLock.set={nv.status_text(status)}")
        if status != 0:
            raise RuntimeError("clock-lock write failed")
        lock_written = True

        status, lock_rb = nv.get_raw(
            handle,
            ID_GPU_GET_CLOCK_BOOST_LOCK,
            CLOCK_LOCK_SIZE,
            CLOCK_LOCK_VERSION,
        )
        log.line(f"clockLock.readback={nv.status_text(status)}")
        if status != 0 or lock_rb is None:
            raise RuntimeError("clock-lock readback failed")
        e6 = read_clock_entry6(lock_rb)
        log.line(
            f"clockLock.index={e6['index']} mode={e6['mode']} "
            f"voltageMv={e6['voltageUV']/1000.0:.3f}"
        )
        if e6["mode"] != 3 or e6["voltageUV"] != target_uv:
            raise RuntimeError("clock-lock readback mismatch")

        # Manual show-effect mode waits for the physical fans.
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

        ready = {
            "status": "READY",
            "gpuName": name,
            "gpuIndex": args.gpu,
            "targetVoltageMv": args.target_voltage_mv,
            "nativeVfFrequencyMHz": native_frequency_khz / 1000,
            "nativeVfSlot": target_slot,
            "fanPercent": None if args.fan_auto else args.fan_percent,
            "fanMode": "automatic" if args.fan_auto else "manual",
            "curveOffsetMHz": curve_offset_mhz,
            "capBandPoints": args.cap_band_points,
            "reportedFanPercent": last_smi["fanPercent"] if last_smi else None,
            "idleClockMHz": last_smi["clockMHz"] if last_smi else None,
            "temperatureC": last_smi["temperatureC"] if last_smi else None,
        }
        atomic_json(args.ready_file, ready)
        log.line("guardState=READY")

        # Stay resident for the entire benchmark.
        while True:
            if args.stop_file.exists():
                log.line("guardStopReason=STOP_FILE")
                break
            if not parent_alive(args.parent_pid):
                log.line("guardStopReason=PARENT_EXIT")
                break
            time.sleep(0.02 if args.fast_control_readback else 0.25)

        exit_code = 0

    except Exception as exc:
        log.line(f"guardErrorType={type(exc).__name__}")
        log.line(f"guardErrorMessage={exc}")
        exit_code = 1

    finally:
        log.line("--- RESTORE ORIGINAL GPU CONTROLS ---")

        restore_ok = True
        if nv is not None and handle is not None:
            if original_lock is not None and lock_written:
                try:
                    s = nv.set_raw(handle, ID_GPU_SET_CLOCK_BOOST_LOCK, original_lock)
                    log.line(f"clockLock.restore.set={nv.status_text(s)}")
                    restore_ok = restore_ok and (s == 0)
                    if s == 0:
                        sr, rb, matched = read_restored_control(
                            lambda: nv.get_raw(handle, ID_GPU_GET_CLOCK_BOOST_LOCK,
                                               CLOCK_LOCK_SIZE, CLOCK_LOCK_VERSION),
                            lambda value: read_clock_entry6(value) == read_clock_entry6(original_lock),
                            args.fast_control_readback,
                        )
                        log.line(f"clockLock.restore.read={nv.status_text(sr)}")
                        if sr == 0 and rb:
                            e6 = read_clock_entry6(rb)
                            log.line(
                                f"clockLock.restore.index={e6['index']} "
                                f"mode={e6['mode']} "
                                f"voltageMv={e6['voltageUV']/1000.0:.3f}"
                            )
                        restore_ok = restore_ok and (matched if args.fast_control_readback else sr == 0)
                except Exception as exc:
                    restore_ok = False
                    log.line(f"clockLock.restore.error={type(exc).__name__}: {exc}")
            else:
                log.line("clockLock.restore=NOT_NEEDED")

            if original_curve_control is not None and curve_written:
                try:
                    s = nv.set_raw(
                        handle,
                        ID_GPU_SET_VF_CURVE_CONTROL,
                        original_curve_control,
                    )
                    log.line(f"curveControl.restore.set={nv.status_text(s)}")
                    restore_ok = restore_ok and (s == 0)

                    if s == 0:
                        sr, rb, matched = read_restored_control(
                            lambda: nv.get_raw(handle, ID_GPU_GET_VF_CURVE_CONTROL,
                                               VF_CONTROL_SIZE, 1, request_mask=mask),
                            lambda value: value == original_curve_control,
                            args.fast_control_readback,
                        )
                        log.line(
                            f"curveControl.restore.read={nv.status_text(sr)}"
                        )
                        restore_ok = restore_ok and matched
                        log.line(f"curveControl.restore.exactMatch={matched}")
                except Exception as exc:
                    restore_ok = False
                    log.line(
                        f"curveControl.restore.error="
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                log.line("curveControl.restore=NOT_NEEDED")

            if original_fan is not None and fan_written:
                try:
                    s = nv.set_raw(handle, ID_GPU_SET_FAN_CONTROL, original_fan)
                    log.line(f"fanControl.restore.set={nv.status_text(s)}")
                    restore_ok = restore_ok and (s == 0)
                    if s == 0:
                        sr, rb, matched = read_restored_control(
                            lambda: nv.get_raw(handle, ID_GPU_GET_FAN_CONTROL, FAN_CONTROL_CLAIMED_SIZE,
                                               1, alloc_size=FAN_CONTROL_ALLOC_SIZE),
                            lambda value: fan_settings(value) == fan_settings(original_fan),
                            args.fast_control_readback,
                        )
                        log.line(f"fanControl.restore.read={nv.status_text(sr)}")
                        if sr == 0 and rb:
                            _, _, entries = decode_fan_control(rb)
                            for e in entries:
                                log.line(
                                    f"fanControl.restore.entry={e['entry']} "
                                    f"index={e['index']} level={e['level']} "
                                    f"mode={e['mode']}"
                                )
                        restore_ok = restore_ok and (matched if args.fast_control_readback else sr == 0)
                except Exception as exc:
                    restore_ok = False
                    log.line(f"fanControl.restore.error={type(exc).__name__}: {exc}")
            else:
                log.line("fanControl.restore=NOT_NEEDED")

            try:
                s = nv.unload()
                log.line(f"NvAPI_Unload={nv.status_text(s)}")
            except Exception as exc:
                restore_ok = False
                log.line(f"NvAPI_Unload.error={exc}")

        done = {
            "status": "DONE" if restore_ok else "RESTORE_WARNING",
            "exitCode": exit_code,
            "restoreOk": restore_ok,
            "generated": datetime.now(timezone.utc).astimezone().isoformat(),
        }
        try:
            atomic_json(args.done_file, done)
        except Exception as exc:
            log.line(f"doneFile.error={exc}")

        try:
            args.ready_file.unlink()
        except FileNotFoundError:
            pass

        log.line(f"guardExitCode={exit_code}")
        log.line(f"restoreOk={restore_ok}")
        log.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
