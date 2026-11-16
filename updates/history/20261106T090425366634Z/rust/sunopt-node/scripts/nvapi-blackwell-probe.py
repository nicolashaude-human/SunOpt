#!/usr/bin/env python3
"""
SunOpt read-only NVIDIA NVAPI / Blackwell probe.

This probe DOES NOT call any write/tuning NVAPI entry point.
It only:
- loads nvapi64.dll
- resolves public/private NVAPI QueryInterface IDs
- initializes NVAPI
- enumerates physical GPUs
- reads identification/architecture data
- reads the V/F-point mask
- reads the live V/F status buffer
- reads the V/F control buffer
- attempts a read-only fan-control query
- records raw buffer hex dumps + binary snapshots

The private NVAPI interfaces are undocumented. Buffer sizes/layout candidates
are intentionally recorded rather than assumed to be correct.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

NVAPI_OK = 0
MAX_PHYSICAL_GPUS = 64
SHORT_STRING_MAX = 64

# Public / stable QueryInterface IDs.
ID_INITIALIZE = 0x0150E828
ID_UNLOAD = 0xD22BDD7E
ID_GET_ERROR_MESSAGE = 0x6C2D048C
ID_ENUM_PHYSICAL_GPUS = 0xE5AC921F
ID_GPU_GET_FULL_NAME = 0xCEEE8E9F
ID_GPU_GET_PCI_IDENTIFIERS = 0x2DDFB66E
ID_GPU_GET_ARCH_INFO = 0xD8265D24
ID_GPU_GET_PSTATES20 = 0x6FF81213

# Private / undocumented tuning IDs. READ functions may be called;
# WRITE functions are resolved only and NEVER invoked by this probe.
ID_GPU_SET_PSTATES20 = 0x0F4DAE6B
ID_GPU_GET_VF_POINTS_INFO = 0x507B4B59
ID_GPU_GET_VF_CURVE_STATUS = 0x21537AD4
ID_GPU_GET_VF_CONTROL = 0x23F1B133
ID_GPU_SET_VF_CONTROL = 0x0733E009
ID_GPU_GET_PERF_LIMITS = 0xE440B867
ID_GPU_SET_CLOCK_BOOST_LOCK = 0x39442CFB
ID_GPU_GET_FAN_STATUS = 0x35AED5E8
ID_GPU_GET_FAN_CONTROL = 0x814B209F
ID_GPU_SET_FAN_CONTROL = 0xA58971A5
ID_GPU_GET_COOLER_SETTINGS = 0xDA141340
ID_GPU_SET_COOLER_LEVELS = 0x891FA0AE

# Raw-buffer sizes from current community Blackwell-capable implementations.
VF_POINTS_INFO_SIZE = 6188      # 0x182C, version 1
VF_STATUS_SIZE = 7208           # 0x1C28, version 1
VF_CONTROL_SIZE = 9248          # 0x2420, version 1

# Fan-control layout from open-source NVAPI tooling.
FAN_CONTROL_CLAIMED_SIZE = 1452  # version 1
FAN_CONTROL_ALLOC_SIZE = 4096

# Two currently observed layout families. We score both instead of trusting one.
VF_STATUS_LAYOUTS = (
    ("simple-nvidia-undervolt", 0x40, 28, 0x04, 0x08, 0x0C),
    ("blackwell-raw-probe",     0x48, 28, 0x00, 0x04, 0x08),
)

VF_CONTROL_LAYOUTS = (
    ("simple-nvidia-undervolt", 0x64, 36, 0x18),
    ("blackwell-raw-probe",     0x44, 36, 0x14),
)

FUNCTIONS = (
    ("NvAPI_Initialize", ID_INITIALIZE, False),
    ("NvAPI_Unload", ID_UNLOAD, False),
    ("NvAPI_GetErrorMessage", ID_GET_ERROR_MESSAGE, False),
    ("NvAPI_EnumPhysicalGPUs", ID_ENUM_PHYSICAL_GPUS, False),
    ("NvAPI_GPU_GetFullName", ID_GPU_GET_FULL_NAME, False),
    ("NvAPI_GPU_GetPCIIdentifiers", ID_GPU_GET_PCI_IDENTIFIERS, False),
    ("NvAPI_GPU_GetArchInfo", ID_GPU_GET_ARCH_INFO, False),
    ("NvAPI_GPU_GetPstates20", ID_GPU_GET_PSTATES20, False),
    ("NvAPI_GPU_SetPstates20", ID_GPU_SET_PSTATES20, True),
    ("ClkVfPointsGetInfo", ID_GPU_GET_VF_POINTS_INFO, False),
    ("ClkVfPointsGetStatus", ID_GPU_GET_VF_CURVE_STATUS, False),
    ("ClkVfPointsGetControl", ID_GPU_GET_VF_CONTROL, False),
    ("ClkVfPointsSetControl", ID_GPU_SET_VF_CONTROL, True),
    ("PerfClientLimitsGetStatus", ID_GPU_GET_PERF_LIMITS, False),
    ("SetClockBoostLock", ID_GPU_SET_CLOCK_BOOST_LOCK, True),
    ("ClientFanCoolersGetStatus", ID_GPU_GET_FAN_STATUS, False),
    ("ClientFanCoolersGetControl", ID_GPU_GET_FAN_CONTROL, False),
    ("ClientFanCoolersSetControl", ID_GPU_SET_FAN_CONTROL, True),
    ("LegacyGetCoolerSettings", ID_GPU_GET_COOLER_SETTINGS, False),
    ("LegacySetCoolerLevels", ID_GPU_SET_COOLER_LEVELS, True),
)


class Reporter:
    def __init__(self, path: Path, append: bool) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._file = self.path.open(mode, encoding="utf-8", newline="\n")

    def line(self, text: str = "") -> None:
        print(text)
        self._file.write(text + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def make_version(size: int, version: int) -> int:
    if size < 4 or size > 0xFFFF:
        raise ValueError(f"invalid NVAPI claimed struct size: {size}")
    return size | (version << 16)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hexdump_lines(data: bytes, width: int = 32):
    for offset in range(0, len(data), width):
        chunk = data[offset:offset + width]
        yield f"{offset:04x}: {chunk.hex()}"


@dataclass
class RawReadResult:
    name: str
    function_id: int
    status: int
    data: Optional[bytes]


class NvApi:
    def __init__(self) -> None:
        self.dll = ctypes.CDLL("nvapi64.dll")
        self.query = self.dll.nvapi_QueryInterface
        self.query.argtypes = [ctypes.c_uint32]
        self.query.restype = ctypes.c_void_p

    def address(self, function_id: int) -> int:
        return int(self.query(function_id) or 0)

    def delegate(self, function_id: int, prototype):
        addr = self.address(function_id)
        if not addr:
            return None
        return prototype(addr)

    def status_text(self, status: int) -> str:
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
        )
        fn = self.delegate(ID_GET_ERROR_MESSAGE, proto)
        if fn is None:
            return f"status={status}"
        buf = ctypes.create_string_buffer(SHORT_STRING_MAX)
        try:
            rc = fn(status, ctypes.cast(buf, ctypes.c_char_p))
            if rc == NVAPI_OK:
                text = buf.value.decode("ascii", errors="replace")
                return f"{status} ({text})"
        except Exception:
            pass
        return f"status={status}"

    def initialize(self) -> int:
        proto = ctypes.CFUNCTYPE(ctypes.c_int)
        fn = self.delegate(ID_INITIALIZE, proto)
        if fn is None:
            raise RuntimeError("NvAPI_Initialize entry point is unavailable")
        return int(fn())

    def unload(self) -> int:
        proto = ctypes.CFUNCTYPE(ctypes.c_int)
        fn = self.delegate(ID_UNLOAD, proto)
        if fn is None:
            return 0
        return int(fn())

    def enum_physical_gpus(self):
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        )
        fn = self.delegate(ID_ENUM_PHYSICAL_GPUS, proto)
        if fn is None:
            raise RuntimeError("NvAPI_EnumPhysicalGPUs entry point is unavailable")

        handles = (ctypes.c_void_p * MAX_PHYSICAL_GPUS)()
        count = ctypes.c_int(0)
        status = int(fn(handles, ctypes.byref(count)))
        return status, [int(handles[i] or 0) for i in range(count.value)]

    def gpu_name(self, handle: int):
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_char_p,
        )
        fn = self.delegate(ID_GPU_GET_FULL_NAME, proto)
        if fn is None:
            return -3, None
        buf = ctypes.create_string_buffer(SHORT_STRING_MAX)
        status = int(fn(ctypes.c_void_p(handle), ctypes.cast(buf, ctypes.c_char_p)))
        name = buf.value.decode("ascii", errors="replace") if status == 0 else None
        return status, name

    def pci_ids(self, handle: int):
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        )
        fn = self.delegate(ID_GPU_GET_PCI_IDENTIFIERS, proto)
        if fn is None:
            return -3, None

        device = ctypes.c_uint32()
        subsystem = ctypes.c_uint32()
        revision = ctypes.c_uint32()
        ext_device = ctypes.c_uint32()

        status = int(fn(
            ctypes.c_void_p(handle),
            ctypes.byref(device),
            ctypes.byref(subsystem),
            ctypes.byref(revision),
            ctypes.byref(ext_device),
        ))
        if status != 0:
            return status, None

        return status, {
            "deviceId": device.value,
            "subSystemId": subsystem.value,
            "revisionId": revision.value,
            "extDeviceId": ext_device.value,
        }

    def raw_gpu_struct(
        self,
        handle: int,
        function_id: int,
        claimed_size: int,
        version: int,
        alloc_size: Optional[int] = None,
        request_mask: bytes = b"",
    ) -> RawReadResult:
        alloc_size = alloc_size or claimed_size
        if claimed_size > alloc_size:
            raise ValueError("claimed_size exceeds alloc_size")
        if 4 + len(request_mask) > alloc_size:
            raise ValueError("request mask exceeds allocation")

        addr = self.address(function_id)
        if not addr:
            return RawReadResult(
                name=f"0x{function_id:08X}",
                function_id=function_id,
                status=-3,
                data=None,
            )

        buf = ctypes.create_string_buffer(alloc_size)
        struct.pack_into("<I", buf, 0, make_version(claimed_size, version))
        if request_mask:
            ctypes.memmove(
                ctypes.addressof(buf) + 4,
                request_mask,
                len(request_mask),
            )

        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        fn = proto(addr)
        status = int(fn(ctypes.c_void_p(handle), ctypes.byref(buf)))

        data = bytes(buf.raw) if status == NVAPI_OK else None
        return RawReadResult(
            name=f"0x{function_id:08X}",
            function_id=function_id,
            status=status,
            data=data,
        )


def mask_slots(mask: bytes):
    slots = []
    for i in range(len(mask) * 8):
        if mask[i // 8] & (1 << (i % 8)):
            slots.append(i)
    return slots


def decode_vf_status(
    data: bytes,
    mask: bytes,
    layout,
):
    name, base, stride, type_off, freq_off, volt_off = layout
    rows = []
    plausible = 0

    for slot in mask_slots(mask):
        off = base + slot * stride
        if off + max(type_off, freq_off, volt_off) + 4 > len(data):
            continue

        clock_type = struct.unpack_from("<I", data, off + type_off)[0]
        freq_khz = struct.unpack_from("<I", data, off + freq_off)[0]
        volt_uv = struct.unpack_from("<I", data, off + volt_off)[0]

        row = {
            "slot": slot,
            "clockType": clock_type,
            "freqKHz": freq_khz,
            "voltUV": volt_uv,
        }
        rows.append(row)

        if (
            clock_type == 0
            and 300_000 <= volt_uv <= 1_300_000
            and 100_000 <= freq_khz <= 5_000_000
        ):
            plausible += 1

    return name, plausible, rows


def decode_control_candidates(data: bytes, mask: bytes):
    out = []
    for name, base, stride, delta_off in VF_CONTROL_LAYOUTS:
        rows = []
        for slot in mask_slots(mask):
            off = base + slot * stride
            if off + delta_off + 4 > len(data):
                continue
            delta = struct.unpack_from("<i", data, off + delta_off)[0]
            rows.append((slot, delta))
        reasonable = sum(1 for _, d in rows if -2_000_000 <= d <= 2_000_000)
        out.append((name, reasonable, rows))
    return out


def dump_raw(
    reporter: Reporter,
    output_dir: Path,
    stem: str,
    result: RawReadResult,
    nvapi: NvApi,
):
    reporter.line(f"{stem}.status={nvapi.status_text(result.status)}")
    if result.data is None:
        return
    reporter.line(f"{stem}.bytes={len(result.data)}")
    reporter.line(f"{stem}.sha256={sha256(result.data)}")

    binary_path = output_dir / f"{stem}.bin"
    binary_path.write_bytes(result.data)
    reporter.line(f"{stem}.binary={binary_path}")

    reporter.line(f"--- {stem} RAW HEX BEGIN ---")
    for line in hexdump_lines(result.data):
        reporter.line(line)
    reporter.line(f"--- {stem} RAW HEX END ---")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "llm.txt",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="overwrite output instead of appending",
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    output_dir = output_path.parent / "nvapi-blackwell-probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(output_path, append=not args.truncate)
    nvapi = None

    try:
        reporter.line("")
        reporter.line("================================================================")
        reporter.line("SUNOPT NVAPI BLACKWELL READ-ONLY PROBE")
        reporter.line("================================================================")
        reporter.line(
            "generated=" + datetime.now(timezone.utc).astimezone().isoformat()
        )
        reporter.line(f"python={sys.version.split()[0]}")
        reporter.line(f"platform={sys.platform}")
        reporter.line(f"gpuIndex={args.gpu}")
        reporter.line("writeCalls=DISABLED_BY_DESIGN")
        reporter.line("")

        if sys.platform != "win32":
            raise RuntimeError("This probe requires Windows.")

        nvapi = NvApi()

        reporter.line("--- QUERYINTERFACE AVAILABILITY ---")
        for name, function_id, is_write in FUNCTIONS:
            addr = nvapi.address(function_id)
            reporter.line(
                f"{name}: id=0x{function_id:08X} "
                f"available={'YES' if addr else 'NO'} "
                f"mode={'WRITE-NOT-CALLED' if is_write else 'READ'}"
            )

        status = nvapi.initialize()
        reporter.line("")
        reporter.line(f"NvAPI_Initialize={nvapi.status_text(status)}")
        if status != NVAPI_OK:
            raise RuntimeError(
                "NvAPI_Initialize failed: " + nvapi.status_text(status)
            )

        status, handles = nvapi.enum_physical_gpus()
        reporter.line(f"NvAPI_EnumPhysicalGPUs={nvapi.status_text(status)}")
        reporter.line(f"physicalGpuCount={len(handles)}")
        if status != NVAPI_OK:
            raise RuntimeError(
                "NvAPI_EnumPhysicalGPUs failed: " + nvapi.status_text(status)
            )
        if not handles:
            raise RuntimeError("No NVIDIA physical GPUs were returned.")
        if args.gpu < 0 or args.gpu >= len(handles):
            raise RuntimeError(
                f"GPU index {args.gpu} out of range 0..{len(handles)-1}"
            )

        for i, handle in enumerate(handles):
            name_status, name = nvapi.gpu_name(handle)
            reporter.line(
                f"gpu[{i}].nameStatus={nvapi.status_text(name_status)}"
            )
            reporter.line(f"gpu[{i}].name={name or '<unavailable>'}")

        handle = handles[args.gpu]
        reporter.line("")
        reporter.line(f"selectedGpuIndex={args.gpu}")
        reporter.line(f"selectedGpuHandle=0x{handle:016X}")

        pci_status, pci = nvapi.pci_ids(handle)
        reporter.line(f"pci.status={nvapi.status_text(pci_status)}")
        if pci:
            for key, value in pci.items():
                reporter.line(f"pci.{key}=0x{value:08X}")

        # Public architecture info: V2, 16 bytes.
        arch = nvapi.raw_gpu_struct(
            handle,
            ID_GPU_GET_ARCH_INFO,
            claimed_size=16,
            version=2,
        )
        reporter.line("")
        reporter.line("--- ARCHITECTURE ---")
        reporter.line(f"arch.status={nvapi.status_text(arch.status)}")
        if arch.data:
            architecture_id = struct.unpack_from("<I", arch.data, 4)[0]
            implementation_id = struct.unpack_from("<I", arch.data, 8)[0]
            revision_id = struct.unpack_from("<I", arch.data, 12)[0]
            reporter.line(f"arch.architectureId=0x{architecture_id:X}")
            reporter.line(f"arch.implementationId=0x{implementation_id:X}")
            reporter.line(f"arch.revisionId=0x{revision_id:X}")
            reporter.line(f"arch.raw={arch.data.hex()}")

        # V/F point mask.
        mask_result = nvapi.raw_gpu_struct(
            handle,
            ID_GPU_GET_VF_POINTS_INFO,
            claimed_size=VF_POINTS_INFO_SIZE,
            version=1,
        )
        reporter.line("")
        reporter.line("--- VF POINT MASK ---")
        dump_raw(
            reporter,
            output_dir,
            "vf-points-info",
            mask_result,
            nvapi,
        )

        if not mask_result.data:
            reporter.line(
                "vfCurveProbe=STOPPED_NO_VF_POINTS_INFO"
            )
            return 2

        mask = mask_result.data[4:36]
        slots = mask_slots(mask)
        reporter.line(f"vfMaskHex={mask.hex()}")
        reporter.line(f"vfMaskSetCount={len(slots)}")
        reporter.line(
            "vfMaskSlots=" + ",".join(str(x) for x in slots)
        )

        # Live effective V/F status.
        vf_status = nvapi.raw_gpu_struct(
            handle,
            ID_GPU_GET_VF_CURVE_STATUS,
            claimed_size=VF_STATUS_SIZE,
            version=1,
            request_mask=mask,
        )
        reporter.line("")
        reporter.line("--- VF STATUS ---")
        dump_raw(
            reporter,
            output_dir,
            "vf-status",
            vf_status,
            nvapi,
        )

        if vf_status.data:
            reporter.line("--- VF STATUS LAYOUT SCORING ---")
            decoded = []
            for layout in VF_STATUS_LAYOUTS:
                name, plausible, rows = decode_vf_status(
                    vf_status.data,
                    mask,
                    layout,
                )
                decoded.append((plausible, name, rows))
                reporter.line(
                    f"layout={name} plausibleCorePoints={plausible} "
                    f"decodedRows={len(rows)}"
                )

            decoded.sort(reverse=True, key=lambda x: x[0])
            best_plausible, best_name, best_rows = decoded[0]
            reporter.line(f"selectedVfStatusLayout={best_name}")
            reporter.line(
                f"selectedVfStatusPlausibleCorePoints={best_plausible}"
            )
            reporter.line("--- VF CURVE DECODED ---")
            for row in best_rows:
                if (
                    row["clockType"] == 0
                    and 300_000 <= row["voltUV"] <= 1_300_000
                ):
                    reporter.line(
                        f"slot={row['slot']} "
                        f"type={row['clockType']} "
                        f"voltage_mV={row['voltUV']/1000.0:.3f} "
                        f"frequency_MHz={row['freqKHz']/1000.0:.3f}"
                    )

        # Editable control table, READ ONLY.
        vf_control = nvapi.raw_gpu_struct(
            handle,
            ID_GPU_GET_VF_CONTROL,
            claimed_size=VF_CONTROL_SIZE,
            version=1,
            request_mask=mask,
        )
        reporter.line("")
        reporter.line("--- VF CONTROL READ-ONLY ---")
        dump_raw(
            reporter,
            output_dir,
            "vf-control",
            vf_control,
            nvapi,
        )

        if vf_control.data:
            reporter.line("--- VF CONTROL LAYOUT CANDIDATES ---")
            for name, reasonable, rows in decode_control_candidates(
                vf_control.data,
                mask,
            ):
                nonzero = [(slot, d) for slot, d in rows if d != 0]
                reporter.line(
                    f"layout={name} reasonableDeltaCount={reasonable} "
                    f"rowCount={len(rows)} nonzeroDeltaCount={len(nonzero)}"
                )
                for slot, delta in nonzero[:64]:
                    reporter.line(
                        f"  slot={slot} freqDeltaKHz={delta}"
                    )

        # Fan control read-only. The private layout can change; allocate
        # generously while claiming the known V1 size.
        fan_control = nvapi.raw_gpu_struct(
            handle,
            ID_GPU_GET_FAN_CONTROL,
            claimed_size=FAN_CONTROL_CLAIMED_SIZE,
            version=1,
            alloc_size=FAN_CONTROL_ALLOC_SIZE,
        )
        reporter.line("")
        reporter.line("--- FAN CONTROL READ-ONLY ---")
        dump_raw(
            reporter,
            output_dir,
            "fan-control",
            fan_control,
            nvapi,
        )

        if fan_control.data:
            count = struct.unpack_from("<I", fan_control.data, 8)[0]
            reporter.line(f"fanControl.claimedCount={count}")
            if count <= 32:
                reporter.line("--- FAN CONTROL DECODED CANDIDATE ---")
                base = 44
                stride = 44
                for i in range(count):
                    off = base + i * stride
                    if off + 12 > len(fan_control.data):
                        break
                    index, level, mode = struct.unpack_from(
                        "<III",
                        fan_control.data,
                        off,
                    )
                    reporter.line(
                        f"fan[{i}].index={index} "
                        f"level={level} mode={mode}"
                    )
            else:
                reporter.line(
                    "fanControl.layoutCandidate=REJECTED_COUNT_OUT_OF_RANGE"
                )

        reporter.line("")
        reporter.line("================================================================")
        reporter.line("PROBE RESULT")
        reporter.line("================================================================")
        reporter.line("writeCallsInvoked=0")
        reporter.line(
            "vfReadPath="
            + ("PASS" if vf_status.data and vf_control.data else "PARTIAL")
        )
        reporter.line(
            "fanReadPath="
            + ("PASS" if fan_control.data else "UNAVAILABLE_OR_LAYOUT_MISMATCH")
        )
        reporter.line(f"rawSnapshotDirectory={output_dir}")
        reporter.line("nextStep=review llm.txt before enabling any write call")
        return 0

    except Exception as exc:
        reporter.line("")
        reporter.line("================================================================")
        reporter.line("PROBE ERROR")
        reporter.line("================================================================")
        reporter.line(f"type={type(exc).__name__}")
        reporter.line(f"message={exc}")
        return 1

    finally:
        if nvapi is not None:
            try:
                status = nvapi.unload()
                reporter.line(f"NvAPI_Unload={nvapi.status_text(status)}")
            except Exception as exc:
                reporter.line(f"NvAPI_Unload.error={exc}")
        reporter.close()


if __name__ == "__main__":
    raise SystemExit(main())
