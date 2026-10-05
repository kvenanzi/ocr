"""Detect the accelerator we are running on and map it to a Colab runtime tag."""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

import psutil


@dataclass
class HardwareInfo:
    tag: str                      # T4, L4, A100-40GB, G4, TPU-v6e-1, cpu, ...
    kind: str                     # gpu | tpu | cpu
    accelerator: str = ""         # raw device name
    vram_gb: float = 0.0          # per device (HBM for TPU)
    compute_capability: float = 0.0
    num_devices: int = 0
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    cpu_model: str = ""
    ram_gb: float = 0.0
    driver: str = ""
    in_colab: bool = False

    @property
    def bf16(self) -> bool:
        return self.kind == "tpu" or self.compute_capability >= 8.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bf16"] = self.bf16
        return d


# HBM per chip, used when the TPU runtime doesn't tell us.
_TPU_HBM_GB = {"v2": 8, "v3": 16, "v4": 32, "v5e": 16, "v5p": 95, "v6e": 32}


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _gpu_tag(name: str, vram_gb: float) -> str:
    n = name.upper()
    if "A100" in n:
        return "A100-80GB" if vram_gb > 60 else "A100-40GB"
    if "RTX PRO 6000" in n or "RTX 6000 PRO" in n:
        return "G4"
    for t in ("H100", "L4", "T4", "V100", "A10G", "L40S"):
        if re.search(rf"\b{t}\b", n):
            return t
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def _detect_nvidia() -> HardwareInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name, mem_mib, cc, driver = [x.strip() for x in out[0].split(",")]
    vram = round(float(mem_mib) / 1024, 1)
    return HardwareInfo(
        tag=_gpu_tag(name, vram), kind="gpu", accelerator=name, vram_gb=vram,
        compute_capability=float(cc), num_devices=len(out), driver=driver,
    )


def _jax_tpu_kind() -> str:
    """Ask JAX (preinstalled on Colab TPU runtimes) for the chip, e.g. 'TPU v6 lite' -> 'v6e-1'."""
    code = "import jax; d = jax.devices(); print(d[0].device_kind, len(d))"
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""
    m = re.match(r"TPU v(\d+)\s*(lite|e|p)?\S*\s+(\d+)$", out, re.I)
    if not m:
        return ""
    suffix = {"lite": "e", "e": "e", "p": "p"}.get((m.group(2) or "").lower(), "")
    return f"v{m.group(1)}{suffix}-{m.group(3)}"


def _detect_tpu() -> HardwareInfo | None:
    accel = os.environ.get("TPU_ACCELERATOR_TYPE", "")
    has_dev = bool(glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*"))
    if not accel and not has_dev and "COLAB_TPU_ADDR" not in os.environ:
        return None
    accel = accel or _jax_tpu_kind()
    # e.g. "v6e-1", "v5litepod-1"
    m = re.match(r"(v\d+[a-z]*)(?:litepod)?-(\d+)", accel.replace("v5litepod", "v5e"))
    gen, n = (m.group(1), int(m.group(2))) if m else ("unknown", len(glob.glob("/dev/accel*")) or 1)
    return HardwareInfo(
        tag=f"TPU-{gen}-{n}", kind="tpu", accelerator=accel or "TPU",
        vram_gb=float(_TPU_HBM_GB.get(gen, 16)), num_devices=n,
    )


def detect() -> HardwareInfo:
    hw = _detect_nvidia() or _detect_tpu() or HardwareInfo(tag="cpu", kind="cpu")
    if os.environ.get("OCRBENCH_HW_TAG"):   # manual override if detection gets it wrong
        hw.tag = os.environ["OCRBENCH_HW_TAG"]
    hw.cpu_model = _cpu_model()
    hw.ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    hw.in_colab = "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content")
    return hw


if __name__ == "__main__":
    import json

    print(json.dumps(detect().to_dict(), indent=2))
