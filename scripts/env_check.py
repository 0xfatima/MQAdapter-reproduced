#!/usr/bin/env python
"""Portable environment report (Windows/Linux). Does not download models."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pkg(n):
    try:
        return version(n)
    except PackageNotFoundError:
        return None


def meminfo_gb():
    try:
        import psutil

        v = psutil.virtual_memory()
        return {"total_gb": v.total / 1e9, "available_gb": v.available / 1e9, "source": "psutil"}
    except Exception:
        pass
    p = Path("/proc/meminfo")
    if p.exists():
        d = {}
        for line in p.read_text().splitlines():
            k, v = line.split(":")
            d[k] = int(v.split()[0]) / 1e6
        return {"total_gb": d.get("MemTotal"), "available_gb": d.get("MemAvailable"), "source": "/proc/meminfo"}
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return {"total_gb": stat.ullTotalPhys / 1e9, "available_gb": stat.ullAvailPhys / 1e9, "source": "GlobalMemoryStatusEx"}
    except Exception as exc:
        return {"total_gb": None, "available_gb": None, "error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "env_report.json"))
    args = ap.parse_args()
    import torch

    from mqa.quantum.reference_paper_vqc import paper_param_count
    from mqa.quantum.torch_vqc import PaperVQC

    rep = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory": meminfo_gb(),
        "disk_free_gb": shutil.disk_usage(str(ROOT)).free / 1e9,
        "packages": {p: pkg(p) for p in ("torch", "torchvision", "torchquantum", "torchquantum-ng", "open_clip_torch", "clip", "numpy", "pillow", "scikit-learn", "matplotlib")},
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "warnings": [],
    }
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        rep.update({"gpu": pr.name, "gpu_mem_gb": pr.total_memory / 1e9})
    else:
        rep["warnings"].append("CUDA not available in this Python process (torch is CPU or driver not visible).")

    vqc = PaperVQC(n_qubits=8)
    x = torch.randn(3, 8)
    y = vqc(x)
    n_params = sum(p.numel() for p in vqc.parameters())
    rep["statevector_vqc"] = {
        "n_params": n_params,
        "expected": paper_param_count(8),
        "output_shape": list(y.shape),
        "finite": bool(torch.isfinite(y).all().item()),
    }
    assert n_params == paper_param_count(8)
    assert tuple(y.shape) == (3, 8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print("ENV_CHECK PASSED")


if __name__ == "__main__":
    main()
