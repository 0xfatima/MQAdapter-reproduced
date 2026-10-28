"""Phase 3a: environment + TorchQuantum verification of the paper's VQC (Eqs. 5-8). Fails loudly; no silent skips.

Target hardware: 12 GB NVIDIA GPU, 64 GB RAM (CUDA used automatically if available).
Checks: package versions/hardware; TorchQuantum builds the 8-qubit circuit (Ry encoding, Ry, ring CRZ, Ry, Pauli-Z);
output shape; parameter count; parity with the numpy oracle (and which CRZ control/target reading TorchQuantum matches);
gradients reach every parameter group; the VQC stays fp32 when called inside an autocast region;
forward/backward timing and peak GPU memory vs batch size (compute estimate for Phase 10).
Run:  python scripts/phase3_env_check.py --out phase3_env_report.json
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn

from mqa.quantum.reference_paper_vqc import paper_param_count, paper_vqc_expvals, ring_edges

N = 8


def pkg(n):
    try:
        return version(n)
    except PackageNotFoundError:
        return None


def meminfo_gb():
    proc = Path("/proc/meminfo")
    if proc.exists():
        d = {}
        for line in proc.read_text().splitlines():
            k, v = line.split(":")
            d[k] = int(v.split()[0]) / 1e6
        return {"total_gb": d.get("MemTotal"), "available_gb": d.get("MemAvailable")}
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
        return {"total_gb": stat.ullTotalPhys / 1e9, "available_gb": stat.ullAvailPhys / 1e9}
    except Exception as exc:
        return {"total_gb": None, "available_gb": None, "error": repr(exc)}


def build_paper_vqc():
    import torchquantum as tq
    import torchquantum.functional as tqf

    class PaperVQC(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.ry1 = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(N)])
            self.crz = nn.ModuleList([tq.CRZ(has_params=True, trainable=True) for _ in ring_edges(N)])
            self.ry2 = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(N)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x):  # x: [B, N] angles
            qdev = tq.QuantumDevice(n_wires=N, bsz=x.shape[0], device=x.device)
            for n in range(N):
                tqf.ry(qdev, wires=n, params=x[:, n])  # Eq. 5 angle encoding
            for n in range(N):
                self.ry1[n](qdev, wires=n)
            for e, (a, b) in enumerate(ring_edges(N)):
                self.crz[e](qdev, wires=[a, b])  # TorchQuantum: wires=[control, target]
            for n in range(N):
                self.ry2[n](qdev, wires=n)
            return self.measure(qdev)  # [B, N] <Z_n>

    return PaperVQC()


def angles(mod_list):
    return np.array([m.params.detach().cpu().numpy().reshape(-1)[0] for m in mod_list], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase3_env_report.json")
    ap.add_argument("--tol", type=float, default=2e-5, help="float32 (esp. CUDA complex64) vs float64 oracle")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    rep = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "memory": meminfo_gb(), "disk_free_gb": shutil.disk_usage(".").free / 1e9,
           "packages": {p: pkg(p) for p in ("torch", "torchvision", "torchquantum", "torchquantum-ng",
                                           "open_clip_torch", "clip", "numpy", "pillow", "scikit-learn")},
           "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_used": dev,
           "warnings": []}
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        rep.update({"gpu": pr.name, "gpu_mem_gb": pr.total_memory / 1e9, "compute_capability": f"{pr.major}.{pr.minor}",
                    "bf16_supported": bool(torch.cuda.is_bf16_supported())})
        if pr.total_memory / 1e9 < 11:
            rep["warnings"].append(f"GPU has {pr.total_memory / 1e9:.1f} GB; expected ~12 GB")
    else:
        rep["warnings"].append("CUDA not available: GPU profile skipped; check the PyTorch CUDA wheel vs `nvidia-smi` driver")
    if rep["memory"]["total_gb"] and rep["memory"]["total_gb"] < 60:
        rep["warnings"].append(f"RAM {rep['memory']['total_gb']:.0f} GB; expected ~64 GB")

    torch.manual_seed(0)
    vqc = build_paper_vqc().to(dev)
    n_params = sum(p.numel() for p in vqc.parameters())
    x = (torch.rand(5, N, device=dev) * 2 - 1)
    out = vqc(x)
    rep["circuit"] = {"n_params_torchquantum": n_params, "n_params_expected_paper": paper_param_count(N),
                      "output_shape": list(out.shape), "output_dtype": str(out.dtype),
                      "output_range": [out.min().item(), out.max().item()]}
    assert n_params == paper_param_count(N), f"param count {n_params} != {paper_param_count(N)}"
    assert tuple(out.shape) == (5, N)

    p1, cz, p2 = angles(vqc.ry1), angles(vqc.crz), angles(vqc.ry2)
    ref_in = x.detach().cpu().numpy()
    out_np = out.detach().cpu().numpy()
    diffs = {d: float(np.max(np.abs(paper_vqc_expvals(ref_in, p1, cz, p2, d) - out_np)))
             for d in ("control_first", "target_first")}
    best = min(diffs, key=diffs.get)
    rep["parity_with_numpy_oracle"] = {"max_abs_diff_by_crz_reading": diffs,
                                       "matching_reading": best if diffs[best] < a.tol else None, "tolerance": a.tol}
    assert diffs[best] < a.tol, f"TorchQuantum matches NEITHER CRZ reading (diffs {diffs}); inspect gate/wire conventions"

    if dev == "cuda":  # D15: the quantum block must run in fp32 even if the surrounding model uses autocast
        ref = vqc(x).detach()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.autocast("cuda", enabled=False):
                y_iso = vqc(x.float())
        rep["autocast_isolated"] = {"dtype": str(y_iso.dtype), "max_abs_diff_vs_fp32": float((y_iso.detach() - ref).abs().max())}
        assert y_iso.dtype == torch.float32 and rep["autocast_isolated"]["max_abs_diff_vs_fp32"] < 1e-6

    vqc.zero_grad()
    loss = (vqc(x) ** 2).sum()
    loss.backward()
    g = {k: float(sum(p.grad.abs().sum().item() for m in getattr(vqc, k) for p in m.parameters()))
         for k in ("ry1", "crz", "ry2")}
    rep["gradients_abs_sum"] = g
    assert all(np.isfinite(v) and v > 0 for v in g.values()), f"some parameter group has no gradient: {g}"

    prof = {}
    batches = (64, 256, 1024) + ((4096,) if dev == "cuda" else ())
    m = build_paper_vqc().to(dev)
    for B in batches:
        xb = torch.rand(B, N, device=dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(4):
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            (m(xb) ** 2).sum().backward()
            if dev == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        prof[f"{dev}_B{B}"] = {"fwd_bwd_median_s": statistics.median(ts[1:]),
                               **({"peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6} if dev == "cuda" else {})}
    rep["profile"] = prof
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("PHASE3a PASSED")


if __name__ == "__main__":
    main()
