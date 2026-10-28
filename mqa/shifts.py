"""Synthetic MRI domain shifts for the domain-generalisation task.

The paper trains on ImageNet and tests on four shifted versions (V2, Sketch, A, R). There is no such suite
for brain MRI, so we build four *deterministic* shifted copies of the MRI test split that mimic real
acquisition differences. They are an analogue, not a reproduction of the paper's datasets:

  noise      Rician-like noise (low SNR / fast sequence)              ~ ImageNet-V2 (mild re-collection)
  bias       smooth multiplicative bias field + gamma (coil / B1 inhomogeneity)
  lowres     2-4x down-sampling + blur + JPEG (old scanner, thick slices, re-encoding)
  sketch     edge map of the image (structure only, no intensity)     ~ ImageNet-Sketch

Each shift is seeded from the image path, so every run sees exactly the same corrupted images.
Real second-site data can be used instead through a dataset spec (see main.py --dg-targets).
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageFilter

SHIFTS = ("noise", "bias", "lowres", "sketch")


def _rng(path: str, name: str) -> np.random.Generator:
    h = hashlib.sha1(f"{name}:{path}".encode()).hexdigest()[:8]
    return np.random.default_rng(int(h, 16))


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _from_gray(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")


def apply_shift(img: Image.Image, name: str, path: str = "", severity: int = 3) -> Image.Image:
    """severity 1..5."""
    s = max(1, min(5, int(severity)))
    rng = _rng(path, name)
    if name == "noise":
        a = _to_gray_array(img)
        sigma = 0.02 + 0.02 * s
        re = a + rng.normal(0, sigma, a.shape)
        im = rng.normal(0, sigma, a.shape)
        return _from_gray(np.sqrt(re**2 + im**2))
    if name == "bias":
        a = _to_gray_array(img)
        h, w = a.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy, xx = yy / max(h - 1, 1) - 0.5, xx / max(w - 1, 1) - 0.5
        c = rng.normal(0, 1, 6)
        field = c[0] * xx + c[1] * yy + c[2] * xx * yy + c[3] * xx**2 + c[4] * yy**2
        field = np.exp(0.15 * s * field / (np.abs(field).max() + 1e-6))
        gamma = float(np.exp(rng.uniform(-0.1, 0.1) * s))
        return _from_gray((a * field) ** gamma)
    if name == "lowres":
        w, h = img.size
        f = 1.5 + 0.5 * s
        small = img.convert("L").resize((max(8, int(w / f)), max(8, int(h / f))), Image.BILINEAR)
        up = small.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=0.3 * s))
        buf = io.BytesIO()
        up.save(buf, format="JPEG", quality=max(10, 60 - 10 * s))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "sketch":
        g = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.0))
        edges = g.filter(ImageFilter.FIND_EDGES)
        a = _to_gray_array(edges)
        a = a / (a.max() + 1e-6)
        return _from_gray(1.0 - np.clip(a * (1.5 + 0.25 * s), 0, 1))  # dark lines on white, like a sketch
    raise ValueError(f"unknown shift {name}; choose from {SHIFTS}")


class ShiftTransform:
    """PIL -> PIL wrapper used by ShiftedDataset (needs the path for deterministic seeding)."""

    def __init__(self, name: str, severity: int = 3):
        self.name, self.severity = name, severity

    def __call__(self, img: Image.Image, path: str = "") -> Image.Image:
        return apply_shift(img, self.name, path, self.severity)
