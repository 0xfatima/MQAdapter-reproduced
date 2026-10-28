#!/usr/bin/env python
"""Create tiny synthetic "MRI-like" image folders so the whole pipeline can be exercised without real data.

    python scripts/make_dummy_dataset.py data/dummy
    python main.py --make-dummy-data data/dummy --backend fake --quick

Images are grey ellipses ("brain") with a bright blob whose position/size depends on the class. They are
NOT medical data and any accuracy measured on them is meaningless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

LAYOUT = {
    "brain_tumor_mri": (["glioma", "meningioma", "notumor", "pituitary"], {"Training": 24, "Testing": 8}),
    "figshare_brain": (["glioma", "meningioma", "pituitary"], {"Testing": 6}),
    "br35h": (["no", "yes"], {"Testing": 6}),
}
BLOB = {"glioma": (0.35, 0.40, 0.12), "meningioma": (0.70, 0.30, 0.08), "pituitary": (0.50, 0.62, 0.06), "yes": (0.4, 0.4, 0.1)}


def _image(cls: str, rng: np.random.Generator, size: int = 128) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size] / size
    brain = (((xx - 0.5) / 0.38) ** 2 + ((yy - 0.5) / 0.45) ** 2 < 1).astype(np.float32) * 0.45
    img = brain + rng.normal(0, 0.05, brain.shape)
    if cls in BLOB:
        cx, cy, r = BLOB[cls]
        cx, cy = cx + rng.normal(0, 0.03), cy + rng.normal(0, 0.03)
        img += 0.5 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2)))
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="L")


def make_all(root: Path, seed: int = 0) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out = {}
    for name, (classes, splits) in LAYOUT.items():
        base = Path(root) / name
        for split, n in splits.items():
            for c in classes:
                d = base / split / c
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    p = d / f"{c}_{i:03d}.png"
                    if not p.exists():
                        _image(c, rng).save(p)
        out[name] = base
    return out


if __name__ == "__main__":
    roots = make_all(Path(sys.argv[1] if len(sys.argv) > 1 else "data/dummy"))
    for k, v in roots.items():
        print(f"{k}: {v}")
