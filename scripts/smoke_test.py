#!/usr/bin/env python
"""End-to-end smoke: tiny synthetic MRI-like images, FakeCLIP (no weight download)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torch.utils.data import DataLoader

from mqa import CLASS_NAMES
from mqa.data.dataset import BrainTumorMRI
from mqa.data.preprocess import build_clip_transform
from mqa.models.factory import build_backbone, build_model
from mqa.training.loop import evaluate_loader, train_model


def write_tiny_dataset(root: Path, n_per=3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for split in ("Training", "Testing"):
        for i, cls in enumerate(CLASS_NAMES):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n_per):
                img = Image.new("L", (80 + i, 90 + j), color=40 + 20 * i + j)
                img.save(d / f"{cls}_{j}.png")


def main():
    tiny = ROOT / "runs" / "_tiny_mri"
    write_tiny_dataset(tiny)
    recs = []
    from mqa.data.dataset import discover_split

    info = discover_split(tiny / "Training")
    recs = info["records"]
    ds = BrainTumorMRI(recs, transform=build_clip_transform(train=True))
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    val = DataLoader(BrainTumorMRI(recs, transform=build_clip_transform(train=False)), batch_size=4)
    device = torch.device("cpu")
    cfg = {"clip_backend": "fake", "model": "mqadapter", "k": 2, "epochs": 1, "quantum_map": "vqc"}
    clip, text, _ = build_backbone(cfg, device)
    model = build_model(cfg, clip, text)
    out = ROOT / "runs" / "smoke"
    summary = train_model(model, loader, val, list(CLASS_NAMES), out_dir=out, epochs=1, lr=1e-3, device="cpu", seed=0, patience=2, max_steps=2)
    ev = evaluate_loader(model, val, device, CLASS_NAMES)
    print(json.dumps({"trainable": model.trainable_parameter_count(), "summary_trained": summary["trained"], "n_eval": ev["n_samples"]}, indent=2))
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
