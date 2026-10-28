#!/usr/bin/env python
"""Evaluate frozen CLIP zero-shot on Testing (no training). Requires CLIP weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqa import CLASS_NAMES
from mqa.data.dataset import BrainTumorMRI, discover_split
from mqa.data.preprocess import build_clip_transform
from mqa.evaluation.metrics import classification_metrics, save_confusion_matrix_png
from mqa.models.factory import build_backbone, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "brain_tumor_mri"))
    ap.add_argument("--clip-backend", default="open_clip")
    ap.add_argument("--download-clip", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "frozen_clip_zeroshot" / "eval"))
    args = ap.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    cfg = {"clip_backend": args.clip_backend, "download_clip": args.download_clip, "model": "frozen_clip"}
    clip, text, _ = build_backbone(cfg, device)
    model = build_model(cfg, clip, text).to(device).eval()
    info = discover_split(Path(args.data_root) / "Testing")
    if info["n_images"] == 0:
        raise SystemExit(f"no Testing images under {args.data_root}")
    loader = DataLoader(BrainTumorMRI(info["records"], transform=build_clip_transform(False)), batch_size=args.batch_size)
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["image"].to(device))["logits"].argmax(-1).cpu().tolist()
            ys.extend(batch["label"].tolist())
            ps.extend(pred)
    metrics = classification_metrics(ys, ps, CLASS_NAMES)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        save_confusion_matrix_png(np.array(metrics["confusion_matrix"]), CLASS_NAMES, str(out / "confusion_matrix.png"))
    except RuntimeError:
        pass
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "n": metrics["n_samples"]}, indent=2))


if __name__ == "__main__":
    main()
