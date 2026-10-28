#!/usr/bin/env python
"""Paper Fig. 1 on MRI: frozen-CLIP Top-1..Top-C accuracy and image-vs-class similarity distribution on Testing.

This tests the paper's motivating claim (high Top-K, low Top-1). With C=4 Top-4 is trivially 100%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mqa import CLASS_NAMES
from mqa.data.dataset import BrainTumorMRI, discover_split
from mqa.data.preprocess import build_clip_transform
from mqa.models.factory import build_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data" / "brain_tumor_mri"))
    ap.add_argument("--clip-backend", default="open_clip")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default=str(ROOT / "runs" / "topk_analysis"))
    args = ap.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    clip, text, _ = build_backbone({"clip_backend": args.clip_backend, "download_clip": True}, device)
    clip.eval()
    info = discover_split(Path(args.data_root) / "Testing")
    loader = DataLoader(BrainTumorMRI(info["records"], transform=build_clip_transform(False)), batch_size=args.batch_size)
    text = F.normalize(text.float(), dim=-1)
    ranks, sims_top = [], []
    with torch.no_grad():
        for b in loader:
            img = F.normalize(clip.encode_image(b["image"].to(device)).float(), dim=-1)
            sim = img @ text.T
            order = sim.argsort(dim=-1, descending=True)
            y = b["label"].to(device)
            ranks.extend((order == y.unsqueeze(-1)).float().argmax(dim=-1).cpu().tolist())
            sims_top.extend(sim.max(dim=-1).values.cpu().tolist())
    ranks = np.asarray(ranks)
    c = len(CLASS_NAMES)
    topk = {f"top{k}": float((ranks < k).mean()) for k in range(1, c + 1)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {"n_images": int(len(ranks)), "topk_accuracy": topk, "mean_top1_similarity": float(np.mean(sims_top))}
    (out / "topk_accuracy.json").write_text(json.dumps(res, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].bar(range(1, c + 1), [100 * topk[f"top{k}"] for k in range(1, c + 1)])
        ax[0].set_xlabel("K"); ax[0].set_ylabel("Top-K accuracy (%)"); ax[0].set_xticks(range(1, c + 1))
        ax[1].hist(sims_top, bins=40)
        ax[1].set_xlabel("max cosine similarity to a class prompt"); ax[1].set_ylabel("count")
        fig.tight_layout(); fig.savefig(out / "topk_fig1_mri.png", dpi=130)
    except Exception as exc:  # plotting is optional
        (out / "plot_skipped.txt").write_text(repr(exc))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
