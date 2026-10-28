#!/usr/bin/env python
"""Paper Fig. 5: t-SNE of learned visual features (Testing). Use --frozen for plain CLIP or --checkpoint for a run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mqa import CLASS_NAMES
from mqa.data.dataset import BrainTumorMRI, discover_split
from mqa.data.preprocess import build_clip_transform
from mqa.models.factory import build_backbone, build_model
from mqa.training.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--data-root", default=str(ROOT / "data" / "brain_tumor_mri"))
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.frozen and not args.checkpoint:
        sys.exit("give --checkpoint runs/<exp>/best.pt or --frozen")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    if args.frozen:
        cfg = {"model": "frozen_clip", "clip_backend": "open_clip", "download_clip": True}
    else:
        cp = Path(args.checkpoint)
        cfg_path = Path(args.config) if args.config else cp.parent / "config_used.json"
        cfg = json.loads(cfg_path.read_text())
    clip, text, tok = build_backbone(cfg, device)
    model = build_model(cfg, clip, text, tok)
    if not args.frozen:
        model.load_state_dict(load_checkpoint(args.checkpoint, map_location=device)["model"])
    model.to(device).eval()
    info = discover_split(Path(args.data_root) / "Testing")
    ds = BrainTumorMRI(info["records"], transform=build_clip_transform(False))
    idx = np.random.default_rng(0).permutation(len(ds))[: args.max_images]
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=32)
    feats, ys = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["image"].to(device))
            feats.append(out["image_features"].float().cpu().numpy())
            ys.extend(b["label"].tolist())
    X, y = np.concatenate(feats), np.asarray(ys)
    from sklearn.manifold import TSNE

    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, n in enumerate(CLASS_NAMES):
        m = y == i
        ax.scatter(emb[m, 0], emb[m, 1], s=4, label=n)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("frozen CLIP" if args.frozen else Path(args.checkpoint).parent.name)
    out_path = Path(args.out or (ROOT / "runs" / ("tsne_frozen.png" if args.frozen else f"tsne_{Path(args.checkpoint).parent.name}.png")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
