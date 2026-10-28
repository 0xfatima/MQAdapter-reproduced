"""Paper analyses that are not training runs.

  topk_analysis   Fig. 1: frozen-CLIP Top-1..Top-5 accuracy and the distribution of image-to-Top-5-class
                  similarities on the test split.
  efficiency      Table VII: trainable parameters (host / MQAdapter / VQC) and inference FPS at batch 100.
  tsne_figure     Fig. 5: t-SNE of image features saved by base-to-new evaluations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqa.datasets import ClassSpace
from mqa.engine.data import RecordDataset, build_transform


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@torch.no_grad()
def topk_analysis(host, records, space: ClassSpace, device, out_dir: Path, cfg: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    host.to(device).eval()
    host.set_class_space(space, "base")
    loader = DataLoader(RecordDataset(records, build_transform(False, cfg.get("augment", "mri"))),
                        batch_size=int(cfg.get("eval_batch_size", 32)), num_workers=int(cfg.get("num_workers", 2)))
    kmax = min(5, len(space))
    hits = np.zeros(kmax)
    sims, n = [], 0
    for b in loader:
        img = host.frozen_image_features(b["image"].to(device))
        s = img @ host.anchor_features.T
        top = s.topk(kmax, dim=-1)
        y = b["label"].to(device).unsqueeze(-1)
        for k in range(kmax):
            hits[k] += float((top.indices[:, : k + 1] == y).any(-1).sum())
        sims.append(top.values.cpu().numpy().ravel())
        n += y.shape[0]
    acc = (hits / max(n, 1) * 100).tolist()
    sims = np.concatenate(sims) if sims else np.zeros(0)
    res = {"top_k_accuracy": {f"top{k + 1}": a for k, a in enumerate(acc)}, "mean_similarity_topk": float(sims.mean()) if sims.size else None,
           "n_images": n, "classes": space.classes}
    (out_dir / "topk_analysis.json").write_text(json.dumps(res, indent=2))
    try:
        plt = _plt()
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].bar(range(1, kmax + 1), acc, color="#9db4e0")
        ax[0].plot(range(1, kmax + 1), acc, "r-o")
        ax[0].set_xticks(range(1, kmax + 1), [f"Top-{k}" for k in range(1, kmax + 1)])
        ax[0].set_ylabel("Accuracy (%)")
        ax[1].hist(sims, bins=40, color="#3b78b5")
        ax[1].set_title(f"Mean sim: {res['mean_similarity_topk']:.4f}")
        fig.suptitle("Frozen CLIP on the test split (paper Fig. 1)")
        fig.tight_layout()
        fig.savefig(out_dir / "fig1_topk.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        res["plot_error"] = str(exc)
    return res


@torch.no_grad()
def efficiency(host, space: ClassSpace, device, batch_size: int = 100, iters: int = 5) -> dict:
    host.to(device).eval()
    host.set_class_space(space, "base")
    total = host.trainable_parameter_count()
    plug = host.plugin_parameter_count()
    vqc = 0
    if host.plugin is not None:
        vqc = sum(p.numel() for p in host.plugin.vqc.parameters() if p.requires_grad)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    fps = None
    try:
        host(x[:2])
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            host(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fps = iters * batch_size / (time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    return {"trainable_params": total, "host_params": total - plug, "mqadapter_params": plug, "vqc_params": vqc,
            "fps": fps, "batch_size": batch_size, "device": str(device)}


def tsne_figure(feature_files: dict[str, dict[str, Path]], out_path: Path, seed: int = 0) -> dict:
    """feature_files: {method: {"base": path, "new": path}} -> grid figure like paper Fig. 5."""
    from sklearn.manifold import TSNE

    plt = _plt()
    methods = [m for m in feature_files if feature_files[m]]
    cols = ["base", "new"]
    fig, axes = plt.subplots(len(methods), 2, figsize=(7, 3.2 * max(1, len(methods))), squeeze=False)
    done = {}
    for r, m in enumerate(methods):
        for c, part in enumerate(cols):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            p = feature_files[m].get(part)
            if p is None or not Path(p).exists():
                ax.set_title(f"{m} / {part}: missing", fontsize=8)
                continue
            d = np.load(p)
            f, y = d["features"], d["labels"]
            perp = max(2, min(30, len(f) // 4))
            emb = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca").fit_transform(f)
            ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=4)
            ax.set_title(f"{m} / {part}", fontsize=9)
            done[f"{m}/{part}"] = str(p)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return done
