"""Evaluate a host on one EvalSet; writes metrics.json, predictions.csv, confusion matrix and (optionally) features."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqa.engine.data import EvalSet, eval_dataset
from mqa.evaluation.metrics import classification_metrics, save_confusion_matrix_png


@torch.no_grad()
def evaluate_set(model, e: EvalSet, cfg: dict, device, out_dir: Path, save_features: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    model.set_class_space(e.space, e.mode)
    loader = DataLoader(eval_dataset(e, cfg.get("augment", "mri")), batch_size=int(cfg.get("eval_batch_size", 32)),
                        shuffle=False, num_workers=int(cfg.get("num_workers", 2)), pin_memory=device.type == "cuda")
    ys, ps, paths, hits, top1, feats, losses = [], [], [], [], [], [], []
    t0, n = time.perf_counter(), 0
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        out = model(x, y)
        losses.append(float(out["loss"]) * y.numel())
        ys += y.tolist()
        ps += out["logits"].argmax(-1).tolist()
        paths += list(batch["path"])
        if out.get("retrieval_hit") is not None:
            hits += out["retrieval_hit"].tolist()
            top1 += (out["topk_idx"][:, 0] == y).tolist()
        if save_features:
            feats.append(out["image_features"].float().cpu().numpy())
        n += y.numel()
    dt = time.perf_counter() - t0
    m = classification_metrics(ys, ps, e.space.classes)
    m.update(eval_set=e.name, group=e.group, mode=e.mode, shift=e.shift, classes=e.space.classes,
             loss=float(sum(losses) / max(n, 1)), images_per_second=n / dt if dt else None)
    if hits:
        h = np.asarray(hits, bool)
        yt, pr = np.asarray(ys), np.asarray(ps)
        m.update(coarse_topk_recall=float(h.mean()), coarse_top1_accuracy=float(np.mean(top1)), k=int(model.k),
                 accuracy_when_retrieved=float((yt[h] == pr[h]).mean()) if h.any() else None,
                 accuracy_when_missed=float((yt[~h] == pr[~h]).mean()) if (~h).any() else None)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_pred", "true_name", "pred_name"])
        for p, t, q in zip(paths, ys, ps):
            w.writerow([p, t, q, e.space.classes[t], e.space.classes[q]])
    if save_features and feats:
        np.savez_compressed(out_dir / "features.npz", features=np.concatenate(feats), labels=np.asarray(ys), classes=np.asarray(e.space.classes))
    try:
        save_confusion_matrix_png(np.array(m["confusion_matrix"]), e.space.classes, str(out_dir / "confusion_matrix.png"))
    except Exception:
        pass
    return m
