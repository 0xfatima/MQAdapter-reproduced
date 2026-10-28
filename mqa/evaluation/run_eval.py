"""Evaluate a model on the protocol's evaluation sets and write metrics, predictions and confusion matrices."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqa import CLASS_NAMES
from mqa.data.dataset import BrainTumorMRI
from mqa.data.preprocess import build_clip_transform
from mqa.evaluation.metrics import save_confusion_matrix_png
from mqa.protocols import harmonic_mean
from mqa.training.loop import evaluate_loader


def evaluate_sets(model, eval_records: dict, eval_sets: dict, device, out_dir: Path, batch_size: int = 16, num_workers: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    results = {}
    for name, classes in eval_sets.items():
        recs = eval_records.get(name, [])
        if not recs:
            raise SystemExit(f"no images for evaluation set '{name}'")
        if hasattr(model, "set_active_classes"):
            model.set_active_classes(classes)
        names = [CLASS_NAMES[i] for i in classes]
        loader = DataLoader(
            BrainTumorMRI(recs, transform=build_clip_transform(train=False)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        m = evaluate_loader(model, loader, device, names)
        paths = m.pop("paths")
        y_true, y_pred = m.pop("y_true"), m.pop("y_pred")
        sub = out_dir / name if len(eval_sets) > 1 else out_dir
        sub.mkdir(parents=True, exist_ok=True)
        m["eval_set"] = name
        m["classes"] = names
        m["n_images"] = len(y_true)
        m["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        m["total_params"] = int(sum(p.numel() for p in model.parameters()))
        (sub / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        rows = [
            {"path": p, "y_true": t, "y_pred": q, "true_name": names[t], "pred_name": names[q]}
            for p, t, q in zip(paths, y_true, y_pred)
        ]
        (sub / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        try:
            save_confusion_matrix_png(np.array(m["confusion_matrix"]), names, str(sub / "confusion_matrix.png"))
        except RuntimeError as exc:
            (sub / "plot_skipped.txt").write_text(str(exc), encoding="utf-8")
        results[name] = m
    if "base" in results and "new" in results:
        b, n = results["base"]["accuracy"], results["new"]["accuracy"]
        summary = {"base_accuracy": b, "new_accuracy": n, "harmonic_mean": harmonic_mean(b, n)}
        (out_dir / "base2new_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        results["base2new"] = summary
    return results
