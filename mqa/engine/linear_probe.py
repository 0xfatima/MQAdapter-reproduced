"""Linear-probe CLIP (Radford et al. 2021): L2-regularised logistic regression on frozen image features.

C is chosen on the validation split (never on test). A linear probe has no text classifier, so it can only
be evaluated on evaluation sets whose classes equal the training classes (not on new classes / other datasets).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqa.engine.data import RecordDataset, build_transform, eval_dataset
from mqa.evaluation.metrics import classification_metrics


@torch.no_grad()
def _features(clip, ds, device, bs=64, nw=2):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    fs, ys = [], []
    for b in loader:
        f = clip.encode_image(b["image"].to(device)).float()
        fs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        ys.append(b["label"].numpy())
    return np.concatenate(fs), np.concatenate(ys)


def run_linear_probe(clip, split, cfg, device, out_dir: Path, log=print) -> dict:
    from sklearn.linear_model import LogisticRegression

    out_dir.mkdir(parents=True, exist_ok=True)
    aug = cfg.get("augment", "mri")
    nw = int(cfg.get("num_workers", 2))
    xtr, ytr = _features(clip, RecordDataset(split.train, build_transform(False, aug)), device, nw=nw)
    xva, yva = _features(clip, RecordDataset(split.val, build_transform(False, aug)), device, nw=nw) if split.val else (xtr, ytr)
    best = (-1.0, None)
    for C in cfg.get("C_grid", [0.01, 0.1, 1.0, 10.0, 100.0]):
        clf = LogisticRegression(C=C, max_iter=2000).fit(xtr, ytr)
        acc = float((clf.predict(xva) == yva).mean())
        log(f"  linear probe C={C}: val acc {acc:.4f}")
        if acc > best[0]:
            best = (acc, C)
    clf = LogisticRegression(C=best[1], max_iter=2000).fit(np.concatenate([xtr, xva]) if split.val else xtr,
                                                          np.concatenate([ytr, yva]) if split.val else ytr)
    results = {}
    for e in split.evals:
        if e.space.classes != split.space.classes:
            results[e.name] = {"eval_set": e.name, "group": e.group, "accuracy": None, "note": "not applicable to unseen classes"}
            continue
        xe, ye = _features(clip, eval_dataset(e, aug), device, nw=nw)
        m = classification_metrics(ye.tolist(), clf.predict(xe).tolist(), e.space.classes)
        m.update(eval_set=e.name, group=e.group, shift=e.shift, mode=e.mode)
        (out_dir / e.name.strip("-")).mkdir(parents=True, exist_ok=True)
        (out_dir / e.name.strip("-") / "metrics.json").write_text(json.dumps(m, indent=2))
        results[e.name] = m
    return {"evals": results, "chosen_C": best[1], "trainable_params": int(clf.coef_.size + clf.intercept_.size)}
