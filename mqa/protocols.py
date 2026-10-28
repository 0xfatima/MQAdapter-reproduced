"""Evaluation protocols (paper Section V-A1) mapped onto the 4-class MRI dataset.

all          train on every class, evaluate on the official Testing split (primary MRI protocol).
base2new     paper's base-to-new generalisation: classes are split in two halves; train on base classes only,
             evaluate on Testing images of base AND new classes (H = harmonic mean). With C=4 each half has
             two classes. Default halves follow CLASS_NAMES order: base={glioma, meningioma},
             new={notumor, pituitary}; override with cfg["base_classes"].
few-shot     cfg["shots"] = N keeps N Training images per class (paper: 1/2/4/8/16), orthogonal to the above.
Cross-dataset transfer and domain generalisation need a second dataset and are not defined for this task.
Testing is never used for selection; validation is carved out of Training only.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mqa import CLASS_NAMES, CLASS_TO_IDX
from mqa.data.dataset import discover_split
from mqa.data.splits import stratified_indices


def fewshot_indices(labels, shots: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[int(y)].append(i)
    picked: list[int] = []
    for y in sorted(by):
        idxs = np.array(by[y])
        rng.shuffle(idxs)
        picked.extend(idxs[: min(shots, len(idxs))].tolist())
    return picked


def class_halves(cfg: dict[str, Any]) -> tuple[list[int], list[int]]:
    names = cfg.get("base_classes")
    if names:
        base = [CLASS_TO_IDX[n] for n in names]
    else:
        base = list(range((len(CLASS_NAMES) + 1) // 2))
    new = [i for i in range(len(CLASS_NAMES)) if i not in base]
    if not new:
        raise ValueError("base-to-new needs at least one new class")
    return base, new


def _remap(records: list[dict], classes: list[int]) -> list[dict]:
    local = {g: i for i, g in enumerate(classes)}
    return [dict(r, global_label=r["label"], label=local[r["label"]]) for r in records if r["label"] in local]


def build_protocol(cfg: dict[str, Any], data_root: Path) -> dict[str, Any]:
    protocol = cfg.get("protocol", "all")
    if protocol not in ("all", "base2new"):
        raise ValueError(f"unknown protocol {protocol}")
    train_info = discover_split(Path(data_root) / "Training")
    if not train_info["exists"] or train_info["n_images"] == 0:
        raise SystemExit(
            f"No training images under {Path(data_root) / 'Training'}. Run scripts/download_dataset.py or point "
            "--data-root at the folder that contains Training/ and Testing/."
        )
    if train_info["missing_classes"]:
        raise SystemExit(f"missing class folders: {train_info['missing_classes']}")
    test_info = discover_split(Path(data_root) / "Testing")

    if protocol == "all":
        train_classes = list(range(len(CLASS_NAMES)))
        eval_sets = {"all": train_classes}
    else:
        base, new = class_halves(cfg)
        train_classes = base
        eval_sets = {"base": base, "new": new}

    seed = int(cfg.get("seed", 0))
    shots = cfg.get("shots")
    pool = [r for r in train_info["records"] if r["label"] in train_classes]
    labels = [r["label"] for r in pool]
    if shots:
        support = fewshot_indices(labels, int(shots), seed)
        pool = [pool[i] for i in support]
        labels = [r["label"] for r in pool]
        tr, va = stratified_indices(labels, float(cfg.get("val_ratio", 0.2)), seed)
    else:
        tr, va = stratified_indices(labels, float(cfg.get("val_ratio", 0.15)), seed)
    train_recs = _remap([pool[i] for i in tr], train_classes)
    val_recs = _remap([pool[i] for i in va], train_classes)

    eval_records = {n: _remap(test_info["records"], cls) for n, cls in eval_sets.items()} if test_info["exists"] else {}
    names = lambda cls: [CLASS_NAMES[i] for i in cls]  # noqa: E731
    info = {
        "protocol": protocol,
        "shots": shots,
        "seed": seed,
        "train_classes": names(train_classes),
        "eval_sets": {n: names(c) for n, c in eval_sets.items()},
        "n_train": len(train_recs),
        "n_val": len(val_recs),
        "train_per_class": {CLASS_NAMES[g]: sum(1 for r in train_recs if r["global_label"] == g) for g in train_classes},
        "val_per_class": {CLASS_NAMES[g]: sum(1 for r in val_recs if r["global_label"] == g) for g in train_classes},
        "test_untouched": True,
    }
    return {
        "train": train_recs,
        "val": val_recs,
        "train_classes": train_classes,
        "eval_sets": eval_sets,
        "eval_records": eval_records,
        "info": info,
    }


def harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a + b) else 0.0
