from __future__ import annotations

from typing import Sequence

import numpy as np

from mqa import CLASS_NAMES


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(class_names)
    cm = confusion_matrix_np(y_true, y_pred, n)
    support = cm.sum(axis=1)
    per = {}
    precs, recs, f1s = [], [], []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p, r, f = _prf(tp, fp, fn)
        per[name] = {"precision": p, "recall": r, "f1": f, "support": int(support[i])}
        precs.append(p)
        recs.append(r)
        f1s.append(f)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(precs)) if precs else 0.0,
        "macro_recall": float(np.mean(recs)) if recs else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "support_total": int(support.sum()),
        "n_samples": int(len(y_true)),
    }


def save_confusion_matrix_png(cm: np.ndarray, class_names: Sequence[str], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to save a confusion-matrix plot") from exc
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
