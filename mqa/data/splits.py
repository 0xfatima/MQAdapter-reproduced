from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def stratified_indices(labels: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Reproducible per-class shuffle. Does not touch an official test split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[int(y)].append(i)
    train, val = [], []
    for y in sorted(by):
        idxs = np.array(by[y], dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = int(round(n * val_ratio))
        n_val = min(max(n_val, 1 if n >= 2 else 0), n - 1 if n >= 2 else 0)
        val.extend(idxs[:n_val].tolist())
        train.extend(idxs[n_val:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
