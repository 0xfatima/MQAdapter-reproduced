"""Task splits for the paper's four protocols (Sec. V-A1) plus a full-data protocol for MRI.

  all       train on every class (all training images, or `shots` per class), test on the test split
  fewshot   = all with shots in {1, 2, 4, 8, 16}
  base2new  classes split in halves (first ceil(C/2) = base, CoOp/CoCoOp convention); train on base (16-shot),
            evaluate base classes in "base" mode and new classes in "novel" mode; report H
  cross     train on all source classes (16-shot), evaluate on each target dataset's own classes ("novel" mode)
  dg        train on the source (16-shot), evaluate on domain-shifted copies of the source test split
            (mqa/shifts.py) and/or real target-domain datasets with the same classes

Validation is always carved out of the training split (stratified) and is only used for monitoring /
optional checkpoint selection. The test split is never used for any choice.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mqa import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD
from mqa.data.preprocess import PadToSquare, RepeatToRGB
from mqa.datasets import ClassSpace, DatasetSpec, load_spec, resolve_classes, scan_split
from mqa.shifts import SHIFTS, ShiftTransform


# ----------------------------------------------------------------------------- transforms
def build_transform(train: bool, augment: str = "mri", size: int = CLIP_IMAGE_SIZE) -> Callable:
    """augment="mri": pad-to-square, resize, small rotation (no flip; laterality can matter).
    augment="clip": CoOp/Dassl recipe for natural images (random resized crop + flip)."""
    norm = [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]
    bicubic = transforms.InterpolationMode.BICUBIC
    if augment == "clip":
        if train:
            ops = [RepeatToRGB(), transforms.RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=bicubic), transforms.RandomHorizontalFlip()]
        else:
            ops = [RepeatToRGB(), transforms.Resize(size, interpolation=bicubic), transforms.CenterCrop(size)]
        return transforms.Compose(ops + norm)
    ops = [RepeatToRGB(), PadToSquare(fill=0), transforms.Resize((size, size), interpolation=bicubic)]
    if train:
        ops.append(transforms.RandomRotation(degrees=8, interpolation=bicubic, fill=0))
    return transforms.Compose(ops + norm)


class RecordDataset(Dataset):
    def __init__(self, records: list[dict], transform: Callable, shift: ShiftTransform | None = None):
        self.records, self.transform, self.shift = list(records), transform, shift

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
        if self.shift is not None:
            im = self.shift(im, r["path"])
        return {"image": self.transform(im), "label": r["label"], "path": r["path"]}


# ----------------------------------------------------------------------------- split helpers
def stratified_split(records: list[dict], ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if ratio <= 0:
        return list(records), []
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    tr, va = [], []
    for y in sorted(by):
        items = by[y]
        idx = rng.permutation(len(items))
        n_val = max(1, int(round(len(items) * ratio))) if len(items) > 1 else 0
        va += [items[i] for i in idx[:n_val]]
        tr += [items[i] for i in idx[n_val:]]
    return tr, va


def sample_shots(records: list[dict], shots: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in records:
        by[r["label"]].append(r)
    out = []
    for y in sorted(by):
        items = by[y]
        idx = rng.permutation(len(items))[: min(shots, len(items))]
        out += [items[i] for i in idx]
    return out


def subset(records: list[dict], keep: list[int]) -> list[dict]:
    """Keep classes `keep` (labels in the full space) and relabel to 0..len(keep)-1."""
    m = {g: i for i, g in enumerate(keep)}
    return [dict(r, label=m[r["label"]], global_label=r["label"]) for r in records if r["label"] in m]


@dataclass
class EvalSet:
    name: str
    records: list[dict]
    space: ClassSpace
    mode: str = "base"          # "base" | "novel"  (MMRL decoupled inference)
    shift: str | None = None    # synthetic domain shift
    group: str = "test"         # base / new / source / target / shifted


@dataclass
class TaskSplit:
    task: str
    train: list[dict]
    val: list[dict]
    space: ClassSpace
    evals: list[EvalSet]
    info: dict[str, Any] = field(default_factory=dict)


def _limit(records: list[dict], n: int | None, seed: int = 0) -> list[dict]:
    if not n or len(records) <= n:
        return records
    rng = np.random.default_rng(seed)
    return [records[i] for i in sorted(rng.permutation(len(records))[:n])]


def build_task(task: str, source: DatasetSpec, cfg: dict, seed: int) -> TaskSplit:
    classes = resolve_classes(source)
    full = ClassSpace(source, classes)
    train_all = scan_split(source, source.train_split, classes)
    test = scan_split(source, source.test_split, classes)
    if not train_all:
        raise SystemExit(f"no training images for {source.name} under {source.root}/{source.train_split}")
    if not test:
        raise SystemExit(f"no test images for {source.name} under {source.root}/{source.test_split}")
    if source.val_split:
        train, val = train_all, scan_split(source, source.val_split, classes)
    else:
        train, val = stratified_split(train_all, float(cfg.get("val_ratio", 0.15)), seed)
    shots = cfg.get("shots")
    lim = cfg.get("limit_eval")
    if task in ("base2new", "cross", "dg") and shots is None:
        shots = 16  # paper: every protocol except few-shot uses 16 shots
    if shots:
        train = sample_shots(train, int(shots), seed)
        val = sample_shots(val, max(1, min(int(shots), 4)) if cfg.get("fewshot_val", True) else 10**9, seed)
    info = {"task": task, "dataset": source.name, "classes": classes, "shots": shots, "seed": seed}

    if task in ("all", "fewshot"):
        evals = [EvalSet("test", _limit(test, lim), full, "base", group="all")]
        sp = TaskSplit(task, train, val, full, evals, info)
    elif task == "base2new":
        m = math.ceil(len(classes) / 2)
        base, new = list(range(m)), list(range(m, len(classes)))
        if not new:
            raise SystemExit("base-to-new needs at least 2 classes")
        bs, ns = ClassSpace(source, [classes[i] for i in base]), ClassSpace(source, [classes[i] for i in new])
        evals = [
            EvalSet("base", _limit(subset(test, base), lim), bs, "base", group="base"),
            EvalSet("new", _limit(subset(test, new), lim), ns, "novel", group="new"),
        ]
        info.update(base=bs.classes, new=ns.classes)
        sp = TaskSplit(task, subset(train, base), subset(val, base), bs, evals, info)
    elif task == "cross":
        evals = [EvalSet(f"{source.name}", _limit(test, lim), full, "base", group="source")]
        for t in cfg.get("cross_targets", []):
            ts = load_spec(t, (cfg.get("target_roots") or {}).get(t if isinstance(t, str) else "", None))
            tcls = resolve_classes(ts)
            recs = scan_split(ts, ts.test_split, tcls) or scan_split(ts, ts.train_split, tcls)
            if not recs:
                print(f"[cross] skipping target {ts.name}: no images under {ts.root}", flush=True)
                continue
            evals.append(EvalSet(ts.name, _limit(recs, lim), ClassSpace(ts, tcls), cfg.get("cross_mode", "novel"), group="target"))
        sp = TaskSplit(task, train, val, full, evals, info)
    elif task == "dg":
        mode = cfg.get("dg_mode", "base")
        evals = [EvalSet(f"{source.name}", _limit(test, lim), full, "base", group="source")]
        for s in cfg.get("dg_shifts", list(SHIFTS)):
            evals.append(EvalSet(f"-{s}", _limit(test, lim), full, mode, shift=s, group="shifted"))
        for t in cfg.get("dg_targets", []):
            ts = load_spec(t)
            tcls = resolve_classes(ts)
            recs = scan_split(ts, ts.test_split, tcls)
            if recs:
                evals.append(EvalSet(ts.name, _limit(recs, lim), ClassSpace(ts, tcls), mode, group="target"))
        sp = TaskSplit(task, train, val, full, evals, info)
    else:
        raise ValueError(f"unknown task {task}")
    sp.info.update(n_train=len(sp.train), n_val=len(sp.val), evals={e.name: len(e.records) for e in sp.evals})
    return sp


def eval_dataset(e: EvalSet, augment: str) -> RecordDataset:
    return RecordDataset(e.records, build_transform(False, augment), ShiftTransform(e.shift, 3) if e.shift else None)
