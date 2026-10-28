from __future__ import annotations

from pathlib import Path

from PIL import Image

from mqa import CLASS_NAMES
from mqa.protocols import build_protocol, class_halves


def _tree(root: Path, n_train=8, n_test=3):
    for split, n in (("Training", n_train), ("Testing", n_test)):
        for cls in CLASS_NAMES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.new("L", (20, 20), color=i).save(d / f"{cls}_{i}.png")


def test_all_protocol(tmp_path):
    _tree(tmp_path)
    p = build_protocol({"protocol": "all", "val_ratio": 0.25, "seed": 0}, tmp_path)
    assert p["train_classes"] == [0, 1, 2, 3]
    assert len(p["train"]) + len(p["val"]) == 32
    assert len(p["eval_records"]["all"]) == 12
    assert not ({r["path"] for r in p["train"]} & {r["path"] for r in p["eval_records"]["all"]})


def test_base_to_new_split(tmp_path):
    _tree(tmp_path)
    p = build_protocol({"protocol": "base2new", "val_ratio": 0.25, "seed": 0}, tmp_path)
    base, new = class_halves({})
    assert p["train_classes"] == base == [0, 1] and new == [2, 3]
    assert {r["global_label"] for r in p["train"] + p["val"]} <= set(base)  # no new-class image in training
    assert {r["label"] for r in p["train"]} == {0, 1}
    assert {r["label"] for r in p["eval_records"]["new"]} == {0, 1}  # remapped to local indices
    assert {r["global_label"] for r in p["eval_records"]["new"]} == set(new)
    assert len(p["eval_records"]["base"]) == 6 and len(p["eval_records"]["new"]) == 6


def test_fewshot(tmp_path):
    _tree(tmp_path, n_train=20)
    p = build_protocol({"protocol": "all", "shots": 4, "val_ratio": 0.25, "seed": 1}, tmp_path)
    assert len(p["train"]) + len(p["val"]) == 16
