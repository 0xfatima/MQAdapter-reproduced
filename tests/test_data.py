from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from mqa import CLASS_NAMES, CLIP_IMAGE_SIZE, CLIP_MEAN, CLASS_TO_IDX
from mqa.data.dataset import BrainTumorMRI, CorruptImageError, discover_split
from mqa.data.inspect_dataset import inspect_dataset
from mqa.data.preprocess import build_clip_transform
from mqa.data.splits import stratified_indices


def _write_tree(root: Path, extra_bad=False):
    for split in ("Training", "Testing"):
        for cls in CLASS_NAMES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(4 if split == "Training" else 2):
                Image.new("L", (30 + i, 40), color=10 * i).save(d / f"{cls}_{i}.png")
    if extra_bad:
        (root / "Training" / "glioma" / "bad.png").write_bytes(b"not-an-image")
        (root / "Training" / "unknown_class").mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(root / "Training" / "unknown_class" / "x.png")


def test_discover_and_label_mapping(tmp_path):
    _write_tree(tmp_path)
    info = discover_split(tmp_path / "Training")
    assert info["missing_classes"] == []
    assert info["n_images"] == 16
    assert info["per_class"]["glioma"] == 4
    labels = {r["class"]: r["label"] for r in info["records"]}
    assert labels["glioma"] == CLASS_TO_IDX["glioma"] == 0
    test = discover_split(tmp_path / "Testing")
    assert test["n_images"] == 8


def test_missing_and_unexpected_classes(tmp_path):
    (tmp_path / "Training" / "glioma").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "Training" / "glioma" / "a.jpg")
    (tmp_path / "Training" / "extra").mkdir()
    info = discover_split(tmp_path / "Training")
    assert "meningioma" in info["missing_classes"]
    assert "extra" in info["unexpected_classes"]


def test_corrupt_image_raises(tmp_path):
    _write_tree(tmp_path, extra_bad=True)
    recs = discover_split(tmp_path / "Training")["records"]
    bad = [r for r in recs if r["path"].endswith("bad.png")]
    ds = BrainTumorMRI(bad, transform=build_clip_transform(False))
    try:
        _ = ds[0]
        assert False, "expected CorruptImageError"
    except CorruptImageError:
        pass


def test_transforms_clip_resolution_and_channels(tmp_path):
    img = Image.new("L", (100, 60), color=128)
    t = build_clip_transform(train=False)(img)
    assert tuple(t.shape) == (3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)
    # normalized: not in [0,1]
    assert t.mean().abs() > 0


def test_stratified_split_reproducible_and_disjoint():
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    a1, b1 = stratified_indices(labels, 0.25, seed=7)
    a2, b2 = stratified_indices(labels, 0.25, seed=7)
    assert a1 == a2 and b1 == b2
    assert set(a1).isdisjoint(b1)
    assert sorted(a1 + b1) == list(range(16))


def test_inspect_reports_missing_root(tmp_path):
    r = inspect_dataset(tmp_path / "nope")
    assert r["exists"] is False


def test_inspect_counts(tmp_path):
    _write_tree(tmp_path)
    r = inspect_dataset(tmp_path, compute_near_dup=True)
    assert r["splits"]["Training"]["n_images"] == 16
    assert r["splits"]["Testing"]["n_images"] == 8
    assert "patient-independent" in r["patient_split"]
