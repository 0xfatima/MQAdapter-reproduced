from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mqa.evaluation.metrics import classification_metrics, confusion_matrix_np
from mqa.models.clip_backbone import FakeCLIP, FakeTokenizer, encode_class_texts
from mqa.models.mqadapter_clip import MQAdapterCLIP
from mqa.training.checkpoint import load_checkpoint, save_checkpoint
from mqa.training.loop import train_model
from mqa.data.dataset import BrainTumorMRI
from mqa.data.preprocess import build_clip_transform
from PIL import Image
from torch.utils.data import DataLoader


def test_metrics_known_values():
    y_true = [0, 0, 1, 1, 2, 3]
    y_pred = [0, 1, 1, 1, 2, 0]
    m = classification_metrics(y_true, y_pred)
    assert m["n_samples"] == 6
    assert m["confusion_matrix"][0][0] == 1
    assert abs(m["accuracy"] - 4 / 6) < 1e-12
    cm = confusion_matrix_np(np.array(y_true), np.array(y_pred), 4)
    assert cm.shape == (4, 4)


def test_checkpoint_roundtrip(tmp_path):
    clip = FakeCLIP()
    text = encode_class_texts(clip, FakeTokenizer())
    model = MQAdapterCLIP(clip, text, k=2)
    x = torch.randn(2, 3, 224, 224)
    before = model(x)["logits"].detach()
    path = tmp_path / "m.pt"
    save_checkpoint(path, {"model": model.state_dict(), "epoch": 3})
    clip2 = FakeCLIP()
    # different random init
    m2 = MQAdapterCLIP(clip2, text, k=2)
    m2.load_state_dict(load_checkpoint(path)["model"], strict=False)
    # clip weights also in state_dict; FakeCLIP was re-inited so strict load on same architecture:
    model2 = MQAdapterCLIP(FakeCLIP(), text, k=2)
    payload = load_checkpoint(path)
    model2.load_state_dict(payload["model"])
    after = model2(x)["logits"].detach()
    # images go through retrieved path; CLIP weights restored from ckpt so logits should match
    assert torch.allclose(before, after, atol=1e-5)
    assert payload["epoch"] == 3


def test_one_optimizer_epoch_smoke(tmp_path):
    root = tmp_path / "data"
    recs = []
    for i, cls in enumerate(["glioma", "meningioma", "notumor", "pituitary"]):
        d = root / cls
        d.mkdir(parents=True)
        for j in range(2):
            p = d / f"{j}.png"
            Image.new("RGB", (64, 64), color=(i * 40, 10, 10)).save(p)
            recs.append({"path": str(p), "class": cls, "label": i})
    ds = BrainTumorMRI(recs, transform=build_clip_transform(False))
    loader = DataLoader(ds, batch_size=4)
    clip = FakeCLIP()
    text = encode_class_texts(clip, FakeTokenizer())
    model = MQAdapterCLIP(clip, text, k=2)
    summary = train_model(
        model,
        loader,
        loader,
        ["glioma", "meningioma", "notumor", "pituitary"],
        out_dir=tmp_path / "run",
        epochs=1,
        lr=1e-3,
        device="cpu",
        seed=0,
        patience=2,
        max_steps=1,
    )
    assert summary["trained"] is True
    assert (tmp_path / "run" / "last.pt").exists()
