from __future__ import annotations

import torch

from mqa.models.clip_backbone import FakeCLIPWithText, FakeTokenizer, encode_class_texts
from mqa.models.mqadapter_clip import MQAdapterCLIP


def _model(**kw):
    clip = FakeCLIPWithText()
    text = encode_class_texts(clip, FakeTokenizer(), device="cpu")
    return MQAdapterCLIP(clip, text, k=2, **kw)


def test_cross_attention_variant_runs_without_vqc_on_anchors():
    m = _model(interaction="cross_attention")
    out = m(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert m.plugin.layers["6"].q_proj.weight.grad is not None


def test_separate_vqc_has_three_circuits():
    shared = _model()
    sep = _model(vqc_sharing="separate")
    n_vqc = lambda mod: sum(p.numel() for n, p in mod.plugin.named_parameters() if n.startswith("vqc"))  # noqa: E731
    assert n_vqc(shared) == 24
    assert n_vqc(sep) == 72
    out = sep(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
    assert torch.isfinite(out["loss"])


def test_active_class_subset_for_standalone():
    m = _model()
    m.set_active_classes([2, 3])
    assert m(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))["logits"].shape == (2, 2)
