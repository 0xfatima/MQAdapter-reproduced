from __future__ import annotations

import torch

from mqa import CLASS_NAMES
from mqa.models.clip_backbone import FakeCLIPWithText, FakeTokenizer, encode_class_texts
from mqa.models.maple import MaPLeCLIP


def _setup(plugin_cfg=None, k=2, depth=3):
    clip = FakeCLIPWithText()
    tok = FakeTokenizer()
    text = encode_class_texts(clip, tok, device="cpu")
    return MaPLeCLIP(clip, tok, CLASS_NAMES, text, n_ctx=2, depth=depth, k=k, plugin_cfg=plugin_cfg), clip


def _plugin():
    return dict(n_qubits=8, gamma=0.001, adapt_layers=(6, 7, 8, 9, 10, 11), quantum_map="vqc")


def test_plain_maple_forward_and_grads():
    model, clip = _setup()
    assert all(not p.requires_grad for p in clip.parameters())
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([0, 3])
    out = model(x, y)
    assert out["logits"].shape == (2, 4) and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.ctx.grad is not None and model.ctx.grad.abs().sum() > 0
    assert model.proj.weight.grad is not None
    assert all(p.grad is not None for p in model.compound_prompts_text)
    assert all(p.grad is None for p in clip.parameters())


def test_maple_mqadapter_forward_and_grads():
    model, _ = _setup(_plugin(), k=2)
    base = _setup()[0].trainable_parameter_count()
    assert model.trainable_parameter_count() - base == 77848  # same plug-in budget as the stand-alone adapter
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([1, 2])
    out = model(x, y)
    assert out["topk_idx"].shape == (2, 2)
    for a in out["alpha"].values():
        assert torch.allclose(a.sum(-1), torch.ones(2), atol=1e-5)
    out["loss"].backward()
    assert model.plugin.text_proj.weight.grad is not None
    assert model.plugin.vqc.phi2.grad is not None and model.plugin.vqc.phi2.grad.abs().sum() > 0


def test_active_classes_for_base_to_new():
    model, _ = _setup(_plugin(), k=3)
    model.set_active_classes([0, 1])
    assert model.k == 2  # clamped to the number of active classes
    out = model(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
    assert out["logits"].shape == (2, 2)
    model.set_active_classes([2, 3])
    out = model(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
    assert out["logits"].shape == (2, 2)
    model.set_active_classes([0, 1, 2, 3])
    assert model(torch.randn(1, 3, 224, 224))["logits"].shape == (1, 4)


def test_clip_stays_in_eval_mode():
    model, clip = _setup()
    model.train()
    assert not clip.training
