from __future__ import annotations

import torch

from mqa.models.adapter import MQAdapterLayer, expected_adapter_param_count
from mqa.models.baselines import CLIPAdapterHead, FrozenCLIPZeroShot
from mqa.models.clip_backbone import FakeCLIP, FakeTokenizer, encode_class_texts
from mqa.models.mqadapter_clip import MQAdapterCLIP
from mqa.quantum.torch_vqc import PaperVQC


def _fake_setup(k=2):
    clip = FakeCLIP()
    tok = FakeTokenizer()
    text = encode_class_texts(clip, tok, device="cpu")
    model = MQAdapterCLIP(clip, text, k=k, quantum_map="vqc", freeze_clip=True)
    return model, clip, text


def test_expected_param_count_matches_078m_assumptions():
    assert expected_adapter_param_count(6) == 6 * (768 * 8 + 8 * 768) + 512 * 8 + 24 == 77848


def test_trainable_count_excludes_frozen_clip():
    model, _, _ = _fake_setup()
    n_train = model.trainable_parameter_count()
    n_clip = sum(p.numel() for p in model.clip.parameters())
    assert n_clip > 0
    assert all(not p.requires_grad for p in model.clip.parameters())
    assert n_train == expected_adapter_param_count(
        n_layers=6,
        visual_dim=768,
        text_dim=512,
        n_qubits=8,
        vqc_params=24,
    )
    assert n_train == 77848


def test_forward_shapes_and_finite_loss():
    model, _, _ = _fake_setup(k=3)
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 2])
    out = model(x, y)
    assert out["logits"].shape == (2, 4)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"])
    assert out["topk_idx"].shape == (2, 3)
    assert 0 <= out["topk_idx"].min() and out["topk_idx"].max() < 4
    for a in out["alpha"].values():
        assert torch.allclose(a.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_k4_retrieves_all_classes():
    model, _, _ = _fake_setup(k=4)
    x = torch.randn(3, 3, 224, 224)
    idx, _, _ = model.retrieve_topk(x)
    for row in idx:
        assert set(row.tolist()) == {0, 1, 2, 3}


def test_k_invalid():
    clip = FakeCLIP()
    tok = FakeTokenizer()
    text = encode_class_texts(clip, tok)
    try:
        MQAdapterCLIP(clip, text, k=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_optimizer_step_changes_adapter_not_clip():
    model, _, _ = _fake_setup(k=2)
    clip_before = {n: p.detach().clone() for n, p in model.clip.named_parameters()}
    ad_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([1, 3])
    opt.zero_grad()
    loss = model(x, y)["loss"]
    loss.backward()
    opt.step()
    for n, p in model.clip.named_parameters():
        assert torch.equal(p.detach(), clip_before[n])
    changed = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if not torch.equal(p.detach(), ad_before[n]):
                changed += 1
    assert changed > 0
    assert torch.isfinite(loss)


def test_layer_residual_shape():
    vqc = PaperVQC(8)
    wt = torch.nn.Linear(512, 8, bias=False)
    layer = MQAdapterLayer(768, 8, 0.001, 6)
    c = torch.randn(4, 768)
    p = torch.randn(4, 2, 512)
    out, alpha = layer(c, p, vqc, wt)
    assert out.shape == c.shape
    assert alpha.shape == (4, 2)


def test_baselines_forward():
    clip = FakeCLIP()
    text = encode_class_texts(clip, FakeTokenizer())
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    z = FrozenCLIPZeroShot(clip, text)(x, y)
    assert z["logits"].shape == (2, 4)
    a = CLIPAdapterHead(clip, text)(x, y)
    assert a["logits"].shape == (2, 4)
    assert CLIPAdapterHead(clip, text).trainable_parameter_count() > 0
