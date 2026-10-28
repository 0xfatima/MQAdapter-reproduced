"""Every host (+/- MQAdapter) on FakeCLIP: forward, backward, label-space switching, decoupled inference."""
import pytest

torch = pytest.importorskip("torch")

from mqa.datasets import ClassSpace, load_spec  # noqa: E402
from mqa.hosts.factory import METHODS, build_host, is_trainable, method_config  # noqa: E402
from mqa.models.clip_backbone import FakeCLIPWithText, FakeTokenizer  # noqa: E402

HOSTS = [m for m in METHODS if m != "linear_probe"]


@pytest.fixture(scope="module")
def clip():
    torch.manual_seed(0)
    m = FakeCLIPWithText().eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def _spaces():
    spec = load_spec("brain_tumor_mri")
    return ClassSpace(spec, ["glioma", "meningioma", "notumor", "pituitary"]), ClassSpace(spec, ["notumor", "pituitary"])


@pytest.mark.parametrize("name", HOSTS + [h + "+mqa" for h in HOSTS])
def test_host_forward_backward(clip, name):
    cfg = method_config(name, {"k": 2})
    host = build_host(name, clip, FakeTokenizer(), cfg)
    full, half = _spaces()
    host.set_class_space(full, "base")
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 3])
    if is_trainable(name):
        host.train()
        out = host(x, y)
        assert out["logits"].shape == (2, 4)
        out["loss"].backward()
        grads = [p.grad for p in host.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient reached the trainable weights"
        if name.endswith("+mqa"):
            assert host.plugin is not None and "topk_idx" in out
        host.on_epoch_end(0, 1)
        host.on_train_end()
    host.eval()
    for mode in ("base", "novel"):
        host.set_class_space(half, mode)
        with torch.no_grad():
            out = host(x, torch.tensor([0, 1]))
        assert out["logits"].shape == (2, 2)
        assert torch.isfinite(out["loss"])


def test_mqadapter_param_count_on_hosts(clip):
    for host_name in ("maple", "promptsrc", "mmrlpp"):
        h = build_host(host_name + "+mqa", clip, FakeTokenizer(), method_config(host_name + "+mqa"))
        assert h.plugin_parameter_count() == 77848  # 6 x (768*8 + 8*768) + 512*8 + 24


def test_mmrl_text_insertion_shifts_eot(clip):
    h = build_host("mmrlpp", clip, FakeTokenizer(), method_config("mmrlpp"))
    full, _ = _spaces()
    h.set_class_space(full)
    w = h.text_features()
    assert w.shape == (4, h.embed_dim)


def test_mmrl_starts_from_clip_projection(clip):
    """SRRA: B factors are zero at init, so f_r uses exactly CLIP's projection on the representation tokens."""
    h = build_host("mmrlpp", clip, FakeTokenizer(), method_config("mmrlpp"))
    assert float(h.proj_r_B.abs().sum()) == 0.0
    assert all(float(b.abs().sum()) == 0.0 for b in h.align_v.B)
