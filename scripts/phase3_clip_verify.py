"""Phase 3b: verify the CLIP ViT-B/16 checkpoint, preprocessing, tensor layout, and the injection mechanism.

Verifies (and records, never assumes): resblock count/width, token layout (seq-first vs batch-first), CLS index,
that final embedding == ln_post(CLS of last block) @ proj, text embedding dim, logit scale, preprocessing constants,
weights fingerprint, that all parameters can be frozen, and that a forward hook can inject a CLS-only residual after block k
changing only blocks >= k.
Run:  python scripts/phase3_clip_verify.py --backend open_clip --out phase3_clip_report.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPTS = ["An MRI scan showing a glioma brain tumor.", "An MRI scan showing a meningioma brain tumor.",
           "A brain MRI showing no tumor.", "An MRI scan showing a pituitary brain tumor."]


def load(backend):
    if backend == "open_clip":
        import open_clip

        model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", precision="fp32")
        return model.eval(), pre, open_clip.get_tokenizer("ViT-B-16")
    import clip

    model, pre = clip.load("ViT-B/16", device="cpu", jit=False)
    return model.float().eval(), pre, clip.tokenize


def fingerprint(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        t = p.detach().double()
        h.update(f"{n}|{tuple(p.shape)}|{t.sum().item():.9e}|{t.abs().sum().item():.9e}".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["open_clip", "clip"], default="open_clip")
    ap.add_argument("--out", default="phase3_clip_report.json")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda (fp32 either way)")
    a = ap.parse_args()
    model, pre, tok = load(a.backend)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device))
    model = model.to(dev)
    rep = {"backend": a.backend, "device": str(dev), "weights_fingerprint": fingerprint(model),
           "n_params_total": sum(p.numel() for p in model.parameters())}

    # preprocessing
    ts = pre.transforms if hasattr(pre, "transforms") else None
    rep["preprocess_repr"] = repr(pre)
    norm = next(t for t in ts if t.__class__.__name__ == "Normalize")
    assert np.allclose(norm.mean, CLIP_MEAN, atol=1e-4) and np.allclose(norm.std, CLIP_STD, atol=1e-4), "unexpected CLIP normalisation"
    img = Image.fromarray((np.random.default_rng(0).random((300, 260, 3)) * 255).astype(np.uint8))
    x = pre(img).unsqueeze(0)
    rep["preprocessed_shape"] = list(x.shape)
    assert tuple(x.shape) == (1, 3, 224, 224)
    x = x.to(dev)

    # frozen
    for p in model.parameters():
        p.requires_grad = False
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # visual structure + hooks
    vis = model.visual
    blocks = vis.transformer.resblocks
    rep["n_blocks"] = len(blocks)
    rep["width"] = vis.transformer.width
    outs = []
    hs = [b.register_forward_hook(lambda m, i, o: outs.append(o.detach().clone())) for b in blocks]
    with torch.no_grad():
        emb = model.encode_image(x)
    for h in hs:
        h.remove()
    shp = list(outs[0].shape)
    layout = "seq_first" if shp[0] == 197 else ("batch_first" if shp[1] == 197 else "unknown")
    rep.update({"block_output_shape": shp, "layout": layout, "image_embedding_shape": list(emb.shape)})
    assert layout != "unknown", f"cannot find 197 tokens in {shp}"
    cls_of = (lambda t: t[0]) if layout == "seq_first" else (lambda t: t[:, 0])
    recon = vis.ln_post(cls_of(outs[-1])) @ vis.proj
    rep["recon_max_abs_diff"] = (recon - emb).abs().max().item()
    assert rep["recon_max_abs_diff"] < 1e-4, "final embedding != ln_post(CLS of last block) @ proj"
    rep["cls_dim"] = int(cls_of(outs[0]).shape[-1])

    # hook injection test: add delta to CLS after block k only
    k, delta = 6, 1e-2
    outs2 = []

    def inject(m, i, o):
        o = o.clone()
        (o[0] if layout == "seq_first" else o[:, 0]).add_(delta)
        return o

    hk = blocks[k].register_forward_hook(inject)
    hs = [b.register_forward_hook(lambda m, i, o: outs2.append(o.detach().clone())) for b in blocks]
    with torch.no_grad():
        emb2 = model.encode_image(x)
    hk.remove()
    [h.remove() for h in hs]
    changed = [bool((a_ - b_).abs().max() > 1e-8) for a_, b_ in zip(outs, outs2)]
    rep["injection_changed_blocks"] = changed
    assert not any(changed[:k]), "blocks before k changed"
    assert all(changed[k + 1:]), "injection did not propagate to later blocks"
    assert (emb - emb2).abs().max() > 1e-8

    # text
    t = tok(PROMPTS).to(dev)
    with torch.no_grad():
        te = model.encode_text(t)
    rep["text_embedding_shape"] = list(te.shape)
    rep["logit_scale_exp"] = float(model.logit_scale.exp())
    rep["tokens_per_prompt_nonpad"] = [int((r != 0).sum()) for r in t]
    assert te.shape == (4, 512)
    rep["zero_shot_logits_on_random_image"] = (model.logit_scale.exp() * torch.nn.functional.normalize(emb, dim=-1)
                                              @ torch.nn.functional.normalize(te, dim=-1).T).tolist()
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k_: v for k_, v in rep.items() if k_ != "preprocess_repr"}, indent=2))
    print("PHASE3b PASSED")


if __name__ == "__main__":
    main()
