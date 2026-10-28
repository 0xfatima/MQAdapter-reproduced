"""Phase 3c: measure GPU memory / throughput of frozen CLIP ViT-B/16 with gradients flowing through blocks first..last.

Stand-in for the real adapter (the real one is not written yet): a trainable [n_blocks, 768] residual is added to the class
token after each adapted block, so autograd traverses the same blocks 6-11 and later as MQAdapter will. The VQC/projections add
little memory; this is an ESTIMATE of headroom, not the final number. Output: phase3_budget_report.json with the largest batch
that stays under --vram_budget_gb for each precision. Later phases read the measured file.
Run:  python scripts/phase3_budget.py --out phase3_budget_report.json
"""
import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from phase3_clip_verify import load

DT = {"bf16": torch.bfloat16, "fp16": torch.float16}


def detect_layout(model):
    shapes = []
    h = model.visual.transformer.resblocks[0].register_forward_hook(lambda m, i, o: shapes.append(list(o.shape)))
    with torch.no_grad():
        model.encode_image(torch.randn(1, 3, 224, 224, device="cuda"))
    h.remove()
    return "seq_first" if shapes[0][0] == 197 else "batch_first"


def run(model, layout, first, last, bs, prec, iters=3):
    delta = torch.nn.Parameter(torch.zeros(last - first + 1, model.visual.transformer.width, device="cuda"))
    hooks = []
    for i, blk in enumerate(model.visual.transformer.resblocks):
        if first <= i <= last:
            def hook(m, inp, o, i=i):
                o2 = o.clone()
                if layout == "seq_first":
                    o2[0] = o2[0] + delta[i - first]
                else:
                    o2[:, 0] = o2[:, 0] + delta[i - first]
                return o2
            hooks.append(blk.register_forward_hook(hook))
    x = torch.randn(bs, 3, 224, 224, device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    try:
        for _ in range(iters + 1):  # first iteration = warm-up
            delta.grad = None
            torch.cuda.synchronize()
            t = time.perf_counter()
            ctx = torch.autocast("cuda", dtype=DT[prec]) if prec in DT else contextlib.nullcontext()
            with ctx:
                emb = model.encode_image(x)
            emb.float().pow(2).mean().backward()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    except torch.cuda.OutOfMemoryError:
        return {"oom": True}
    finally:
        for h in hooks:
            h.remove()
    med = statistics.median(ts[1:])
    return {"oom": False, "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "sec_per_iter": med, "img_per_s": bs / med}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["open_clip", "clip"], default="open_clip")
    ap.add_argument("--first", type=int, default=6)
    ap.add_argument("--last", type=int, default=11)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--vram_budget_gb", type=float, default=10.5)
    ap.add_argument("--out", default="phase3_budget_report.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: this profile needs the GPU (check the PyTorch CUDA wheel vs driver)")
    model, _, _ = load(a.backend)
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad = False
    layout = detect_layout(model)
    precs = ["fp32", "fp16"] + (["bf16"] if torch.cuda.is_bf16_supported() else [])
    rep = {"gpu": torch.cuda.get_device_name(0), "layout": layout, "blocks": [a.first, a.last],
           "budget_gb": a.vram_budget_gb, "precisions_run": precs,
           "bf16_skipped_reason": None if "bf16" in precs else "GPU does not support bf16", "results": {}, "max_batch_within_budget": {}}
    for prec in precs:
        rep["results"][prec], best = {}, 0
        for bs in a.batch_sizes:
            r = run(model, layout, a.first, a.last, bs, prec)
            rep["results"][prec][bs] = r
            print(prec, bs, r, flush=True)
            if r["oom"]:
                break
            if r["peak_gb"] <= a.vram_budget_gb:
                best = bs
        rep["max_batch_within_budget"][prec] = best
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["max_batch_within_budget"], indent=2))
    print("PHASE3c PASSED")


if __name__ == "__main__":
    main()
