#!/usr/bin/env python
"""Paper Table VII style efficiency: trainable params and FPS (batch of random images, fp32).

  python scripts/efficiency.py --configs configs/maple.json configs/maple_mqadapter.json configs/mqadapter_k2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mqa.models.factory import build_backbone, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(ROOT / "runs" / "efficiency.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))
    rows = []
    for path in args.configs:
        cfg = json.loads(Path(path).read_text())
        clip, text, tok = build_backbone(cfg, device)
        model = build_model(cfg, clip, text, tok).to(device).eval()
        x = torch.randn(args.batch_size, 3, 224, 224, device=device)
        bs = args.batch_size
        try:
            with torch.no_grad():
                model(x[:2])  # warm-up
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            fps = args.iters * bs / (time.perf_counter() - t0)
        except torch.cuda.OutOfMemoryError:
            fps = None
            torch.cuda.empty_cache()
        rows.append({"config": Path(path).stem, "model": cfg.get("model"), "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "fps": fps, "batch_size": bs, "device": str(device)})
        print(rows[-1], flush=True)
        del model, clip
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
