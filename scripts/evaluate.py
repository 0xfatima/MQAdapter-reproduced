#!/usr/bin/env python
"""Evaluate a checkpoint on the official Testing split. Do not tune hyper-parameters here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mqa import CLASS_NAMES
from mqa.data.dataset import discover_split
from mqa.evaluation.run_eval import evaluate_sets
from mqa.models.factory import build_backbone, build_model
from mqa.protocols import build_protocol
from mqa.training.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="json used at train time (default: sibling config_used.json)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="Testing", choices=["Testing", "Training"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config_used.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    data_root = Path(args.data_root or cfg.get("data_root") or (ROOT / "data" / "brain_tumor_mri"))
    out_dir = Path(args.out_dir or ckpt_path.parent / "eval")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    proto = build_protocol(cfg, data_root)
    eval_records, eval_sets = proto["eval_records"], proto["eval_sets"]
    if args.split == "Training":  # sanity check only: evaluates all Training images of each evaluation set
        from mqa.protocols import _remap

        info = discover_split(data_root / "Training")
        eval_records = {n: _remap(info["records"], c) for n, c in eval_sets.items()}

    clip, text, tok = build_backbone(cfg, device)
    model = build_model(cfg, clip, text, tok)
    model.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"], strict=True)
    model.to(device).eval()
    res = evaluate_sets(model, eval_records, eval_sets, device, out_dir, batch_size=args.batch_size, num_workers=int(cfg.get("num_workers", 0)))
    print(json.dumps({k: (v["accuracy"] if "accuracy" in v else v) for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
