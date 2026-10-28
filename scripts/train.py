#!/usr/bin/env python
"""Train one configured experiment.

  python scripts/train.py --config configs/mqadapter_k2.json --data-root data/brain_tumor_mri
  python scripts/train.py --config configs/maple_mqadapter.json --set k=3 gamma=0.002 seed=1
  python scripts/train.py --config configs/maple_mqadapter.json --protocol base2new --eval-after

The official Testing split is only read when --eval-after is given (or via scripts/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from mqa import CLASS_NAMES
from mqa.data.dataset import BrainTumorMRI
from mqa.data.preprocess import build_clip_transform
from mqa.evaluation.run_eval import evaluate_sets
from mqa.models.factory import build_backbone, build_model
from mqa.protocols import build_protocol, fewshot_indices  # noqa: F401  (fewshot_indices re-exported for old imports)
from mqa.training.loop import train_model


def parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_cfg(path: str | None, overrides: dict) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cpu_smoke.json"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--protocol", default=None, choices=[None, "all", "base2new"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--download-clip", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-after", action="store_true", help="evaluate best.pt on Testing when training ends")
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE", help="override any config key")
    args = ap.parse_args()

    overrides = {
        "data_root": args.data_root, "out_dir": args.out_dir, "device": args.device, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "max_steps": args.max_steps, "protocol": args.protocol,
        "seed": args.seed, "shots": args.shots, "download_clip": True if args.download_clip else None,
    }
    overrides.update(parse_overrides(args.set))
    cfg = load_cfg(args.config, overrides)
    data_root = Path(cfg.get("data_root") or (ROOT / "data" / "brain_tumor_mri"))
    out_dir = Path(cfg.get("out_dir") or (ROOT / "runs" / Path(args.config).stem))
    device = resolve_device(cfg.get("device", "auto"))

    proto = build_protocol(cfg, data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(proto["info"], indent=2), encoding="utf-8")
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 0))
    pin = device.type == "cuda"
    train_loader = DataLoader(BrainTumorMRI(proto["train"], transform=build_clip_transform(train=True)), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=len(proto["train"]) > bs)
    val_loader = DataLoader(BrainTumorMRI(proto["val"], transform=build_clip_transform(train=False)), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    clip, text, tok = build_backbone(cfg, device)
    model = build_model(cfg, clip, text, tok)
    if hasattr(model, "set_active_classes"):
        model.set_active_classes(proto["train_classes"])
    train_names = [CLASS_NAMES[i] for i in proto["train_classes"]]
    summary = train_model(
        model, train_loader, val_loader, train_names, out_dir=out_dir,
        epochs=int(cfg.get("epochs", 5)), lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.01)), device=str(device),
        seed=int(cfg.get("seed", 0)), patience=int(cfg.get("patience", 5)), max_steps=cfg.get("max_steps"),
        scheduler=cfg.get("scheduler", "cosine"), mixed_precision=bool(cfg.get("mixed_precision", False)),
        resume=args.resume, optimizer=cfg.get("optimizer", "adamw"), momentum=float(cfg.get("momentum", 0.9)),
        warmup_epochs=int(cfg.get("warmup_epochs", 0)), warmup_lr=float(cfg.get("warmup_lr", 1e-5)),
        grad_clip=cfg.get("grad_clip"),
    )
    print(f"trained {summary['completed_epochs']} epochs, trainable params {summary['env']['trainable_params']}")
    if args.eval_after:
        best = out_dir / "best.pt"
        if best.exists():
            from mqa.training.checkpoint import load_checkpoint

            model.load_state_dict(load_checkpoint(best, map_location=device)["model"], strict=True)
        res = evaluate_sets(model, proto["eval_records"], proto["eval_sets"], device, out_dir / "eval", batch_size=int(cfg.get("eval_batch_size", 16)), num_workers=nw)
        print(json.dumps({k: (v["accuracy"] if "accuracy" in v else v) for k, v in res.items()}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
