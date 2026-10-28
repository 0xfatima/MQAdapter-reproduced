"""Generic trainer for every host (Eq. 16 or the host's own objective).

Schedule: constant warm-up lr for `warmup_epochs`, then cosine over the remaining epochs (Dassl/CoOp style,
stepped per epoch). Model selection: "last" (what all the papers report) or "best_val" (validation loss).
Only the non-CLIP weights are saved.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def lr_at(epoch: int, epochs: int, base: float, warmup: int, warmup_lr: float, schedule: str) -> float:
    if epoch < warmup:
        return warmup_lr
    if schedule == "none":
        return base
    span = max(epochs - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * (epoch - warmup) / span))


def trainable_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("clip.")}


@torch.no_grad()
def val_loss(model, loader, device, max_batches=None) -> tuple[float | None, float | None]:
    if loader is None or len(loader.dataset) == 0:
        return None, None
    model.eval()
    losses, correct, n = [], 0, 0
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = batch["image"].to(device), batch["label"].to(device)
        out = model(x, y)
        losses.append(float(out["loss"]))
        correct += int((out["logits"].argmax(-1) == y).sum())
        n += y.numel()
    return (float(np.mean(losses)) if losses else None), (correct / n if n else None)


def train_host(model, train_ds, val_ds, cfg: dict, device: torch.device, out_dir: Path, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("host has no trainable parameters")
    bs = int(cfg.get("batch_size", 4))
    nw = int(cfg.get("num_workers", 2))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=len(train_ds) > bs)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("eval_batch_size", 32)), shuffle=False, num_workers=nw, pin_memory=pin) if val_ds is not None and len(val_ds) else None

    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 5e-4))
    if cfg.get("optimizer", "sgd") == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    epochs = int(cfg.get("epochs", 10))
    warmup = int(cfg.get("warmup_epochs", 0))
    use_amp = bool(cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    max_steps = cfg.get("max_steps")
    selection = cfg.get("selection", "last")
    grad_clip = cfg.get("grad_clip")

    hist, best, best_state, step = [], float("inf"), None, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        cur = lr_at(ep, epochs, lr, warmup, float(cfg.get("warmup_lr", 1e-5)), cfg.get("scheduler", "cosine"))
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        run, parts = [], {}
        stop = False
        for batch in train_loader:
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        out = model(x, y)
                    scaler.scale(out["loss"]).backward()
                    if grad_clip:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
                    scaler.step(opt)
                    scaler.update()
                else:
                    out = model(x, y)
                    out["loss"].backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
                    opt.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM at batch_size={bs}; rerun with --set batch_size={max(1, bs // 2)}") from None
            run.append(float(out["loss"]))
            for k, v in (out.get("loss_parts") or {}).items():
                parts.setdefault(k, []).append(v)
            step += 1
            if max_steps and step >= int(max_steps):
                stop = True
                break
        model.on_epoch_end(ep, epochs)
        vl, va = val_loss(model, val_loader, device, cfg.get("val_max_batches"))
        rec = {"epoch": ep, "lr": cur, "train_loss": float(np.mean(run)) if run else None, "val_loss": vl, "val_acc": va,
               **{f"train_{k}": float(np.mean(v)) for k, v in parts.items()}}
        hist.append(rec)
        log(f"  epoch {ep + 1}/{epochs} lr {cur:.2e} train {rec['train_loss']:.4f} val_loss {vl if vl is None else round(vl, 4)} val_acc {va if va is None else round(va, 4)}")
        if selection == "best_val" and vl is not None and vl < best:
            best, best_state = vl, copy.deepcopy(trainable_state(model))
        if stop:
            break
    model.on_train_end()
    if selection == "best_val" and best_state is not None:
        model.load_state_dict(best_state, strict=False)
    train_time = time.perf_counter() - t0
    torch.save(trainable_state(model), out_dir / "weights.pt")
    summary = {"epochs_run": len(hist), "steps": step, "train_seconds": train_time, "history": hist, "selection": selection,
               "trainable_params": int(sum(p.numel() for p in params))}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
