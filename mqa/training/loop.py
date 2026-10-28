"""Training loop (Eq. 16 objective) for stand-alone MQAdapter, MaPLe(+MQAdapter) and baselines.

Optimisers: AdamW (default, stand-alone adapter) or SGD+momentum (MaPLe's own recipe). LR schedule is a
stateless function of the epoch (optional constant warm-up, then cosine), so resuming needs no scheduler state.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mqa.evaluation.metrics import classification_metrics
from mqa.seed import set_seed
from mqa.training.checkpoint import load_checkpoint, save_checkpoint


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "path": batch.get("path"),
        "class_name": batch.get("class_name"),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device, class_names) -> dict:
    model.eval()
    ys, ps, losses, hits, top1_hits, paths = [], [], [], [], [], []
    t0 = time.perf_counter()
    n = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        out = model(batch["image"], batch["label"])
        losses.append(float(out["loss"].item()))
        ys.extend(batch["label"].cpu().tolist())
        if batch.get("path") is not None:
            paths.extend(list(batch["path"]))
        ps.extend(out["logits"].argmax(dim=-1).cpu().tolist())
        if out.get("retrieval_hit") is not None:
            hits.extend(out["retrieval_hit"].cpu().tolist())
            top1_hits.extend((out["topk_idx"][:, 0] == batch["label"]).cpu().tolist())
        n += batch["label"].size(0)
    elapsed = time.perf_counter() - t0
    metrics = classification_metrics(ys, ps, class_names)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    metrics["inference_seconds"] = elapsed
    metrics["images_per_second"] = n / elapsed if elapsed else None
    if hits:  # coarse-stage diagnostics: how often the true class is among the Top-K anchors
        metrics["coarse_topk_recall"] = float(np.mean(hits))
        metrics["coarse_top1_accuracy"] = float(np.mean(top1_hits))
        yt, pr = np.asarray(ys), np.asarray(ps)
        h = np.asarray(hits, dtype=bool)
        metrics["accuracy_when_retrieved"] = float((yt[h] == pr[h]).mean()) if h.any() else None
        metrics["accuracy_when_missed"] = float((yt[~h] == pr[~h]).mean()) if (~h).any() else None
    metrics["y_true"] = ys
    metrics["y_pred"] = ps
    metrics["paths"] = paths
    return metrics


def lr_at_epoch(epoch: int, epochs: int, base_lr: float, schedule: str, warmup_epochs: int, warmup_lr: float) -> float:
    if epoch < warmup_epochs:
        return warmup_lr
    if schedule == "none":
        return base_lr
    if schedule == "cosine":
        span = max(epochs - warmup_epochs, 1)
        return 0.5 * base_lr * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / span))
    raise ValueError(schedule)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    out_dir: str | Path,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = "auto",
    seed: int = 0,
    patience: int = 5,
    min_delta: float = 1e-4,
    max_steps: int | None = None,
    scheduler: str = "cosine",
    mixed_precision: bool = False,
    resume: str | None = None,
    log_every: int = 20,
    optimizer: str = "adamw",
    momentum: float = 0.9,
    warmup_epochs: int = 0,
    warmup_lr: float = 1e-5,
    grad_clip: float | None = None,
) -> dict:
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _device(device)
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters (frozen_clip is evaluated with scripts/eval_zeroshot.py)")
    if optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer}")

    use_amp = bool(mixed_precision and dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    start_epoch, best_val, best_acc = 0, float("inf"), -1.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    if resume:
        ckpt = load_checkpoint(resume, map_location=dev)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        history = ckpt.get("history", history)

    env = {
        "device": str(dev),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "optimizer": optimizer,
        "warmup_epochs": warmup_epochs,
        "mixed_precision": use_amp,
        "trainable_params": int(sum(p.numel() for p in params)),
        "note": "VQC runs in fp32 even under autocast (PaperVQC/TorchQuantumVQC disable autocast internally).",
    }
    (out_dir / "run_config.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    global_step, stale = 0, 0
    for epoch in range(start_epoch, epochs):
        cur_lr = lr_at_epoch(epoch, epochs, lr, scheduler, warmup_epochs, warmup_lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        model.train()
        running = []
        stop_now = False
        for batch in train_loader:
            batch = _move_batch(batch, dev)
            opt.zero_grad(set_to_none=True)
            try:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = model(batch["image"], batch["label"])["loss"]
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss = model(batch["image"], batch["label"])["loss"]
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(params, grad_clip)
                    opt.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory. Lower batch_size in the config (try 8, then 4); the 12 GB budget "
                    "allows roughly batch 16-32 for the stand-alone adapter and 8-16 for MaPLe."
                ) from None
            running.append(float(loss.item()))
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                stop_now = True
                break
        train_loss = float(np.mean(running)) if running else None
        val = evaluate_loader(model, val_loader, dev, class_names)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])

        improved = val["loss"] is not None and (best_val - val["loss"]) > min_delta
        if improved:
            best_val, best_acc, stale = val["loss"], val["accuracy"], 0
        else:
            stale += 1
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "best_val": best_val,
            "history": history,
            "env": env,
        }
        save_checkpoint(out_dir / "last.pt", payload)
        if improved:
            save_checkpoint(out_dir / "best.pt", payload)
            (out_dir / "best_val_metrics.json").write_text(
                json.dumps({k: v for k, v in val.items() if k not in ("y_true", "y_pred")}, indent=2), encoding="utf-8"
            )
        with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "lr": cur_lr, "train_loss": train_loss, "val_loss": val["loss"], "val_acc": val["accuracy"]}) + "\n")
        print(f"epoch {epoch:03d} lr {cur_lr:.2e} train {train_loss:.4f} val {val['loss']:.4f} acc {val['accuracy']:.4f}", flush=True)
        if stale >= patience:
            env["early_stop_epoch"] = epoch
            break
        if stop_now:
            break

    summary = {
        "best_val_loss": best_val if best_val < float("inf") else None,
        "best_val_acc": best_acc if best_acc >= 0 else None,
        "history": history,
        "env": env,
        "completed_epochs": len(history["train_loss"]),
        "trained": len(history["train_loss"]) > 0,
        "evaluated_on_held_out_test": False,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
