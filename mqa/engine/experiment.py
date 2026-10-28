"""One experiment = (task, method, seed, shots, variant): split -> build host -> train -> evaluate -> result.json.

Results live in  <out_root>/<task>/<method>[__<variant>]/shots<S>/seed<N>/result.json  and are skipped on rerun
unless --force. Nothing in here invents numbers: every value in result.json comes from an evaluation pass.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from mqa.datasets import load_spec
from mqa.engine.data import RecordDataset, build_task, build_transform
from mqa.engine.evaluate import evaluate_set
from mqa.engine.linear_probe import run_linear_probe
from mqa.engine.trainer import train_host
from mqa.hosts.factory import build_host, is_trainable, method_config, split_name
from mqa.seed import set_seed

# Few-shot epochs. CoOp, PromptSRC and MMRL++ values follow their papers/configs; MaPLe, CoCoOp,
# CLIP-Adapter and the stand-alone adapter are our choice (marked in docs/REPRODUCTION_STATUS.md).
FEWSHOT_EPOCHS = {
    "coop": {1: 50, 2: 100, 4: 100, 8: 200, 16: 200},
    "cocoop": {s: 10 for s in (1, 2, 4, 8, 16)},
    "clip_adapter": {s: 200 for s in (1, 2, 4, 8, 16)},
    "maple": {s: 50 for s in (1, 2, 4, 8, 16)},
    "promptsrc": {s: 50 for s in (1, 2, 4, 8, 16)},
    "mmrl": {s: 100 for s in (1, 2, 4, 8, 16)},
    "mmrlpp": {s: 100 for s in (1, 2, 4, 8, 16)},
    "zeroshot": {s: 50 for s in (1, 2, 4, 8, 16)},  # stand-alone MQAdapter
}


def harmonic(a, b):
    return 2 * a * b / (a + b) if a is not None and b is not None and (a + b) > 0 else None


class Engine:
    def __init__(self, cfg: dict, log=print):
        self.cfg = dict(cfg)
        self.log = log
        self.spec = load_spec(cfg.get("dataset", "brain_tumor_mri"), cfg.get("data_root"))
        dev = cfg.get("device", "auto")
        self.device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev))
        self.out_root = Path(cfg.get("out_root", "runs/main"))
        self._clip = None
        self._tok = None

    # ------------------------------------------------------------------ backbone (loaded once)
    def backbone(self):
        if self._clip is None:
            backend = self.cfg.get("backend", "open_clip")
            if backend == "fake":
                from mqa.models.clip_backbone import FakeCLIPWithText, FakeTokenizer

                torch.manual_seed(0)
                self._clip, self._tok = FakeCLIPWithText().to(self.device).eval(), FakeTokenizer()
            else:
                from mqa.models.clip_backbone import load_clip

                clip, _, tok = load_clip(backend, device=self.device, download=bool(self.cfg.get("download_clip", True)))
                self._clip, self._tok = clip.to(self.device).eval(), tok
            for p in self._clip.parameters():
                p.requires_grad = False
        return self._clip, self._tok

    # ------------------------------------------------------------------ config policy
    def run_config(self, task: str, method: str, shots, overrides: dict | None) -> dict:
        base, _ = split_name(method)
        cfg = method_config(method, None, fewshot=(task == "fewshot"))
        if task == "fewshot" and base in FEWSHOT_EPOCHS:
            cfg["epochs"] = FEWSHOT_EPOCHS[base].get(int(shots), cfg.get("epochs", 10))
        if task == "all" and not shots:
            cfg["epochs"] = int(self.cfg.get("full_epochs", 5))
            cfg["batch_size"] = max(int(cfg.get("batch_size", 4)), int(self.cfg.get("full_batch_size", 16)))
        for k in ("augment", "num_workers", "eval_batch_size", "val_ratio", "mixed_precision", "selection", "limit_eval",
                  "cross_targets", "dg_targets", "dg_shifts", "dg_mode", "cross_mode", "target_roots"):
            if k in self.cfg and self.cfg[k] is not None:
                cfg[k] = self.cfg[k]
        cfg.update({k: v for k, v in (self.cfg.get("overrides") or {}).items() if v is not None})
        cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
        if self.cfg.get("quick"):
            cfg.update(epochs=min(int(cfg.get("epochs", 1)), int(self.cfg.get("quick_epochs", 1))), max_steps=int(self.cfg.get("quick_steps", 3)),
                       warmup_epochs=0, val_max_batches=1, limit_eval=cfg.get("limit_eval") or 32)
        if self.cfg.get("epochs_scale"):
            cfg["epochs"] = max(1, int(round(cfg.get("epochs", 1) * float(self.cfg["epochs_scale"]))))
        return cfg

    def run_dir(self, task, method, seed, shots, variant) -> Path:
        name = method + (f"__{variant}" if variant else "")
        return self.out_root / task / name / f"shots{shots or 'all'}" / f"seed{seed}"

    # ------------------------------------------------------------------ one experiment
    def run(self, task: str, method: str, seed: int = 1, shots=None, overrides: dict | None = None, variant: str = "") -> dict:
        out_dir = self.run_dir(task, method, seed, shots, variant)
        res_path = out_dir / "result.json"
        if res_path.exists() and not self.cfg.get("force"):
            self.log(f"[skip] {res_path} exists")
            return json.loads(res_path.read_text())
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.run_config(task, method, shots, overrides)
        cfg["shots"] = shots
        self.log(f"\n=== {task} | {method}{' [' + variant + ']' if variant else ''} | shots={shots or 'all'} | seed={seed} | epochs={cfg.get('epochs')}")
        set_seed(seed)
        split = build_task(task, self.spec, cfg, seed)
        (out_dir / "split.json").write_text(json.dumps(split.info, indent=2))
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        clip, tok = self.backbone()
        base, mqa = split_name(method)
        t0 = time.perf_counter()
        result = {"task": task, "method": method, "host": base, "mqadapter": mqa, "variant": variant, "seed": seed,
                  "shots": shots, "dataset": self.spec.name, "classes": split.space.classes, "evals": {}}
        if base == "linear_probe":
            lp = run_linear_probe(clip, split, cfg, self.device, out_dir / "eval", self.log)
            result["evals"] = {k: _slim(v) for k, v in lp["evals"].items()}
            result["trainable_params"] = lp["trainable_params"]
        else:
            set_seed(seed)
            host = build_host(method, clip, tok, cfg)
            host.set_class_space(split.space, "base")
            if is_trainable(method):
                aug = cfg.get("augment", "mri")
                summ = train_host(host, RecordDataset(split.train, build_transform(True, aug)),
                                  RecordDataset(split.val, build_transform(False, aug)) if split.val else None,
                                  cfg, self.device, out_dir, self.log)
                result.update(train_seconds=summ["train_seconds"], epochs_run=summ["epochs_run"])
            result["trainable_params"] = host.trainable_parameter_count()
            result["mqadapter_params"] = host.plugin_parameter_count()
            result["k_effective"] = {}
            for e in split.evals:
                m = evaluate_set(host, e, cfg, self.device, out_dir / "eval" / e.name.strip("-"),
                                 save_features=(task == "base2new" or bool(self.cfg.get("save_features"))))
                result["evals"][e.name] = _slim(m)
                result["k_effective"][e.name] = int(host.k) if host.plugin is not None else None
                self.log(f"  [{e.name}] acc {m['accuracy'] * 100:.2f}  macro-F1 {m['macro_f1'] * 100:.2f}"
                         + (f"  Top-{host.k} recall {m['coarse_topk_recall'] * 100:.1f}" if "coarse_topk_recall" in m else ""))
            del host
        result["wall_seconds"] = time.perf_counter() - t0
        result["summary"] = summarise(task, result["evals"])
        res_path.write_text(json.dumps(result, indent=2))
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def _slim(m: dict) -> dict:
    keep = ("accuracy", "macro_f1", "group", "mode", "shift", "eval_set", "n_samples", "coarse_topk_recall",
            "coarse_top1_accuracy", "accuracy_when_retrieved", "accuracy_when_missed", "images_per_second", "note", "k")
    return {k: m.get(k) for k in keep if k in m}


def summarise(task: str, evals: dict) -> dict:
    acc = {k: (v.get("accuracy") * 100 if v.get("accuracy") is not None else None) for k, v in evals.items()}
    if task == "base2new":
        b, n = acc.get("base"), acc.get("new")
        return {"base": b, "new": n, "H": harmonic(b, n)}
    groups = {k: v.get("group") for k, v in evals.items()}
    if task in ("cross", "dg"):
        src = [acc[k] for k in acc if groups[k] == "source"]
        tgt = [acc[k] for k in acc if groups[k] in ("target", "shifted") and acc[k] is not None]
        return {"source": src[0] if src else None, "targets": {k: acc[k] for k in acc if groups[k] != "source"},
                "target_avg": sum(tgt) / len(tgt) if tgt else None}
    return {"accuracy": acc.get("test"), "macro_f1": (evals.get("test") or {}).get("macro_f1")}
