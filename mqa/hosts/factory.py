"""Method registry: name -> (host class, default hyper-parameters).

Defaults follow each method's paper / public config; the MQAdapter values follow the MQAdapter paper
(Sec. V-A2): 8 qubits, insertion from block 6 to the last, K=5, gamma=0.001 (base-to-new) and the per-host
few-shot gammas (MaPLe 0.001, PromptSRC 0.0001, MMRL++ 0.002). Anything can be overridden from main.py.

"+mqa" suffix = the same host with the MQAdapter plug-in, e.g. "mmrlpp+mqa".
"""
from __future__ import annotations

import copy

from mqa.hosts.maple import MaPLeHost
from mqa.hosts.mmrl import MMRLHost, MMRLPlusPlusHost
from mqa.hosts.promptsrc import PromptSRCHost
from mqa.hosts.simple import CLIPAdapterHost, CLIPHost, CoCoOpHost, CoOpHost

MQA_DEFAULTS = dict(
    n_qubits=8, adapt_layers=[6, 7, 8, 9, 10, 11], k=5, gamma=0.001, quantum_map="vqc",
    crz_dir="control_first", vqc_init="zeros", interaction="quantum", vqc_sharing="shared",
)
FEWSHOT_GAMMA = {"maple": 0.001, "promptsrc": 0.0001, "mmrlpp": 0.002, "mmrl": 0.002}

METHODS: dict[str, dict] = {
    # name          host class          training recipe (+ host hyper-parameters)
    "zeroshot": dict(cls=CLIPHost, train=False, cfg=dict()),
    "linear_probe": dict(cls=None, train="sklearn", cfg=dict(C_grid=[0.01, 0.1, 1.0, 10.0, 100.0])),
    "clip_adapter": dict(cls=CLIPAdapterHost, cfg=dict(optimizer="sgd", lr=0.002, batch_size=32, epochs=50, warmup_epochs=1, reduction=4, adapter_ratio=0.2)),
    "coop": dict(cls=CoOpHost, cfg=dict(optimizer="sgd", lr=0.002, batch_size=32, epochs=50, warmup_epochs=1, n_ctx=16, ctx_init="")),
    "cocoop": dict(cls=CoCoOpHost, cfg=dict(optimizer="sgd", lr=0.002, batch_size=4, epochs=10, warmup_epochs=1, n_ctx=4, ctx_init="a photo of a")),
    "maple": dict(cls=MaPLeHost, cfg=dict(optimizer="sgd", lr=0.0035, batch_size=4, epochs=5, warmup_epochs=1, n_ctx=2, prompt_depth=9, ctx_init="a photo of a")),
    "promptsrc": dict(cls=PromptSRCHost, cfg=dict(
        optimizer="sgd", lr=0.0025, batch_size=4, epochs=20, warmup_epochs=1, n_ctx_text=4, n_ctx_vision=4,
        prompt_depth_text=9, prompt_depth_vision=9, ctx_init="a photo of a", text_loss_weight=25.0,
        image_loss_weight=10.0, gpa_mean=15, gpa_std=1.0)),
    "mmrl": dict(cls=MMRLHost, cfg=dict(
        optimizer="adamw", lr=0.001, weight_decay=0.01, batch_size=4, epochs=10, warmup_epochs=0, rep_tokens=5,
        rep_dim=512, rep_layer=6, mmrl_alpha=0.7, mmrl_lambda=0.5, srra=False, prc=False)),
    "mmrlpp": dict(cls=MMRLPlusPlusHost, cfg=dict(
        optimizer="adamw", lr=0.001, weight_decay=0.01, batch_size=4, epochs=10, warmup_epochs=0, rep_tokens=5,
        rep_dim=512, rep_layer=6, mmrl_alpha=0.7, mmrl_lambda=0.5, mmrl_beta=0.9, rank_aligner=4, rank_proj=64,
        srra=True, prc=True)),
}
# Stand-alone MQAdapter on frozen CLIP (no prompt-learning host) = "zeroshot+mqa"; give it a training recipe.
STANDALONE_MQA = dict(optimizer="adamw", lr=0.001, weight_decay=0.01, batch_size=8, epochs=20, warmup_epochs=0)
COMMON = dict(scheduler="cosine", warmup_lr=1e-5, momentum=0.9, weight_decay=5e-4, grad_clip=None, mixed_precision=False)


def split_name(name: str) -> tuple[str, bool]:
    base, _, suffix = name.partition("+")
    if suffix not in ("", "mqa"):
        raise ValueError(f"unknown suffix in {name!r} (only '+mqa')")
    if base not in METHODS:
        raise ValueError(f"unknown method {base!r}; choose from {sorted(METHODS)} (optionally +mqa)")
    return base, suffix == "mqa"


def method_config(name: str, overrides: dict | None = None, fewshot: bool = False) -> dict:
    """Merge COMMON <- method defaults <- MQAdapter defaults (if +mqa) <- overrides."""
    base, mqa = split_name(name)
    cfg = copy.deepcopy(COMMON)
    cfg.update(copy.deepcopy(METHODS[base]["cfg"]))
    if mqa:
        cfg.update(copy.deepcopy(MQA_DEFAULTS))
        cfg["mqadapter"] = True
        if base == "zeroshot":
            cfg.update(STANDALONE_MQA)
        if fewshot and base in FEWSHOT_GAMMA:
            cfg["gamma"] = FEWSHOT_GAMMA[base]
    cfg["method"] = name
    cfg["host"] = base
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def is_trainable(name: str) -> bool:
    base, mqa = split_name(name)
    if base == "zeroshot":
        return mqa
    return METHODS[base].get("train", True) is not False


def build_host(name: str, clip_model, tokenizer, cfg: dict):
    base, _ = split_name(name)
    cls = METHODS[base]["cls"]
    if cls is None:
        raise ValueError(f"{name} is not a torch host (handled by the engine)")
    return cls(clip_model, tokenizer, cfg)
