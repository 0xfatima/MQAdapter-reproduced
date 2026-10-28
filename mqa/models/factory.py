from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mqa import CLASS_NAMES, CLASS_PROMPTS, DEFAULT_ADAPT_LAYERS, DEFAULT_GAMMA
from mqa.models.baselines import CLIPAdapterHead, FrozenCLIPZeroShot
from mqa.models.clip_backbone import FakeCLIPWithText, FakeTokenizer, encode_class_texts, load_clip
from mqa.models.maple import MaPLeCLIP
from mqa.models.mqadapter_clip import MQAdapterCLIP

MODEL_NAMES = (
    "frozen_clip",
    "clip_adapter",
    "mqadapter",          # stand-alone MQAdapter on frozen CLIP
    "classical_adapter",  # same topology, MLP instead of VQC
    "maple",              # host only
    "maple_mqadapter",    # MaPLe + MQAdapter (paper setting)
)


def build_backbone(cfg: dict[str, Any], device: torch.device):
    backend = cfg.get("clip_backend", "open_clip")
    if backend == "fake":
        model = FakeCLIPWithText().to(device)
        tok = FakeTokenizer()
        return model, encode_class_texts(model, tok, device=device), tok
    clip, _preprocess, tok = load_clip(backend=backend, device=device, download=bool(cfg.get("download_clip", True)))
    clip = clip.to(device)
    text = encode_class_texts(clip, tok, class_names=CLASS_NAMES, prompts=CLASS_PROMPTS, device=device)
    return clip, text, tok


def _plugin_kwargs(cfg: dict[str, Any], default_map: str) -> dict[str, Any]:
    return dict(
        n_qubits=int(cfg.get("n_qubits", 8)),
        gamma=float(cfg.get("gamma", DEFAULT_GAMMA)),
        adapt_layers=tuple(cfg.get("adapt_layers", list(DEFAULT_ADAPT_LAYERS))),
        quantum_map=cfg.get("quantum_map", default_map),
        crz_dir=cfg.get("crz_dir", "control_first"),
        mlp_hidden=int(cfg.get("mlp_hidden", 8)),
        vqc_init=cfg.get("vqc_init", "zeros"),
        interaction=cfg.get("interaction", "quantum"),
        vqc_sharing=cfg.get("vqc_sharing", "shared"),
    )


def build_model(cfg: dict[str, Any], clip: nn.Module, text_features: torch.Tensor, tokenizer=None) -> nn.Module:
    name = cfg.get("model", "mqadapter")
    k = int(cfg.get("k", 4))
    if name in ("mqadapter", "classical_adapter"):
        kw = _plugin_kwargs(cfg, "vqc" if name == "mqadapter" else "mlp")
        return MQAdapterCLIP(
            clip_model=clip,
            text_features=text_features,
            k=k,
            freeze_clip=bool(cfg.get("freeze_clip", True)),
            **kw,
        )
    if name in ("maple", "maple_mqadapter"):
        if tokenizer is None:
            raise ValueError("MaPLe needs the CLIP tokenizer: pass build_backbone()'s third return value")
        plugin_cfg = _plugin_kwargs(cfg, "vqc") if name == "maple_mqadapter" else None
        return MaPLeCLIP(
            clip,
            tokenizer,
            CLASS_NAMES,
            anchor_text_features=text_features,
            n_ctx=int(cfg.get("n_ctx", 2)),
            depth=int(cfg.get("prompt_depth", 9)),
            ctx_init=cfg.get("ctx_init", "a photo of a"),
            k=k,
            plugin_cfg=plugin_cfg,
        )
    if name in ("frozen_clip", "zeroshot"):
        return FrozenCLIPZeroShot(clip, text_features)
    if name in ("clip_adapter", "clipadapter"):
        return CLIPAdapterHead(
            clip,
            text_features,
            bottleneck=int(cfg.get("bottleneck", 64)),
            residual=float(cfg.get("clip_adapter_residual", 0.2)),
        )
    raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
