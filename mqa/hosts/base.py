"""Common machinery for every fine-tuning method ("host") in the end-to-end engine.

A host owns a frozen CLIP and (optionally) the MQAdapter plug-in. It exposes:
  set_class_space(space, mode)   switch label space at any time (base / new / cross-dataset targets).
                                 mode="base" (seen classes) or "novel" (unseen classes / new task); only
                                 MMRL(++) behaves differently between the two (its decoupled inference).
  forward(images, labels=None)   -> {"logits", "loss" (if labels), "image_features", Top-K diagnostics}
  on_epoch_end / on_train_end    hooks (PromptSRC uses them for Gaussian prompt aggregation)

The vision and text towers are run block-by-block with optional hooks, so prompt methods can replace
tokens, MMRL can insert representation tokens, and the MQAdapter plug-in can rewrite the class token after
selected blocks (paper Eq. 15). The coarse stage (paper Sec. IV-A) always uses the *unprompted* frozen CLIP
image embedding and the hand-written prompts of the current label space.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa import DEFAULT_ADAPT_LAYERS, DEFAULT_GAMMA
from mqa.datasets import ClassSpace
from mqa.models.clip_backbone import freeze_module, run_block, transformer_batch_first, visual_pool, visual_prefix
from mqa.models.plugin import MQAdapterPlugin


def plugin_kwargs(cfg: dict) -> dict:
    return dict(
        n_qubits=int(cfg.get("n_qubits", 8)),
        gamma=float(cfg.get("gamma", DEFAULT_GAMMA)),
        adapt_layers=tuple(cfg.get("adapt_layers", list(DEFAULT_ADAPT_LAYERS))),
        quantum_map=cfg.get("quantum_map", "vqc"),
        crz_dir=cfg.get("crz_dir", "control_first"),
        mlp_hidden=int(cfg.get("mlp_hidden", 8)),
        vqc_init=cfg.get("vqc_init", "zeros"),
        interaction=cfg.get("interaction", "quantum"),
        vqc_sharing=cfg.get("vqc_sharing", "shared"),
    )


def causal_mask(length: int, device, dtype=torch.float32) -> torch.Tensor:
    return torch.full((length, length), float("-inf"), device=device, dtype=dtype).triu_(1)


class HostBase(nn.Module):
    method = "base"
    trainable = True

    def __init__(self, clip_model: nn.Module, tokenizer, cfg: dict):
        super().__init__()
        self.clip = clip_model
        freeze_module(self.clip)
        self.clip.eval()
        self.tokenizer = tokenizer
        self.cfg = dict(cfg)
        visual = clip_model.visual
        self.vis_width = int(getattr(visual.transformer, "width", visual.conv1.out_channels))
        self.txt_width = int(clip_model.token_embedding.embedding_dim)
        proj = getattr(clip_model, "text_projection", None)
        if isinstance(proj, nn.Linear):
            self.embed_dim = int(proj.out_features)
        elif proj is not None:
            self.embed_dim = int(proj.shape[1])
        else:
            self.embed_dim = int(getattr(visual, "output_dim", 512))
        self.bf_vis = transformer_batch_first(visual.transformer)
        self.bf_txt = transformer_batch_first(clip_model.transformer) if hasattr(clip_model, "transformer") else False
        self.n_vis_blocks = len(visual.transformer.resblocks)
        self.n_txt_blocks = len(clip_model.transformer.resblocks) if hasattr(clip_model, "transformer") else 0
        self.k_requested = int(cfg.get("k", 5))
        self.eval_mode = "base"
        self.space: ClassSpace | None = None
        self._mask_cache: dict[int, torch.Tensor] = {}
        self.plugin = None
        if cfg.get("mqadapter", False):
            self.plugin = MQAdapterPlugin(
                visual_width=self.vis_width, text_dim=self.embed_dim, n_blocks=self.n_vis_blocks, **plugin_kwargs(cfg)
            )

    # ------------------------------------------------------------------ utilities
    @property
    def device(self) -> torch.device:
        return next(self.clip.parameters()).device

    def _set_buf(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value, persistent=False)

    def logit_scale(self) -> torch.Tensor:
        if hasattr(self.clip, "logit_scale"):
            return self.clip.logit_scale.exp().float()
        return torch.tensor(100.0, device=self.device)

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        t = self.tokenizer(texts)
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        return t.to(self.device)

    @torch.no_grad()
    def encode_texts_frozen(self, texts: list[str]) -> torch.Tensor:
        return F.normalize(self.clip.encode_text(self.tokenize(texts)).float(), dim=-1)

    @torch.no_grad()
    def frozen_image_features(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.clip.encode_image(images).float(), dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # the backbone never leaves eval mode (no dropout / BN updates)
        return self

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def plugin_parameter_count(self) -> int:
        return 0 if self.plugin is None else sum(p.numel() for p in self.plugin.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ label space
    def set_class_space(self, space: ClassSpace, mode: str = "base") -> None:
        if mode not in ("base", "novel"):
            raise ValueError(mode)
        self.space = space
        self.eval_mode = mode
        self._set_buf("anchor_features", self.encode_texts_frozen(space.hand_prompts))  # [C, d]
        self._on_class_space(space)

    def _on_class_space(self, space: ClassSpace) -> None:  # hosts rebuild prompt tokens here
        pass

    def n_classes(self) -> int:
        return 0 if self.space is None else len(self.space)

    @property
    def k(self) -> int:
        return max(1, min(self.k_requested, self.n_classes()))

    # ------------------------------------------------------------------ coarse stage (paper IV-A)
    @torch.no_grad()
    def retrieve_topk(self, frozen_img: torch.Tensor):
        sim = frozen_img @ self.anchor_features.T
        vals, idx = sim.topk(self.k, dim=-1)
        return idx, vals, sim

    # ------------------------------------------------------------------ towers
    def run_visual(
        self,
        images: torch.Tensor,
        topk_idx: torch.Tensor | None = None,
        extra_tokens: torch.Tensor | None = None,
        before: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        after: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Return final tokens [B, L, D] and the per-layer MQAdapter attention maps."""
        visual = self.clip.visual
        x, _ = visual_prefix(visual, images, extra_tokens=extra_tokens)
        use_plugin = self.plugin is not None and topk_idx is not None
        p_all = self.plugin.encode_anchors(self.anchor_features) if use_plugin else None
        alphas = {}
        for i, blk in enumerate(visual.transformer.resblocks):
            if before is not None:
                x = before(i, x)
            x = run_block(blk, x, self.bf_vis)
            if after is not None:
                x = after(i, x)
            if use_plugin:
                x, a = self.plugin.apply(i, x, p_all, topk_idx)
                if a is not None:
                    alphas[i] = a
        return x, alphas

    def pool_visual(self, tokens: torch.Tensor) -> torch.Tensor:
        return visual_pool(self.clip.visual, tokens)

    def text_mask(self, length: int) -> torch.Tensor | None:
        base = getattr(self.clip, "attn_mask", None)
        if base is not None and base.shape[-1] == length:
            return base
        if length not in self._mask_cache:
            self._mask_cache[length] = causal_mask(length, self.device)
        return self._mask_cache[length].to(self.device)

    def run_text(
        self,
        x: torch.Tensor,
        eot_idx: torch.Tensor,
        before: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        after: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        eot_shift: int = 0,
    ) -> torch.Tensor:
        """x: token embeddings [N, 77, d_t] (before positional embedding). Returns normalized [N, d]."""
        clip = self.clip
        x = x + clip.positional_embedding.to(x.dtype)
        for i, blk in enumerate(clip.transformer.resblocks):
            if before is not None:
                x = before(i, x)
            x = run_block(blk, x, self.bf_txt, attn_mask=self.text_mask(x.shape[1]))
            if after is not None:
                x = after(i, x)
        x = clip.ln_final(x)
        x = x[torch.arange(x.shape[0], device=x.device), eot_idx + eot_shift]
        proj = clip.text_projection
        x = proj(x) if isinstance(proj, nn.Module) else x @ proj
        return F.normalize(x.float(), dim=-1)

    # ------------------------------------------------------------------ prompt helpers
    def build_prompt_tokens(self, prefix_text: str, n_ctx: int, displays: list[str], name: str = "") -> None:
        """Tokenize "<prefix> <class>." and store the frozen pieces around the n_ctx learnable slots."""
        prompts = [f"{prefix_text} {d}." for d in displays]
        tok = self.tokenize(prompts)
        with torch.no_grad():
            emb = self.clip.token_embedding(tok).float()
        self._set_buf(f"{name}token_prefix", emb[:, :1].detach())
        self._set_buf(f"{name}token_suffix", emb[:, 1 + n_ctx :].detach())
        self._set_buf(f"{name}tokenized", tok)

    def init_ctx(self, ctx_init: str | None, n_ctx: int, std: float = 0.02) -> tuple[torch.Tensor, str]:
        """Context vectors and the textual prefix used for tokenization (MaPLe/PromptSRC/CoOp convention)."""
        if ctx_init and n_ctx <= len(ctx_init.split()):
            with torch.no_grad():
                emb = self.clip.token_embedding(self.tokenize([ctx_init])).float()
            return emb[0, 1 : 1 + n_ctx].detach().clone(), ctx_init
        ctx = torch.empty(n_ctx, self.txt_width, device=self.device)
        nn.init.normal_(ctx, std=std)
        return ctx, " ".join(["X"] * n_ctx)

    # ------------------------------------------------------------------ outputs
    def finish(self, out: dict, labels: torch.Tensor | None, topk=None) -> dict:
        if topk is not None:
            idx, vals, sim = topk
            out.update(topk_idx=idx, topk_sim=vals, frozen_sim=sim)
            if labels is not None:
                out["retrieval_hit"] = (idx == labels.unsqueeze(-1)).any(dim=-1)
        if labels is not None and "loss" not in out:
            out["loss"] = F.cross_entropy(out["logits"], labels)
        return out

    def on_epoch_end(self, epoch: int, n_epochs: int) -> None:
        pass

    def on_train_end(self) -> None:
        pass
