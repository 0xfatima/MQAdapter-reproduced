"""Baseline hosts.

  CLIPHost        frozen CLIP with hand-written prompts (zero-shot). With `mqadapter: true` this is the
                  stand-alone MQAdapter on frozen CLIP (no prompt learning).
  CLIPAdapterHost CLIP-Adapter (Gao et al., IJCV 2024): residual 2-layer MLP on the image feature, ratio 0.2.
  CoOpHost        CoOp (Zhou et al., IJCV 2022): learnable shared text context.
  CoCoOpHost      CoCoOp (Zhou et al., CVPR 2022): context + image-conditioned meta-net bias.

All accept the MQAdapter plug-in on the vision tower (the paper calls MQAdapter plug-and-play; the paper's
own tables only plug it into MaPLe / PromptSRC / MMRL++).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa.hosts.base import HostBase


class _VisionMixin:
    def image_features(self, images: torch.Tensor):
        """Adapted (if the plug-in is on) or frozen image features + Top-K info."""
        topk = None
        if self.plugin is not None:
            topk = self.retrieve_topk(self.frozen_image_features(images))
            tokens, alphas = self.run_visual(images, topk_idx=topk[0])
            z = F.normalize(self.pool_visual(tokens).float(), dim=-1)
            return z, topk, alphas
        return self.frozen_image_features(images), None, {}


class CLIPHost(_VisionMixin, HostBase):
    method = "clip"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        self.ensemble = bool(cfg.get("zs_ensemble", False))
        self.trainable = self.plugin is not None

    def _on_class_space(self, space):
        if self.ensemble:
            feats = []
            for c, d in zip(space.classes, space.display):
                t = self.encode_texts_frozen([tpl.format(d) for tpl in space.templates])
                feats.append(F.normalize(t.mean(0), dim=-1))
            self._set_buf("classifier", torch.stack(feats))
        else:
            self._set_buf("classifier", self.anchor_features.clone())

    def forward(self, images, labels=None):
        z, topk, alphas = self.image_features(images)
        logits = self.logit_scale() * z @ self.classifier.T
        return self.finish({"logits": logits, "image_features": z, "alpha": alphas}, labels, topk)


class CLIPAdapterHost(_VisionMixin, HostBase):
    method = "clip_adapter"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        d, red = self.embed_dim, int(cfg.get("reduction", 4))
        self.adapter = nn.Sequential(
            nn.Linear(d, d // red, bias=False), nn.ReLU(inplace=True), nn.Linear(d // red, d, bias=False), nn.ReLU(inplace=True)
        )
        self.ratio = float(cfg.get("adapter_ratio", 0.2))

    def forward(self, images, labels=None):
        z, topk, alphas = self.image_features(images)
        z = F.normalize(self.ratio * self.adapter(z) + (1 - self.ratio) * z, dim=-1)
        logits = self.logit_scale() * z @ self.anchor_features.T
        return self.finish({"logits": logits, "image_features": z, "alpha": alphas}, labels, topk)


class CoOpHost(_VisionMixin, HostBase):
    method = "coop"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        self.n_ctx = int(cfg.get("n_ctx", 16))
        ctx, self.prefix_text = self.init_ctx(cfg.get("ctx_init", ""), self.n_ctx)
        self.ctx = nn.Parameter(ctx)

    def _on_class_space(self, space):
        self.build_prompt_tokens(self.prefix_text, self.n_ctx, space.display)

    def text_features(self) -> torch.Tensor:
        c = self.token_prefix.shape[0]
        x = torch.cat([self.token_prefix, self.ctx.unsqueeze(0).expand(c, -1, -1), self.token_suffix], dim=1)
        return self.run_text(x, self.tokenized.argmax(dim=-1))

    def forward(self, images, labels=None):
        z, topk, alphas = self.image_features(images)
        logits = self.logit_scale() * z @ self.text_features().T
        return self.finish({"logits": logits, "image_features": z, "alpha": alphas}, labels, topk)


class CoCoOpHost(_VisionMixin, HostBase):
    method = "cocoop"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        self.n_ctx = int(cfg.get("n_ctx", 4))
        ctx, self.prefix_text = self.init_ctx(cfg.get("ctx_init", "a photo of a"), self.n_ctx)
        self.ctx = nn.Parameter(ctx)
        d = self.embed_dim
        self.meta_net = nn.Sequential(nn.Linear(d, d // 16), nn.ReLU(inplace=True), nn.Linear(d // 16, self.txt_width))

    def _on_class_space(self, space):
        self.build_prompt_tokens(self.prefix_text, self.n_ctx, space.display)

    def forward(self, images, labels=None):
        z, topk, alphas = self.image_features(images)
        bias = self.meta_net(z)  # [B, d_t]
        c = self.token_prefix.shape[0]
        eot = self.tokenized.argmax(dim=-1)
        scale = self.logit_scale()
        rows = []
        for i in range(z.shape[0]):  # per-image prompts, as in the official implementation
            ctx_i = (self.ctx + bias[i]).unsqueeze(0).expand(c, -1, -1)
            x = torch.cat([self.token_prefix, ctx_i, self.token_suffix], dim=1)
            t = self.run_text(x, eot)
            rows.append(scale * z[i : i + 1] @ t.T)
        logits = torch.cat(rows, dim=0)
        return self.finish({"logits": logits, "image_features": z, "alpha": alphas}, labels, topk)
