"""MaPLe (Khattak et al., CVPR 2023) host with optional MQAdapter (paper host #1).

Deep coupled prompts: learnable text context at the input (n_ctx=2, init "a photo of a"), text prompts replaced
in blocks 1..depth-1 (depth=9), and vision prompts that are linear projections (512->768) of the text prompts,
appended to the patch sequence and replaced in the same blocks. Official recipe: SGD lr 0.0035, batch 4,
5 epochs (base-to-new), fp16 (fp32 here).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa.hosts.base import HostBase


class MaPLeHost(HostBase):
    method = "maple"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        self.n_ctx = int(cfg.get("n_ctx", 2))
        self.depth = int(cfg.get("prompt_depth", 9))
        if not 1 <= self.depth <= min(self.n_vis_blocks, self.n_txt_blocks):
            raise ValueError(f"prompt_depth {self.depth} invalid")
        ctx, self.prefix_text = self.init_ctx(cfg.get("ctx_init", "a photo of a"), self.n_ctx)
        self.ctx = nn.Parameter(ctx)
        self.proj = nn.Linear(self.txt_width, self.vis_width)
        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_ctx, self.txt_width, device=self.device) * 0.02) for _ in range(self.depth - 1)]
        )
        self.compound_prompt_projections = nn.ModuleList([nn.Linear(self.txt_width, self.vis_width) for _ in range(self.depth - 1)])

    def _on_class_space(self, space):
        self.build_prompt_tokens(self.prefix_text, self.n_ctx, space.display)

    def text_features(self) -> torch.Tensor:
        c = self.token_prefix.shape[0]
        x = torch.cat([self.token_prefix, self.ctx.unsqueeze(0).expand(c, -1, -1), self.token_suffix], dim=1)

        def before(i, x):
            if 0 < i < self.depth:
                deep = self.compound_prompts_text[i - 1].unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([x[:, :1], deep.to(x.dtype), x[:, 1 + self.n_ctx :]], dim=1)
            return x

        return self.run_text(x, self.tokenized.argmax(dim=-1), before=before)

    def image_features(self, images, topk_idx):
        b = images.shape[0]
        vis_ctx = self.proj(self.ctx).unsqueeze(0).expand(b, -1, -1)

        def before(i, x):
            if 0 < i < self.depth:
                deep = self.compound_prompt_projections[i - 1](self.compound_prompts_text[i - 1])
                x = torch.cat([x[:, : -self.n_ctx], deep.unsqueeze(0).expand(b, -1, -1).to(x.dtype)], dim=1)
            return x

        tokens, alphas = self.run_visual(images, topk_idx=topk_idx, extra_tokens=vis_ctx, before=before)
        return F.normalize(self.pool_visual(tokens).float(), dim=-1), alphas

    def forward(self, images, labels=None):
        topk = self.retrieve_topk(self.frozen_image_features(images)) if self.plugin is not None else None
        z, alphas = self.image_features(images, None if topk is None else topk[0])
        logits = self.logit_scale() * z @ self.text_features().T
        return self.finish({"logits": logits, "image_features": z, "alpha": alphas}, labels, topk)
