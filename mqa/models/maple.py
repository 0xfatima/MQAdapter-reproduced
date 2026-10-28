"""MaPLe host (Khattak et al., CVPR 2023) with optional MQAdapter plug-in (the paper's main setting).

MaPLe = deep, coupled text/vision prompts on a frozen CLIP:
  * learnable text context `ctx` (n_ctx x 512) at the input of the text tower;
  * vision prompts are *not* free: they are linear projections of the text prompts (512 -> 768), so the two
    branches stay coupled;
  * prompts are re-injected in blocks 1 .. depth-1 (text: replace tokens 1..n_ctx; vision: replace the last n_ctx tokens).
Re-implemented from the MaPLe paper/official repo on the open_clip API. It has not been run against the official
code here; see docs/IMPLEMENTATION_AUDIT.md.

MQAdapter plug-in (paper Section IV-C): after visual block l (l in adapt_layers) the class token is rewritten by
`MQAdapterPlugin.apply`. Anchors are the Top-K frozen-CLIP class embeddings (hand-written prompts, no grad);
retrieval uses the *unprompted* frozen CLIP image embedding, as the coarse stage prescribes.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa.models.clip_backbone import (
    freeze_module,
    run_block,
    tokenize_prompts,
    transformer_batch_first,
    visual_pool,
    visual_prefix,
)
from mqa.models.mqadapter_clip import ActiveClassMixin
from mqa.models.plugin import MQAdapterPlugin

DEFAULT_DISPLAY_NAMES = {
    "glioma": "glioma brain tumor",
    "meningioma": "meningioma brain tumor",
    "notumor": "brain with no tumor",
    "pituitary": "pituitary brain tumor",
}


class MaPLeCLIP(ActiveClassMixin, nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        tokenizer,
        class_names: Sequence[str],
        anchor_text_features: torch.Tensor,
        display_names: dict[str, str] | None = None,
        n_ctx: int = 2,
        depth: int = 9,
        ctx_init: str = "a photo of a",
        k: int = 2,
        plugin_cfg: dict | None = None,
    ):
        """
        anchor_text_features: [C, d_t] frozen-CLIP embeddings of fixed hand-written prompts. Used only for the
            coarse Top-K retrieval and as MQAdapter anchors; never trained.
        plugin_cfg: kwargs for MQAdapterPlugin (None = plain MaPLe).
        """
        super().__init__()
        self.clip = clip_model
        freeze_module(self.clip)
        self.tokenizer = tokenizer
        self.class_names = list(class_names)
        self.display = dict(DEFAULT_DISPLAY_NAMES if display_names is None else display_names)
        self.n_ctx = int(n_ctx)
        self.depth = int(depth)
        self.prefix_text = ctx_init
        self.k_requested = int(k)
        self._init_classes(anchor_text_features)

        visual = clip_model.visual
        self.vis_width = int(getattr(visual.transformer, "width", visual.conv1.out_channels))
        self.txt_width = int(clip_model.token_embedding.embedding_dim)
        self.bf_vis = transformer_batch_first(visual.transformer)
        self.bf_txt = transformer_batch_first(clip_model.transformer)
        n_vis_blocks = len(visual.transformer.resblocks)
        if not (1 <= self.depth <= min(n_vis_blocks, len(clip_model.transformer.resblocks))):
            raise ValueError(f"depth {self.depth} invalid for this CLIP")

        device = next(clip_model.parameters()).device
        # --- learnable prompts -------------------------------------------------------------------------
        with torch.no_grad():
            tok = tokenize_prompts(tokenizer, [ctx_init], device)
            emb = clip_model.token_embedding(tok)
        words = len(ctx_init.split())
        if self.n_ctx <= words:
            ctx = emb[0, 1 : 1 + self.n_ctx].detach().clone().float()
        else:
            ctx = torch.randn(self.n_ctx, self.txt_width, device=device) * 0.02
        self.ctx = nn.Parameter(ctx)
        self.proj = nn.Linear(self.txt_width, self.vis_width)  # shallow text -> vision coupling
        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_ctx, self.txt_width, device=device) * 0.02) for _ in range(self.depth - 1)]
        )
        self.compound_prompt_projections = nn.ModuleList(
            [nn.Linear(self.txt_width, self.vis_width) for _ in range(self.depth - 1)]
        )
        self._build_prompt_buffers(self.class_names)

        # --- optional MQAdapter -------------------------------------------------------------------------
        self.plugin = None
        if plugin_cfg is not None:
            self.plugin = MQAdapterPlugin(
                visual_width=self.vis_width,
                text_dim=int(anchor_text_features.shape[-1]),
                n_blocks=n_vis_blocks,
                **plugin_cfg,
            )
        if hasattr(self.clip, "logit_scale"):
            self.clip.logit_scale.requires_grad = False

    # ---------------------------------------------------------------------------------------------------
    def _set_buf(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value, persistent=False)

    def _build_prompt_buffers(self, names: Sequence[str]) -> None:
        device = self.ctx.device
        prompts = [f"{self.prefix_text} {self.display.get(n, n)}." for n in names]
        tokens = tokenize_prompts(self.tokenizer, prompts, device)
        with torch.no_grad():
            emb = self.clip.token_embedding(tokens)
        self._set_buf("token_prefix", emb[:, :1].detach())
        self._set_buf("token_suffix", emb[:, 1 + self.n_ctx :].detach())
        self._set_buf("tokenized", tokens)

    def set_active_classes(self, indices: Sequence[int]) -> None:
        super().set_active_classes(indices)
        self._build_prompt_buffers([self.class_names[i] for i in indices])

    @property
    def k(self) -> int:
        return min(self.k_requested, self.n_classes())

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # --- text tower with deep prompts ------------------------------------------------------------------
    def encode_text_prompted(self) -> torch.Tensor:
        clip = self.clip
        c = self.token_prefix.shape[0]
        ctx = self.ctx.unsqueeze(0).expand(c, -1, -1)
        x = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)  # [C, 77, 512]
        x = x + clip.positional_embedding.to(x.dtype)
        attn_mask = getattr(clip, "attn_mask", None)
        for i, blk in enumerate(clip.transformer.resblocks):
            if 0 < i < self.depth:
                deep = self.compound_prompts_text[i - 1].unsqueeze(0).expand(c, -1, -1)
                x = torch.cat([x[:, :1], deep, x[:, 1 + self.n_ctx :]], dim=1)
            x = run_block(blk, x, self.bf_txt, attn_mask=attn_mask)
        x = clip.ln_final(x)
        x = x[torch.arange(c, device=x.device), self.tokenized.argmax(dim=-1)]
        proj = clip.text_projection
        x = proj(x) if isinstance(proj, nn.Module) else x @ proj
        return F.normalize(x.float(), dim=-1)

    # --- vision tower with deep prompts (+ optional MQAdapter) -------------------------------------------
    @torch.no_grad()
    def retrieve_topk(self, images: torch.Tensor):
        img = F.normalize(self.clip.encode_image(images).float(), dim=-1)
        sim = img @ self.text_features.T  # fixed anchor features of the active classes
        vals, idx = sim.topk(self.k, dim=-1)
        return idx, vals, sim

    def encode_image_prompted(self, images: torch.Tensor, topk_idx: torch.Tensor | None):
        visual = self.clip.visual
        b = images.shape[0]
        vis_ctx = self.proj(self.ctx).unsqueeze(0).expand(b, -1, -1)
        x, _ = visual_prefix(visual, images, extra_tokens=vis_ctx)
        p_all = self.plugin.encode_anchors(self.text_features) if self.plugin is not None else None
        alphas = {}
        for i, blk in enumerate(visual.transformer.resblocks):
            if 0 < i < self.depth:
                deep = self.compound_prompt_projections[i - 1](self.compound_prompts_text[i - 1])
                x = torch.cat([x[:, : -self.n_ctx], deep.unsqueeze(0).expand(b, -1, -1)], dim=1)
            x = run_block(blk, x, self.bf_vis)
            if self.plugin is not None:
                x, alpha = self.plugin.apply(i, x, p_all, topk_idx)
                if alpha is not None:
                    alphas[i] = alpha
        return visual_pool(visual, x), alphas

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        idx = vals = sim_frozen = None
        if self.plugin is not None:
            idx, vals, sim_frozen = self.retrieve_topk(images)
        z, alphas = self.encode_image_prompted(images, idx)
        z = F.normalize(z.float(), dim=-1)
        text = self.encode_text_prompted()
        scale = self.clip.logit_scale.exp() if hasattr(self.clip, "logit_scale") else torch.tensor(100.0, device=z.device)
        logits = scale * (z @ text.T)
        out = {"logits": logits, "image_features": z, "alpha": alphas, "topk_idx": idx, "topk_sim": vals, "frozen_sim": sim_frozen}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
            if idx is not None:
                out["retrieval_hit"] = (idx == labels.unsqueeze(-1)).any(dim=-1)
        return out
