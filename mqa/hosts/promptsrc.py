"""PromptSRC (Khattak et al., ICCV 2023) host with optional MQAdapter (paper host #2).

Re-implemented from the PromptSRC paper and the hyper-parameters of its public configs
(vit_b16_c2_ep20_batch4_4+4ctx): independent (uncoupled) deep prompts, 4 text + 4 vision tokens,
depth 9 on both towers, text context initialised from "a photo of a".

Self-regulation (paper Sec. 3.2):
  L_SCL-image  = w_img  * L1(f_prompted_image, f_frozen_image)                      (w_img  = 10)
  L_SCL-text   = w_text * L1(g_prompted_text,  mean_t g_frozen(template_t(class)))   (w_text = 25)
  L_SCL-logits = KL(p_prompted || p_frozen)  (sum over the batch / numel, as in the official code)
  loss = CE + L_SCL-image + L_SCL-text + L_SCL-logits
Gaussian Prompt Aggregation (GPA): after every epoch the prompt weights are accumulated with Gaussian
weights N(mu=gpa_mean, sigma=gpa_std) over epochs; the aggregate replaces the weights at the end of training.
Only the prompt parameters are aggregated; the MQAdapter weights (if any) keep their final values.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa.hosts.base import HostBase


class PromptSRCHost(HostBase):
    method = "promptsrc"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        self.n_ctx_t = int(cfg.get("n_ctx_text", 4))
        self.n_ctx_v = int(cfg.get("n_ctx_vision", 4))
        self.depth_t = int(cfg.get("prompt_depth_text", 9))
        self.depth_v = int(cfg.get("prompt_depth_vision", 9))
        self.w_text = float(cfg.get("text_loss_weight", 25.0))
        self.w_img = float(cfg.get("image_loss_weight", 10.0))
        self.gpa_mean = cfg.get("gpa_mean")
        self.gpa_std = float(cfg.get("gpa_std", 1.0))
        dev = self.device
        ctx, self.prefix_text = self.init_ctx(cfg.get("ctx_init", "a photo of a"), self.n_ctx_t)
        self.ctx = nn.Parameter(ctx)
        self.deep_text = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_ctx_t, self.txt_width, device=dev) * 0.02) for _ in range(self.depth_t - 1)]
        )
        self.vis_ctx = nn.Parameter(torch.randn(self.n_ctx_v, self.vis_width, device=dev) * 0.02)
        self.deep_vis = nn.ParameterList(
            [nn.Parameter(torch.randn(self.n_ctx_v, self.vis_width, device=dev) * 0.02) for _ in range(self.depth_v - 1)]
        )
        self._gpa_sum: dict[str, torch.Tensor] | None = None
        self._gpa_wsum = 0.0

    # ---------------------------------------------------------------- label space
    def _on_class_space(self, space):
        self.build_prompt_tokens(self.prefix_text, self.n_ctx_t, space.display)
        teach = []
        for d in space.display:  # textual diversity: average of the frozen features over templates
            t = self.encode_texts_frozen([tpl.format(d) for tpl in space.templates])
            teach.append(t.mean(0))
        self._set_buf("teacher_text", torch.stack(teach))

    # ---------------------------------------------------------------- towers
    def text_features(self):
        c = self.token_prefix.shape[0]
        x = torch.cat([self.token_prefix, self.ctx.unsqueeze(0).expand(c, -1, -1), self.token_suffix], dim=1)

        def before(i, x):
            if 0 < i < self.depth_t:
                deep = self.deep_text[i - 1].unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([x[:, :1], deep.to(x.dtype), x[:, 1 + self.n_ctx_t :]], dim=1)
            return x

        return self.run_text(x, self.tokenized.argmax(dim=-1), before=before)

    def image_features(self, images, topk_idx):
        b = images.shape[0]

        def before(i, x):
            if 0 < i < self.depth_v:
                deep = self.deep_vis[i - 1].unsqueeze(0).expand(b, -1, -1)
                x = torch.cat([x[:, : -self.n_ctx_v], deep.to(x.dtype)], dim=1)
            return x

        tokens, alphas = self.run_visual(
            images, topk_idx=topk_idx, extra_tokens=self.vis_ctx.unsqueeze(0).expand(b, -1, -1), before=before
        )
        return F.normalize(self.pool_visual(tokens).float(), dim=-1), alphas

    def forward(self, images, labels=None):
        frozen = self.frozen_image_features(images)
        topk = self.retrieve_topk(frozen) if self.plugin is not None else None
        z, alphas = self.image_features(images, None if topk is None else topk[0])
        t = self.text_features()
        scale = self.logit_scale()
        logits = scale * z @ t.T
        out = {"logits": logits, "image_features": z, "alpha": alphas}
        if labels is not None:
            ce = F.cross_entropy(logits, labels)
            if self.training:
                zs_logits = scale * frozen @ F.normalize(self.teacher_text, dim=-1).T
                l_text = F.l1_loss(t, self.teacher_text, reduction="mean") * self.w_text
                l_img = F.l1_loss(z, frozen, reduction="mean") * self.w_img
                l_kl = F.kl_div(
                    F.log_softmax(logits, dim=1), F.log_softmax(zs_logits, dim=1), reduction="sum", log_target=True
                ) / logits.numel()
                out["loss"] = ce + l_text + l_img + l_kl
                out["loss_parts"] = {"ce": ce.item(), "scl_text": l_text.item(), "scl_image": l_img.item(), "scl_logits": l_kl.item()}
            else:
                out["loss"] = ce
        return self.finish(out, labels, topk)

    # ---------------------------------------------------------------- GPA
    def _prompt_state(self) -> dict[str, torch.Tensor]:
        names = ("ctx", "vis_ctx", "deep_text", "deep_vis")
        return {n: p.detach().clone() for n, p in self.named_parameters() if n.split(".")[0] in names}

    def on_epoch_end(self, epoch: int, n_epochs: int) -> None:
        mean = float(self.gpa_mean) if self.gpa_mean is not None else max(1.0, 0.75 * n_epochs)
        e = epoch + 1  # 1-based like the official code
        w = math.exp(-((e - mean) ** 2) / (2 * self.gpa_std**2)) / (self.gpa_std * math.sqrt(2 * math.pi))
        state = self._prompt_state()
        if self._gpa_sum is None:
            self._gpa_sum = {k: torch.zeros_like(v) for k, v in state.items()}
        for k, v in state.items():
            self._gpa_sum[k] += w * v
        self._gpa_wsum += w

    def on_train_end(self) -> None:
        if not self._gpa_sum or self._gpa_wsum <= 1e-12:
            return  # e.g. run shorter than the Gaussian's support: keep the last weights
        params = dict(self.named_parameters())
        with torch.no_grad():
            for k, v in self._gpa_sum.items():
                params[k].copy_(v / self._gpa_wsum)
        self._gpa_sum, self._gpa_wsum = None, 0.0
