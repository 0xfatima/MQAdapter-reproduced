"""MMRL (CVPR 2025) and MMRL++ (IJCV 2026) hosts, Guo & Gu, with optional MQAdapter (paper host #3).

Implemented from the MMRL++ paper (arXiv:2505.10088), Sec. 3, Eqs. 7-34:
  * shared learnable representation space R in R^{K x d_r}, N(0, 0.02), K=5, d_r=512
  * representation aligners map R to visual (d_r->768) and textual (d_r->512) representation tokens that are
    inserted into blocks J..L (J=6, 1-indexed; 0-indexed blocks 5..11 -> 7 aligners per modality):
      vision: [c, R, E]   text: [BOS, R, T, EOS] (causal mask extended, EOS index shifted by K)
  * MMRL   : one independent linear aligner per layer; R re-inserted at every layer (Eq. 10/13);
             representation patch projection P_v^r trainable, initialised from CLIP's projection.
  * MMRL++ : SRRA (Eqs. 26-28): W_i = W_shared + A_i B_i (rank r1=4), and W_r = W_v^c(frozen) + A_r B_r (r2=64)
             PRC  (Eqs. 30-34): input of block i>J = beta * R_i + (1-beta) * O_{i-1}  (beta=0.9)
  * features: f_c = P_v^c(c_L) (frozen projection), f_r = P_v^r(mean_k R_L^v); text w = P_t(EOS)
  * loss (Eq. 22): alpha*CE(f_c) + (1-alpha)*CE(f_r) + lambda*(L_cos^v(f_c, f_0) + L_cos^t(w, w_0)), alpha=0.7
  * inference (Eqs. 23-24): base classes p = alpha*p(f_c) + (1-alpha)*p(f_r); novel classes / new tasks p(f_c)

Not checked against the official code (github.com/yunncheng/MMRL). Choices that the paper leaves open and that
are made here: ln_post is applied to every representation token before mean pooling; the LoRA factors use
A ~ Kaiming-uniform, B = 0, so training starts from W_shared / W_v^c; lambda defaults to 0.5 (the paper tunes
it per dataset, Table A1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa.hosts.base import HostBase


class LowRankAligners(nn.Module):
    """SRRA: one shared linear map + per-layer low-rank residuals A_i B_i."""

    def __init__(self, d_in: int, d_out: int, n_layers: int, rank: int):
        super().__init__()
        self.shared = nn.Linear(d_in, d_out)
        self.A = nn.ParameterList([nn.Parameter(torch.empty(d_in, rank)) for _ in range(n_layers)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(rank, d_out)) for _ in range(n_layers)])
        for a in self.A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, R: torch.Tensor, j: int) -> torch.Tensor:
        return self.shared(R) + (R @ self.A[j]) @ self.B[j]


class IndependentAligners(nn.Module):
    """MMRL: an independent linear aligner per layer."""

    def __init__(self, d_in: int, d_out: int, n_layers: int):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n_layers)])

    def forward(self, R: torch.Tensor, j: int) -> torch.Tensor:
        return self.maps[j](R)


class MMRLHost(HostBase):
    method = "mmrl"

    def __init__(self, clip_model, tokenizer, cfg):
        super().__init__(clip_model, tokenizer, cfg)
        self.n_rep = int(cfg.get("rep_tokens", 5))
        self.d_r = int(cfg.get("rep_dim", 512))
        self.J = int(cfg.get("rep_layer", 6))  # 1-indexed start layer (paper J)
        self.alpha = float(cfg.get("mmrl_alpha", 0.7))
        self.lam = float(cfg.get("mmrl_lambda", 0.5))
        self.srra = bool(cfg.get("srra", False))
        self.prc = bool(cfg.get("prc", False))
        self.beta = float(cfg.get("mmrl_beta", 0.9)) if self.prc else 1.0
        self.use_vision = bool(cfg.get("rep_vision", True))
        self.use_text = bool(cfg.get("rep_text", True))
        n_layers = min(self.n_vis_blocks, self.n_txt_blocks) - (self.J - 1)
        if n_layers < 1:
            raise ValueError(f"rep_layer J={self.J} too deep")
        self.first = self.J - 1  # first 0-indexed block that receives tokens
        self.R = nn.Parameter(torch.randn(self.n_rep, self.d_r, device=self.device) * 0.02)
        r1, r2 = int(cfg.get("rank_aligner", 4)), int(cfg.get("rank_proj", 64))
        if self.srra:
            self.align_v = LowRankAligners(self.d_r, self.vis_width, n_layers, r1)
            self.align_t = LowRankAligners(self.d_r, self.txt_width, n_layers, r1)
            self.proj_r_A = nn.Parameter(torch.empty(self.vis_width, r2))
            nn.init.kaiming_uniform_(self.proj_r_A, a=math.sqrt(5))
            self.proj_r_B = nn.Parameter(torch.zeros(r2, self.embed_dim))
        else:
            self.align_v = IndependentAligners(self.d_r, self.vis_width, n_layers)
            self.align_t = IndependentAligners(self.d_r, self.txt_width, n_layers)
            self.proj_r = nn.Parameter(self.clip.visual.proj.detach().clone().float())
        if not self.use_vision:
            for p in self.align_v.parameters():
                p.requires_grad = False
        if not self.use_text:
            for p in self.align_t.parameters():
                p.requires_grad = False

    # ---------------------------------------------------------------- label space
    def _on_class_space(self, space):
        tok = self.tokenize(space.hand_prompts)
        with torch.no_grad():
            emb = self.clip.token_embedding(tok).float()
        self._set_buf("text_emb", emb)
        self._set_buf("tokenized", tok)

    # ---------------------------------------------------------------- insertion hook shared by both towers
    def _inserter(self, aligner, batch: int, enabled: bool):
        K = self.n_rep

        def before(i, x):
            if not enabled or i < self.first:
                return x
            j = i - self.first
            Ri = aligner(self.R, j).to(x.dtype).unsqueeze(0).expand(batch, -1, -1)
            if i == self.first:
                return torch.cat([x[:, :1], Ri, x[:, 1:]], dim=1)
            tok = self.beta * Ri + (1.0 - self.beta) * x[:, 1 : 1 + K] if self.prc else Ri
            return torch.cat([x[:, :1], tok, x[:, 1 + K :]], dim=1)

        return before

    def text_features(self) -> torch.Tensor:
        c = self.text_emb.shape[0]
        before = self._inserter(self.align_t, c, self.use_text)
        shift = self.n_rep if self.use_text else 0
        return self.run_text(self.text_emb, self.tokenized.argmax(dim=-1), before=before, eot_shift=shift)

    def image_features(self, images, topk_idx):
        b = images.shape[0]
        before = self._inserter(self.align_v, b, self.use_vision)
        tokens, alphas = self.run_visual(images, topk_idx=topk_idx, before=before)
        f_c = F.normalize(self.pool_visual(tokens).float(), dim=-1)
        if not self.use_vision:
            return f_c, f_c, alphas
        rep = self.clip.visual.ln_post(tokens[:, 1 : 1 + self.n_rep]).mean(dim=1)
        if self.srra:
            f_r = rep @ self.clip.visual.proj + (rep @ self.proj_r_A) @ self.proj_r_B
        else:
            f_r = rep @ self.proj_r
        return f_c, F.normalize(f_r.float(), dim=-1), alphas

    def forward(self, images, labels=None):
        frozen = self.frozen_image_features(images)
        topk = self.retrieve_topk(frozen) if self.plugin is not None else None
        f_c, f_r, alphas = self.image_features(images, None if topk is None else topk[0])
        w = self.text_features()
        scale = self.logit_scale()
        lc, lr = scale * f_c @ w.T, scale * f_r @ w.T
        if self.training:
            logits = lc
        elif self.eval_mode == "base":  # Eq. 23: combine probabilities
            p = self.alpha * lc.softmax(-1) + (1 - self.alpha) * lr.softmax(-1)
            logits = p.clamp_min(1e-12).log()
        else:  # Eq. 24: class token only
            logits = lc
        out = {"logits": logits, "image_features": f_c, "rep_features": f_r, "alpha": alphas}
        if labels is not None:
            if self.training:
                ce_c, ce_r = F.cross_entropy(lc, labels), F.cross_entropy(lr, labels)
                cos_v = (1 - (f_c * frozen).sum(-1)).mean()
                cos_t = (1 - (w * self.anchor_features).sum(-1)).mean()
                out["loss"] = self.alpha * ce_c + (1 - self.alpha) * ce_r + self.lam * (cos_v + cos_t)
                out["loss_parts"] = {"ce_c": ce_c.item(), "ce_r": ce_r.item(), "cos_v": cos_v.item(), "cos_t": cos_t.item()}
            else:
                out["loss"] = F.nll_loss(logits, labels) if self.eval_mode == "base" else F.cross_entropy(logits, labels)
        return self.finish(out, labels, topk)


class MMRLPlusPlusHost(MMRLHost):
    method = "mmrlpp"

    def __init__(self, clip_model, tokenizer, cfg):
        cfg = dict(cfg)
        cfg.setdefault("srra", True)
        cfg.setdefault("prc", True)
        super().__init__(clip_model, tokenizer, cfg)
