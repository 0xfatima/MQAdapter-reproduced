"""Stand-alone MQAdapter on frozen CLIP (paper adapter + Eq. 16; no prompt-learning host)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa import DEFAULT_ADAPT_LAYERS, DEFAULT_GAMMA, N_QUBITS
from mqa.models.adapter import expected_adapter_param_count
from mqa.models.clip_backbone import (
    freeze_module,
    run_block,
    transformer_batch_first,
    visual_pool,
    visual_prefix,
)
from mqa.models.plugin import MQAdapterPlugin


class ActiveClassMixin:
    """Lets one model be evaluated on a subset of classes (base-to-new protocol)."""

    def _init_classes(self, text_features: torch.Tensor):
        self.register_buffer("all_text_features", F.normalize(text_features.float(), dim=-1), persistent=False)
        self.register_buffer("text_features", self.all_text_features.clone(), persistent=False)
        self.active_classes = list(range(text_features.shape[0]))

    def set_active_classes(self, indices: Sequence[int]) -> None:
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=self.all_text_features.device)
        self.active_classes = [int(i) for i in indices]
        self.text_features = self.all_text_features.index_select(0, idx)

    def n_classes(self) -> int:
        return int(self.text_features.shape[0])


class MQAdapterCLIP(ActiveClassMixin, nn.Module):
    """Coarse Top-K retrieval (frozen CLIP) + fine MQAdapter on visual class tokens.

    Logits: logit_scale * cos(z^v, z^t_i) over all active classes (Eq. 16).
    """

    def __init__(
        self,
        clip_model: nn.Module,
        text_features: torch.Tensor,
        k: int = 4,
        n_qubits: int = N_QUBITS,
        gamma: float = DEFAULT_GAMMA,
        adapt_layers: Sequence[int] = DEFAULT_ADAPT_LAYERS,
        quantum_map: str = "vqc",
        crz_dir: str = "control_first",
        freeze_clip: bool = True,
        freeze_logit_scale: bool = True,
        mlp_hidden: int = 8,
        vqc_init: str = "zeros",
        interaction: str = "quantum",
        vqc_sharing: str = "shared",
    ):
        super().__init__()
        self.clip = clip_model
        self.k_requested = int(k)
        self.n_qubits = int(n_qubits)
        n_classes = text_features.shape[0]
        if not (1 <= self.k_requested <= n_classes):
            raise ValueError(f"K={self.k_requested} is incompatible with C={n_classes}")
        self._init_classes(text_features)
        visual = clip_model.visual
        width = int(getattr(getattr(visual, "transformer", visual), "width", visual.conv1.out_channels))
        self.visual_width = width
        self.text_dim = int(text_features.shape[-1])
        self.batch_first = transformer_batch_first(visual.transformer)
        n_blocks = len(visual.transformer.resblocks)
        self.plugin = MQAdapterPlugin(
            visual_width=width,
            text_dim=self.text_dim,
            n_blocks=n_blocks,
            n_qubits=n_qubits,
            gamma=gamma,
            adapt_layers=adapt_layers,
            quantum_map=quantum_map,
            crz_dir=crz_dir,
            mlp_hidden=mlp_hidden,
            vqc_init=vqc_init,
            interaction=interaction,
            vqc_sharing=vqc_sharing,
        )
        self.adapt_layers = self.plugin.adapt_layers
        if freeze_clip:
            freeze_module(self.clip)
        if freeze_logit_scale and hasattr(self.clip, "logit_scale"):
            self.clip.logit_scale.requires_grad = False
        self._expected_params = expected_adapter_param_count(
            n_layers=len(self.adapt_layers),
            visual_dim=width,
            text_dim=self.text_dim,
            n_qubits=n_qubits,
            vqc_params=sum(p.numel() for p in self.plugin.vqc.parameters()),
        )

    @property
    def k(self) -> int:
        return min(self.k_requested, self.n_classes())

    @property
    def k_equals_c(self) -> bool:
        return self.k == self.n_classes()

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # frozen backbone never leaves eval mode
        return self

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_clip_parameter_count(self) -> int:
        return sum(p.numel() for p in self.clip.parameters())

    @torch.no_grad()
    def retrieve_topk(self, images: torch.Tensor):
        """Frozen CLIP similarities -> Top-K class indices (paper Section IV-A)."""
        img = F.normalize(self.clip.encode_image(images).float(), dim=-1)
        sim = img @ self.text_features.T
        vals, idx = sim.topk(self.k, dim=-1)
        return idx, vals, sim

    def encode_image_adapted(self, images: torch.Tensor, idx: torch.Tensor):
        visual = self.clip.visual
        tokens, _ = visual_prefix(visual, images)  # [N, L, D]
        p_all = self.plugin.encode_anchors(self.text_features)  # [C, N]
        alphas = {}
        for i, blk in enumerate(visual.transformer.resblocks):
            tokens = run_block(blk, tokens, self.batch_first)
            tokens, alpha = self.plugin.apply(i, tokens, p_all, idx)
            if alpha is not None:
                alphas[i] = alpha
        return visual_pool(visual, tokens), {"alpha": alphas, "idx": idx}

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        idx, vals, sim_frozen = self.retrieve_topk(images)
        z, extra = self.encode_image_adapted(images, idx)
        z = F.normalize(z.float(), dim=-1)
        scale = self.clip.logit_scale.exp() if hasattr(self.clip, "logit_scale") else torch.tensor(100.0, device=z.device)
        logits = scale * (z @ self.text_features.T)
        out = {
            "logits": logits,
            "image_features": z,
            "topk_idx": idx,
            "topk_sim": vals,
            "frozen_sim": sim_frozen,
            "alpha": extra["alpha"],
        }
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
            out["retrieval_hit"] = (idx == labels.unsqueeze(-1)).any(dim=-1)
        return out

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)["logits"].argmax(dim=-1)
