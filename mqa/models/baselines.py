"""Comparable baselines: frozen CLIP zero-shot and a CLIP-Adapter-style MLP head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mqa.models.clip_backbone import freeze_module
from mqa.models.mqadapter_clip import ActiveClassMixin


class FrozenCLIPZeroShot(ActiveClassMixin, nn.Module):
    """Frozen CLIP cosine classifier over the same class prompts. No trainable params."""

    def __init__(self, clip_model: nn.Module, text_features: torch.Tensor):
        super().__init__()
        self.clip = clip_model
        freeze_module(self.clip)
        self._init_classes(text_features)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        img = F.normalize(self.clip.encode_image(images).float(), dim=-1)
        scale = self.clip.logit_scale.exp() if hasattr(self.clip, "logit_scale") else torch.tensor(100.0, device=img.device)
        logits = scale * (img @ self.text_features.T)
        out = {"logits": logits, "image_features": img}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out

    def trainable_parameter_count(self) -> int:
        return 0


class CLIPAdapterHead(ActiveClassMixin, nn.Module):
    """CLIP-Adapter-style visual MLP on the *final* embedding (Gao et al. 2024).

    This is NOT MQAdapter. Residual: z' = (1-α) z + α MLP(z), then cosine CE.
    Default bottleneck matches a light adapter; document param count at runtime.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        text_features: torch.Tensor,
        bottleneck: int = 64,
        residual: float = 0.2,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip = clip_model
        if freeze_clip:
            freeze_module(self.clip)
        self._init_classes(text_features)
        dim = int(text_features.shape[-1])
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim, bias=True),
        )
        self.residual = float(residual)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> dict:
        z = self.clip.encode_image(images).float()
        z = F.normalize(z, dim=-1)
        adapted = F.normalize((1.0 - self.residual) * z + self.residual * self.mlp(z), dim=-1)
        scale = self.clip.logit_scale.exp() if hasattr(self.clip, "logit_scale") else torch.tensor(100.0, device=z.device)
        logits = scale * (adapted @ self.text_features.T)
        out = {"logits": logits, "image_features": adapted}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
