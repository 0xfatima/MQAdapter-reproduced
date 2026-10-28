"""MQAdapter as a host-agnostic plug-in (paper Section IV-C, Eqs. 11-15).

A host (frozen CLIP, MaPLe, ...) runs its own visual transformer and calls
`plugin.apply(layer_idx, tokens, anchors, topk_idx)` after each block. The plug-in only ever rewrites
the class token (index 0), exactly as in Eq. 15, and returns the other tokens untouched.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mqa import DEFAULT_ADAPT_LAYERS, DEFAULT_GAMMA, N_QUBITS
from mqa.models.adapter import MQAdapterLayer
from mqa.quantum.torch_vqc import QuantumMap


class MQAdapterPlugin(nn.Module):
    def __init__(
        self,
        visual_width: int,
        text_dim: int,
        n_blocks: int,
        n_qubits: int = N_QUBITS,
        gamma: float = DEFAULT_GAMMA,
        adapt_layers: Sequence[int] = DEFAULT_ADAPT_LAYERS,
        quantum_map: str = "vqc",
        crz_dir: str = "control_first",
        mlp_hidden: int = 8,
        vqc_init: str = "zeros",
        interaction: str = "quantum",
        vqc_sharing: str = "shared",
    ):
        super().__init__()
        if vqc_sharing not in ("shared", "separate"):
            raise ValueError(f"vqc_sharing must be shared|separate, got {vqc_sharing}")
        self.n_qubits = int(n_qubits)
        self.adapt_layers = tuple(int(i) for i in adapt_layers)
        self.interaction = interaction
        self.vqc_sharing = vqc_sharing

        def make():
            return QuantumMap(kind=quantum_map, n_qubits=n_qubits, crz_dir=crz_dir, mlp_hidden=mlp_hidden, vqc_init=vqc_init)

        self.vqc = make()
        if vqc_sharing == "separate":  # Table V ablation: one circuit per role instead of one shared circuit
            self.vqc_text = make()
            self.vqc_fused = make()
        self.text_proj = nn.Linear(text_dim, n_qubits, bias=False)  # W^t, shared across layers
        self.layers = nn.ModuleDict()
        for i in self.adapt_layers:
            if i < 0 or i >= n_blocks:
                raise ValueError(f"adapt layer {i} outside [0, {n_blocks})")
            self.layers[str(i)] = MQAdapterLayer(visual_width, n_qubits, gamma, layer_id=i, interaction=interaction)

    def encode_anchors(self, text_features: torch.Tensor) -> torch.Tensor:
        """Eq. 11-12 for all C class embeddings at once -> [C, N].

        Each vector goes through the circuit independently, so gathering Top-K afterwards is
        exactly equivalent to encoding only the K retrieved vectors.
        """
        p_bar = self.text_proj(text_features)
        if self.interaction == "cross_attention":
            return p_bar
        return getattr(self, "vqc_text", self.vqc)(p_bar)

    def apply(self, layer_idx: int, tokens_nld: torch.Tensor, p_all: torch.Tensor, topk_idx: torch.Tensor):
        """Return (tokens, alpha). tokens are [N, L, D]; no-op if layer_idx is not adapted."""
        key = str(layer_idx)
        if key not in self.layers:
            return tokens_nld, None
        p = p_all[topk_idx]  # [B, K, N]
        new_cls, alpha = self.layers[key](
            tokens_nld[:, 0],
            None,
            self.vqc,
            self.text_proj,
            precomputed_p_tilde=p,
            vqc_text=getattr(self, "vqc_text", None),
            vqc_fused=getattr(self, "vqc_fused", None),
        )
        tokens = torch.cat([new_cls.unsqueeze(1), tokens_nld[:, 1:]], dim=1)  # out-of-place: autograd-safe
        return tokens, alpha

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
