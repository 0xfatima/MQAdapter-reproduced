"""Fine-grained MQAdapter block (paper Section IV-C, Eqs. 11–15)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MQAdapterLayer(nn.Module):
    """Per-layer maps W^v_l and W_l. Shared W^t and f_phi are owned by the parent.

    interaction="quantum"         paper Eqs. 12-15 (VQC on visual token, anchors and fused feature)
    interaction="cross_attention" Table V ablation: conventional single-head cross-attention in the
                                  classical 8-d space (no VQC); query = visual token, key/value = anchors.
    """

    def __init__(self, visual_dim: int, n_qubits: int, gamma: float, layer_id: int, interaction: str = "quantum"):
        super().__init__()
        if interaction not in ("quantum", "cross_attention"):
            raise ValueError(f"unknown interaction {interaction}")
        self.layer_id = layer_id
        self.n_qubits = n_qubits
        self.gamma = float(gamma)
        self.interaction = interaction
        self.visual_proj = nn.Linear(visual_dim, n_qubits, bias=False)
        self.out_proj = nn.Linear(n_qubits, visual_dim, bias=False)
        if interaction == "cross_attention":
            self.q_proj = nn.Linear(n_qubits, n_qubits, bias=False)
            self.k_proj = nn.Linear(n_qubits, n_qubits, bias=False)
            self.v_proj = nn.Linear(n_qubits, n_qubits, bias=False)

    def forward(
        self,
        c: torch.Tensor,
        anchors: torch.Tensor | None,
        vqc: nn.Module,
        text_proj: nn.Linear,
        precomputed_p_tilde: torch.Tensor | None = None,
        vqc_text: nn.Module | None = None,
        vqc_fused: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        c: [B, d_v]; anchors: [B, K, d_t] (ignored if precomputed_p_tilde is set)
        precomputed_p_tilde: [B, K, N]. quantum mode: VQC-encoded anchors; cross_attention mode: classical P-bar.
        vqc / vqc_text / vqc_fused: visual / anchor / fused-feature circuits (all the same module when shared).
        Returns (c_hat, alpha).
        """
        n = self.n_qubits
        vqc_text = vqc_text if vqc_text is not None else vqc
        vqc_fused = vqc_fused if vqc_fused is not None else vqc
        c_bar = self.visual_proj(c)
        scale = math.sqrt(n)

        if self.interaction == "cross_attention":
            if precomputed_p_tilde is None:
                p_bar = text_proj(anchors)
            else:
                p_bar = precomputed_p_tilde
            q = self.q_proj(c_bar)
            k = self.k_proj(p_bar)
            v = self.v_proj(p_bar)
            alpha = torch.softmax(torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / scale, dim=-1)
            h = torch.bmm(alpha.unsqueeze(1), v).squeeze(1)
            return c + self.gamma * self.out_proj(h), alpha

        c_tilde = vqc(c_bar)
        if precomputed_p_tilde is None:
            bsz, k_, _ = anchors.shape
            p_bar = text_proj(anchors)
            p_tilde = vqc_text(p_bar.reshape(bsz * k_, n)).reshape(bsz, k_, n)
        else:
            p_tilde = precomputed_p_tilde
        attn_logits = torch.bmm(p_tilde, c_tilde.unsqueeze(-1)).squeeze(-1) / scale
        alpha = torch.softmax(attn_logits, dim=-1)
        h = torch.bmm(alpha.unsqueeze(1), p_tilde).squeeze(1)
        h_q = vqc_fused(h)
        return c + self.gamma * self.out_proj(h_q), alpha


def expected_adapter_param_count(
    n_layers: int,
    visual_dim: int = 768,
    text_dim: int = 512,
    n_qubits: int = 8,
    vqc_params: int = 24,
    include_text_proj: bool = True,
) -> int:
    """Bias-free 768→8 and 8→768 per layer + shared 512→8 + VQC.

    For 6 layers this is 77,848 ≈ 0.078M under D2/D5 — counted at runtime, not hard-coded.
    """
    per_layer = visual_dim * n_qubits + n_qubits * visual_dim
    wt = text_dim * n_qubits if include_text_proj else 0
    return n_layers * per_layer + wt + vqc_params
