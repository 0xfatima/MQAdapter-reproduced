"""Differentiable statevector VQC matching Eqs. 5–8 and the NumPy oracle.

TorchQuantum is optional. This module is the training implementation: it is a real
8-qubit statevector circuit (256 amplitudes), not a classical MLP.
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from mqa.quantum.reference_paper_vqc import paper_param_count, ring_edges

CRZDir = Literal["control_first", "target_first"]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def apply_ry(state: torch.Tensor, theta: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply batched Ry(theta) to `wire`. state: [B, 2, 2, ..., 2], theta: [B]."""
    # Ry = [[c, -s], [s, c]], c=cos(θ/2), s=sin(θ/2)
    half = theta.to(dtype=_real_dtype(state.dtype)) * 0.5
    c = torch.cos(half)
    s = torch.sin(half)
    moved = state.movedim(wire + 1, 1)
    s0, s1 = moved[:, 0], moved[:, 1]
    view = (theta.shape[0],) + (1,) * (s0.dim() - 1)
    c = c.view(view).to(dtype=s0.dtype)
    s = s.view(view).to(dtype=s0.dtype)
    n0 = c * s0 - s * s1
    n1 = s * s0 + c * s1
    return torch.stack((n0, n1), dim=1).movedim(1, wire + 1)


def apply_crz(state: torch.Tensor, theta: torch.Tensor, control: int, target: int) -> torch.Tensor:
    """CRZ(control→target, θ): |0⟩⟨0|⊗I + |1⟩⟨1|⊗RZ(θ), RZ=diag(e^{-iθ/2}, e^{iθ/2})."""
    n = state.dim() - 1
    bsz = state.shape[0]
    flat = state.reshape(bsz, 2**n)
    idx = torch.arange(2**n, device=state.device)
    cbit = (idx >> (n - 1 - control)) & 1
    tbit = (idx >> (n - 1 - target)) & 1
    half = theta.to(dtype=_real_dtype(state.dtype)).to(device=state.device).view(bsz, 1) * 0.5
    # phase = 1 (control 0); e^{-iθ/2} (control 1, target 0); e^{iθ/2} (control 1, target 1)
    c_on = cbit.bool().unsqueeze(0)
    t_on = tbit.bool().unsqueeze(0)
    angle = torch.where(c_on, torch.where(t_on, half, -half), torch.zeros_like(half))
    cd = state.dtype if state.is_complex() else _complex_dtype(state.dtype)
    phase = torch.polar(torch.ones_like(angle), angle).to(dtype=cd)
    return (flat * phase).reshape(state.shape)


def pauli_z_expvals(state: torch.Tensor) -> torch.Tensor:
    """⟨Z_w⟩ for each wire. state [B,2,...,2] -> [B,N] real."""
    n = state.dim() - 1
    outs = []
    for w in range(n):
        moved = state.movedim(w + 1, 1)
        rest = tuple(range(1, moved[:, 0].dim()))
        p0 = moved[:, 0].abs().square().sum(dim=rest)
        p1 = moved[:, 1].abs().square().sum(dim=rest)
        outs.append(p0 - p1)
    return torch.stack(outs, dim=-1).real


def run_paper_circuit(
    x: torch.Tensor,
    phi1: torch.Tensor,
    crz: torch.Tensor,
    phi2: torch.Tensor,
    crz_dir: CRZDir = "control_first",
) -> torch.Tensor:
    """x: [B, N] real angles -> [B, N] ⟨Z⟩."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    bsz, n = x.shape
    cd = _complex_dtype(x.dtype)
    state = torch.zeros((bsz,) + (2,) * n, dtype=cd, device=x.device)
    state[(slice(None),) + (0,) * n] = 1.0
    for i in range(n):
        state = apply_ry(state, x[:, i], i)
    for i in range(n):
        state = apply_ry(state, phi1[i].expand(bsz), i)
    for e, (a, b) in enumerate(ring_edges(n)):
        c, t = (a, b) if crz_dir == "control_first" else (b, a)
        state = apply_crz(state, crz[e].expand(bsz), c, t)
    for i in range(n):
        state = apply_ry(state, phi2[i].expand(bsz), i)
    out = pauli_z_expvals(state).to(dtype=_real_dtype(x.dtype))
    return out.squeeze(0) if squeeze else out


class PaperVQC(nn.Module):
    """Shared variational circuit of the paper (Eqs. 5–8). Parameters: 3N."""

    def __init__(
        self,
        n_qubits: int = 8,
        crz_dir: CRZDir = "control_first",
        init: str = "zeros",
    ):
        super().__init__()
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2 for ring entanglement")
        self.n_qubits = n_qubits
        self.crz_dir: CRZDir = crz_dir
        self.phi1 = nn.Parameter(torch.zeros(n_qubits))
        self.crz = nn.Parameter(torch.zeros(n_qubits))
        self.phi2 = nn.Parameter(torch.zeros(n_qubits))
        if init == "small":
            nn.init.uniform_(self.phi1, -0.02, 0.02)
            nn.init.uniform_(self.crz, -0.02, 0.02)
            nn.init.uniform_(self.phi2, -0.02, 0.02)
        elif init != "zeros":
            raise ValueError(f"unknown VQC init {init}")

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, crz_dir={self.crz_dir}, n_params={self.n_params}"

    @property
    def n_params(self) -> int:
        return paper_param_count(self.n_qubits)

    def grouped_parameters(self) -> dict[str, nn.Parameter]:
        return {"ry1": self.phi1, "crz": self.crz, "ry2": self.phi2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., N] -> [..., N] Pauli-Z expectations in [-1, 1]."""
        orig = x.shape
        n = self.n_qubits
        if orig[-1] != n:
            raise ValueError(f"last dim must be n_qubits={n}, got {orig}")
        x_flat = x.reshape(-1, n)
        # D15: the circuit always runs in fp32/fp64, even when the caller is inside autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_flat = x_flat.float() if x_flat.dtype in (torch.float16, torch.bfloat16) else x_flat
            out = run_paper_circuit(x_flat, self.phi1, self.crz, self.phi2, self.crz_dir)
        return out.to(dtype=_real_dtype(x.dtype)).reshape(orig)


class CosBiasMap(nn.Module):
    """Classical control: f(x)=cos(x+b). Isolates VQC entanglement (see D14)."""

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x + self.bias)


class MLPMap(nn.Module):
    """Classical control: 8→h→8 MLP. Not a quantum circuit."""

    def __init__(self, n_qubits: int = 8, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, n_qubits, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityMap(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FrozenRandomVQC(nn.Module):
    """VQC with frozen random parameters (entanglement present, not trained)."""

    def __init__(self, n_qubits: int = 8, seed: int = 0, crz_dir: CRZDir = "control_first"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.inner = PaperVQC(n_qubits=n_qubits, crz_dir=crz_dir, init="zeros")
        with torch.no_grad():
            self.inner.phi1.copy_(torch.randn(n_qubits, generator=g) * 0.5)
            self.inner.crz.copy_(torch.randn(n_qubits, generator=g) * 0.5)
            self.inner.phi2.copy_(torch.randn(n_qubits, generator=g) * 0.5)
        for p in self.inner.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


def QuantumMap(
    kind: str = "vqc",
    n_qubits: int = 8,
    crz_dir: CRZDir = "control_first",
    mlp_hidden: int = 8,
    frozen_vqc_seed: int = 0,
    vqc_init: str = "zeros",
) -> nn.Module:
    """Factory for the f_φ slot. `vqc` is the paper circuit; others are documented ablations."""
    kind = kind.lower()
    if kind == "vqc":
        return PaperVQC(n_qubits=n_qubits, crz_dir=crz_dir, init=vqc_init)
    if kind == "identity":
        return IdentityMap()
    if kind in ("cos", "cos_bias"):
        return CosBiasMap(n_qubits=n_qubits)
    if kind == "mlp":
        return MLPMap(n_qubits=n_qubits, hidden=mlp_hidden)
    if kind in ("frozen_vqc", "frozen_random_vqc"):
        return FrozenRandomVQC(n_qubits=n_qubits, seed=frozen_vqc_seed, crz_dir=crz_dir)
    if kind in ("tq", "tq_vqc", "torchquantum"):
        from mqa.quantum.tq_vqc import TorchQuantumVQC

        return TorchQuantumVQC(n_qubits=n_qubits, crz_dir=crz_dir, init=vqc_init)
    if kind == "linear":
        return nn.Sequential(nn.Linear(n_qubits, n_qubits, bias=False), nn.Tanh())
    raise ValueError(f"unknown quantum_map kind: {kind}")
