"""Parity and gradients: PaperVQC vs NumPy oracle (Eqs. 5–8)."""
from __future__ import annotations

import numpy as np
import torch

from mqa.quantum.reference_paper_vqc import paper_param_count, paper_vqc_expvals
from mqa.quantum.torch_vqc import PaperVQC, QuantumMap, run_paper_circuit


def _np_params(vqc: PaperVQC):
    return (
        vqc.phi1.detach().cpu().numpy().astype(np.float64),
        vqc.crz.detach().cpu().numpy().astype(np.float64),
        vqc.phi2.detach().cpu().numpy().astype(np.float64),
    )


def test_param_count_and_shape():
    vqc = PaperVQC(n_qubits=8)
    assert sum(p.numel() for p in vqc.parameters()) == paper_param_count(8) == 24
    y = vqc(torch.randn(5, 8))
    assert y.shape == (5, 8)
    assert torch.isfinite(y).all()
    assert y.abs().max() <= 1 + 1e-5


def test_parity_control_first_float64():
    torch.manual_seed(0)
    vqc = PaperVQC(n_qubits=8, crz_dir="control_first", init="small").double()
    x = (torch.rand(4, 8, dtype=torch.float64) * 2 - 1)
    y = vqc(x).detach().cpu().numpy()
    p1, cz, p2 = _np_params(vqc)
    ref = paper_vqc_expvals(x.numpy(), p1, cz, p2, "control_first")
    assert np.max(np.abs(y - ref)) < 1e-10


def test_parity_target_first():
    torch.manual_seed(1)
    vqc = PaperVQC(n_qubits=6, crz_dir="target_first", init="small").double()
    x = torch.randn(2, 6, dtype=torch.float64)
    y = vqc(x).detach().numpy()
    p1, cz, p2 = _np_params(vqc)
    ref = paper_vqc_expvals(x.numpy(), p1, cz, p2, "target_first")
    assert np.max(np.abs(y - ref)) < 1e-10


def test_phi2_zero_output_is_cos():
    n = 5
    x = torch.tensor(np.linspace(-1, 1, n), dtype=torch.float64)
    phi1 = torch.arange(n, dtype=torch.float64) * 0.1
    crz = torch.randn(n, dtype=torch.float64)
    phi2 = torch.zeros(n, dtype=torch.float64)
    y = run_paper_circuit(x.unsqueeze(0), phi1, crz, phi2)[0]
    assert torch.allclose(y, torch.cos(x + phi1), atol=1e-10)


def test_gradients_every_gate_group():
    vqc = PaperVQC(n_qubits=8, init="small")
    x = torch.randn(3, 8, requires_grad=True)
    loss = (vqc(x) ** 2).sum()
    loss.backward()
    assert vqc.phi1.grad is not None and vqc.phi1.grad.abs().sum() > 0
    assert vqc.crz.grad is not None and vqc.crz.grad.abs().sum() > 0
    assert vqc.phi2.grad is not None and vqc.phi2.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_quantum_map_factory_kinds():
    x = torch.randn(2, 8)
    for kind in ("vqc", "identity", "cos_bias", "mlp", "frozen_vqc", "linear"):
        m = QuantumMap(kind)
        y = m(x)
        assert y.shape == (2, 8)
        assert torch.isfinite(y).all()
