"""TorchQuantum backend vs the NumPy oracle. Skipped when torchquantum is not installed."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torchquantum", reason="torchquantum not installed; `pip install torchquantum`")

from mqa.quantum.reference_paper_vqc import paper_vqc_expvals  # noqa: E402
from mqa.quantum.tq_vqc import TorchQuantumVQC  # noqa: E402
from mqa.quantum.torch_vqc import PaperVQC  # noqa: E402


@pytest.mark.parametrize("crz_dir", ["control_first", "target_first"])
def test_tq_matches_oracle_and_statevector(crz_dir):
    torch.manual_seed(0)
    tqv = TorchQuantumVQC(8, crz_dir=crz_dir, init="small")
    ref = PaperVQC(8, crz_dir=crz_dir, init="zeros")
    with torch.no_grad():
        for name in ("phi1", "crz", "phi2"):
            getattr(ref, name).copy_(getattr(tqv, name))
    x = torch.rand(4, 8) * 2 - 1
    y_tq = tqv(x).detach().numpy()
    y_sv = ref(x).detach().numpy()
    y_np = paper_vqc_expvals(
        x.numpy().astype(np.float64),
        tqv.phi1.detach().numpy().astype(np.float64),
        tqv.crz.detach().numpy().astype(np.float64),
        tqv.phi2.detach().numpy().astype(np.float64),
        crz_dir,
    )
    assert np.max(np.abs(y_sv - y_np)) < 1e-4
    assert np.max(np.abs(y_tq - y_np)) < 1e-4, "TorchQuantum disagrees with the equation-level oracle"


def test_tq_gradients_reach_all_parameters():
    tqv = TorchQuantumVQC(8, init="small")
    x = torch.randn(3, 8, requires_grad=True)
    (tqv(x) ** 2).sum().backward()
    for p in (tqv.phi1, tqv.crz, tqv.phi2):
        assert p.grad is not None and p.grad.abs().sum() > 0
