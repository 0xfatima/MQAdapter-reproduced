"""The paper's shared VQC built with TorchQuantum (the library named in Section V-A2).

Same circuit and same parameter layout as `PaperVQC` (phi1, crz, phi2), so weights can be copied
between the two backends and `tests/test_tq_vqc.py` can check them against each other and the NumPy oracle.

NOT executed in the authoring sandbox (no network, no torchquantum). Run
    pytest -q tests/test_tq_vqc.py
on your machine; the test is skipped automatically when torchquantum is missing and fails loudly if
the TorchQuantum circuit disagrees with the oracle for both CRZ readings.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mqa.quantum.reference_paper_vqc import paper_param_count, ring_edges


class TorchQuantumVQC(nn.Module):
    def __init__(self, n_qubits: int = 8, crz_dir: str = "control_first", init: str = "zeros"):
        super().__init__()
        try:
            import torchquantum  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "torchquantum is not installed. `pip install torchquantum` (Python <= 3.11 recommended; "
                "on 3.12 try `pip install torchquantum-ng`) or use quantum_map='vqc' (PyTorch statevector)."
            ) from exc
        self.n_qubits = n_qubits
        self.crz_dir = crz_dir
        self.phi1 = nn.Parameter(torch.zeros(n_qubits))
        self.crz = nn.Parameter(torch.zeros(n_qubits))
        self.phi2 = nn.Parameter(torch.zeros(n_qubits))
        if init == "small":
            for p in (self.phi1, self.crz, self.phi2):
                nn.init.uniform_(p, -0.02, 0.02)
        elif init != "zeros":
            raise ValueError(f"unknown VQC init {init}")

    @property
    def n_params(self) -> int:
        return paper_param_count(self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torchquantum as tq
        import torchquantum.functional as tqf

        orig = x.shape
        n = self.n_qubits
        if orig[-1] != n:
            raise ValueError(f"last dim must be {n}, got {orig}")
        x_flat = x.reshape(-1, n)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_flat = x_flat.float()
            bsz = x_flat.shape[0]
            qdev = tq.QuantumDevice(n_wires=n, bsz=bsz, device=x.device)

            def col(v: torch.Tensor) -> torch.Tensor:  # scalar parameter -> [B, 1]
                return v.reshape(1, 1).expand(bsz, 1)

            for i in range(n):  # Eq. 5 angle encoding
                tqf.ry(qdev, wires=i, params=x_flat[:, i : i + 1])
            for i in range(n):  # Ry(phi^(1))
                tqf.ry(qdev, wires=i, params=col(self.phi1[i]))
            for e, (a, b) in enumerate(ring_edges(n)):  # ring CRZ
                c, t = (a, b) if self.crz_dir == "control_first" else (b, a)
                tqf.crz(qdev, wires=[c, t], params=col(self.crz[e]))
            for i in range(n):  # Ry(phi^(2))
                tqf.ry(qdev, wires=i, params=col(self.phi2[i]))
            out = tq.MeasureAll(tq.PauliZ)(qdev)  # Eq. 8: <Z_n> per qubit -> [B, N]
        return out.to(x.dtype if x.dtype.is_floating_point else torch.float32).reshape(orig)
