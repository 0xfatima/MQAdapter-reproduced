"""Independent numpy statevector oracle of the MQAdapter VQC exactly as written in the paper (Eqs. 5-8).

    |psi(x)>  = (x)_n Ry(x_n)|0>                      (angle encoding, Eq. 5)
    U(phi)    = (x)_n Ry(phi2_n) . prod_{(n,m) in E} CRZ(phi_nm) . (x)_n Ry(phi1_n)     (Eq. 6)
    f(x)      = [<Z_1>, ..., <Z_N>]  on U(phi)|psi(x)>  (Eqs. 7-8)
    E         = {(n,n+1)} U {(N,1)}   (ring: N edges)  => trainable parameters = 3N (24 for N=8)

Used ONLY as a test oracle / for analysis; not for training. Wire i == tensor axis i (wire 0 most significant).
CRZ(c->t, theta) = |0><0|(x)I + |1><1|(x)RZ(theta), RZ(theta)=exp(-i theta Z/2). The paper does not state which ring
endpoint is control vs target; `crz_dir` selects the reading ("control_first": (n, n+1) = control n; "target_first": swapped).
"""
from __future__ import annotations

import numpy as np


def ring_edges(n: int):
    return [(i, (i + 1) % n) for i in range(n)]


def paper_param_count(n: int) -> int:
    return n + len(ring_edges(n)) + n  # phi1 + CRZ ring + phi2


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply1(state, gate, wire):
    return np.moveaxis(np.tensordot(gate, state, axes=([1], [wire])), 0, wire)


def _crz(state, control, target, theta):
    out = state.copy()
    sl = [slice(None)] * state.ndim
    sl[control] = 1
    sub = out[tuple(sl)]
    t_axis = target - (1 if target > control else 0)
    shape = [1] * sub.ndim
    shape[t_axis] = 2
    out[tuple(sl)] = sub * np.exp(np.array([-0.5j * theta, 0.5j * theta])).reshape(shape)
    return out


def expval(state, wire, basis="Z"):
    psi = np.moveaxis(state, wire, 0).reshape(2, -1)
    a, b = psi[0], psi[1]
    if basis == "Z":
        return float(np.sum(np.abs(a) ** 2) - np.sum(np.abs(b) ** 2))
    cross = np.sum(np.conj(a) * b)
    return float(2 * cross.real) if basis == "X" else float(2 * cross.imag)


def encode(x):
    n = len(x)
    st = np.zeros((2,) * n, dtype=np.complex128)
    st[(0,) * n] = 1.0
    for i in range(n):
        st = _apply1(st, _ry(x[i]), i)
    return st


def final_state(x, phi1, crz, phi2, crz_dir="control_first"):
    n = len(x)
    st = encode(x)
    for i in range(n):
        st = _apply1(st, _ry(phi1[i]), i)
    for e, (a, b) in enumerate(ring_edges(n)):
        c, t = (a, b) if crz_dir == "control_first" else (b, a)
        st = _crz(st, c, t, crz[e])
    for i in range(n):
        st = _apply1(st, _ry(phi2[i]), i)
    return st


def paper_vqc_expvals(x, phi1, crz, phi2, crz_dir="control_first"):
    """x: [N] or [B,N] -> [N] or [B,N] of <Z_n>."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return np.stack([paper_vqc_expvals(r, phi1, crz, phi2, crz_dir) for r in x])
    st = final_state(x, phi1, crz, phi2, crz_dir)
    return np.array([expval(st, i, "Z") for i in range(len(x))])
