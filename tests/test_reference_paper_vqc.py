"""numpy-only: properties of the paper's circuit (Eqs. 5-8) that the implementation tests will rely on."""
import numpy as np

from mqa.quantum.reference_paper_vqc import (encode, expval, final_state, paper_param_count, paper_vqc_expvals,
                                              ring_edges, _apply1, _crz, _ry)


def _rand(n, seed=0):
    r = np.random.default_rng(seed)
    return r.uniform(-1, 1, n), r.normal(size=n), r.normal(size=n), r.normal(size=n)


def test_parameter_count_is_3N_and_24_for_8_qubits():
    assert paper_param_count(8) == 24 and len(ring_edges(8)) == 8
    assert ring_edges(4) == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_state_normalised_and_outputs_bounded():
    x, p1, c, p2 = _rand(8)
    st = final_state(x, p1, c, p2)
    assert abs(np.linalg.norm(st) - 1) < 1e-12
    y = paper_vqc_expvals(x, p1, c, p2)
    assert y.shape == (8,) and np.all(np.abs(y) <= 1 + 1e-9)
    assert paper_vqc_expvals(np.stack([x, x]), p1, c, p2).shape == (2, 8)


def test_phi1_is_redundant_with_encoding_angle():
    # Ry(phi1) after Ry(x) == Ry(x + phi1): phi1 only shifts the input angle.
    x, p1, c, p2 = _rand(6, 1)
    a = paper_vqc_expvals(x, p1, c, p2)
    b = paper_vqc_expvals(x + p1, np.zeros(6), c, p2)
    assert np.allclose(a, b, atol=1e-12)


def test_crz_cannot_affect_Z_outputs_when_phi2_is_zero():
    # CRZ is diagonal => commutes with Z measurement => output = cos(x + phi1) exactly.
    x, p1, c, _ = _rand(6, 2)
    y = paper_vqc_expvals(x, p1, c, np.zeros(6))
    assert np.allclose(y, np.cos(x + p1), atol=1e-12)


def test_crz_matches_analytic_phases_and_fixes_control_target_convention():
    # |+>_c |+>_t (Ry(pi/2)|0> = |+>); CRZ(c->t, th): <X_t>=(1+cos th)/2, <Y_t>=sin(th)/2, <X_c>=cos(th/2)
    th = 0.9
    st = encode(np.array([np.pi / 2, np.pi / 2]))
    st = _crz(st, 0, 1, th)
    assert abs(expval(st, 1, "X") - (1 + np.cos(th)) / 2) < 1e-12
    assert abs(expval(st, 1, "Y") - np.sin(th) / 2) < 1e-12
    assert abs(expval(st, 0, "X") - np.cos(th / 2)) < 1e-12
    # CRZ on the |1>_c branch only: control=|0> leaves target untouched
    st0 = _crz(encode(np.array([0.0, np.pi / 2])), 0, 1, th)
    assert abs(expval(st0, 1, "X") - 1) < 1e-12


def test_crz_direction_readings_differ_when_phi2_nonzero():
    x, p1, c, p2 = _rand(5, 3)
    a = paper_vqc_expvals(x, p1, c, p2, "control_first")
    b = paper_vqc_expvals(x, p1, c, p2, "target_first")
    assert np.max(np.abs(a - b)) > 1e-3  # the ambiguity is observable => torchquantum parity test can resolve it
