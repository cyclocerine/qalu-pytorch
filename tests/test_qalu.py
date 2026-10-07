"""
Unit Tests for Q-ALU Repository
Author: Faiq Hammam Mutaqin (2026)
SSRN: https://ssrn.com/abstract=7365098
DOI: 10.2139/ssrn.7365098
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qalu import QALU, FastQALU, QALU2

def test_origin_continuity():
    act = QALU()
    assert abs(act.forward(np.array([0.0]))[0]) < 1e-12
    print("test_origin_continuity PASSED")

def test_positive_momentum():
    act = QALU(alpha=1.0, beta=1.0)
    y_small = act.forward(np.array([0.1]))[0]
    y_large = act.forward(np.array([10.0]))[0]
    assert y_small > 0.1 # accelerating
    assert abs(y_large - 20.0) < 1e-2 # asymptotic to 2*alpha*x
    print("test_positive_momentum PASSED")

def test_negative_risk_damping():
    act = QALU(gamma=1.5, delta=0.5)
    # Small negative is damped quadratically
    y_tiny = act.forward(np.array([-0.01]))[0]
    assert abs(y_tiny) < 0.001
    # Large negative is linear cut-loss
    y_crash = act.forward(np.array([-10.0]))[0]
    assert abs(y_crash - (-15.0)) < 1e-2
    print("test_negative_risk_damping PASSED")

def test_global_monotonicity():
    act = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
    x = np.linspace(-10.0, 10.0, 2000)
    y = act.forward(x)
    dy = np.diff(y)
    assert np.all(dy >= -1e-8), "Monotonicity violation detected!"
    print("test_global_monotonicity PASSED")

def test_fastqalu_equivalence():
    q = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
    fq = FastQALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
    x = np.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0])
    y_q = q.forward(x)
    y_fq = fq.forward(x)
    # FastQALU preserves qualitative dual regime dynamics
    assert np.all(np.sign(y_q) == np.sign(y_fq))
    assert abs(y_fq[3]) < 1e-12 # origin continuity at 0
    print("test_fastqalu_equivalence PASSED")

def test_gradient_overshoot_peak():
    # Numerical derivative of negative branch at x = -2*delta
    gamma, delta = 1.5, 0.5
    x_star = -2.0 * delta # -1.0
    act = QALU(alpha=1.0, beta=1.0, gamma=gamma, delta=delta)
    eps = 1e-5
    grad_num = (act.forward(np.array([x_star + eps]))[0] - act.forward(np.array([x_star - eps]))[0]) / (2.0 * eps)
    expected_peak = gamma * (1.0 + np.exp(-2.0)) # ~1.135335 * gamma
    assert abs(grad_num - expected_peak) < 1e-3, f"Expected {expected_peak}, got {grad_num}"
    print(f"test_gradient_overshoot_peak PASSED (Peak: {grad_num:.4f} vs Asymptote: {gamma})")

def test_qalu2_c1_continuity():
    act = QALU2(alpha=1.0, beta=1.0, risk_premium=0.25, delta=0.5)
    eps = 1e-6
    d_pos = (act.forward(np.array([eps]))[0] - act.forward(np.array([0.0]))[0]) / eps
    d_neg = (act.forward(np.array([0.0]))[0] - act.forward(np.array([-eps]))[0]) / eps
    assert abs(d_pos) < 1e-4, f"Expected d_pos -> 0, got {d_pos}"
    assert abs(d_neg) < 1e-4, f"Expected d_neg -> 0, got {d_neg}"
    print("test_qalu2_c1_continuity PASSED")

def test_torch_autograd():
    try:
        import torch
    except ImportError:
        print("test_torch_autograd SKIPPED (PyTorch not installed)")
        return
    act = QALU()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = act(x)
    assert not torch.isnan(y).any()
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print("test_torch_autograd PASSED")

if __name__ == '__main__':
    test_origin_continuity()
    test_positive_momentum()
    test_negative_risk_damping()
    test_global_monotonicity()
    test_fastqalu_equivalence()
    test_gradient_overshoot_peak()
    test_qalu2_c1_continuity()
    test_torch_autograd()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
