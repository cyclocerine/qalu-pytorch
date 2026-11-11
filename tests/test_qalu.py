"""
Unit Tests for Q-ALU Repository
Author: Faiq Hammam Mutaqin (2026)
SSRN: https://ssrn.com/abstract=7365098
DOI: 10.2139/ssrn.7365098

Runs under pytest, or directly (``python tests/test_qalu.py``) with NumPy only.
The mathematical assertions run against ``qalu.reference`` (always available);
the PyTorch classes are tested separately and skipped when torch is absent.
"""

import math
import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from qalu import reference as ref
from qalu.reference import QALU, FastQALU, QALU2, QALU_Ultra

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Default parameters used throughout the paper.
P = dict(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
X = np.linspace(-10.0, 10.0, 40001)


def test_origin_continuity():
    for act in (QALU(**P), QALU2(), QALU_Ultra(), FastQALU(**P)):
        assert abs(float(act.forward(np.array([0.0]))[0])) < 1e-12
    print("test_origin_continuity PASSED")


def test_positive_momentum():
    act = QALU(alpha=1.0, beta=1.0)
    y_small = act.forward(np.array([0.1]))[0]
    y_large = act.forward(np.array([10.0]))[0]
    assert y_small > 0.1  # accelerating above the linear envelope
    assert abs(y_large - 20.0) < 1e-2  # asymptotic to 2*alpha*x
    print("test_positive_momentum PASSED")


def test_negative_risk_damping():
    act = QALU(gamma=1.5, delta=0.5)
    y_tiny = act.forward(np.array([-0.01]))[0]
    assert abs(y_tiny) < 0.001  # quadratic dead zone
    y_crash = act.forward(np.array([-10.0]))[0]
    assert abs(y_crash - (-15.0)) < 1e-2  # linear cut-loss tail, slope gamma
    print("test_negative_risk_damping PASSED")


def test_global_monotonicity():
    for act in (QALU(**P), QALU2(), QALU_Ultra(), FastQALU(**P)):
        dy = np.diff(act.forward(X))
        assert np.all(dy >= -1e-8), f"Monotonicity violation in {type(act).__name__}"
    print("test_global_monotonicity PASSED")


def test_fastqalu_negative_accuracy():
    """FastQALU must track the exact branch, not merely share its sign."""
    q, fq = QALU(**P), FastQALU(**P)
    x = -np.linspace(1e-3, 20.0, 20000)
    err = np.max(np.abs(fq.forward(x) - q.forward(x)) / np.abs(q.forward(x)))
    assert err < 6e-3, f"FastQALU negative branch relative error {err:.2%} > 0.6%"
    print(f"test_fastqalu_negative_accuracy PASSED (max rel err {err * 100:.3f}%)")


def test_fastqalu_positive_accuracy():
    """The rsqrt surrogate of tanh is coarser; bound it explicitly."""
    q, fq = QALU(**P), FastQALU(**P)
    x = np.linspace(1e-3, 20.0, 20000)
    err = np.max(np.abs(fq.forward(x) - q.forward(x)) / np.abs(q.forward(x)))
    assert err < 8e-2, f"FastQALU positive branch relative error {err:.2%} > 8%"
    print(f"test_fastqalu_positive_accuracy PASSED (max rel err {err * 100:.3f}%)")


def test_fastqalu_preserves_gradient_overshoot():
    """The surrogate must keep the downside overshoot, not just monotonicity."""
    fq = FastQALU(**P)
    x = -np.linspace(0.5, 4.0, 200001)
    grad = np.gradient(fq.forward(x), x)
    peak = grad.max()
    peak_at = x[np.argmax(grad)]
    expected = P["gamma"] * (1.0 + math.exp(-2.0))
    assert abs(peak - expected) / expected < 5e-3, f"peak {peak:.4f} vs {expected:.4f}"
    assert abs(peak_at + 2.0 * P["delta"]) < 0.1, f"peak at x={peak_at:.3f}, expected -1.0"
    print(f"test_fastqalu_preserves_gradient_overshoot PASSED "
          f"(peak {peak:.4f} at x={peak_at:.3f}, exact {expected:.4f} at x=-1.0)")


def test_gradient_overshoot_peak():
    gamma, delta = P["gamma"], P["delta"]
    x_star = -2.0 * delta
    act = QALU(**P)
    eps = 1e-5
    grad_num = (act.forward(np.array([x_star + eps]))[0]
                - act.forward(np.array([x_star - eps]))[0]) / (2.0 * eps)
    expected = gamma * (1.0 + math.exp(-2.0))
    assert abs(grad_num - expected) < 1e-3, f"Expected {expected}, got {grad_num}"
    print(f"test_gradient_overshoot_peak PASSED (Peak: {grad_num:.4f} vs Asymptote: {gamma})")


def test_qalu2_c1_continuity():
    act = QALU2(alpha=1.0, beta=1.0, risk_premium=0.25, delta=0.5)
    eps = 1e-6
    d_pos = (act.forward(np.array([eps]))[0] - act.forward(np.array([0.0]))[0]) / eps
    d_neg = (act.forward(np.array([0.0]))[0] - act.forward(np.array([-eps]))[0]) / eps
    assert abs(d_pos) < 1e-4, f"Expected d_pos -> 0, got {d_pos}"
    assert abs(d_neg) < 1e-4, f"Expected d_neg -> 0, got {d_neg}"
    print("test_qalu2_c1_continuity PASSED")


def test_ultra_gamma_and_volatility_conditioning():
    act = QALU_Ultra(alpha=1.0, beta=1.0, delta=0.5,
                     fee_threshold=0.01, loss_aversion=2.25)
    # gamma = 2 * alpha * (1 + loss_aversion), not 2 * alpha * loss_aversion
    assert abs(act.gamma - 6.5) < 1e-12, f"gamma = {act.gamma}"
    # Wider volatility ratio widens the dead zone => smaller |f| for a small loss
    small_loss = np.array([-0.02])
    y_calm = abs(float(act.forward(small_loss, vol_ratio=1.0)[0]))
    y_stress = abs(float(act.forward(small_loss, vol_ratio=3.0)[0]))
    assert y_stress < y_calm
    print("test_ultra_gamma_and_volatility_conditioning PASSED")


def test_parameter_roundtrip():
    """softplus(inv_softplus(v)) must reproduce v, defaults included."""
    from qalu.core import _inv_softplus
    for v in (1e-3, 0.25, 0.5, 1.0, 1.5, 2.25, 6.5, 100.0):
        raw = _inv_softplus(v)
        recovered = math.log1p(math.exp(-abs(raw))) + max(raw, 0.0)  # stable softplus
        assert abs(recovered + 1e-4 - v) < 1e-6 * max(1.0, v), f"{v} -> {recovered + 1e-4}"
    print("test_parameter_roundtrip PASSED")


def test_inv_softplus_rejects_nonpositive():
    from qalu.core import _inv_softplus
    for bad in (0.0, -1.0):
        try:
            _inv_softplus(bad)
        except ValueError:
            continue
        raise AssertionError(f"inv_softplus({bad}) should raise ValueError")
    print("test_inv_softplus_rejects_nonpositive PASSED")


def test_fast_neg_shape_has_no_cancellation():
    """1 - 1/(1+n) cancels in FP32; n/(1+n) must not."""
    f32 = np.float32
    worst_naive, worst_stable = 0.0, 0.0
    # Small-u regime only: here the naive form cancels, while the truncation
    # error of the approximant itself is O(u) < 1e-6 and cannot mask the effect.
    for uv in (1e-6, 1e-7, 1e-8):
        u = f32(uv)
        n = u * (f32(1) + u * (f32(0.55) + u * (f32(0.05) + f32(0.10) * u)))
        naive = f32(1) - f32(1) / (f32(1) + n)
        stable = n / (f32(1) + n)
        worst_naive = max(worst_naive, abs(float(naive) - uv) / uv)
        worst_stable = max(worst_stable, abs(float(stable) - uv) / uv)
    assert worst_naive > 0.01, "expected the naive form to cancel badly"
    assert worst_stable < 1e-6, worst_stable
    print(f"test_fast_neg_shape_has_no_cancellation PASSED "
          f"(FP32 worst case: stable {worst_stable:.2e} vs naive {worst_naive:.2e})")


def test_fastqalu_analytic_backward():
    """The hand-written backward must match autograd of the same expression."""
    if not HAS_TORCH:
        print("test_fastqalu_analytic_backward SKIPPED (PyTorch not installed)")
        return
    from qalu import FastQALU as TFastQALU
    torch.manual_seed(0)
    x = torch.randn(5000, dtype=torch.float64, requires_grad=True)
    TFastQALU()(x).sum().backward()

    xr = x.detach().clone().requires_grad_(True)
    xp = xr.clamp(min=0.0)
    xn = xr.clamp(max=0.0)
    up = xp / 1.0
    tp = up * torch.rsqrt(1.0 + up * up)
    pos = xp * (1.0 + tp)
    un = -xn / 0.5
    n = un * (1.0 + un * (0.55 + un * (0.05 + 0.10 * un)))
    neg = 1.5 * xn * (n / (1.0 + n))
    torch.where(xr >= 0.0, pos, neg).sum().backward()

    err = (x.grad - xr.grad).abs().max().item()
    assert err < 1e-10, err
    print(f"test_fastqalu_analytic_backward PASSED (max |analytic - autograd| = {err:.2e})")


def test_torch_autograd():
    if not HAS_TORCH:
        print("test_torch_autograd SKIPPED (PyTorch not installed)")
        return
    from qalu import QALU as TQALU
    act = TQALU()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = act(x)
    assert not torch.isnan(y).any()
    y.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    print("test_torch_autograd PASSED")


def test_torch_extreme_inputs_are_nan_free():
    if not HAS_TORCH:
        print("test_torch_extreme_inputs_are_nan_free SKIPPED (PyTorch not installed)")
        return
    import qalu.core as core
    for cls in ("QALU", "QALU2", "QALU_Ultra", "FastQALU"):
        act = getattr(core, cls)()
        x = torch.tensor([-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6], requires_grad=True)
        y = act(x)
        assert torch.isfinite(y).all(), f"{cls} forward not finite"
        y.sum().backward()
        assert torch.isfinite(x.grad).all(), f"{cls} backward not finite"
        x.grad = None
    print("test_torch_extreme_inputs_are_nan_free PASSED")


def test_torch_trainable_parameters_get_gradients():
    if not HAS_TORCH:
        print("test_torch_trainable_parameters_get_gradients SKIPPED (PyTorch not installed)")
        return
    from qalu import QALU as TQALU
    act = TQALU(trainable=True)
    x = torch.linspace(-3.0, 3.0, 128)
    act(x).pow(2).mean().backward()
    for name in ("_raw_alpha", "_raw_beta", "_raw_gamma", "_raw_delta"):
        grad = getattr(act, name).grad
        assert grad is not None and torch.isfinite(grad).all(), name
    print("test_torch_trainable_parameters_get_gradients PASSED")


def test_torch_matches_numpy():
    """The shipped PyTorch classes must reproduce the NumPy reference."""
    if not HAS_TORCH:
        print("test_torch_matches_numpy SKIPPED (PyTorch not installed)")
        return
    import qalu.core as core
    x = np.linspace(-5.0, 5.0, 1001)
    for cls in ("QALU", "QALU2", "QALU_Ultra", "FastQALU"):
        y_np = getattr(ref, cls)().forward(x)
        module = getattr(core, cls)()
        y_torch = module(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        assert np.allclose(y_np, y_torch, atol=1e-5, rtol=1e-5), cls
    print("test_torch_matches_numpy PASSED")


if __name__ == '__main__':
    test_origin_continuity()
    test_positive_momentum()
    test_negative_risk_damping()
    test_global_monotonicity()
    test_fastqalu_negative_accuracy()
    test_fastqalu_positive_accuracy()
    test_fastqalu_preserves_gradient_overshoot()
    test_gradient_overshoot_peak()
    test_qalu2_c1_continuity()
    test_ultra_gamma_and_volatility_conditioning()
    test_parameter_roundtrip()
    test_inv_softplus_rejects_nonpositive()
    test_fast_neg_shape_has_no_cancellation()
    test_torch_autograd()
    test_torch_extreme_inputs_are_nan_free()
    test_torch_trainable_parameters_get_gradients()
    test_torch_matches_numpy()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
