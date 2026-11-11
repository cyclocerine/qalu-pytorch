"""
Q-ALU: pure NumPy reference implementation.

This module is the mathematical ground truth of the package: every PyTorch
class in ``qalu.core`` must reproduce these numbers bit-for-bit up to FP32
rounding (``tests/test_qalu.py::test_torch_matches_numpy``). It has no
dependency on PyTorch, so it also runs where torch is unavailable.

Conventions (defaults used throughout the paper):
    alpha = 1.0, beta = 1.0, gamma = 1.5, delta = 0.5
"""

import numpy as np

# Coefficients of the [4/4] Pade-type surrogate of 1 - exp(-u); see FastQALU.
_FAST_C2 = 0.55
_FAST_C3 = 0.05
_FAST_C4 = 0.10


def fast_neg_shape(u):
    """Rational surrogate r(u) ~= 1 - exp(-u) for u >= 0.

    r(0) = 0, r'(0) = 1, r(inf) = 1, denominator >= 1 for every u >= 0.
    Evaluated as n / (1 + n) rather than 1 - 1 / (1 + n): the latter loses up
    to 4.6% relative accuracy in FP32 for u ~ 1e-6 through cancellation.
    """
    u = np.asarray(u, dtype=np.float64)
    n = u * (1.0 + u * (_FAST_C2 + u * (_FAST_C3 + _FAST_C4 * u)))
    return n / (1.0 + n)


class QALU:
    """f(x) = alpha*x*(1 + tanh(x/beta)) for x >= 0, -gamma*x*expm1(x/delta) for x < 0."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.5, delta=0.5):
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_pos = np.maximum(x, 0.0)
        x_neg = np.minimum(x, 0.0)
        pos = self.alpha * x_pos * (1.0 + np.tanh(x_pos / self.beta))
        neg = -self.gamma * x_neg * np.expm1(x_neg / self.delta)
        return np.where(x >= 0.0, pos, neg)

    def __call__(self, x):
        return self.forward(x)


class QALU2:
    """C1-continuous variant: alpha*x*(tanh + tanh^2) for x >= 0, gamma = 2*alpha*(1+risk_premium)."""

    def __init__(self, alpha=1.0, beta=1.0, risk_premium=0.25, delta=0.5):
        self.alpha, self.beta = alpha, beta
        self.risk_premium = risk_premium
        self.gamma = 2.0 * alpha * (1.0 + risk_premium)
        self.delta = delta

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        t = np.tanh(np.maximum(x, 0.0) / self.beta)
        pos = self.alpha * np.maximum(x, 0.0) * (t + t**2)
        neg = -self.gamma * np.minimum(x, 0.0) * np.expm1(np.minimum(x, 0.0) / self.delta)
        return np.where(x >= 0.0, pos, neg)

    def __call__(self, x):
        return self.forward(x)


class QALU_Ultra:
    """Volatility- and fee-conditioned variant: beta/delta widen with vol_ratio."""

    def __init__(self, alpha=1.0, beta=1.0, delta=0.5,
                 fee_threshold=0.01, loss_aversion=2.25):
        self.alpha, self.beta, self.delta = alpha, beta, delta
        self.fee_threshold = float(fee_threshold)
        self.loss_aversion = loss_aversion
        self.gamma = 2.0 * alpha * (1.0 + loss_aversion)

    def forward(self, x, vol_ratio=1.0):
        x = np.asarray(x, dtype=np.float64)
        eff_beta = (self.beta + self.fee_threshold) * vol_ratio
        eff_delta = (self.delta + self.fee_threshold) * vol_ratio
        t = np.tanh(np.maximum(x, 0.0) / eff_beta)
        pos = self.alpha * np.maximum(x, 0.0) * (t + t**2)
        neg = -self.gamma * np.minimum(x, 0.0) * np.expm1(np.minimum(x, 0.0) / eff_delta)
        return np.where(x >= 0.0, pos, neg)

    def __call__(self, x, vol_ratio=1.0):
        return self.forward(x, vol_ratio)


class FastQALU:
    """Transcendental-free surrogate of QALU (see paper, Eq. 12)."""

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.5, delta=0.5):
        self.alpha, self.beta = float(alpha), float(beta)
        self.gamma, self.delta = float(gamma), float(delta)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_pos = np.maximum(x, 0.0)
        x_neg = np.minimum(x, 0.0)
        u_p = x_pos / self.beta
        t_p = u_p / np.sqrt(1.0 + u_p * u_p)
        pos = self.alpha * x_pos * (1.0 + t_p)
        neg = self.gamma * x_neg * fast_neg_shape(-x_neg / self.delta)
        return np.where(x >= 0.0, pos, neg)

    def __call__(self, x):
        return self.forward(x)


QALU_Adaptive = QALU_Ultra
