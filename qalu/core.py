"""
Q-ALU: Quant Asymmetric Leaky Unit: Core Implementations
Author: Faiq Hammam Mutaqin (2026)
SSRN: https://ssrn.com/abstract=7365098
DOI: 10.2139/ssrn.7365098

Contains:
- QALU: production reference implementation (NaN-free, FP32 stable)
- QALU2: C1-continuous variant (smooth origin)
- QALU_Ultra: volatility- and fee-conditioned variant
- FastQALU: transcendental-free rational surrogate

All classes exist in two flavours: a PyTorch module (autograd, trainable
parameters) and a NumPy fallback with identical mathematics, used when
PyTorch is not installed.
"""

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

# The trainable parameters are parameterised through softplus so that they stay
# strictly positive. The floor keeps delta away from 0 (which would turn x/delta
# into +-inf and resurrect the 0 * inf = NaN hazard) and is subtracted again in
# _inv_softplus so that softplus(raw) + FLOOR reproduces the requested value
# exactly instead of biasing every default by 1e-4.
_SOFTPLUS_FLOOR = 1e-4

# Saturation point of the FastQALU rational branch. For u >= 1e4 the surrogate
# equals 1.0 to within FP32 resolution, so clamping there costs no accuracy and
# prevents u^4 from overflowing to +inf (which would give inf/inf = NaN).
_FAST_U_MAX = 1e4


def _inv_softplus(value: float) -> float:
    """Inverse of ``softplus(v) + _SOFTPLUS_FLOOR`` for ``value > FLOOR``.

    ``log(exp(v) - 1)`` overflows to +inf for v > 88 in FP32 and returns NaN for
    v <= 0. The piecewise form below stays finite over the whole legal domain.
    """
    target = float(value) - _SOFTPLUS_FLOOR
    if not target > 0.0:
        raise ValueError(
            f"Q-ALU parameters must be > {_SOFTPLUS_FLOOR}, got {value!r}"
        )
    if target < 1.0:
        return math.log(math.expm1(target))
    return target + math.log1p(-math.exp(-target))


if HAS_TORCH:
    class _PositiveParamModule(nn.Module):
        """Base class wiring strictly-positive parameters through softplus."""

        def _init_params(self, trainable: bool, **values: float) -> None:
            self.trainable = trainable
            for name, value in values.items():
                raw = torch.tensor(_inv_softplus(value), dtype=torch.float32)
                if trainable:
                    setattr(self, "_raw_" + name, nn.Parameter(raw))
                else:
                    self.register_buffer("_raw_" + name, raw)

        def _positive(self, name: str) -> torch.Tensor:
            return F.softplus(getattr(self, "_raw_" + name)) + _SOFTPLUS_FLOOR

    class QALU(_PositiveParamModule):
        """
        Q-ALU:
        f(x) = alpha * x * (1 + tanh(x / beta))      for x >= 0
        f(x) = -gamma * x * expm1(x / delta)         for x < 0
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                     gamma: float = 1.5, delta: float = 0.5,
                     trainable: bool = True):
            super(QALU, self).__init__()
            self._init_params(trainable, alpha=alpha, beta=beta,
                              gamma=gamma, delta=delta)

        @property
        def alpha(self) -> torch.Tensor:
            return self._positive("alpha")

        @property
        def beta(self) -> torch.Tensor:
            return self._positive("beta")

        @property
        def gamma(self) -> torch.Tensor:
            return self._positive("gamma")

        @property
        def delta(self) -> torch.Tensor:
            return self._positive("delta")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Clamping keeps the unselected branch finite, so that
            # 0 * grad(unselected) can never become 0 * inf = NaN.
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)

            pos_term = self.alpha * x_pos * (1.0 + torch.tanh(x_pos / self.beta))
            neg_term = -self.gamma * x_neg * torch.expm1(x_neg / self.delta)
            return torch.where(x >= 0.0, pos_term, neg_term)


    class QALU2(_PositiveParamModule):
        """
        Q-ALU 2.0 (C1 continuous, smooth transition at the origin):
        f(x) = alpha * x * (tanh(x / beta) + tanh^2(x / beta))  for x >= 0
        f(x) = -gamma * x * expm1(x / delta)                    for x < 0
        where gamma = 2 * alpha * (1 + risk_premium)
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                     risk_premium: float = 0.25, delta: float = 0.5,
                     trainable: bool = True):
            super(QALU2, self).__init__()
            self._init_params(trainable, alpha=alpha, beta=beta,
                              risk_premium=risk_premium, delta=delta)

        @property
        def alpha(self) -> torch.Tensor:
            return self._positive("alpha")

        @property
        def beta(self) -> torch.Tensor:
            return self._positive("beta")

        @property
        def risk_premium(self) -> torch.Tensor:
            return self._positive("risk_premium")

        @property
        def gamma(self) -> torch.Tensor:
            return 2.0 * self.alpha * (1.0 + self.risk_premium)

        @property
        def delta(self) -> torch.Tensor:
            return self._positive("delta")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)

            u_p = x_pos / self.beta
            t_p = torch.tanh(u_p)
            pos_term = self.alpha * x_pos * (t_p + t_p * t_p)

            u_n = x_neg / self.delta
            neg_term = -self.gamma * x_neg * torch.expm1(u_n)
            return torch.where(x >= 0.0, pos_term, neg_term)


    class QALU_Ultra(_PositiveParamModule):
        """
        Q-ALU Ultra: volatility- and fee-conditioned variant.

        beta_eff  = (beta + fee_threshold) * vol_ratio
        delta_eff = (delta + fee_threshold) * vol_ratio
        gamma     = 2 * alpha * (1 + loss_aversion)

        The effective thresholds widen with the fee floor and with realised
        volatility, so the dead zone grows exactly when microstructure noise
        and transaction costs make small signals unprofitable.
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0, delta: float = 0.5,
                     fee_threshold: float = 0.01, loss_aversion: float = 2.25,
                     trainable: bool = True):
            super(QALU_Ultra, self).__init__()
            self._init_params(trainable, alpha=alpha, beta=beta, delta=delta,
                              loss_aversion=loss_aversion)
            self.fee_threshold = float(fee_threshold)

        @property
        def alpha(self) -> torch.Tensor:
            return self._positive("alpha")

        @property
        def beta(self) -> torch.Tensor:
            return self._positive("beta")

        @property
        def delta(self) -> torch.Tensor:
            return self._positive("delta")

        @property
        def loss_aversion(self) -> torch.Tensor:
            return self._positive("loss_aversion")

        @property
        def gamma(self) -> torch.Tensor:
            return 2.0 * self.alpha * (1.0 + self.loss_aversion)

        def forward(self, x: torch.Tensor, vol_ratio: float = 1.0) -> torch.Tensor:
            eff_beta = (self.beta + self.fee_threshold) * vol_ratio
            eff_delta = (self.delta + self.fee_threshold) * vol_ratio

            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)

            u_p = x_pos / eff_beta
            t_p = torch.tanh(u_p)
            pos_term = self.alpha * x_pos * (t_p + t_p * t_p)

            u_n = x_neg / eff_delta
            neg_term = -self.gamma * x_neg * torch.expm1(u_n)
            return torch.where(x >= 0.0, pos_term, neg_term)


    class _FastQALUFunction(torch.autograd.Function):
        """FastQALU with a hand-written analytic backward.

        The autograd graph of the rational branch would save n, 1+n and the
        quotient as separate tensors; the closed form below needs only x, so
        the backward pass allocates one saved tensor instead of five.
        """

        @staticmethod
        def forward(ctx, x, alpha, beta, gamma, delta):
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)

            u_p = x_pos / beta
            t_p = u_p * torch.rsqrt(1.0 + u_p * u_p)
            pos_term = alpha * x_pos * (1.0 + t_p)

            # Saturated at u = 1e4: there n ~ 1e15, so n/(1+n) is exactly 1.0 in
            # FP32 and the value equals the asymptote gamma*x. Without the clamp
            # u^4 overflows to inf for |x| > ~1e7 and inf/inf yields NaN.
            u_n = torch.clamp(-x_neg / delta, max=_FAST_U_MAX)
            n = u_n * (1.0 + u_n * (0.55 + u_n * (0.05 + 0.10 * u_n)))
            neg_term = gamma * x_neg * (n / (1.0 + n))

            ctx.save_for_backward(x)
            ctx.consts = (alpha, beta, gamma, delta)
            return torch.where(x >= 0.0, pos_term, neg_term)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            alpha, beta, gamma, delta = ctx.consts

            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)

            # d/dx [alpha*x*(1 + u/sqrt(1+u^2))] = alpha*(1 + t + u/(1+u^2)^{3/2})
            u_p = x_pos / beta
            inv_p = torch.rsqrt(1.0 + u_p * u_p)
            d_pos = alpha * (1.0 + u_p * inv_p + u_p * inv_p * inv_p * inv_p)

            # u = -x/delta, r = n/(1+n) => df/dx = gamma*(r + u*r')
            # r' = n'/(1+n)^2 = (n'/n) * r * (1-r), so with p = n/u and q = n':
            #     df/dx = gamma * (r + (q/p) * r * (1-r))
            # This form never forms n*(1+n) (which overflows in FP32 for u > ~1e5)
            # and stays finite at u = 0 because p(0) = q(0) = 1.
            u_n = torch.clamp(-x_neg / delta, max=_FAST_U_MAX)
            n = u_n * (1.0 + u_n * (0.55 + u_n * (0.05 + 0.10 * u_n)))
            p = 1.0 + u_n * (0.55 + u_n * (0.05 + 0.10 * u_n))   # n / u
            q = 1.0 + u_n * (1.1 + u_n * (0.15 + 0.4 * u_n))     # n'
            r = n / (1.0 + n)
            d_neg = gamma * (r + (q / p) * r * (1.0 - r))

            return grad_output * torch.where(x >= 0.0, d_pos, d_neg), None, None, None, None

    class FastQALU(nn.Module):
        """
        FastQALU: transcendental-free rational surrogate.

        Positive branch: tanh(u) is replaced by u / sqrt(1 + u^2) (rsqrt, one
        reciprocal square root, no transcendental unit).

        Negative branch: 1 - exp(-u) is replaced by N(u) / (1 + N(u)) with
        N(u) = u * (1 + u * (0.55 + u * (0.05 + 0.10 * u))), i.e. the [4/4]
        Pade-type approximant obtained by substituting the degree-4 Taylor
        polynomial of exp(u) for exp(u). Properties:
            r(0) = 0, r'(0) = 1, r(inf) = 1, denominator >= 1 for all u >= 0
            max relative error 0.53% in value and 0.74% in gradient on [0, 20]
            preserves the downside gradient overshoot: +13.12% at u ~ 1.97
            (exact form: +13.53% at u = 2)
        Written as N/(1+N) rather than 1 - 1/(1+N) on purpose: the latter loses
        4.6% relative accuracy in FP32 at u ~ 1e-6 through cancellation, 19% at
        1e-7 and 100% at 1e-8.
        The backward pass is analytic (see _FastQALUFunction), which keeps one
        saved tensor instead of the five an eager graph would retain.
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                     gamma: float = 1.5, delta: float = 0.5):
            super(FastQALU, self).__init__()
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.gamma = float(gamma)
            self.delta = float(delta)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _FastQALUFunction.apply(x, self.alpha, self.beta,
                                           self.gamma, self.delta)

    QALU_Adaptive = QALU_Ultra

else:
    # Pure NumPy fallback: identical mathematics, no autograd.
    from .reference import (  # noqa: F401
        QALU, QALU2, QALU_Ultra, FastQALU, QALU_Adaptive,
    )
