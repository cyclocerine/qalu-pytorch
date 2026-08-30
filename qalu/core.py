"""
Q-ALU: Quant Asymmetric Leaky Unit: Core Implementations
Author: Faiq Hammam Mutaqin (2026)
SSRN: https://ssrn.com/abstract=7365098
DOI: 10.2139/ssrn.7365098

Contains:
- QALU: Production Ready, NaN-Free, FP32 Stable
- FastQALU: Sub-microsecond Algebraic Rational Approximation
- QALU_Adaptive: Dynamic Volatility Conditioned and Fee-Aware
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _DummyModule:
        def __init__(self, *args, **kwargs):
            pass
    nn = type('nn', (), {'Module': _DummyModule, 'Parameter': lambda x: x})

import numpy as np

if HAS_TORCH:
    class QALU(nn.Module):
        """
        Q-ALU:
        f(x) = alpha * x * (1 + tanh(x / beta))      for x >= 0
        f(x) = -gamma * x * expm1(x / delta)         for x < 0
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                     gamma: float = 1.5, delta: float = 0.5, 
                     trainable: bool = True):
            super(QALU, self).__init__()
            self.trainable = trainable

            def inv_softplus(val: float) -> torch.Tensor:
                return torch.log(torch.exp(torch.tensor(val, dtype=torch.float32)) - 1.0)

            if trainable:
                self._raw_alpha = nn.Parameter(inv_softplus(alpha))
                self._raw_beta = nn.Parameter(inv_softplus(beta))
                self._raw_gamma = nn.Parameter(inv_softplus(gamma))
                self._raw_delta = nn.Parameter(inv_softplus(delta))
            else:
                self.register_buffer('_raw_alpha', inv_softplus(alpha))
                self.register_buffer('_raw_beta', inv_softplus(beta))
                self.register_buffer('_raw_gamma', inv_softplus(gamma))
                self.register_buffer('_raw_delta', inv_softplus(delta))

        @property
        def alpha(self) -> torch.Tensor:
            return F.softplus(self._raw_alpha) + 1e-4

        @property
        def beta(self) -> torch.Tensor:
            return F.softplus(self._raw_beta) + 1e-4

        @property
        def gamma(self) -> torch.Tensor:
            return F.softplus(self._raw_gamma) + 1e-4

        @property
        def delta(self) -> torch.Tensor:
            return F.softplus(self._raw_delta) + 1e-4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Clamping prevents 0 * inf = NaN on GPU Autograd unselected branch
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)
            
            pos_term = self.alpha * x_pos * (1.0 + torch.tanh(x_pos / self.beta))
            neg_term = -self.gamma * x_neg * torch.expm1(x_neg / self.delta)
            return torch.where(x >= 0.0, pos_term, neg_term)


    class QALU2(nn.Module):
        """
        Q-ALU 2.0 (C1 Continuous Smooth Transition):
        f(x) = alpha * x * (tanh(x / beta) + tanh^2(x / beta)) for x >= 0
        f(x) = -gamma * x * expm1(x / delta)                   for x < 0
        where gamma = 2 * alpha * (1 + risk_premium)
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                     risk_premium: float = 0.25, delta: float = 0.5, 
                     trainable: bool = True):
            super(QALU2, self).__init__()
            self.trainable = trainable

            def inv_softplus(val: float) -> torch.Tensor:
                return torch.log(torch.exp(torch.tensor(val, dtype=torch.float32)) - 1.0)

            if trainable:
                self._raw_alpha = nn.Parameter(inv_softplus(alpha))
                self._raw_beta = nn.Parameter(inv_softplus(beta))
                self._raw_risk_premium = nn.Parameter(inv_softplus(risk_premium))
                self._raw_delta = nn.Parameter(inv_softplus(delta))
            else:
                self.register_buffer('_raw_alpha', inv_softplus(alpha))
                self.register_buffer('_raw_beta', inv_softplus(beta))
                self.register_buffer('_raw_risk_premium', inv_softplus(risk_premium))
                self.register_buffer('_raw_delta', inv_softplus(delta))

        @property
        def alpha(self) -> torch.Tensor:
            return F.softplus(self._raw_alpha) + 1e-4

        @property
        def beta(self) -> torch.Tensor:
            return F.softplus(self._raw_beta) + 1e-4

        @property
        def gamma(self) -> torch.Tensor:
            premium = F.softplus(self._raw_risk_premium) + 1e-4
            return 2.0 * self.alpha * (1.0 + premium)

        @property
        def delta(self) -> torch.Tensor:
            return F.softplus(self._raw_delta) + 1e-4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)
            
            u_p = x_pos / self.beta
            t_p = torch.tanh(u_p)
            pos_term = self.alpha * x_pos * (t_p + t_p * t_p)
            
            u_n = x_neg / self.delta
            neg_term = -self.gamma * x_neg * torch.expm1(u_n)
            return torch.where(x >= 0.0, pos_term, neg_term)


    class QALU_Ultra(nn.Module):
        """
        Q-ALU Ultra: Quant-Adaptive HFT Activation Function
        Dynamically conditions noise thresholds on real-time market volatility.
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0, delta: float = 0.5,
                     fee_threshold: float = 0.01, loss_aversion: float = 2.25,
                     trainable: bool = True):
            super(QALU_Ultra, self).__init__()
            self.trainable = trainable
            self.fee_threshold = float(fee_threshold)

            def inv_softplus(val: float) -> torch.Tensor:
                return torch.log(torch.exp(torch.tensor(val, dtype=torch.float32)) - 1.0)

            if trainable:
                self._raw_alpha = nn.Parameter(inv_softplus(alpha))
                self._raw_beta = nn.Parameter(inv_softplus(beta))
                self._raw_delta = nn.Parameter(inv_softplus(delta))
                self._raw_loss_aversion = nn.Parameter(inv_softplus(loss_aversion))
            else:
                self.register_buffer('_raw_alpha', inv_softplus(alpha))
                self.register_buffer('_raw_beta', inv_softplus(beta))
                self.register_buffer('_raw_delta', inv_softplus(delta))
                self.register_buffer('_raw_loss_aversion', inv_softplus(loss_aversion))

        @property
        def alpha(self) -> torch.Tensor:
            return F.softplus(self._raw_alpha) + 1e-4

        @property
        def beta(self) -> torch.Tensor:
            return F.softplus(self._raw_beta) + 1e-4

        @property
        def delta(self) -> torch.Tensor:
            return F.softplus(self._raw_delta) + 1e-4

        @property
        def gamma(self) -> torch.Tensor:
            la = F.softplus(self._raw_loss_aversion) + 1.0
            return 2.0 * self.alpha * la

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


    class FastQALU(nn.Module):
        """
        FastQALU: Sub-microsecond Rational Approximation without transcendentals.
        """
        def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                     gamma: float = 1.5, delta: float = 0.5):
            super(FastQALU, self).__init__()
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.gamma = float(gamma)
            self.delta = float(delta)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_pos = torch.clamp(x, min=0.0)
            x_neg = torch.clamp(x, max=0.0)
            
            u_p = x_pos / self.beta
            t_p = u_p * torch.rsqrt(1.0 + u_p * u_p)
            pos_term = self.alpha * x_pos * (1.0 + t_p)
            
            u_n = x_neg / self.delta
            neg_term = self.gamma * x_neg * (-u_n / (1.0 - 0.5 * u_n))
            return torch.where(x >= 0.0, pos_term, neg_term)

    QALU_Adaptive = QALU_Ultra

else:
    # Pure NumPy implementation if PyTorch is not installed in the environment
    class QALU:
        def __init__(self, alpha=1.0, beta=1.0, gamma=1.5, delta=0.5):
            self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        def forward(self, x):
            x = np.asarray(x, dtype=np.float64)
            pos = self.alpha * np.maximum(x, 0.0) * (1.0 + np.tanh(np.maximum(x, 0.0) / self.beta))
            neg = -self.gamma * np.minimum(x, 0.0) * np.expm1(np.minimum(x, 0.0) / self.delta)
            return np.where(x >= 0.0, pos, neg)
        def __call__(self, x):
            return self.forward(x)

    class QALU2:
        def __init__(self, alpha=1.0, beta=1.0, risk_premium=0.25, delta=0.5):
            self.alpha, self.beta = alpha, beta
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

    QALU_Ultra = QALU2
    QALU_Adaptive = QALU_Ultra
    FastQALU = QALU
