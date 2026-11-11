# Q-ALU: Quant Asymmetric Leaky Unit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![PyPI](https://img.shields.io/pypi/v/qalu-pytorch.svg)](https://pypi.org/project/qalu-pytorch/)
[![SSRN](https://img.shields.io/badge/SSRN-7365098-blue.svg)](https://ssrn.com/abstract=7365098)
[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.7365098-blue.svg)](http://dx.doi.org/10.2139/ssrn.7365098)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Q_ALU_Research_Paper.pdf)

> **A Market Psychology Aware Activation Function with Downside Risk Protection for High Frequency Quantitative Trading and Deep Reinforcement Learning**

Developed by **Faiq Hammam Mutaqin** (*Independent Researcher, Indonesia*).

---

### Official Paper Publication

> **Mutaqin, Faiq Hammam**, *Q-ALU: Quant Asymmetric Leaky Unit: A Market Psychology Aware Activation Function with Downside Risk Protection for High Frequency Quantitative Trading and Deep Reinforcement Learning* (August 13, 2026). Available at SSRN: [https://ssrn.com/abstract=7365098](https://ssrn.com/abstract=7365098) or [http://dx.doi.org/10.2139/ssrn.7365098](http://dx.doi.org/10.2139/ssrn.7365098).
>
> **Research Paper PDF:** [`paper/Q_ALU_Research_Paper.pdf`](paper/Q_ALU_Research_Paper.pdf)

---

## Overview

Traditional deep learning activation functions (such as **ReLU**, **GELU**, and **Swish/SiLU**) were fundamentally designed for computer vision and natural language processing under the implicit assumption of isotropic, symmetric feature representations. However, in quantitative finance and algorithmic portfolio management:

1. **Market price discovery is fundamentally asymmetric:** Downside sell-offs (panic liquidations, margin cascades) occur with substantially higher velocity and severity than bullish accumulation phases, as formalized by Kahneman and Tversky behavioral Prospect Theory.
2. **Microstructure noise causes fee erosion and overtrading:** Pervasive high-frequency noise (such as bid-ask bounce and flickering quotes) produces spurious activations near zero, misleading policy gradient algorithms (e.g. Proximal Policy Optimization / PPO) into excessive turnover.
3. **Dead-neuron downside blindness in ReLU:** Standard ReLU completely zeros out negative activations ($f'(x) = 0$ for $x < 0$). In policy gradient architectures, when a position suffers a drawdown, negative hidden features cannot backpropagate error gradients, blinding the policy network to tail losses until liquidation.

**Q-ALU** directly embeds downside risk protection and market psychology into neural network calculus.

---

## Mathematical Formulation

### Formal Definition
Let $x \in \mathbb{R}$ denote the pre-activation input. The Quant Asymmetric Leaky Unit (Q-ALU) is defined piecewise as:

$$
f(x) = \begin{cases} 
\alpha \cdot x \cdot \left( 1 + \tanh\left(\frac{x}{\beta}\right) \right), & \text{for } x \ge 0 \\ 
\gamma \cdot x \cdot \left( 1 - \exp\left(\frac{x}{\delta}\right) \right), & \text{for } x < 0 
\end{cases}
$$

* **Positive Domain ($x \ge 0$):** Bullish momentum riding regime accelerating toward terminal slope $2\alpha$.
* **Negative Domain ($x < 0$):** Downside risk protection and quadratic drawdown barrier.

Where $\alpha, \beta, \gamma, \delta > 0$:
* **$\alpha$ (Bullish Momentum Scaler):** Dictates initial response gain to positive returns, accelerating toward terminal slope $2\alpha$.
* **$\beta$ (Positive Damping Scale):** Governs transition rate from linear response to accelerated momentum riding.
* **$\gamma$ (Risk / Panic Scaler):** Controls the asymptotic penalty gradient for severe negative drawdowns.
* **$\delta$ (Micro-Noise Drawdown Threshold):** Establishes the quadratic noise-filtering deadband around zero.

Using the floating-point identity $\mathrm{expm1}(u) = e^u - 1$, the negative branch is evaluated with full machine precision:

$$
f(x) = -\gamma \cdot x \cdot \mathrm{expm1}\left(\frac{x}{\delta}\right), \quad \text{for } x < 0
$$

### Key Mathematical Theorems
1. **Origin Continuity ($C^0$):** $\lim_{x \to 0^+} f(x) = \lim_{x \to 0^-} f(x) = f(0) = 0$.
2. **Global Monotonicity:** $f'(x) \ge 0 \quad (\forall x \in \mathbb{R})$, eliminating artificial local minima wells.
3. **Dynamic Gradient Overshoot:** The downside gradient is monotone and bounded: it rises through the quadratic band, reaches the full linear penalty $\gamma$ at $x = -\delta$, peaks at $f'(-2\delta) \approx 1.1353\gamma$ (**$+13.53\%$** above the asymptote) and then relaxes back toward $\gamma$ as the loss deepens. The amplification is a localized window around $-2\delta$, not an unbounded spike, so severe losses get a 13.5% stronger error signal than a pure linear cut-loss without risking exploding gradients.
4. **FastQALU Rational Approximant:** For FPGA and order book engines without transcendental units:

$$
f_{\text{fast}}(x) = \begin{cases} 
\alpha \cdot x \cdot \left( 1 + \frac{u}{\sqrt{1 + u^2}} \right), & \text{for } x \ge 0,\; u = x/\beta \\ 
\gamma \cdot x \cdot \frac{N(u)}{1 + N(u)}, & \text{for } x < 0,\; u = -x/\delta
\end{cases}
$$

with $N(u) = u\left(1 + u\left(0.55 + u\left(0.05 + 0.10u\right)\right)\right)$, a $[4/4]$ Padé-type approximant of $1 - e^{-u}$ that satisfies $N(0)=0$, $N'(0)=1$ and $1 + N(u) \ge 1$. Maximum relative error is **0.53% in value** and **0.74% in gradient** on $u \in [0, 20]$; the downside overshoot is preserved at **$+13.12\%$** (exact: $+13.53\%$).

---

## Visual Analysis

<p align="center">
  <img src="paper/figures/fig1_activation_curves.png" width="48%" alt="Activation Curves" />
  <img src="paper/figures/fig2_gradient_dynamics.png" width="48%" alt="Gradient Dynamics" />
</p>

* **Left (Fig. 1):** Q-ALU compared with baseline activations (ReLU, LeakyReLU, GELU, Swish).
* **Right (Fig. 2):** First derivative dynamics showing the Downside Alarm Peak (+13.53% at $x = -2\delta$) and Bullish Momentum Peak (+10.0% at $x \approx 1.2\beta$).

<p align="center">
  <img src="paper/figures/fig5_equity_curves.png" width="95%" alt="Cumulative Equity Performance" />
</p>

* **Fig. 5:** Out-of-sample cumulative equity across 4 market regimes (Choppy, Breakout, Flash Crash, Recovery) under realistic 1.0 bps fee per turnover.

---

## Quickstart

### Installation

Install via pip:
```bash
pip install qalu-pytorch
```

Or install from source:
```bash
git clone https://github.com/cyclocerine/qalu-pytorch.git
cd qalu-pytorch
pip install -e .
```

### PyTorch Usage
```python
import torch
import torch.nn as nn
from qalu import QALU, FastQALU

class TradingPolicyNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5), # Q-ALU activation
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 32),
            FastQALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5), # FastQALU
            nn.Linear(32, action_dim) # Short, Flat, Long
        )

    def forward(self, x):
        return self.net(x)

# Instantiate model
model = TradingPolicyNet()
x = torch.randn(16, 10)
logits = model(x)
print("Output logits shape:", logits.shape)
```

### Numerical Safety in PyTorch Autograd
Standard implementations using `torch.where` evaluate both branches eagerly, causing $0 \times \infty = \text{NaN}$ in backward passes for large inputs ($x > 88.7$). Q-ALU strictly pre-clamps inputs to valid mathematical domains, guaranteeing zero NaN gradients across all precision levels (FP32, FP16, BF16).

---

## Empirical Trading Benchmarks

> **Status: negative result, reported as measured.** The benchmark below is fully reproducible (`experiments/trading_benchmark.py --epochs 300 --json results/trading_benchmark.json --plot ...`), uses five paired seeds and identical initial weights per activation. It **does not** support the claim that Q-ALU outperforms standard activations. The figures circulated in an earlier revision (+75.85% versus +5.43% for ReLU, baselines collapsing to 0–8 trades, a 2,834-bar BTC/USDT study and a 1D-CNN/LSTM study) could not be reproduced from this repository and are withdrawn; see §"Reproducibility Note" in the paper.

### Table I: Out-of-Sample Quantitative Benchmark (5,000 Ticks, 1.0 bps Turnover Fee, 5 Paired Seeds, Mean ± Std)

| Activation Function | Out-of-Sample Return (%) | Per-Step Sharpe | Per-Step Sortino | Profit Factor | Max Drawdown (%) | Total Trades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q-ALU** | +373.51 ± 27.39 | +0.489 ± 0.030 | +1.083 ± 0.129 | 4.37 ± 0.46 | 6.02 ± 1.33 | 718.2 ± 23.4 |
| **ReLU** | +378.81 ± 40.32 | +0.498 ± 0.054 | +1.099 ± 0.232 | 4.56 ± 0.84 | 4.91 ± 1.47 | 761.2 ± 39.9 |
| **LeakyReLU** | +383.70 ± 42.02 | +0.506 ± 0.057 | +1.140 ± 0.250 | 4.69 ± 0.86 | 4.48 ± 1.65 | 756.4 ± 46.1 |
| **GELU** | +393.58 ± 16.23 | +0.521 ± 0.014 | +1.182 ± 0.098 | 4.86 ± 0.25 | 4.81 ± 1.70 | 737.2 ± 88.7 |
| **SiLU (Swish)** | **+396.08 ± 20.61** | **+0.525 ± 0.025** | **+1.201 ± 0.086** | **4.94 ± 0.41** | **4.47 ± 1.46** | 733.2 ± 96.8 |

What the numbers say, with paired differences over identical seeds and initial weights ($n = 5$, two-sided 5% critical value $2.78$):

* **No turnover collapse for any activation.** Individual runs trade 574–819 times out of 5,000; the means span 718–761, and the lowest mean is Q-ALU's.
* **Q-ALU trails on return and risk-adjusted return.** −20.1 pp of return versus GELU ($t = -3.65$) and −22.6 pp versus SiLU ($t = -4.51$); Sortino −0.099 ($t = -2.20$) and −0.118 ($t = -3.55$). Against ReLU and LeakyReLU the return gap is not significant ($t = -0.58$ and $-0.87$).
* **The one consistent Q-ALU effect is lower turnover, with deeper drawdowns.** 43 fewer trades than ReLU ($t = -3.59$), but the highest maximum drawdown of the five (6.02% versus 4.47% for SiLU; paired $+1.55$ pp, $t = +2.42$).
* **Reproduce it yourself:** `experiments/trading_benchmark.py --epochs 300 --json results/trading_benchmark.json --plot paper/latex/figures/fig5_equity_curves.pdf` writes both the JSON behind this table and the equity-curve figure.

### Table II: Hardware Latency and Throughput (100,000 FP32 Elements, Single Thread, Best of 3)

| Activation | Forward (ms) | Backward (ms) | Throughput (Million/s) |
| :--- | :---: | :---: | :---: |
| **ReLU** | 0.334 | 2.547 | 299.29 |
| **LeakyReLU** | 1.696 | 2.661 | 58.98 |
| **SiLU** | 3.375 | 4.415 | 29.63 |
| **GELU** | 2.678 | 4.234 | 37.34 |
| **FastQALU** | **10.384** | **22.600** | **9.63** |
| **Q-ALU 2.0** | 15.764 | 20.413 | 6.34 |
| **Q-ALU Ultra** | 16.427 | 14.807 | 6.09 |
| **Q-ALU** | 17.064 | 17.881 | 5.86 |

FastQALU is 1.64× faster than transcendental Q-ALU in the forward pass and 1.27× faster end-to-end, but GELU is 3.9× faster still: the algebraic surrogate pays off where transcendental units are unavailable or the kernel is fused, not as a general throughput win.

---

## High-Performance C++ and CUDA Kernels

For latency-sensitive deployment (no transcendental units in the target datapath, or a fused kernel), production kernels are available in `qalu/csrc/`:
* `qalu.hpp`: Single-header C++20 implementation with OpenMP SIMD vectorization.
* `qalu_cuda.cu`: NVIDIA Tensor Core optimized CUDA kernels supporting FP32, FP16, and BF16 execution.

---

## Repository Structure

```text
qalu-pytorch/
├── qalu/
│   ├── __init__.py          # Package initialization and exports
│   ├── core.py              # PyTorch and NumPy implementations (QALU, FastQALU)
│   └── csrc/
│       ├── qalu.hpp         # C++20 header-only implementation
│       └── qalu_cuda.cu     # CUDA FP32/FP16 forward and backward kernels
├── paper/
│   ├── Q_ALU_Research_Paper.pdf  # 8-page research paper PDF
│   ├── Q_ALU_Research_Paper.md   # Markdown research manuscript
│   ├── references.bib            # 22 peer-reviewed citations
│   ├── figures/                  # Vector PDF and 300 DPI PNG figures
│   └── latex/                    # Full IEEEtran LaTeX source code
├── examples/
│   ├── quickstart.py        # PyTorch and NumPy quickstart guide
│   └── ppo_trading_demo.py  # PPO actor-critic financial trading demo
├── tests/
│   └── test_qalu.py         # Unit tests (continuity, monotonicity, overshoot, FastQALU)
├── pyproject.toml           # PEP 517/518 build specification
├── setup.py                 # Package setup and build configuration
├── LICENSE                  # MIT License
└── README.md                # Documentation and benchmark overview
```

---

## Citation

If you use Q-ALU in your research, trading algorithms, or production systems, please cite the official SSRN publication:

```bibtex
@article{mutaqin2026qalu,
  title={Q-ALU: Quant Asymmetric Leaky Unit: A Market Psychology Aware Activation Function with Downside Risk Protection for High Frequency Quantitative Trading and Deep Reinforcement Learning},
  author={Mutaqin, Faiq Hammam},
  journal={SSRN Electronic Journal},
  year={2026},
  month={August},
  doi={10.2139/ssrn.7365098},
  url={https://ssrn.com/abstract=7365098}
}
```

Text citation:
> Mutaqin, F. H. (2026). *Q-ALU: Quant Asymmetric Leaky Unit: A Market Psychology Aware Activation Function with Downside Risk Protection for High Frequency Quantitative Trading and Deep Reinforcement Learning*. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.7365098

---

## License

MIT License. Copyright (c) 2026 Faiq Hammam Mutaqin.

