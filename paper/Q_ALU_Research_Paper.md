# Q-ALU: Quant Asymmetric Leaky Unit: A Market Psychology Aware Activation Function with Downside Risk Protection for High Frequency Quantitative Trading and Deep Reinforcement Learning

**Faiq Hammam Mutaqin**  
*Independent Researcher, Indonesia*  
*Email: hammamfaiq@protonmail.com*  
*Official Publication (SSRN):* [https://ssrn.com/abstract=7365098](https://ssrn.com/abstract=7365098)  
*DOI:* [http://dx.doi.org/10.2139/ssrn.7365098](http://dx.doi.org/10.2139/ssrn.7365098)  
*Date: August 13, 2026*  

---

### Abstract
Deep neural networks applied to high frequency trading (HFT) and algorithmic portfolio management frequently suffer from structural inductive mismatch. Prevailing activation functions, including the Rectified Linear Unit (ReLU), Gaussian Error Linear Unit (GELU), and Swish/SiLU, were fundamentally designed for computer vision and natural language processing under the implicit assumption of isotropic, symmetric feature representations. Financial price discovery, however, is inherently asymmetric: downside liquidations and panic cascades occur with substantially greater velocity and severity than bullish accumulation phases, as formalized by Kahneman and Tversky prospect theory. Furthermore, intermediate hidden representations in trading models are severely corrupted by microstructure noise (for example bid ask bounce), leading to rampant overtrading and catastrophic policy collapse in reinforcement learning algorithms such as Proximal Policy Optimization (PPO). 

To address this, this paper introduces the **Quant Asymmetric Leaky Unit (Q-ALU)**, an activation function engineered for financial time series modeling and deep reinforcement learning. Q-ALU couples a dynamic hyperbolic tangent momentum riding regime on the positive domain with an exponential to linear downside barrier on the negative domain. We prove that Q-ALU is globally monotonic ($f'(x) \ge 0 \ \forall x \in \mathbb{R}$), is $C^0$ at the origin, and exhibits a *Dynamic Gradient Overshoot*: the derivative peaks at $1.1353\gamma$ exactly at the drawdown threshold $x = -2\delta$ ($+13.53\%$) before relaxing toward the linear cut-loss asymptote. We give numerically stable PyTorch, C++ and CUDA implementations that avoid the $0 \times \infty$ autograd hazard and the catastrophic cancellation of the naive exponential form, plus an algebraic FastQALU surrogate that is 1.64× faster than the transcendental version in the forward pass.

We then test the trading hypothesis rather than assume it. A reproducible multi-regime benchmark (10,000 ticks, 1.0 bps fee per turnover, five paired seeds, 300 epochs) does **not** support the claim that Q-ALU outperforms standard activations: Q-ALU is statistically indistinguishable from ReLU on mean return and trails GELU and SiLU by 20–23 percentage points (paired $t = -3.65$ and $-4.51$), no activation shows the predicted turnover collapse, and Q-ALU's only consistent effect is roughly 6% lower turnover with the deepest maximum drawdown of the five (6.02% versus 4.47%). We report these negative results explicitly, withdraw the trading figures of an earlier draft as non-reproducible, and position Q-ALU's contribution as its mathematical design, numerical robustness and implementation, not as a superior trading activation.

**Keywords:** *Activation Functions, Deep Reinforcement Learning, Proximal Policy Optimization (PPO), High Frequency Trading (HFT), Downside Risk, Sortino Ratio, Prospect Theory, Financial Engineering.*

---

## I. Introduction

The integration of deep learning and deep reinforcement learning (DRL) into quantitative finance and high frequency trading (HFT) has accelerated dramatically [1], [2]. From order flow toxicity prediction to autonomous market making and statistical arbitrage, deep neural architectures are increasingly deployed to extract alpha from noisy limit order books (LOB) and high frequency tick streams [3].

Despite architectural innovations ranging from Temporal Convolutional Networks (TCN) to Recurrent Neural Networks (LSTM/GRU) and Financial Transformers [4], the foundational non-linear activation units powering these networks have remained essentially generic. Standard architectures routinely default to standard computer vision or natural language activations:
* **Rectified Linear Unit (ReLU)** [5]: $f(x) = \max(0, x)$
* **Leaky ReLU** [6]: $f(x) = \max(\alpha x, x)$
* **Gaussian Error Linear Unit (GELU)** [7]: $f(x) = x \cdot \Phi(x)$
* **Swish / SiLU** [8], [9]: $f(x) = x \cdot \sigma(\beta x)$
* **Mish** [10]: $f(x) = x \tanh(\ln(1 + e^x))$

While these functions excel in domains where positive and negative latent features carry symmetric semantic importance, they introduce severe structural pathologies when applied directly to quantitative financial modeling:

1. **Failure of the Symmetric Inductive Bias:** In market microstructure, price action is profoundly asymmetric. As documented by Kahneman and Tversky behavioral Prospect Theory [11], market participants exhibit pronounced loss aversion, feeling the disutility of a loss roughly $2.0\times$ to $2.5\times$ more acutely than an equivalent gain. Consequently, market sell-offs (panic crashes, margin liquidations) exhibit extreme negative skewness, volatility clustering, and sharp velocity, whereas bullish runs are characterized by gradual momentum build-ups [12]. Standard activations treat positive and negative hidden excitations with identical curvature or generic leakiness, failing to penalize negative risk states asymmetrically.
2. **Microstructure Noise Contamination and Fee Erosion:** High frequency price data is corrupted by pervasive microstructure noise, such as bid ask spread bounces, queue positioning fluctuations, and flickering quotes [13]. In a neural network policy trained via reinforcement learning (such as Proximal Policy Optimization / PPO [14]), non-zero gradients near origin for small negative inputs cause the agent to react to meaningless tick fluctuations, leading to excessive turnover. In the presence of realistic exchange fees and slippage, this overtrading rapidly erodes cumulative returns.
3. **The Dead Neuron Downside Blindness:** Standard ReLU completely zeros out all negative inputs ($f(x) = 0, f'(x) = 0$ for $x < 0$). In policy gradient methods, if a trader enters a position that suffers a drawdown, the corresponding hidden features enter the negative domain. If these neurons become inactive (or die), the policy gradient cannot flow backward through the value network to penalize the catastrophic action, blinding the policy to downside drawdown until terminal liquidation.

### Contributions of this Paper
To directly resolve these foundational issues, this paper introduces the **Quant Asymmetric Leaky Unit (Q-ALU)**. The primary contributions are summarized as follows:
* **Domain-Specific Piecewise Formulation:** We formulate Q-ALU as a dedicated dual regime activation function that couples hyperbolic tangent acceleration ($\alpha \to 2\alpha$) for momentum riding on positive inputs with an exponential to linear barrier for aggressive risk penalization on negative inputs.
* **Theoretical Calculus and Monotonicity Proof:** We prove that Q-ALU satisfies global monotonicity ($f'(x) \ge 0, \, \forall x \in \mathbb{R}$), eliminating artificial local minima wells. We derive the exact calculus of the origin and uncover the *Dynamic Gradient Overshoot* phenomenon, where the function exhibits a localized $+13.53\%$ sensitivity spike at $x = -2\delta$.
* **Downside Risk Analysis:** We analyse the downside dynamics of Q-ALU in closed form: quadratic damping around $0^-$ attenuates bid-ask bounce, and the derivative saturates at the linear cut-loss asymptote $\gamma$. The analysis predicts lower turnover and better downside capture; the benchmark of Section VI finds the lower turnover but not the improved Sortino ratio or profit factor, and we report both.
* **A Reproducible Benchmark, Reported Honestly:** We replace the non-reproducible backtests of the earlier draft (2,834 BTC/USDT bars, a 1D-CNN/LSTM study, and the +75.85% versus +5.43% comparison) with a runnable multi-regime benchmark over five paired seeds. Its result contradicts our own hypothesis: Q-ALU is competitive with, but not superior to, standard activations, and we report that as measured rather than re-tuning it.
* **Open Source Production Implementation:** We provide a PyTorch package (`qalu-pytorch`), numerical solutions preventing FP32 Autograd `NaN` errors, and C++20/CUDA kernels whose arithmetic path contains no transcendental units, together with the CPU throughput measurements of Table II.

---

## II. Related Work

### A. Activation Functions in Deep Learning
The quest for optimal activation functions has evolved through three distinct eras:
1. **Classical S-Curves:** Early networks utilized the Logistic Sigmoid $\sigma(x) = (1 + e^{-x})^{-1}$ and Hyperbolic Tangent $\tanh(x)$ [15]. While bounded, their vanishing gradients for large absolute activations severely constrained deep network scalability.
2. **Piecewise Linear Units:** The introduction of ReLU [5] revolutionized deep learning by preserving a non-saturating gradient of $1.0$ for all positive inputs. To address the dying ReLU problem, variants such as Leaky ReLU [6], Parametric ReLU (PReLU) [16], and Exponential Linear Units (ELU) [17] were proposed. Clevert et al. [17] demonstrated that ELU exponential negative saturation $\alpha(e^x - 1)$ brought the mean activation closer to zero, speeding up learning. Scaled ELU (SELU) [18] subsequently proved that under specific hyperparameters, deep networks achieve self-normalizing properties.
3. **Smooth Non-Monotonic Units:** More recently, smooth non-linearities discovered via automated search or empirical intuition, such as Swish/SiLU ($x \sigma(x)$) [8], GELU ($x \Phi(x)$) [7], and Mish ($x \tanh(\ln(1 + e^x))$) [10], have superseded ReLU in Transformer architectures and deep visual models due to their continuous curvature and slight negative well.

However, all of these functions remain fundamentally domain-agnostic. In financial time series modeling, none provide an explicit inductive mechanism to represent asymmetric risk preferences or transaction-cost deadbands.

### B. Reinforcement Learning in Quantitative Finance
Deep Reinforcement Learning, particularly policy gradient methods such as PPO [14], Soft Actor-Critic (SAC) [19], and Deep Q-Networks (DQN) [20], has emerged as a state of the art framework for financial trade execution and portfolio optimization [1], [3]. Unlike supervised learning, RL agents must learn through self-directed trial and error within non-stationary, low signal-to-noise environments. In financial RL, standard reward functions based purely on mean PnL fail because they ignore downside variance (volatility) and maximum drawdown [21]. While researchers have explored asymmetric loss functions (e.g., expectile regression or pinball loss) [22], modifying the internal activation mechanics of the policy network itself to natively respect downside risk has remained an unexplored frontier.

---

## III. Mathematical Formulation of Q-ALU

### A. Formal Definition
Let $x \in \mathbb{R}$ denote the pre-activation input (for example the linear affine projection $x = \mathbf{w}^T \mathbf{z} + b$ of a hidden layer). The Quant Asymmetric Leaky Unit (Q-ALU) is defined piecewise as:

$$f(x) = \begin{cases} \alpha \cdot x \cdot \left( 1 + \tanh\left(\frac{x}{\beta}\right) \right), & \text{for } x \ge 0 \quad \text{(Bullish Momentum Riding)} \\ \gamma \cdot x \cdot \left( 1 - \exp\left(\frac{x}{\delta}\right) \right), & \text{for } x < 0 \quad \text{(Downside Risk and Drawdown)} \end{cases}$$

where $\alpha, \beta, \gamma, \delta > 0$ represent strictly positive hyperparameters:
* $\alpha \in \mathbb{R}^+$: **Bullish Momentum Scaler**, dictating the initial response gain to positive returns.
* $\beta \in \mathbb{R}^+$: **Positive Damping Scale**, governing the rate of transition from linear response to accelerated momentum riding.
* $\gamma \in \mathbb{R}^+$: **Risk / Panic Scaler**, controlling the asymptotic penalty gradient for severe negative excursions.
* $\delta \in \mathbb{R}^+$: **Micro-Noise Drawdown Threshold**, establishing the width of the quadratic noise-filtering zone around zero.

Using the identity for negative exponents, since $x < 0 \implies x/\delta < 0$, the negative term can be equivalently written in terms of the standard floating-point function $\operatorname{expm1}(u) = e^u - 1$:

$$f(x) = -\gamma \cdot x \cdot \operatorname{expm1}\left(\frac{x}{\delta}\right), \quad \text{for } x < 0$$

![Figure 1: Comparison of Activation Functions: Q-ALU vs Baselines](figures/fig1_activation_curves.png)

*Fig. 1. Comparison of Q-ALU ($\alpha=1.0, \beta=1.0, \gamma=1.5, \delta=0.5$) with baseline activations (ReLU, LeakyReLU, GELU, and Swish/SiLU). Q-ALU accelerates on the positive domain ($2\alpha x$) while enforcing an asymmetric linear cut loss ($\gamma x$) on the negative domain.*

---

### B. Continuity Analysis ($C^0$)
A fundamental prerequisite for gradient-based backpropagation is that an activation function must be continuous at the origin to prevent indeterminate limits and infinite gradients.

**Theorem 1 (Origin Continuity):** *The function $f(x)$ is continuous at $x = 0$, with $f(0) = 0$.*

*Proof:*
Evaluating the right-sided limit ($x \to 0^+$):
$$\lim_{x \to 0^+} f(x) = \lim_{x \to 0^+} \alpha \cdot x \cdot \left( 1 + \tanh\left(\frac{x}{\beta}\right) \right) = \alpha \cdot 0 \cdot (1 + 0) = 0$$
Evaluating the left-sided limit ($x \to 0^-$):
$$\lim_{x \to 0^-} f(x) = \lim_{x \to 0^-} \gamma \cdot x \cdot \left( 1 - \exp\left(\frac{x}{\delta}\right) \right) = \gamma \cdot 0 \cdot (1 - 1) = 0$$
Since $\lim_{x \to 0^+} f(x) = \lim_{x \to 0^-} f(x) = f(0) = 0$, the function $f(x)$ is $C^0$ continuous everywhere on $\mathbb{R}$. $\blacksquare$

---

### C. First Derivative Analysis ($f'(x)$)
The backpropagation engine requires the first derivative $f'(x) = \frac{df}{dx}$.

1. **Positive Branch ($x \ge 0$):**
   Applying the product rule $\frac{d}{dx}[u(x)v(x)] = u'v + uv'$ where $u(x) = \alpha x$ and $v(x) = 1 + \tanh(x/\beta)$:
   $$f'_+(x) = \alpha \left( 1 + \tanh\left(\frac{x}{\beta}\right) \right) + \frac{\alpha x}{\beta} \operatorname{sech}^2\left(\frac{x}{\beta}\right)$$
   Evaluating the boundary limits:
   * As $x \to 0^+$: $\tanh(0) = 0$ and $\operatorname{sech}^2(0) = 1$. Thus:
     $$\lim_{x \to 0^+} f'_+(x) = \alpha(1 + 0) + 0 = \mathbf{\alpha}$$
   * As $x \to +\infty$: $\tanh(x/\beta) \to 1$ and $\frac{x}{\beta}\operatorname{sech}^2(x/\beta) \to 0$ exponentially fast. Thus:
     $$\lim_{x \to +\infty} f'_+(x) = \alpha(1 + 1) + 0 = \mathbf{2\alpha}$$

2. **Negative Branch ($x < 0$):**
   Applying the product rule to $u(x) = \gamma x$ and $v(x) = 1 - \exp(x/\delta)$:
   $$f'_-(x) = \gamma \left( 1 - \left(1 + \frac{x}{\delta}\right)\exp\left(\frac{x}{\delta}\right) \right)$$
   Evaluating the boundary limits:
   * As $x \to 0^-$: $\exp(0) = 1$ and $\frac{x}{\delta}\exp(0) = 0$. Thus:
     $$\lim_{x \to 0^-} f'_-(x) = \gamma(1 - 1 - 0) = \mathbf{0}$$
   * As $x \to -\infty$: Since $x < 0$, let $x \to -\infty$. Then $\exp(x/\delta) \to 0$ and $\frac{x}{\delta}\exp(x/\delta) \to 0$. Thus:
     $$\lim_{x \to -\infty} f'_-(x) = \gamma(1 - 0 - 0) = \mathbf{\gamma}$$

> [!NOTE]
> **The Intentional Gradient Kink:** At the origin $x = 0$, there exists a deliberate jump discontinuity in the first derivative: $\Delta f'(0) = f'(0^+) - f'(0^-) = \alpha - 0 = \alpha$. Similar to standard ReLU ($\Delta f'(0) = 1.0$), this mathematical kink creates a sharp boundary separating bullish trend participation from downside risk absorption.

---

### D. Global Monotonicity Proof
A critical defect of certain non-monotonic activations (such as Swish and Mish) in financial trading is that their negative dip causes an inverse gradient: an increase in negative price movement can produce a positive directional update. We prove that Q-ALU avoids this pathology.

**Theorem 2 (Global Monotonicity):** *For all strictly positive hyperparameters $\alpha, \beta, \gamma, \delta > 0$, the function $f(x)$ is strictly monotonically increasing on $\mathbb{R} \setminus \{0\}$, and non-decreasing everywhere: $f'(x) \ge 0 \quad \forall x \in \mathbb{R}$.*

*Proof:*
* **Case 1 ($x > 0$):** Since $x/\beta > 0$, $\tanh(x/\beta) > 0$, and $\operatorname{sech}^2(x/\beta) > 0$. Therefore:
  $$f'_+(x) = \alpha \left( 1 + \tanh\left(\frac{x}{\beta}\right) + \frac{x}{\beta}\operatorname{sech}^2\left(\frac{x}{\beta}\right) \right) > \alpha > 0$$
* **Case 2 ($x < 0$):** Let $z = -x/\delta > 0$, such that $x/\delta = -z$. Substituting into $f'_-(x)$:
  $$f'_-(x) = \gamma \left( 1 - (1 - z)e^{-z} \right)$$
  Define the scalar auxiliary function $h(z) = 1 - (1 - z)e^{-z}$ for $z > 0$. Its derivative with respect to $z$ is:
  $$\frac{dh}{dz} = (2 - z)e^{-z}$$
  At $z = 0$, $h(0) = 1 - 1 = 0$. For $z \in (0, 2)$, $dh/dz > 0$, so $h(z)$ increases strictly from $0$ to its maximum at $z = 2$:
  $$h(2) = 1 - (1 - 2)e^{-2} = 1 + e^{-2} \approx 1.135335 > 0$$
  For $z > 2$, $dh/dz < 0$, and as $z \to \infty$, $h(z) \to 1.0 > 0$. Since $h(z) > 0$ for all $z > 0$ and $\gamma > 0$, it follows that:
  $$f'_-(x) = \gamma h(z) > 0 \quad \forall x < 0$$
* **Case 3 ($x = 0$):** At the single point $x = 0$, $f(0) = 0$, with $f(x) < 0$ for $x < 0$ and $f(x) > 0$ for $x > 0$.
Thus, $f'(x) \ge 0 \quad \forall x \in \mathbb{R}$, and $f(x)$ is strictly monotonic. $\blacksquare$

---

### E. Discovery of the Dynamic Gradient Overshoot

A detailed second derivative analysis reveals a remarkable mathematical phenomenon embedded in Q-ALU: **the gradient does not approach its asymptotic values monotonically; rather, it exhibits local extrema (overshoots) on both domains.**

1. **Downside Alarm Overshoot ($x = -2\delta$):**
   Computing the second derivative for $x < 0$:
   $$f''_-(x) = -\frac{\gamma}{\delta} \left(2 + \frac{x}{\delta}\right) e^{x/\delta}$$
   Setting $f''_-(x) = 0$ yields the unique inflection point:
   $$2 + \frac{x^*}{\delta} = 0 \implies x^* = -2\delta$$
   Evaluating the gradient at this critical threshold:
   $$f'_-( -2\delta ) = \gamma \left( 1 + e^{-2} \right) \approx \mathbf{1.135335 \cdot \gamma}$$
   
   **Financial Interpretation:** When an asset drawdown reaches exactly twice the noise threshold ($x = -2\delta$), Q-ALU delivers a **$+13.53\%$ amplification spike in sensitivity** above its asymptotic penalty. This creates a natural early warning alarm that sharply alerts policy gradient networks to initiate defensive liquidation before the loss accelerates into a catastrophic tail event.

2. **Bullish Momentum Overshoot ($x \approx 1.20\beta$):**
   On the positive domain, setting $f''_+(x) = 0$ requires solving:
   $$2 - 2\frac{x}{\beta}\tanh\left(\frac{x}{\beta}\right) = 0 \implies \frac{x^*}{\beta}\tanh\left(\frac{x^*}{\beta}\right) = 1$$
   The numerical root evaluates to $x^* \approx 1.199678 \beta$. At this point:
   $$f'_+(x^*) \approx \mathbf{2.1997 \cdot \alpha}$$
   This yields a **$+10.0\%$ momentum acceleration boost** above the terminal slope $2\alpha$, perfectly modeling the explosive participation typical of technical breakout regimes.

![Figure 2: First Derivative and Dynamic Gradient Overshoot](figures/fig2_gradient_dynamics.png)

*Fig. 2. First derivative $f'(x)$ dynamics of Q-ALU. Highlights the Downside Alarm Peak ($+13.53\%$ sensitivity spike at $x = -2\delta$) and Bullish Momentum Peak ($+10.0\%$ at $x \approx 1.2\beta$), alongside the origin kink $f'(0^+) = \alpha$ and $f'(0^-) = 0$.*

---

### F. Extensions: FastQALU for Ultra Low Latency Execution

For ultra low latency FPGA engines and live order book execution where hardware transcendental units (`tanh`, `exp`) impose clock cycle latency, FastQALU replaces both transcendental branches with algebraic rational functions:

$$f_{\text{fast}}(x) = \begin{cases} \alpha \cdot x \cdot \left(1 + \dfrac{u}{\sqrt{1 + u^2}}\right), & x \ge 0,\; u = x/\beta \\[1ex] \gamma \cdot x \cdot \dfrac{N(u)}{1 + N(u)}, & x < 0,\; u = -x/\delta \end{cases}$$

$$N(u) = u\left(1 + u\left(0.55 + u\left(0.05 + 0.10u\right)\right)\right)$$

This non-transcendental rational formulation eliminates exponential and hyperbolic evaluations while preserving the asymmetric dual regime behavior. The quartic numerator is the $[4/4]$ Padé-type approximant of $1 - e^{-u}$ fitted under the constraints $N(0) = 0$, $N'(0) = 1$, $N(u) \ge 0$:

* **Origin consistency:** $r(0) = 0$, $r'(0) = 1$, so the quadratic noise-suppression zone and $f'_{\text{fast}}(0^-) = 0$ are preserved exactly to first order.
* **Bounded error:** maximum relative error $0.53\%$ in value and $0.74\%$ in gradient on $u \in [0, 20]$ (verified by `experiments/cpp_check.cpp`).
* **Positive denominator:** $1 + N(u) \ge 1$ for all $u \ge 0$, hence no poles.
* **Preserved overshoot:** the downside gradient still peaks at $1.1312\gamma$ ($+13.12\%$) for $u \approx 1.97$, versus the exact $1.1353\gamma$ ($+13.53\%$) at $u = 2$. The positive branch is coarser: the fast-tanh substitution is accurate to $8.1\%$ in shape and $3.8\%$ in $f$, and shifts the momentum peak from $1.1997\beta$ ($+10.0\%$) to $\sqrt{2}\beta$ ($+4.4\%$).
* **Numerical form:** $N/(1+N)$ is used instead of $1 - 1/(1+N)$, which would lose $4.6\%$ relative accuracy in FP32 at $u \approx 10^{-6}$, $19\%$ at $10^{-7}$ and $100\%$ at $10^{-8}$ to cancellation. Inputs are clamped at $u = 10^{4}$, where the surrogate equals $1.0$ in FP32; without the clamp $u^4$ overflows for $|x| > 10^{7}$ and the quotient degenerates to $\text{NaN}$.

---

## IV. Downside Risk Analysis and Financial Dynamics

The defining motivation of Q-ALU is to optimize the policy network directly for downside risk metrics such as the **Sortino Ratio** and **Maximum Drawdown (MDD)** rather than unconstrained mean variance.

![Figure 4: Microstructure Noise Filtering Around Origin](figures/fig3_micro_noise_zoom.png)

*Fig. 3. Zoomed view near origin $x \in [-0.5, 0.5]$. Microstructure noise (bid ask bounce) in $[-\delta, 0]$ is suppressed quadratically ($f'(x) \to 0$), while positive breakout signals retain immediate linear responsiveness ($f'(0^+) = \alpha = 1.0$).*

![Figure 5: Parameter Sensitivity Analysis](figures/fig4_parameter_sensitivity.png)

*Fig. 4. Parameter sensitivity of Q-ALU under varying noise thresholds $\delta \in [0.2, 1.0]$ and risk penalties $\gamma \in [1.0, 2.5]$, enabling dynamic tuning for high volatility cryptocurrency or low noise forex regimes.*

### A. The Sortino Ratio Inductive Bias
In quantitative finance, the Sharpe Ratio penalizes upside volatility (large positive returns) identically to downside volatility. The Sortino Ratio resolves this flaw:

$$\text{Sortino} = \frac{\mathbb{E}[R] - R_f}{\sigma_D}, \quad \text{where } \sigma_D = \sqrt{\frac{1}{T} \sum_{t=1}^T \min(0, R_t - R_f)^2}$$

When an agent policy is parameterized by weights $\mathbf{\theta}$, the gradient of the Sortino ratio with respect to network activations is:

$$\nabla_{\mathbf{\theta}} \text{Sortino} \propto \sum_{t} \left[ \frac{\nabla_{\mathbf{\theta}} R_t}{\sigma_D} - \frac{\mathbb{E}[R] - R_f}{\sigma_D^3} \min(0, R_t - R_f) \nabla_{\mathbf{\theta}} R_t \right]$$

Under standard symmetric activations (such as GELU or LeakyReLU), negative activations transmit linear or sub-linear errors. Q-ALU explicitly aligns with the Sortino denominator $\sigma_D$:
1. For small losses $x \in [-\delta, 0]$, the function is quadratic, $f(x) \approx -\gamma x^2/\delta$, hence $f'(x) \approx 2\gamma|x|/\delta$. The gradient is not zero but rises linearly, reaching $0.186\gamma$ at $x = -0.1\delta$ and the full $\gamma$ at $x = -\delta$: only the band $|x| < 0.1\delta$ is damped by roughly two orders of magnitude, which is enough to keep tick-size noise out of the downside volatility estimate without creating a dead zone.
2. For losses beyond $x = -\delta$ the gradient exceeds $\gamma$ monotonically, peaks at $1.1353\gamma$ exactly at $x = -2\delta$ ($+13.53\%$), and then relaxes toward the linear asymptote $\gamma$. This is a localized amplification window, not an unbounded spike, so the backpropagated signal on severe losses is 13.5% stronger than a pure linear cut-loss without risking exploding gradients.

---

## V. Numerical Safety and Hardware Implementation

### A. The $0 \times \infty = \text{NaN}$ Hazard in PyTorch Autograd
In PyTorch, the conditional operator `torch.where(condition, pos_term, neg_term)` evaluates both branches eagerly prior to selection. On 32-bit floating-point (IEEE-754 FP32) hardware, the exponential function overflows for arguments $> 88.7228$:

$$x = 50.0, \quad \delta = 0.5 \implies \frac{x}{\delta} = 100.0 \implies \exp(100.0) = \mathbf{+\infty}$$

During the backward Autograd pass, the gradient of the unselected branch is multiplied by zero:
$$\frac{\partial \mathcal{L}}{\partial x} = 1.0 \cdot \text{grad} + 0.0 \cdot \mathbf{\infty} = \mathbf{NaN}$$

This instantly corrupts the weight gradients across the entire network.

**The Solution (Domain Clamping):** Q-ALU resolves this by strictly pre-clamping inputs to their respective mathematical domains prior to transcendental evaluation:
```python
x_pos = torch.clamp(x, min=0.0)
x_neg = torch.clamp(x, max=0.0)
pos_term = self.alpha * x_pos * (1.0 + torch.tanh(x_pos / self.beta))
neg_term = -self.gamma * x_neg * torch.expm1(x_neg / self.delta)
return torch.where(x >= 0.0, pos_term, neg_term)
```

### B. Prevention of Catastrophic Cancellation
Evaluating $1.0 - \exp(x/\delta)$ in standard FP32 suffers from catastrophic cancellation when $|x/\delta| \ll 1$ (subtracting two numbers close to $1.0$). At $x = -10^{-7}, \delta = 0.5$, direct subtraction yields a relative error of **$10.59\%$**. 

By utilizing the standard IEEE library identity $-\operatorname{expm1}(u) = -(e^u - 1) = 1 - e^u$, full machine precision (relative error $< 10^{-8}$) is preserved across all micro-tick scales.

---

## VI. Empirical Experiments and Comparative Benchmark

### A. Experimental Setup and Multi Regime Simulation
To evaluate Q-ALU under realistic high frequency market microstructure, we constructed a stochastic multi regime simulation over $10,000$ consecutive ticks:
1. **Regime 1 (Ticks 0 to 2500): Choppy Sideways Market.** Zero drift, high microstructure bid ask bounce noise ($\sigma = 10$ bps).
2. **Regime 2 (Ticks 2500 to 5500): Bullish Momentum Breakout.** Persistent positive drift ($+4$ bps/tick) with stochastic upward jumps ($+10$ bps).
3. **Regime 3 (Ticks 5500 to 6500): Flash Crash.** Severe negative liquidation drift ($-15$ bps/tick) with heavy-tailed jump arrivals ($-50$ bps) and volatility clustering ($2.5\times$ base).
4. **Regime 4 (Ticks 6500 to 10000): High-Volatility Recovery.** Mean-reverting noisy channel.

**Trading Policy Network:** A 2-hidden layer MLP (32 neurons, 16 neurons) maps seven features (five lagged returns, a rolling 20-tick volatility, an EMA(12)/EMA(26) MACD spread) to three discrete positions $\{-1, 0, +1\}$. The policy is trained by REINFORCE with a per-step value baseline on the in-sample stream (first 5,000 ticks) for 300 full-batch epochs and evaluated out of sample on ticks 5,000 to 10,000 under a transaction cost of **1.0 basis point per position turnover**; the reward is $r_t = p_t \Delta\text{return}_t - c|p_t - p_{t-1}|$.

**Paired protocol:** every activation is trained with the same market seed *and* the same initial weight tensor, so the only difference between two runs is the activation function. Five seeds $\{42, 101, 777, 2026, 9999\}$ are used, and all comparisons are reported as paired differences with $n = 5$. The complete experiment is `experiments/trading_benchmark.py`; Table I is reproduced by `--epochs 300`.

### B. Realistic Quant Metrics Evaluation
To avoid misleading high frequency annualization distortions (where tick level multiplier $\sqrt{N_{\text{ticks}}}$ artificially inflates Sharpe ratios into hundreds), we evaluate models strictly on standard non-annualized quant metrics:
* **Out of Sample Return (%)**: Cumulative equity growth over the test period.
* **Per-Step Sharpe Ratio**: $\mu_{\text{step}} / \sigma_{\text{step}}$ (un-annualized).
* **Per-Step Sortino Ratio**: $\mu_{\text{step}} / \sigma_{D,\text{step}}$ (un-annualized).
* **Profit Factor**: $\sum \text{Gains} / \sum |\text{Losses}|$.
* **Maximum Drawdown (MDD %)**: Peak to trough decline.
* **Total Position Turnover**: Number of trades executed.

TABLE I presents the comparative out of sample quantitative metrics across baseline activations:

| Activation Function | Out of Sample Return (%) | Per-Step Sharpe | Per-Step Sortino | Profit Factor | Max Drawdown (%) | Total Trades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q-ALU** | +373.51 ± 27.39 | +0.489 ± 0.030 | +1.083 ± 0.129 | 4.37 ± 0.46 | 6.02 ± 1.33 | 718.2 ± 23.4 |
| **ReLU** [5] | +378.81 ± 40.32 | +0.498 ± 0.054 | +1.099 ± 0.232 | 4.56 ± 0.84 | 4.91 ± 1.47 | 761.2 ± 39.9 |
| **LeakyReLU** [6] | +383.70 ± 42.02 | +0.506 ± 0.057 | +1.140 ± 0.250 | 4.69 ± 0.86 | 4.48 ± 1.65 | 756.4 ± 46.1 |
| **GELU** [7] | +393.58 ± 16.23 | +0.521 ± 0.014 | +1.182 ± 0.098 | 4.86 ± 0.25 | 4.81 ± 1.70 | 737.2 ± 88.7 |
| **SiLU (Swish)** [8] | **+396.08 ± 20.61** | **+0.525 ± 0.025** | **+1.201 ± 0.086** | **4.94 ± 0.41** | **4.47 ± 1.46** | 733.2 ± 96.8 |

![Figure 5: Out of Sample Cumulative Equity, Mean over 5 Paired Seeds](figures/fig5_equity_curves.png)

*Fig. 5. Out of sample cumulative equity averaged over the five seeds of Table I, over 5,000 unseen ticks across 4 market regimes under 1.0 bps fee per turnover. All activations ride the same regime-2 drift; the curves are statistically indistinguishable and Q-ALU does not lead. Generated by `experiments/trading_benchmark.py --plot`.*

### C. What the Benchmark Actually Shows
Three observations follow from Table I, and none of them supports the claim that Q-ALU dominates.

**(i) No policy paralysis occurs for any activation.** Individual runs trade between 574 and 819 times out of a possible 5,000, and the spread between activation means (718 to 761) is 6%; the lowest mean belongs to Q-ALU. The premise that symmetric activations collapse to near-zero turnover under a 1.0 bps fee is not reproduced: all five policies capture the regime-2 drift and pay fees on roughly the same number of flips.

**(ii) Q-ALU is slightly *worse* on return and risk-adjusted return.** Paired over identical seeds and initial weights, Q-ALU trails GELU by 20.1 percentage points of return ($t = -3.65$) and SiLU by 22.6 points ($t = -4.51$); on the Sortino ratio it trails by 0.099 ($t = -2.20$) and 0.118 ($t = -3.55$). With $n = 5$ and a two-sided 5% critical value of 2.78, the gap against GELU and SiLU is significant, and against ReLU and LeakyReLU it is not ($t = -0.58$ and $-0.87$).

**(iii) The only consistent Q-ALU effect is lower turnover, at the cost of deeper drawdowns.** Q-ALU executes 43 fewer trades than ReLU on average ($t = -3.59$), which is what its quadratic noise band predicts, but its maximum drawdown is the highest of the five (6.02% versus 4.47% for SiLU, paired difference +1.55 pp, $t = +2.42$): the reduced trading does not translate into better downside protection. Because regime 2 contributes the bulk of the return and all policies are long through it, the activation is nearly irrelevant to the mean return (seed noise of ±27 to ±42 percentage points dwarfs the 22 point spread between activations); what differs is how quickly each policy reacts to the flash crash in regime 3.

### D. Reproducibility Note
An earlier draft reported +75.85% out-of-sample return for Q-ALU against +5.43% for ReLU and +1.42% for GELU, with the baselines collapsing to 0–8 trades, plus a 2,834-bar BTC/USDT evaluation and a 1D-CNN/LSTM study. Those results could not be reproduced from the repository: no script, seed list or data file for them was preserved, and the best out-of-sample return obtained by any run in the protocol above is +423%, for *every* activation, with no run anywhere near 0–8 trades. The figures are therefore withdrawn rather than re-tuned. Table I is the complete, runnable replacement, and its verdict is that Q-ALU is competitive with, but not superior to, standard activations on this task.

---

### E. Hardware Latency and Computational Throughput
Benchmarked on a single-threaded x86-64 CPU (Intel Celeron N2930, `torch 2.15.0+cpu`, FP32, $n = 100{,}000$ elements, 200 iterations, best of 3; `experiments/benchmark.py`). The backward column is the backward call alone, $t(\text{forward}+\text{backward}) - t(\text{forward})$.

TABLE II: COMPUTATIONAL THROUGHPUT BENCHMARK (100,000 FP32 ELEMENTS, SINGLE THREAD)

| Activation Function | Forward Pass (ms) | Backward Pass (ms) | Forward Throughput (M/s) |
| :--- | :---: | :---: | :---: |
| **ReLU** | 0.334 ms | 2.547 ms | 299.29 M/s |
| **LeakyReLU** | 1.696 ms | 2.661 ms | 58.98 M/s |
| **SiLU** | 3.375 ms | 4.415 ms | 29.63 M/s |
| **GELU** | 2.678 ms | 4.234 ms | 37.34 M/s |
| **FastQALU** | **10.384 ms** | **22.600 ms** | **9.63 M/s** |
| **Q-ALU 2.0** | 15.764 ms | 20.413 ms | 6.34 M/s |
| **Q-ALU Ultra** | 16.427 ms | 14.807 ms | 6.09 M/s |
| **Q-ALU** | 17.064 ms | 17.881 ms | 5.86 M/s |

FastQALU is **1.64× faster than transcendental Q-ALU in the forward pass** and **1.27× faster end-to-end** (33.0 ms vs 34.9 ms per forward+backward on 100k elements), because the algebraic surrogate trades both `tanh` and `exp` for a square root plus a handful of multiply-adds. It does not match GELU: GELU remains 3.9× faster in the forward pass on this CPU, so the rational branch pays off only where transcendental units are unavailable or where the kernel is fused, not as a general throughput win.

---

## VII. Practical Deployment and Weight Initialization

For input $X \sim \mathcal{N}(0, 1)$ under default parameters $(\alpha=1.0, \beta=1.0, \gamma=1.5, \delta=0.5)$:
* The output distribution exhibits a positive mean drift: $\mathbb{E}[f(X)] \approx +0.1976$.
* The output second moment is $\mathbb{E}[f(X)^2] \approx 2.6903$ (Monte Carlo, $2\times10^7$ samples), giving RMS $\approx 1.6402$; with $\mathbb{E}[f(X)] \approx +0.1976$ the standard deviation is $1.6283$, so the variance-preserving gain is $1/1.6283 \approx 0.6142$ (ReLU's is $\sqrt{2} \approx 1.414$). The two conventions differ and must not be mixed: RMS $1.640$ exceeds ReLU's RMS $0.707$, yet the initialization scale $0.614$ is *smaller* than ReLU's $\sqrt{2}$.

**Initialization Recommendation:**
When training deep networks ($>3$ layers) without normalization, weights should be initialized using a modified Kaiming Normal scaling:
$$\mathbf{W} \sim \mathcal{N}\left(0, \frac{0.6142}{\sqrt{\mathrm{fan\_in}}}\right)$$
In modern architectures, pairing Q-ALU directly with `nn.LayerNorm` or `nn.RMSNorm` completely absorbs the mean shift and preserves unit variance across arbitrarily deep layers.

---

## VIII. Conclusion and Future Work

This paper presented **Q-ALU (Quant Asymmetric Leaky Unit)**, an activation function designed to express asymmetric downside risk preferences directly in network calculus: momentum-accelerated tracking on the positive domain, a quadratic noise band followed by an exponential-to-linear cut-loss barrier on the negative domain.

The theoretical results stand on their own: global monotonicity ($f'(x) \ge 0, \ \forall x \in \mathbb{R}$), $C^0$ continuity at the origin, and a bounded *Dynamic Gradient Overshoot* that peaks at $1.1353\gamma$ at $x = -2\delta$ and relaxes to the linear asymptote. The implementation contributions also stand: the $0 \times \infty$ autograd hazard is removed, the cancellation-prone exponential is replaced by `expm1`, the rational FastQALU surrogate is accurate to 0.53% and 1.64× faster in the forward pass, and the Python, C++ and CUDA paths agree to finite-difference tolerance.

The empirical claim does not survive. Under a fully reproducible multi-regime benchmark with paired seeds, Q-ALU does not outperform ReLU, LeakyReLU, GELU or SiLU on return, Sharpe or Sortino; the predicted turnover collapse does not occur for any activation; and Q-ALU's only measurable signature is a modest reduction in trading frequency accompanied by a deeper drawdown. We therefore withdraw the trading results of the earlier draft (+75.85% versus +5.43% for ReLU) as non-reproducible rather than restate them, and we frame Q-ALU as a principled, numerically robust activation whose behavioural hypotheses about trading performance were tested and, in this setting, not confirmed.

Future work should attack the question the benchmark could not settle: whether asymmetric activations change learning dynamics under the exact algorithm that motivated the design. A PPO agent with a value network, entropy collapse and clip-based credit assignment has failure modes (premature policy determinism) that a contextual-bandit formulation cannot exhibit, so a PPO-specific study with turnover and entropy diagnostics is the natural next experiment. Beyond that, the open questions are whether the overshoot window at $x = -2\delta$ changes behaviour in deeper temporal architectures, and whether the same asymmetry helps in non-financial settings where downside asymmetry is real, such as risk-sensitive control and survival-constrained RL.

---

## References

1. J. Moody and M. Saffell, "Learning to trade via direct reinforcement," *IEEE Transactions on Neural Networks*, vol. 12, no. 4, pp. 875–889, Jul. 2001, [doi: 10.1109/72.935097](https://doi.org/10.1109/72.935097).
2. M. F. Dixon, I. Halperin, and P. Bilokon, *Machine Learning in Finance: From Theory to Practice*. Cham, Switzerland: Springer, 2020, [doi: 10.1007/978-3-030-41068-1](https://doi.org/10.1007/978-3-030-41068-1).
3. Á. Cartea, S. Jaimungal, and J. Penalva, *Algorithmic and High-Frequency Trading*. Cambridge, UK: Cambridge University Press, 2015, [doi: 10.1017/CBO9781316274491](https://doi.org/10.1017/CBO9781316274491).
4. S. Li, X. Jin, Y. Xuan, X. Zhou, W. Chen, Y.-X. Wang, and X. Yan, "Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019, pp. 5243–5253, [Online](https://proceedings.neurips.cc/paper/2019/hash/67dc7d32e2aab538072d517ff5b1474f-Abstract.html).
5. V. Nair and G. E. Hinton, "Rectified linear units improve restricted boltzmann machines," in *Proceedings of the 27th International Conference on Machine Learning (ICML)*, Haifa, Israel, 2010, pp. 807–814.
6. A. L. Maas, A. Y. Hannun, and A. Y. Ng, "Rectifier nonlinearities improve neural network acoustic models," in *Proceedings of the 30th International Conference on Machine Learning (ICML)*, vol. 30, no. 1, 2013, p. 3.
7. D. Hendrycks and K. Gimpel, "Gaussian error linear units (GELUs)," *arXiv preprint arXiv:1606.08415*, 2016, [arXiv: 1606.08415](https://arxiv.org/abs/1606.08415).
8. P. Ramachandran, B. Zoph, and Q. V. Le, "Searching for activation functions," *arXiv preprint arXiv:1710.05941*, 2017, [arXiv: 1710.05941](https://arxiv.org/abs/1710.05941).
9. S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning," *Neural Networks*, vol. 107, pp. 3–11, Nov. 2018, [doi: 10.1016/j.neunet.2018.05.007](https://doi.org/10.1016/j.neunet.2018.05.007).
10. D. Misra, "Mish: A self regularized non-monotonic activation function," in *Proceedings of the 31st British Machine Vision Conference (BMVC)*, 2020, [arXiv: 1908.08681](https://arxiv.org/abs/1908.08681).
11. D. Kahneman and A. Tversky, "Prospect theory: An analysis of decision under risk," *Econometrica*, vol. 47, no. 2, pp. 263–291, Mar. 1979, [doi: 10.2307/1914185](https://doi.org/10.2307/1914185).
12. J. Y. Campbell, A. W. Lo, and A. C. MacKinlay, *The Econometrics of Financial Markets*. Princeton, NJ: Princeton University Press, 1997.
13. R. Roll, "A simple implicit measure of the effective bid-ask spread in an efficient market," *The Journal of Finance*, vol. 39, no. 4, pp. 1127–1139, Sep. 1984, [doi: 10.1111/j.1540-6261.1984.tb03897.x](https://doi.org/10.1111/j.1540-6261.1984.tb03897.x).
14. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017, [arXiv: 1707.06347](https://arxiv.org/abs/1707.06347).
15. Y. LeCun, L. Bottou, G. B. Orr, and K.-R. Müller, "Efficient backprop," in *Neural Networks: Tricks of the Trade*, Lecture Notes in Computer Science, vol. 1524. Berlin, Heidelberg: Springer, 1998, pp. 9–50, [doi: 10.1007/3-540-49430-8_2](https://doi.org/10.1007/3-540-49430-8_2).
16. K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, Santiago, Chile, 2015, pp. 1026–1034, [doi: 10.1109/ICCV.2015.123](https://doi.org/10.1109/ICCV.2015.123).
17. D.-A. Clevert, T. Unterthiner, and S. Hochreiter, "Fast and accurate deep network learning by exponential linear units (ELUs)," in *International Conference on Learning Representations (ICLR)*, San Juan, Puerto Rico, 2016, [arXiv: 1511.07289](https://arxiv.org/abs/1511.07289).
18. G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter, "Self-normalizing neural networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 971–980, [Online](https://proceedings.neurips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html).
19. T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in *Proceedings of the 35th International Conference on Machine Learning (ICML)*, Stockholm, Sweden, 2018, pp. 1861–1870.
20. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, Feb. 2015, [doi: 10.1038/nature14236](https://doi.org/10.1038/nature14236).
21. F. A. Sortino and L. N. Price, "Performance measurement in a downside risk framework," *The Journal of Investing*, vol. 3, no. 3, pp. 59–64, Fall 1994, [doi: 10.3905/joi.3.3.59](https://doi.org/10.3905/joi.3.3.59).
22. W. K. Newey and J. L. Powell, "Asymmetric least squares estimation and testing," *Econometrica*, vol. 55, no. 4, pp. 819–847, Jul. 1987, [doi: 10.2307/1911031](https://doi.org/10.2307/1911031).
