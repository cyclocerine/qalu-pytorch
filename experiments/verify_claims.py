"""Numeric audit of every closed-form number quoted in the paper.

Checks
  1. E[f(X)], E[f(X)^2], effective gain and sigma for X ~ N(0,1), defaults
     (alpha=1, beta=1, gamma=1.5, delta=0.5) against the values printed in
     Section "Practical Deployment and Weight Initialization".
  2. The same quantities for ReLU, to justify the "sqrt(2) for ReLU" comparison.
  3. The catastrophic-cancellation claim at x = -1e-7, delta = 0.5 in FP32.
  4. The downside overshoot 1 + e^-2 and the momentum peak 2.1997*alpha.

Run: python experiments/verify_claims.py
"""
import numpy as np

ALPHA, BETA, GAMMA, DELTA = 1.0, 1.0, 1.5, 0.5


def f(x, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA):
    x = np.asarray(x, dtype=np.float64)
    pos = alpha * x * (1.0 + np.tanh(x / beta))
    neg = -gamma * x * np.expm1(x / delta)
    return np.where(x >= 0.0, pos, neg)


def hermite(n):
    """E[g(X)] for X ~ N(0,1) via Gauss-Hermite with weight 1/sqrt(2pi)."""
    t, w = np.polynomial.hermite_e.hermegauss(n)
    return t, w / np.sqrt(2.0 * np.pi)


def moments(g, n=200):
    t, w = hermite(n)
    vals = g(t)
    m1 = float(np.dot(w, vals))
    m2 = float(np.dot(w, vals * vals))
    return m1, m2


def mc_moments(g, n=20_000_000, seed=0):
    rng = np.random.default_rng(seed)
    s1 = s2 = 0.0
    done = 0
    chunk = 2_000_000
    while done < n:
        k = min(chunk, n - done)
        v = g(rng.standard_normal(k))
        s1 += float(np.sum(v))
        s2 += float(np.sum(v * v))
        done += k
    return s1 / n, s2 / n


def main():
    print("=" * 68)
    print("1. Q-ALU output moments, X ~ N(0,1), defaults")
    print("=" * 68)
    for n in (80, 120, 200, 300):
        m1, m2 = moments(f, n)
        print(f"  GH n={n:4d}: E[f] = {m1:+.6f}  E[f^2] = {m2:.6f}  "
              f"sigma = {np.sqrt(m2):.6f}  gain = {np.sqrt(m2):.4f}")
    m1, m2 = mc_moments(f)
    print(f"  Monte Carlo 2e7: E[f] = {m1:+.6f}  E[f^2] = {m2:.6f}  "
          f"sigma = {np.sqrt(m2):.6f}")
    print("  paper claims : E[f] = +0.1974  E[f^2] = 2.6902  sigma = 0.6097")

    print()
    print("=" * 68)
    print("2. ReLU reference")
    print("=" * 68)
    relu = lambda x: np.maximum(x, 0.0)
    m1r, m2r = moments(relu, 200)
    print(f"  ReLU: E[relu] = {m1r:.6f} (1/sqrt(2pi) = {1/np.sqrt(2*np.pi):.6f}), "
          f"E[relu^2] = {m2r:.6f}, gain = {np.sqrt(m2r):.6f} (sqrt2 = {np.sqrt(2):.6f})")

    print()
    print("=" * 68)
    print("3. Catastrophic cancellation, x = -1e-7, delta = 0.5 (FP32)")
    print("=" * 68)
    for x in (-1e-5, -1e-6, -1e-7, -1e-8):
        u = np.float32(x) / np.float32(DELTA)
        naive = np.float32(1.0) - np.exp(u)              # 1 - exp(u)
        stable = -np.expm1(u)                            # expm1
        exact = -(np.expm1(np.float64(x) / DELTA))
        rel_naive = abs(float(naive) - exact) / abs(exact)
        rel_stable = abs(float(stable) - exact) / abs(exact)
        print(f"  x={x:+.0e}: naive rel err = {rel_naive*100:7.3f}%   "
              f"expm1 rel err = {rel_stable*100:.3e}%")
    print("  paper claims 10.59% relative error for direct subtraction at x=-1e-7")

    print()
    print("=" * 68)
    print("4. Gradient landmarks")
    print("=" * 68)
    h = lambda z: 1.0 - (1.0 - z) * np.exp(-z)
    print(f"  h(2) = 1 + e^-2 = {h(2.0):.6f}  (+{(h(2.0)-1)*100:.2f}% over asymptote)")
    # momentum peak: u*tanh(u) = 1
    from scipy.optimize import brentq
    u_star = brentq(lambda u: u * np.tanh(u) - 1.0, 0.5, 2.0)
    grad = lambda u: 1.0 + np.tanh(u) + u * (1.0 - np.tanh(u) ** 2)
    print(f"  u* = {u_star:.6f}, f'(u*)/alpha = {grad(u_star):.6f} "
          f"(+{(grad(u_star)-2)*100:.2f}% over 2*alpha)")
    print("  paper claims x* = 1.199678 beta and 2.1997 alpha")


if __name__ == "__main__":
    main()
