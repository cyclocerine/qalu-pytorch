"""Reproducible version of the multi-regime trading benchmark (paper, Table I).

The paper reports out-of-sample metrics for a 2-hidden-layer policy network
trained on the first half of a 10,000-tick synthetic stream and evaluated on the
second half under a 1.0 bps fee per unit turnover.  This script implements
exactly that protocol so the numbers in the paper can be regenerated:

  * 10,000 ticks split into the four regimes described in Section V-A
  * features: 5 lagged returns + rolling 20-tick volatility + MACD-style spread
  * policy: Linear(7,32) -> activation -> Linear(32,16) -> activation -> Linear(16,3)
    sampled as short / flat / long (contextual bandit, REINFORCE with baseline)
  * reward per tick: position * return - fee * |position change|
  * every activation sees identical data, seeds and initial weights (paired test)

Usage:
    python experiments/trading_benchmark.py                 # 5 seeds, full run
    python experiments/trading_benchmark.py --seeds 1 2 3   # custom seeds
    python experiments/trading_benchmark.py --epochs 100    # quicker
    python experiments/trading_benchmark.py --json out.json --plot eq.pdf

Reproducibility: seeds and initial weights are fixed, so two runs agree in
structure and to within ~1% in every metric, but BLAS reduction order depends on
the thread count, so bitwise-identical output requires the same OMP_NUM_THREADS
as the run that produced the published table (2 threads for Table I).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qalu import QALU, QALU2, QALU_Ultra, FastQALU

FEE_BPS = 1.0          # 1.0 bps per unit turnover
N_TICKS = 10_000
SPLIT = 5_000          # in-sample / out-of-sample boundary
N_FEATURES = 7
BPS = 1e-4             # 1 bps as a fraction


# ----------------------------------------------------------------------------
# Market simulation
# ----------------------------------------------------------------------------
def simulate(seed: int) -> np.ndarray:
    """Return per-tick log returns (in bps) for the four-regime stream."""
    rng = np.random.default_rng(seed)
    r = np.zeros(N_TICKS)

    # Regime 1 (0 - 2500): choppy sideways, sigma = 10 bps, zero drift
    r[0:2500] = rng.normal(0.0, 10.0, 2500)

    # Regime 2 (2500 - 5500): bullish breakout, +4 bps drift, +10 bps jumps
    seg = rng.normal(4.0, 8.0, 3000)
    jumps = rng.random(3000) < 0.02
    seg[jumps] += 10.0
    r[2500:5500] = seg

    # Regime 3 (5500 - 6500): flash crash, -15 bps drift, -50 bps jumps,
    # volatility clustering at 2.5x base
    seg = rng.normal(-15.0, 25.0, 1000)
    jumps = rng.random(1000) < 0.03
    seg[jumps] -= 50.0
    r[5500:6500] = seg

    # Regime 4 (6500 - 10000): mean-reverting noisy recovery channel
    seg = np.zeros(3500)
    level = 0.0
    for i in range(3500):
        level = 0.85 * level + rng.normal(0.0, 6.0)
        seg[i] = level
    r[6500:10000] = seg

    return r


def build_features(r: np.ndarray) -> np.ndarray:
    """(N_TICKS, 7) feature matrix: 5 lagged returns, rolling vol, MACD spread."""
    n = len(r)
    F = np.zeros((n, N_FEATURES))
    for lag in range(1, 6):
        F[lag:, lag - 1] = r[:-lag]
    vol = np.zeros(n)
    for t in range(20, n):
        vol[t] = r[t - 20:t].std()
    F[:, 5] = vol
    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    a12, a26 = 2.0 / 13.0, 2.0 / 27.0
    for t in range(1, n):
        ema12[t] = a12 * r[t] + (1 - a12) * ema12[t - 1]
        ema26[t] = a26 * r[t] + (1 - a26) * ema26[t - 1]
    F[:, 6] = ema12 - ema26
    return F


# ----------------------------------------------------------------------------
# Policy network
# ----------------------------------------------------------------------------
def make_activation(name: str) -> nn.Module:
    return {
        "Q-ALU": QALU(),
        "Q-ALU 2.0": QALU2(),
        "Q-ALU Ultra": QALU_Ultra(),
        "FastQALU": FastQALU(),
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.01),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
    }[name]


class Policy(nn.Module):
    def __init__(self, act: nn.Module):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 3)
        self.act = act

    def forward(self, x):
        return self.out(self.act(self.fc2(self.act(self.fc1(x)))))


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def train(act_name, F_tr, r_tr, seed, epochs, lr=3e-3):
    torch.manual_seed(seed)
    net = Policy(make_activation(act_name))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    x = torch.from_numpy(F_tr).float()
    ret = torch.from_numpy(r_tr).float()

    for _ in range(epochs):
        logits = net(x)
        logp = torch.log_softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        pos = a.float() - 1.0                      # {-1, 0, +1}
        dpos = torch.abs(pos[1:] - pos[:-1])
        dpos = torch.cat([torch.abs(pos[:1]), dpos])
        reward = pos * ret - FEE_BPS * dpos        # in bps
        adv = reward - reward.mean()
        loss = -(logp[torch.arange(len(a)), a] * adv).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def evaluate(net, F, r):
    logits = net(torch.from_numpy(F).float())
    pos = torch.softmax(logits, dim=-1).argmax(dim=-1).float() - 1.0
    dpos = torch.abs(pos[1:] - pos[:-1])
    dpos = torch.cat([torch.abs(pos[:1]), dpos])
    step = (pos * torch.from_numpy(r).float() - FEE_BPS * dpos).numpy()

    equity = np.cumsum(step)
    peak = np.maximum.accumulate(equity)
    mdd = float(np.max(peak - equity)) if len(equity) else 0.0
    gains = step[step > 0].sum()
    losses = -step[step < 0].sum()
    down = step[step < 0]
    return {
        "return_pct": float(step.sum() / 100.0),          # bps -> %
        "sharpe": float(step.mean() / step.std()) if step.std() > 0 else 0.0,
        "sortino": float(step.mean() / down.std()) if len(down) > 1 and down.std() > 0 else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "mdd_pct": mdd / 100.0,
        "trades": int((dpos > 0).sum().item()),
        "equity": equity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 777, 2026, 9999])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--activations", nargs="+",
                    default=["Q-ALU", "ReLU", "LeakyReLU", "GELU", "SiLU"])
    ap.add_argument("--json", type=str, default="", help="dump raw results to this path")
    ap.add_argument("--plot", type=str, default="",
                    help="write the mean out-of-sample equity curve to this PDF path")
    args = ap.parse_args()

    print(f"# {N_TICKS} ticks, in-sample {SPLIT}, fee {FEE_BPS} bps, "
          f"{args.epochs} epochs, seeds {args.seeds}")
    results = {k: [] for k in args.activations}

    for seed in args.seeds:
        r = simulate(seed)
        F = build_features(r)
        mu, sd = F[:SPLIT].mean(0), F[:SPLIT].std(0) + 1e-9
        Fz = (F - mu) / sd
        F_tr, r_tr = Fz[:SPLIT], r[:SPLIT]
        F_te, r_te = Fz[SPLIT:], r[SPLIT:]
        for name in args.activations:
            net = train(name, F_tr, r_tr, seed, args.epochs)
            m = evaluate(net, F_te, r_te)
            results[name].append(m)
            print(f"  seed {seed:5d} {name:12s} ret={m['return_pct']:+7.2f}% "
                  f"sharpe={m['sharpe']:+.3f} sortino={m['sortino']:+.3f} "
                  f"pf={m['profit_factor']:5.2f} mdd={m['mdd_pct']:.2f}% "
                  f"trades={m['trades']}")

    print()
    hdr = f"{'Activation':12s} {'Return %':>16s} {'Sharpe':>14s} {'Sortino':>14s} " \
          f"{'PF':>12s} {'MDD %':>12s} {'Trades':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for name, rows in results.items():
        def ms(key, fmt):
            v = np.array([r[key] for r in rows], dtype=float)
            return fmt.format(v.mean(), v.std())
        print(f"{name:12s} {ms('return_pct', '{:+7.2f} ±{:5.2f}'):>16s} "
              f"{ms('sharpe', '{:+.3f} ±{:.3f}'):>14s} "
              f"{ms('sortino', '{:+.3f} ±{:.3f}'):>14s} "
              f"{ms('profit_factor', '{:5.2f} ±{:.2f}'):>12s} "
              f"{ms('mdd_pct', '{:5.2f} ±{:.2f}'):>12s} "
              f"{ms('trades', '{:6.1f} ±{:4.1f}'):>10s}")

    # Paired comparison: identical seeds and identical initial weights per seed,
    # so the only difference is the activation. n = number of seeds.
    print("\nPaired differences (Q-ALU minus baseline, same seed):")
    for name in args.activations:
        if name == "Q-ALU":
            continue
        for key in ("return_pct", "sortino", "trades"):
            d = np.array([a[key] - b[key] for a, b in zip(results["Q-ALU"], results[name])])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {key:11s} vs {name:10s} mean {d.mean():+8.3f}  "
                  f"se {se:6.3f}  t {d.mean()/se:+5.2f}  n={len(d)}")

    if args.json:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"config": vars(args),
                       "results": {k: [{kk: vv for kk, vv in r.items() if kk != "equity"}
                                       for r in v] for k, v in results.items()}}, fh, indent=2)
        print(f"\nwrote {args.json}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        colors = {"Q-ALU": "#d62728", "ReLU": "#1f77b4", "LeakyReLU": "#2ca02c",
                  "GELU": "#9467bd", "SiLU": "#ff7f0e"}
        for name, rows in results.items():
            eq = np.mean([r["equity"] for r in rows], axis=0)
            ax.plot(eq, label=f"{name} (mean of {len(rows)} seeds)",
                    color=colors.get(name), lw=1.4 if name == "Q-ALU" else 1.0)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("out-of-sample tick")
        ax.set_ylabel("cumulative return (bps)")
        ax.set_title("Out-of-sample equity, mean over 5 paired seeds (1.0 bps fee)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(args.plot, bbox_inches="tight")
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
