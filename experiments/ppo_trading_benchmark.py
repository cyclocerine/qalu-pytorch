"""PPO version of the trading benchmark, using the algorithm named in the paper.

The paper's discussion section raises "policy paralysis" (turnover collapse) as
a hypothesis about Proximal Policy Optimization agents.  This script trains a
PPO agent with a value network on 64 synthetic market paths and evaluates it
greedily on the five held-out paths whose seeds are the ones used in the paper
(42, 101, 777, 2026, 9999).  Policy and value networks share the same
activation function, so the only difference between runs is the activation.

STATUS: experimental, and not the source of any number in the paper.  Table I is
produced by the lighter REINFORCE benchmark in trading_benchmark.py, because a
full PPO sweep (5 activations x 5 seeds) did not finish on the 4-core Celeron
used for the other experiments: the run was killed after ~20 minutes per seed
with no epoch completed.  Treat this script as a starting point for a GPU host,
not as a verified result.

Usage:
    python experiments/ppo_trading_benchmark.py --seeds 42 --rollouts 2   # smoke
    python experiments/ppo_trading_benchmark.py                           # full
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qalu import QALU, QALU2, QALU_Ultra, FastQALU
from trading_benchmark import (FEE_BPS, N_FEATURES, SPLIT, build_features,
                               evaluate, simulate)

GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-3
EPOCHS_PER_ROLLOUT = 4
MINIBATCH = 2048


def make_activation(name: str) -> nn.Module:
    return {
        "Q-ALU": QALU,
        "Q-ALU 2.0": QALU2,
        "Q-ALU Ultra": QALU_Ultra,
        "FastQALU": FastQALU,
        "ReLU": nn.ReLU,
        "LeakyReLU": lambda: nn.LeakyReLU(0.01),
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
    }[name]()


class ActorCritic(nn.Module):
    def __init__(self, act_name: str):
        super().__init__()
        self.pi1 = nn.Linear(N_FEATURES, 32)
        self.pi2 = nn.Linear(32, 16)
        self.pi_out = nn.Linear(16, 3)
        self.v1 = nn.Linear(N_FEATURES, 32)
        self.v2 = nn.Linear(32, 16)
        self.v_out = nn.Linear(16, 1)
        self.act_pi = make_activation(act_name)
        self.act_v = make_activation(act_name)

    def policy(self, x):
        return self.pi_out(self.act_pi(self.pi2(self.act_pi(self.pi1(x)))))

    def value(self, x):
        return self.v_out(self.act_v(self.v2(self.act_v(self.v1(x))))).squeeze(-1)


def collect(net, F, r):
    """One full rollout over all envs. F: (E,T,D), r: (E,T)."""
    E, T, D = F.shape
    obs = torch.zeros(E, T, D)
    act = torch.zeros(E, T, dtype=torch.long)
    logp = torch.zeros(E, T)
    val = torch.zeros(E, T)
    rew = torch.zeros(E, T)
    prev = torch.zeros(E)
    with torch.no_grad():
        for t in range(T):
            x = F[:, t]
            logits = net.policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            pos = a.float() - 1.0
            dpos = torch.abs(pos - prev)
            prev = pos
            obs[:, t] = x
            act[:, t] = a
            logp[:, t] = dist.log_prob(a)
            val[:, t] = net.value(x)
            rew[:, t] = pos * r[:, t] - FEE_BPS * dpos
    return obs, act, logp, val, rew


def gae(rew, val):
    E, T = rew.shape
    adv = torch.zeros(E, T)
    last = torch.zeros(E)
    for t in reversed(range(T)):
        nxt = val[:, t + 1] if t + 1 < T else torch.zeros(E)
        delta = rew[:, t] + GAMMA * nxt - val[:, t]
        last = delta + GAMMA * LAM * last
        adv[:, t] = last
    return adv, adv + val


def train(act_name, F, r, seed, rollouts):
    torch.manual_seed(seed)
    net = ActorCritic(act_name)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(rollouts):
        obs, act, logp_old, val_old, rew = collect(net, F, r)
        adv, ret = gae(rew, val_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        E, T, D = obs.shape
        flat = lambda z: z.reshape(-1, *z.shape[2:])
        o, a, lp_old, ad, rt = flat(obs), flat(act), flat(logp_old), flat(adv), flat(ret)
        n = o.shape[0]
        for _ in range(EPOCHS_PER_ROLLOUT):
            perm = torch.randperm(n)
            for i in range(0, n, MINIBATCH):
                idx = perm[i:i + MINIBATCH]
                logits = net.policy(o[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a[idx])
                ratio = (lp - lp_old[idx]).exp()
                s1 = ratio * ad[idx]
                s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ad[idx]
                pi_loss = -torch.min(s1, s2).mean()
                v = net.value(o[idx])
                v_loss = ((v - rt[idx]) ** 2).mean()
                loss = pi_loss + VF_COEF * v_loss - ENT_COEF * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 777, 2026, 9999])
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--activations", nargs="+",
                    default=["Q-ALU", "ReLU", "LeakyReLU", "GELU", "SiLU"])
    args = ap.parse_args()

    # 64 training paths, five held-out evaluation paths
    trF, trR = [], []
    for s in range(1000, 1000 + args.n_envs):
        r = simulate(s)
        trR.append(r[:SPLIT])
        trF.append(build_features(r)[:SPLIT])
    trF = np.stack(trF)
    trR = np.stack(trR)
    mu, sd = trF.reshape(-1, N_FEATURES).mean(0), trF.reshape(-1, N_FEATURES).std(0) + 1e-9
    F = torch.from_numpy((trF - mu) / sd).float()
    R = torch.from_numpy(trR).float()

    print(f"# PPO: {args.n_envs} train paths x {SPLIT} ticks, {args.rollouts} rollouts, "
          f"gamma={GAMMA}, lam={LAM}, clip={CLIP}, fee={FEE_BPS} bps")
    results = {k: [] for k in args.activations}
    for seed in args.seeds:
        r = simulate(seed)
        Fev = ((build_features(r) - mu) / sd)[SPLIT:]
        rev = r[SPLIT:]
        for name in args.activations:
            net = train(name, F, R, seed, args.rollouts)
            m = evaluate(net.policy, Fev, rev)
            results[name].append(m)
            print(f"  seed {seed:5d} {name:12s} ret={m['return_pct']:+8.2f}% "
                  f"sharpe={m['sharpe']:+.3f} sortino={m['sortino']:+.3f} "
                  f"pf={m['profit_factor']:5.2f} mdd={m['mdd_pct']:5.2f}% "
                  f"trades={m['trades']}", flush=True)

    print()
    hdr = (f"{'Activation':12s} {'Return %':>17s} {'Sharpe':>15s} {'Sortino':>15s} "
           f"{'PF':>13s} {'MDD %':>13s} {'Trades':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for name, rows in results.items():
        def ms(key, fmt):
            v = np.array([x[key] for x in rows], dtype=float)
            return fmt.format(v.mean(), v.std())
        print(f"{name:12s} {ms('return_pct', '{:+8.2f} ±{:5.2f}'):>17s} "
              f"{ms('sharpe', '{:+.3f} ±{:.3f}'):>15s} "
              f"{ms('sortino', '{:+.3f} ±{:.3f}'):>15s} "
              f"{ms('profit_factor', '{:5.2f} ±{:.2f}'):>13s} "
              f"{ms('mdd_pct', '{:5.2f} ±{:.2f}'):>13s} "
              f"{ms('trades', '{:6.1f} ±{:4.1f}'):>12s}")

    # paired comparison against the strongest baseline, same seeds
    base = max((k for k in results if k != "Q-ALU"),
               key=lambda k: np.mean([x["return_pct"] for x in results[k]]))
    d = np.array([a["return_pct"] - b["return_pct"]
                  for a, b in zip(results["Q-ALU"], results[base])])
    print(f"\npaired Q-ALU - {base} return: mean {d.mean():+.2f} pp, "
          f"sd {d.std(ddof=1):.2f}, t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}, n={len(d)}")


if __name__ == "__main__":
    main()
