"""
Reproducible throughput benchmark for Q-ALU and baseline activations.

Produces the numbers in Table II of the paper (forward / backward latency and
forward throughput on 100,000 FP32 elements). Runs on CPU with a single thread
so that the comparison reflects algorithmic cost rather than BLAS threading.

Usage:
    python experiments/benchmark.py [--n 100000] [--iters 200] [--threads 1]

Every activation is wrapped as an nn.Module so that autograd cost is included
in the backward column. Backward latency is measured as
(forward+backward) - forward, which is the cost of the graph traversal itself.
"""

import argparse
import platform
import time

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from qalu import QALU, QALU2, QALU_Ultra, FastQALU


class Wrapped(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fwd(module, x, iters):
    for _ in range(max(10, iters // 10)):
        module(x)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        module(x)
    _sync()
    return (time.perf_counter() - t0) / iters


def time_fwd_bwd(module, x, iters):
    def step():
        inp = x.clone().requires_grad_(True)
        module(inp).sum().backward()

    for _ in range(max(10, iters // 10)):
        step()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    acts = [
        ("ReLU", nn.ReLU()),
        ("LeakyReLU", nn.LeakyReLU(0.01)),
        ("SiLU", nn.SiLU()),
        ("GELU", nn.GELU()),
        ("Q-ALU", Wrapped(QALU())),
        ("QALU2", Wrapped(QALU2())),
        ("QALU-Ultra", Wrapped(QALU_Ultra())),
        ("FastQALU", Wrapped(FastQALU())),
    ]

    x = torch.randn(args.n, dtype=torch.float32)
    rows = []
    for name, module in acts:
        fwd = min(time_fwd(module, x, args.iters) for _ in range(args.repeats)) * 1e3
        both = min(time_fwd_bwd(module, x, args.iters) for _ in range(args.repeats)) * 1e3
        bwd = max(both - fwd, 0.0)
        rows.append((name, fwd, bwd, args.n / (fwd * 1e3)))

    print(f"# host: {platform.processor() or platform.machine()}")
    print(f"# torch {torch.__version__} | threads={args.threads} | n={args.n} | iters={args.iters} | best of {args.repeats}")
    print(f"{'Activation':<14}{'Fwd (ms)':>10}{'Bwd (ms)':>10}{'Fwd M/s':>10}{'Bwd M/s':>10}")
    for name, fwd, bwd, mps in rows:
        print(f"{name:<14}{fwd:>10.3f}{bwd:>10.3f}{mps:>10.2f}{args.n / (bwd * 1e3):>10.2f}")

    relu = next(r for r in rows if r[0] == "ReLU")
    q = next(r for r in rows if r[0] == "Q-ALU")
    fq = next(r for r in rows if r[0] == "FastQALU")
    print(f"\nFastQALU vs Q-ALU: forward {q[1] / fq[1]:.2f}x, "
          f"forward+backward {(q[1] + q[2]) / (fq[1] + fq[2]):.2f}x")
    print(f"Q-ALU vs ReLU: forward+backward {(q[1] + q[2]) / (relu[1] + relu[2]):.2f}x slower")


if __name__ == "__main__":
    main()
