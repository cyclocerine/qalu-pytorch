"""
Q-ALU Quickstart Example
Demonstrates how to use Q-ALU in a PyTorch neural network.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    import torch.nn as nn
    from qalu import QALU, FastQALU

    class TradingMLP(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5), # Q-ALU activation
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 32),
                FastQALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5), # FastQALU for low-latency inference
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    model = TradingMLP()
    x = torch.randn(32, 10)
    out = model(x)
    print("Forward pass successful!")
    print("Output shape:", out.shape)

    loss = out.sum()
    loss.backward()
    print("Backward pass successful! Gradient norm:", model.net[0].weight.grad.norm().item())

except ImportError:
    print("PyTorch not installed in this environment. Showing NumPy interface:")
    import numpy as np
    from qalu import QALU
    act = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = act.forward(x)
    print("Input: ", x)
    print("Output:", y)
