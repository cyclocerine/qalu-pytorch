"""
PPO Trading Demo: Integrating Q-ALU into a Deep Reinforcement Learning Actor-Critic Architecture
Author: Faiq Hammam Mutaqin (2026)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

# Simple NumPy PPO Policy Network using Q-ALU
class QALUPolicyNetwork:
    def __init__(self, state_dim=6, action_dim=3):
        from qalu import QALU
        self.w1 = np.random.randn(state_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.act1 = QALU(alpha=1.0, beta=1.0, gamma=1.5, delta=0.5)
        
        self.w2 = np.random.randn(32, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        
    def forward(self, state):
        z1 = np.dot(state, self.w1) + self.b1
        h1 = self.act1.forward(z1)
        logits = np.dot(h1, self.w2) + self.b2
        # Softmax policy
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

if __name__ == '__main__':
    policy = QALUPolicyNetwork()
    # Dummy market observation: [return_lag1, return_lag2, ema_diff, volatility, spread, position]
    obs = np.array([0.002, -0.001, 0.0005, 0.015, 0.0002, 1.0])
    action_probs = policy.forward(obs)
    print("Action Probabilities [Short, Flat, Long]:", action_probs)
    print("Best Action:", ["SHORT", "FLAT", "LONG"][np.argmax(action_probs)])
