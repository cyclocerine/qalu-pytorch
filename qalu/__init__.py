"""
Q-ALU: Quant Asymmetric Leaky Unit: A Market Psychology Aware Activation Function with Downside Risk Protection for High Frequency Quantitative Trading and Deep Reinforcement Learning
Author: Faiq Hammam Mutaqin (2026)
SSRN: https://ssrn.com/abstract=7365098
DOI: 10.2139/ssrn.7365098

Official PyTorch package for Q-ALU activation functions.
"""

from .core import QALU, FastQALU, QALU_Adaptive

__version__ = "1.0.0"
__author__ = "Faiq Hammam Mutaqin"
__all__ = ["QALU", "FastQALU", "QALU_Adaptive"]
