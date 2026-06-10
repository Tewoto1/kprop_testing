"""analysis/trained_to_0/activations.py -- sampling the input Gaussian ball.

`gaussian_batch` is how the train-to-zero study "covers the input ball": draw many
x ~ N(0, std^2 I_d). Activation collection itself is the general
`analysis.common.collect_activations`, re-exported here so the study modules can
keep importing both from one place.
"""
from __future__ import annotations
from typing import Optional
import torch

from ..common import collect_activations  # re-exported

__all__ = ["gaussian_batch", "collect_activations"]


def gaussian_batch(dim: int, n: int, std: float = 1.0,
                   seed: Optional[int] = None, device="cpu") -> torch.Tensor:
    """n samples covering the input ball: x ~ N(0, std^2 I_dim). Shape (n, dim)."""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    x = torch.randn(n, dim, generator=g) * std
    return x.to(device)
