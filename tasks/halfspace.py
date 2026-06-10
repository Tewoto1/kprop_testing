"""tasks/halfspace.py -- the "separate ONE half-space" task.

The model must classify whether a Gaussian input lies on one side of a single
fixed random hyperplane through (offset) space: target y = 1[x . w > b] for a
unit normal w and scalar offset b, both drawn ONCE at construction and then held
fixed. Output is scalar (one half-space), trained with MSE to the 0/1 indicator.

Unlike train-to-zero (whose output mean -> 0), here the output mean tends to
E[y] = Phi(-b / ||projection||) != 0, so it produces a genuinely different learned
weight geometry -- a useful second case to study and to stress-test a mechanistic
predictor against.
"""
from __future__ import annotations
from typing import Optional
import torch

from .base import Task


class HalfspaceTask(Task):
    """Target y = 1[x . w > b], x ~ N(0, input_std^2 I_d). One fixed random half-space.

    The hyperplane (unit normal `w`, offset `b ~ N(0, offset_std^2)`) is drawn once
    using `seed` (if given) so the task is reproducible, and stored on CPU; it is
    moved onto the batch device in `sample_batch`.
    """

    output_dim = 1

    def __init__(self, input_dim: int, *, offset_std: float = 1.0, input_std: float = 1.0,
                 seed: Optional[int] = None):
        self.input_dim = input_dim
        self.offset_std = offset_std
        self.input_std = input_std
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        w = torch.randn(input_dim, generator=g)
        self.w = w / w.norm().clamp_min(1e-12)          # unit normal (input_dim,)
        self.b = float(offset_std * torch.randn(1, generator=g).item())  # scalar offset

    def sample_batch(self, batch_size, device):
        x = torch.randn(batch_size, self.input_dim, device=device) * self.input_std
        w = self.w.to(device)
        y = (x @ w - self.b > 0).to(x.dtype).unsqueeze(1)   # (batch, 1) in {0,1}
        return x, y
