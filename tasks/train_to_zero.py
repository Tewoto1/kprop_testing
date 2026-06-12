"""tasks/train_to_zero.py -- the "train an MLP to output 0" task.

Inputs are standard Gaussian vectors x ~ N(0, input_std^2 I_d). The target is the
zero vector, so the loss is simply E[||f(x)||^2]. With no weight decay the network
is not *forced* to shrink its weights -- it must find some geometry that sends a
whole Gaussian ball to ~0. That emergent structure is what the `analysis` package
(and especially `analysis.trained_to_0`) dissects. Data is resampled fresh every
step (infinite data), which is exactly "covering the Gaussian ball" rather than
memorising a finite set.
"""
from __future__ import annotations
import torch

from .base import Task


class ZeroTask(Task):
    """x ~ N(0, input_std^2 I_d); target = 0. Loss = E[||f(x)||^2] (MSE to 0)."""

    def __init__(self, input_dim: int, output_dim: int = 1, input_std: float = 1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_std = input_std

    def sample_batch(self, batch_size, device):
        x = torch.randn(batch_size, self.input_dim, device=device) * self.input_std
        y = torch.zeros(batch_size, self.output_dim, device=device)
        return x, y

    def loss(self, output, target):
        """E[||f(x)||^2]: MSE of the output against the zero target."""
        return torch.mean((output - target) ** 2)
