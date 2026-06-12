"""tasks/distillation.py -- distill the output of another network.

The student is trained (MSE) to match the output of a *fixed* teacher network on
x ~ N(0, input_std^2 I_d): target = teacher(x). The teacher here is a frozen,
randomly-initialized `model.MLP` (a self-contained "other NN" -- no checkpoint
needed), so the task probes how the student reproduces a known target function
rather than collapsing to 0. Use `random_teacher(...)` to build one, or pass any
`model.MLP` instance you like (e.g. one loaded from a checkpoint).
"""
from __future__ import annotations
from typing import Optional
import torch

from model import MLP, ModelConfig
from .base import Task


def random_teacher(input_dim: int, hidden_dim: int, depth: int, *, output_dim: int = 1,
                   activation: str = "relu", bias: bool = False,
                   seed: Optional[int] = None) -> MLP:
    """Build a frozen, randomly-initialized teacher MLP (eval mode, grad disabled)."""
    cfg = ModelConfig(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth,
                      output_dim=output_dim, bias=bias, final_bias=bias,
                      activation=activation, seed=seed)
    teacher = cfg.build().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


class DistillTask(Task):
    """Target = teacher(x), x ~ N(0, input_std^2 I_d). MSE student-vs-teacher.

    `teacher` is any `model.MLP`; it is frozen and used under no-grad. Input/output
    dims are read off the teacher's config, so a distilling student should be built
    with a matching `input_dim`/`output_dim`.
    """

    def __init__(self, teacher: MLP, *, input_std: float = 1.0):
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.input_dim = teacher.cfg.input_dim
        self.output_dim = teacher.cfg.output_dim
        self.input_std = input_std
        self._device = None

    def sample_batch(self, batch_size, device):
        if self._device != device:                  # move teacher onto the batch device once
            self.teacher.to(device)
            self._device = device
        x = torch.randn(batch_size, self.input_dim, device=device) * self.input_std
        with torch.no_grad():
            y = self.teacher(x)
        return x, y

    def loss(self, output, target):
        """MSE of the student output against the teacher's output."""
        return torch.mean((output - target) ** 2)
