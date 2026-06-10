"""tasks/base.py -- the Task interface.

A `Task` is a pure data+objective definition: it knows how to draw an
`(input, target)` batch and how to score a model's output against that target.
It deliberately knows *nothing* about optimization -- the training loop lives in
the `training` package, which consumes any `Task`. This keeps `tasks` free of a
trainer dependency so a task can be reused for evaluation, plotting, or a custom
loop.

Subclasses implement `sample_batch`; the default `loss` is plain MSE, which is
correct for every task shipped here (train-to-zero, single half-space, and
distillation are all MSE objectives, differing only in their targets). Override
`loss` for a non-MSE objective.
"""
from __future__ import annotations
from typing import Tuple
import torch


class Task:
    """Base task. Subclasses implement `sample_batch`; default loss is MSE.

    A task also advertises the input/output dimensionality it produces via
    `input_dim` / `output_dim` so callers can size a matching model.
    """

    input_dim: int
    output_dim: int = 1

    def sample_batch(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(x, y)` of shape `(batch_size, input_dim)` / `(batch_size, output_dim)`."""
        raise NotImplementedError

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean-squared error between model output and target (the default objective)."""
        return torch.mean((output - target) ** 2)
