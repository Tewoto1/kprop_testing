"""tasks/base.py -- the Task interface.

A `Task` is a pure data+objective definition: it knows how to draw an
`(input, target)` batch and how to score a model's output against that target.
It deliberately knows *nothing* about optimization -- the training loop lives in
the `training` package, which consumes any `Task`. This keeps `tasks` free of a
trainer dependency so a task can be reused for evaluation, plotting, or a custom
loop.

`Task` is an abstract base class: both `sample_batch` (how to draw a batch) and
`loss` (how to score an output against a target) are `@abstractmethod`, so each
task owns its own objective rather than inheriting a shared one. A subclass that
forgets either method -- or `Task` itself -- cannot be instantiated (the failure
is at construction with a clear message, rather than deep inside the training
loop on the first batch). The tasks shipped here all happen to use MSE, but each
declares that explicitly so a non-MSE task can drop in without special-casing.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import torch


class Task(ABC):
    """Base task. Subclasses implement `sample_batch` and `loss`.

    A task also advertises the input/output dimensionality it produces via
    `input_dim` / `output_dim` so callers can size a matching model.
    """

    input_dim: int
    output_dim: int = 1

    @abstractmethod
    def sample_batch(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(x, y)` of shape `(batch_size, input_dim)` / `(batch_size, output_dim)`."""
        raise NotImplementedError

    @abstractmethod
    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scalar objective scoring `output` against `target` for this task."""
        raise NotImplementedError
