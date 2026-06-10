"""training -- the optimization loop and grid runner.

`Trainer`/`train_model` run any `tasks.Task` against a `model.MLP` and checkpoint
the result; `train_grid` (and the `python -m training.run` CLI) sweep a
depth x width x seed grid for a chosen task.
"""
from .trainer import Trainer, TrainConfig, train_model
from .run import train_grid, build_task

__all__ = ["Trainer", "TrainConfig", "train_model", "train_grid", "build_task"]
