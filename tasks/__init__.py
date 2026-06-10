"""tasks -- pure task definitions (data + loss) for the trained-case study.

Each task draws `(x, y)` batches and scores a model with `loss` (MSE by default).
Training itself lives in the `training` package, which runs any of these tasks.

Tasks:
    ZeroTask       -- output 0 on a Gaussian ball                (train_to_zero.py)
    HalfspaceTask  -- separate ONE fixed random half-space        (halfspace.py)
    DistillTask    -- match a frozen random teacher MLP's output  (distillation.py)
"""
from .base import Task
from .train_to_zero import ZeroTask
from .halfspace import HalfspaceTask
from .distillation import DistillTask, random_teacher

__all__ = ["Task", "ZeroTask", "HalfspaceTask", "DistillTask", "random_teacher"]
