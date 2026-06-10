"""utils.py -- small shared helpers (device selection, seeding, numpy conv).

Imported by both `training/` and `analysis/`, so it lives at the repo root
alongside the top-level packages.
"""
from __future__ import annotations
import random
import numpy as np
import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """Return a torch device. 'auto' picks cuda -> mps -> cpu."""
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)
