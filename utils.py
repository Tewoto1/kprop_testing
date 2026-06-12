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


_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def pick_dtype(prefer: str = "auto") -> torch.dtype:
    """Resolve a dtype string. The repo policy is FLOAT32 for training/inference
    (GPU-friendly; ~10-30x faster on CUDA than float64) and float64 only where
    accuracy demands it (kprop internals, eigendecompositions, MC accumulators).
    'auto' -> float32."""
    if prefer in (None, "auto"):
        return torch.float32
    if isinstance(prefer, torch.dtype):
        return prefer
    if prefer in _DTYPES:
        return _DTYPES[prefer]
    raise ValueError(f"unknown dtype {prefer!r}; choose from {list(_DTYPES)} or 'auto'")


def enable_fast_matmul(device: torch.device) -> None:
    """On CUDA, allow TF32 tensor-core matmuls for float32 (big speedup on
    Ampere+; ~1e-3 relative precision, fine for training -- measurement paths
    that need exactness use float64 anyway). No-op on cpu/mps."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


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
