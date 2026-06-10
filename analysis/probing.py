"""analysis/probing.py -- linear probes on activations.

Fit a linear map from a representation to a target and report how well the target
is *linearly* decodable -- the standard interpretability probe.

  * `linear_probe(X, y, ridge)` -- least-squares (optionally ridge) fit X -> y with a
    bias term; returns weights/bias, per-output R², and (for 0/1 targets) accuracy.
  * `probe_layer(model, x, y, layer)` -- collect a hidden layer's activations for
    inputs `x` and probe them against targets `y`.

Pure torch (no sklearn dependency).
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import torch

from .common import collect_activations


def linear_probe(X, y, *, ridge: float = 0.0) -> Dict:
    """Fit a linear probe `X @ W + b ≈ y` by (ridge) least squares.

    X: (n, d), y: (n,) or (n, k). Returns weights (d, k), bias (k,), per-output R²,
    predictions, and -- when y is binary {0,1} -- threshold-0.5 accuracy.
    """
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    n, d = X.shape
    Xa = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)        # bias column
    A = Xa.T @ Xa + ridge * torch.eye(d + 1, dtype=X.dtype)
    W = torch.linalg.solve(A, Xa.T @ y)                               # (d+1, k)
    pred = Xa @ W
    resid = ((y - pred) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0).clamp_min(1e-30)
    r2 = (1.0 - resid / tot).numpy()

    out = {
        "weights": W[:d].numpy(),
        "bias": W[d].numpy(),
        "r2": r2,
        "predictions": pred.numpy(),
    }
    is_binary = bool(((y == 0) | (y == 1)).all().item())
    if is_binary:
        out["accuracy"] = ((pred > 0.5).double() == y).double().mean(0).numpy()
    return out


def probe_layer(model, x, y, layer: int = -1, *, use_post: bool = True, ridge: float = 0.0) -> Dict:
    """Probe a hidden layer's activations (for inputs `x`) against targets `y`.

    `layer` indexes hidden layers (negative from the end). Returns the
    `linear_probe` result plus the probed layer index.
    """
    acts = collect_activations(model, x)
    keys = sorted(acts["post"].keys())
    li = keys[layer] if layer < 0 else layer
    A = acts["post"][li] if use_post else acts["pre"][li]
    result = linear_probe(A, y, ridge=ridge)
    result["layer"] = li
    return result
