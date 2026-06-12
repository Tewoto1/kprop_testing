"""analysis/common.py -- shared primitives for the analysis tools.

Two groups, used by the circuit tools (`pca`, `ablation`, `patching`,
`attribution`, `weight_structure`):

  * activation collection: `collect_activations` (forward once, return every
    layer's pre/post on CPU) and `run_with_intervention` (a hook-free forward that
    edits a chosen hidden layer's post-activation -- the causal-intervention
    primitive behind ablation and patching).
  * linear algebra: `mean_cov`, `eig_spectrum`, `effective_rank`, `svd_spectrum`,
    `cosine`.

Everything assumes the stable `model.MLP` interface (`hidden_layers`, `act`,
`readout`, and `forward(x, return_activations=True)`).
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import torch

from utils import to_numpy  # re-exported for convenience

__all__ = ["to_numpy", "collect_activations", "run_with_intervention",
           "mean_cov", "eig_spectrum", "effective_rank", "svd_spectrum", "cosine"]


# ---- activations -------------------------------------------------------

@torch.no_grad()
def collect_activations(model, x: torch.Tensor) -> Dict:
    """Forward `x` (eval, no grad) and return everything on CPU:

        {"input": (n, in), "pre": {layer: (n, w)}, "post": {layer: (n, w)},
         "output": (n, out)}   -- pre/post are the pre-/post-activation tensors.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    _, acts = model(x.to(device), return_activations=True)
    if was_training:
        model.train()
    cpu = lambda t: t.detach().cpu()
    return {
        "input": cpu(acts["input"]),
        "pre": {k: cpu(v) for k, v in acts["pre"].items()},
        "post": {k: cpu(v) for k, v in acts["post"].items()},
        "output": cpu(acts["output"]),
    }


@torch.no_grad()
def run_with_intervention(model, x: torch.Tensor,
                          interventions: Optional[Dict[int, Callable]] = None) -> torch.Tensor:
    """Forward `x` while editing chosen hidden layers' POST-activations.

    `interventions` maps a hidden-layer index to `fn(post_activation) -> tensor`
    (same shape). This mirrors `model.forward` exactly (no hooks), so it is the
    intervention primitive for ablation (zero/mean a neuron) and activation
    patching (substitute another input's activations). Returns the output tensor.
    """
    interventions = interventions or {}
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    h = x.to(device)
    for i, layer in enumerate(model.hidden_layers):
        a = model.act(layer(h))
        if i in interventions:
            a = interventions[i](a)
        h = a
    out = model.readout(h)
    if was_training:
        model.train()
    return out


# ---- linear algebra ----------------------------------------------------

def mean_cov(X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical mean (D,) and covariance (D, D) of the rows of X (N, D)."""
    X = X if isinstance(X, torch.Tensor) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
    Xc = X - X.mean(0, keepdim=True)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    return X.mean(0).numpy(), cov.numpy()


def eig_spectrum(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and eigenvectors (columns) of a symmetric cov."""
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def effective_rank(eigvals: np.ndarray) -> Dict[str, float]:
    """How many directions carry the variance, three ways (all =D isotropic, ~1 rank-1):

    participation_ratio = (sum l)^2 / sum l^2 ; stable_rank = sum l / l_max ;
    entropy_rank = exp(spectral entropy).
    """
    ev = np.clip(np.asarray(eigvals, dtype=np.float64), 0, None)
    s = ev.sum()
    if s <= 0:
        return {"participation_ratio": 0.0, "stable_rank": 0.0, "entropy_rank": 0.0}
    p = ev / s
    nz = p[p > 0]
    return {
        "participation_ratio": float((s ** 2) / (np.sum(ev ** 2) + 1e-30)),
        "stable_rank": float(s / (ev.max() + 1e-30)),
        "entropy_rank": float(np.exp(-np.sum(nz * np.log(nz)))),
    }


def svd_spectrum(W: torch.Tensor) -> Dict:
    """SVD of a weight matrix W (out, in): singular values/vectors + effective rank
    (computed on the squared singular values = eigenvalues of W Wᵀ)."""
    Wd = W.detach().cpu().float() if isinstance(W, torch.Tensor) else torch.as_tensor(np.asarray(W))
    U, S, Vh = torch.linalg.svd(Wd, full_matrices=False)
    s = S.numpy()
    return {"U": U.numpy(), "S": s, "Vh": Vh.numpy(), "effective_rank": effective_rank(s ** 2)}


def cosine(a, b) -> float:
    """Cosine similarity between two vectors (tensors / arrays / lists)."""
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).flatten()
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).flatten()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))
