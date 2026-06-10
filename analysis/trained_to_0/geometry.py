"""analysis/trained_to_0/geometry.py -- first-layer covariance & example tracking.

Study-specific geometry for the train-to-zero net (general linear-algebra
primitives -- mean_cov, eig_spectrum, effective_rank, svd_spectrum, cosine -- live
in `analysis.common` and are re-exported here):

  * "What is the covariance of the Gaussian after the first layer?"
        -> layer_gaussian_stats(model)[0]["pre_cov"]  (empirical)
           and ["analytic_pre_cov"] = std^2 * W1 W1^T.
  * "Are most inputs aligned to a direction? Which one?"
        -> top eigenvector / top_var_fraction / effective rank of that cov.
  * "How does the geometry of example vectors change layer by layer?"
        -> propagate_examples(model, vectors).
"""
from __future__ import annotations
from typing import Dict, List
import torch

from ..common import mean_cov, eig_spectrum, effective_rank, svd_spectrum, cosine
from .activations import gaussian_batch, collect_activations

__all__ = ["mean_cov", "eig_spectrum", "effective_rank", "svd_spectrum", "cosine",
           "layer_gaussian_stats", "propagate_examples"]


# ---- per-layer Gaussian statistics -------------------------------------

def layer_gaussian_stats(model, n: int = 8192, std: float = 1.0,
                         seed: int = 0) -> Dict[int, Dict]:
    """Push a Gaussian ball through the model; for each hidden layer summarise
    the pre-/post-activation distribution and its principal directions.

    For layer 0 we also return the analytic pre-activation covariance
    std^2 * W1 W1^T (a linear map of a Gaussian is Gaussian, so this should
    match the empirical pre_cov up to sampling noise)."""
    dim = model.cfg.input_dim
    x = gaussian_batch(dim, n, std=std, seed=seed)
    acts = collect_activations(model, x)
    stats: Dict[int, Dict] = {}
    for i in sorted(acts["pre"].keys()):
        pre, post = acts["pre"][i], acts["post"][i]
        pre_mean, pre_cov = mean_cov(pre)
        pre_eigs, pre_vecs = eig_spectrum(pre_cov)
        stats[i] = {
            "pre_mean": pre_mean,
            "pre_cov": pre_cov,
            "pre_eigvals": pre_eigs,
            "pre_eigvecs": pre_vecs,
            "pre_effrank": effective_rank(pre_eigs),
            "top_direction": pre_vecs[:, 0],
            "top_var_fraction": float(pre_eigs[0] / (pre_eigs.sum() + 1e-30)),
            "post_mean": post.mean(0).numpy(),
            "post_rms": float(post.pow(2).mean().sqrt()),
        }
    # analytic check for the first layer
    W1 = model.hidden_layers[0].weight.detach().cpu().float()
    stats[0]["analytic_pre_cov"] = ((std ** 2) * (W1 @ W1.T)).numpy()
    return stats


# ---- following example vectors through the network ---------------------

def propagate_examples(model, vectors: torch.Tensor) -> List[Dict]:
    """Track how specific input vectors deform layer by layer: pre/post norm,
    fraction of coordinates surviving each ReLU, and (when in_dim==width) the
    cosine of the pre-activation to the original input vector."""
    acts = collect_activations(model, vectors)
    x = acts["input"]
    records: List[Dict] = []
    for r in range(vectors.shape[0]):
        layers = []
        for i in sorted(acts["pre"].keys()):
            pre, post = acts["pre"][i][r], acts["post"][i][r]
            same_shape = pre.shape == x[r].shape
            layers.append({
                "pre_norm": float(pre.norm()),
                "post_norm": float(post.norm()),
                "frac_surviving_relu": float((pre > 0).float().mean()),
                "cos_to_input_pre": cosine(pre, x[r]) if same_shape else None,
            })
        records.append({
            "input_norm": float(x[r].norm()),
            "layers": layers,
            "output": acts["output"][r].numpy(),
        })
    return records
