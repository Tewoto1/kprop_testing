"""analysis/pca.py -- PCA of activations and SVD of weights.

Two standard "how many directions matter" probes, usable on any MLP:

  * `activation_pca(model, x, layer)` -- principal components of a hidden layer's
    activations over an input batch: eigen-spectrum, explained-variance ratios,
    and effective rank of the representation.
  * `weight_spectrum(model)` -- the singular-value spectrum and effective rank of
    every weight matrix (how low-rank each layer's map is).

Run either directly; each returns a dict (and the caller prints what it wants).
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from .common import collect_activations, mean_cov, eig_spectrum, effective_rank, svd_spectrum


def activation_pca(model, x, layer: int = -1, *, use_post: bool = True, top_k: int = 10) -> Dict:
    """PCA of hidden-layer activations over the batch `x`.

    `layer` indexes hidden layers (negative counts from the end; -1 = last hidden).
    `use_post` selects post-activation (default) vs pre-activation. Returns the
    eigen-spectrum (descending), components (columns), explained-variance ratios,
    effective rank, and the top-k components of the centered activation covariance.
    """
    acts = collect_activations(model, x)
    keys = sorted(acts["post"].keys())
    li = keys[layer] if layer < 0 else layer
    A = acts["post"][li] if use_post else acts["pre"][li]
    mean, cov = mean_cov(A)
    eigs, vecs = eig_spectrum(cov)
    total = float(eigs.sum()) + 1e-30
    return {
        "layer": li,
        "use_post": use_post,
        "mean": mean,
        "eigvals": eigs,
        "components": vecs,                              # columns are PCs
        "explained_variance_ratio": eigs / total,
        "top_explained": float(eigs[0] / total),
        "effective_rank": effective_rank(eigs),
        "top_components": vecs[:, :top_k],
    }


def weight_spectrum(model) -> Dict[str, Dict]:
    """SVD spectrum + effective rank of every weight matrix (forward order).

    Keys are the layer names from `model.named_weights()` (hidden0..hiddenN, readout).
    """
    return {name: svd_spectrum(W) for name, W in model.named_weights()}
