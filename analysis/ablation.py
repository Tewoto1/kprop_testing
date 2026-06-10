"""analysis/ablation.py -- knock out neurons and measure the effect on the output.

The classic causal-importance probe: set one or more hidden neurons to a fixed value
(0, or their batch-mean activation) and see how much the output moves.

  * `ablate(model, x, layer, neurons, mode)` -- ablate a chosen set, return baseline
    vs ablated output and their difference.
  * `neuron_importance(model, x, layer, mode)` -- ablate each neuron in a layer one
    at a time and rank them by mean |Δoutput| (O(width) forward passes).

Uses `common.run_with_intervention` (hook-free), so it works on the plain `model.MLP`.
"""
from __future__ import annotations
from typing import Dict, Iterable
import numpy as np
import torch

from .common import collect_activations, run_with_intervention


def _ablation_fn(idx: torch.Tensor, mode: str, post_mean: torch.Tensor):
    def fn(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        if mode == "zero":
            a[:, idx] = 0.0
        elif mode == "mean":
            a[:, idx] = post_mean.to(a.device, a.dtype)[idx]
        else:
            raise ValueError(f"mode must be 'zero' or 'mean' (got {mode!r})")
        return a
    return fn


def ablate(model, x, layer: int, neurons: Iterable[int], *, mode: str = "zero") -> Dict:
    """Ablate `neurons` at hidden `layer`; return baseline/ablated outputs and Δ.

    `mode`: "zero" (set activations to 0) or "mean" (set to their batch-mean).
    `mean_abs_delta` summarizes the output change.
    """
    acts = collect_activations(model, x)
    base_out = acts["output"]
    post_mean = acts["post"][layer].mean(0)
    idx = torch.as_tensor(list(neurons), dtype=torch.long)
    abl_out = run_with_intervention(model, x, {layer: _ablation_fn(idx, mode, post_mean)}).detach().cpu()
    delta = abl_out - base_out
    return {
        "layer": layer, "mode": mode, "neurons": idx.tolist(),
        "baseline_output": base_out.numpy(),
        "ablated_output": abl_out.numpy(),
        "delta": delta.numpy(),
        "mean_abs_delta": float(delta.abs().mean()),
    }


def neuron_importance(model, x, layer: int, *, mode: str = "zero", top_k: int = 10) -> Dict:
    """Single-neuron knockout importance for every neuron in `layer`.

    Ablates each neuron alone and records mean |Δoutput|; returns the per-neuron
    importance vector, a descending ranking, and the top-k neurons.
    """
    acts = collect_activations(model, x)
    base_out = acts["output"]
    post_mean = acts["post"][layer].mean(0)
    width = acts["post"][layer].shape[1]
    importance = np.zeros(width)
    for j in range(width):
        idx = torch.tensor([j], dtype=torch.long)
        out = run_with_intervention(model, x, {layer: _ablation_fn(idx, mode, post_mean)}).detach().cpu()
        importance[j] = float((out - base_out).abs().mean())
    ranking = np.argsort(importance)[::-1]
    return {
        "layer": layer, "mode": mode,
        "importance": importance,
        "ranking": ranking,
        "top_k": ranking[:top_k].tolist(),
    }
