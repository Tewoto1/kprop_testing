"""analysis/relu_gating.py -- how the ReLUs gate the Gaussian.

Answers:
  * Is the model pushing pre-activations negative so the ReLU zeros them?
        -> gating_stats: pre_mean < 0, frac_zeroed > baseline ~0.5, pre_skew.
  * What fraction of each layer is zeroed vs a random-init baseline?
        -> gating_stats vs baseline_gating.
  * (If biases are enabled -- they are OFF by default) do strongly negative
    biases gate neurons off regardless of input?  -> bias_stats.
  * (depth 3) Are the zeros of consecutive layers in "similar or different
    directions" -- do the gating masks overlap, dead neurons persist?
        -> mask_overlap + per_neuron_active_rate in gating_stats.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch

from .activations import gaussian_batch, collect_activations
from model import ModelConfig


def _skew(t: torch.Tensor) -> float:
    t = t.float()
    m, s = t.mean(), t.std()
    if s == 0:
        return 0.0
    return float(((t - m) ** 3).mean() / (s ** 3))


def gating_stats(model, n: int = 8192, std: float = 1.0, seed: int = 0) -> Dict[int, Dict]:
    """Per-hidden-layer ReLU statistics over a Gaussian ball."""
    dim = model.cfg.input_dim
    x = gaussian_batch(dim, n, std=std, seed=seed)
    acts = collect_activations(model, x)
    out: Dict[int, Dict] = {}
    for i in sorted(acts["pre"].keys()):
        pre = acts["pre"][i]                        # (N, W)
        active = (pre > 0).float()                  # gating mask
        per_neuron_active = active.mean(0).numpy()  # P(neuron on)
        pre_mean_per_neuron = pre.mean(0)
        out[i] = {
            "frac_zeroed": float((pre <= 0).float().mean()),
            "frac_active": float(active.mean()),
            "per_neuron_active_rate": per_neuron_active,
            "dead_rate": float((per_neuron_active < 0.01).mean()),       # ~always off
            "saturated_on_rate": float((per_neuron_active > 0.99).mean()),  # ~always on
            "pre_mean": float(pre.mean()),                              # < 0 => pushed negative
            "pre_mean_per_neuron": pre_mean_per_neuron.numpy(),
            "frac_neurons_neg_mean": float((pre_mean_per_neuron < 0).float().mean()),
            "pre_skew": _skew(pre.flatten()),
        }
    return out


def baseline_gating(model, n: int = 8192, std: float = 1.0, seed: int = 0) -> Dict[int, Dict]:
    """Same stats for a freshly INITIALISED net of the same shape -- the random
    reference (expect ~50% zeroed, pre_mean ~ 0)."""
    c = model.cfg
    fresh = ModelConfig(input_dim=c.input_dim, hidden_dim=c.hidden_dim, depth=c.depth,
                        output_dim=c.output_dim, bias=c.bias, final_bias=c.final_bias,
                        activation=c.activation, seed=12345).build()
    return gating_stats(fresh, n=n, std=std, seed=seed)


def bias_stats(model) -> Dict[str, Optional[Dict]]:
    """Distribution of biases per layer. Strongly negative biases switch
    neurons off independent of the input."""
    out: Dict[str, Optional[Dict]] = {}
    for name, b in model.named_biases():
        if b is None:
            out[name] = None
            continue
        bb = b.detach().cpu().numpy()
        out[name] = {"mean": float(bb.mean()), "std": float(bb.std()),
                     "frac_negative": float((bb < 0).mean()), "values": bb}
    return out


def mask_overlap(model, n: int = 8192, std: float = 1.0, seed: int = 0) -> Dict:
    """For depth >= 2: how related are the gating masks of consecutive layers?

    For each adjacent pair (i, i+1) of equal width, the mean per-sample Jaccard
    overlap of the "active neuron" sets. High overlap => the same coordinates
    tend to stay on/off across layers ("similar directions"); low overlap =>
    each layer re-gates a different set ("different directions"). Keyed by the
    string "i->j" so the result is JSON/printable."""
    dim = model.cfg.input_dim
    x = gaussian_batch(dim, n, std=std, seed=seed)
    acts = collect_activations(model, x)
    layers = sorted(acts["pre"].keys())
    res: Dict[str, Dict] = {}
    for a, b in zip(layers[:-1], layers[1:]):
        ma = (acts["pre"][a] > 0).float()
        mb = (acts["pre"][b] > 0).float()
        if ma.shape[1] == mb.shape[1]:
            inter = (ma * mb).sum(1)
            union = ((ma + mb) > 0).float().sum(1).clamp(min=1)
            jacc = float((inter / union).mean())
        else:
            jacc = None
        res[f"{a}->{b}"] = {"jaccard_active_sets": jacc,
                            "active_rate_a": float(ma.mean()),
                            "active_rate_b": float(mb.mean())}
    return res
