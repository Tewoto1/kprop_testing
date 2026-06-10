"""analysis/attribution.py -- direct contributions and the output lens.

  * `direct_contributions(model, x)` -- the readout output decomposed over the last
    hidden activations: term wᵢ·zᵢ per neuron. Reports the mean contribution of each
    neuron, the gross (E[Σ|wᵢzᵢ|]) vs net (E|out|) magnitudes, and their ratio (the
    "direct logit attribution" analog -- ratio →1 = no cancellation, →0 = neurons
    cancel).
  * `output_lens(model, x)` -- project every hidden layer's post-activation through
    the readout, i.e. "what would the output be if we read off here?" (a logit-lens
    analog). Valid because every hidden layer has width = readout in-features.

Both return dicts; run directly.
"""
from __future__ import annotations
from typing import Dict, Optional
import torch


def direct_contributions(model, x) -> Dict:
    """Per-neuron contribution wᵢ·zᵢ of the last hidden layer to each output."""
    from .common import collect_activations
    acts = collect_activations(model, x)
    last = max(acts["post"].keys())
    z = acts["post"][last]                                   # (n, w)
    Wr = model.readout.weight.detach().cpu()                 # (out, w)
    terms = torch.einsum("ow,nw->now", Wr, z)                # (n, out, w)
    gross = terms.abs().sum(2).mean(0)                       # (out,)
    net = acts["output"].abs().mean(0)                       # (out,)
    return {
        "layer": last,
        "mean_contribution": terms.mean(0).numpy(),          # (out, w)
        "per_neuron_abs_mean": terms.abs().mean(0).numpy(),  # (out, w)
        "gross": gross.numpy(),                              # E[ sum_i |w_i z_i| ]
        "net": net.numpy(),                                  # E[ |out| ]
        "cancellation_ratio": (net / (gross + 1e-30)).numpy(),
    }


def output_lens(model, x) -> Dict[int, Optional[Dict]]:
    """Project each hidden layer's post-activation through the readout.

    Returns {layer: {"mean": (out,), "rms": float}} -- the "as-if final" output from
    reading off at each layer. A layer whose width != readout in-features maps to None.
    """
    from .common import collect_activations
    acts = collect_activations(model, x)
    Wr = model.readout.weight.detach().cpu()                 # (out, w)
    br = model.readout.bias.detach().cpu() if model.readout.bias is not None else None
    res: Dict[int, Optional[Dict]] = {}
    for li in sorted(acts["post"].keys()):
        a = acts["post"][li]
        if a.shape[1] != Wr.shape[1]:
            res[li] = None
            continue
        proj = a @ Wr.T
        if br is not None:
            proj = proj + br
        res[li] = {"mean": proj.mean(0).numpy(), "rms": float(proj.pow(2).mean().sqrt())}
    return res
