"""analysis/output_layer.py -- what is the *last* layer doing?

Beyond "the readout weights are small", a readout can drive the output to ~0 by
three distinguishable mechanisms:

  (a) shrinking     -- ||readout|| small and/or ||z|| small.
  (b) orthogonality -- readout (nearly) perpendicular to E[z], so the mean
                       activation contributes ~0.
  (c) cancellation  -- individual terms w_i * z_i are large but sum to ~0; the
                       layer "mixes" activations so they cancel in MSE.

These functions quantify each so you can say which mix the model uses.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import torch

from .activations import gaussian_batch, collect_activations
from .geometry import cosine


def readout_norms(model) -> Dict:
    """Frobenius norm of every weight matrix (readout vs hidden layers) plus the
    per-output-row norms of the readout (the literal 'weight size of the last
    layer')."""
    norms = {name: float(W.detach().norm()) for name, W in model.named_weights()}
    Wr = model.readout.weight.detach().cpu()
    return {"weight_frob_norms": norms,
            "readout_row_norms": Wr.norm(dim=1).numpy(),
            "readout_frob": float(Wr.norm())}


@torch.no_grad()
def output_decomposition(model, n: int = 8192, std: float = 1.0, seed: int = 0) -> Dict:
    """Decompose out = sum_i w_i z_i + b over the last hidden activations z.

    cancellation_ratio = E|out| / E[sum_i |w_i z_i|]  per output.
        ~1  => no cancellation (output small only because terms are small)
        ~0  => heavy cancellation/mixing (big terms summing to nearly nothing)
    Also reports alignment of the readout with the mean activation E[z]
    (mechanism b) and the magnitudes involved (mechanism a)."""
    dim = model.cfg.input_dim
    x = gaussian_batch(dim, n, std=std, seed=seed)
    acts = collect_activations(model, x)
    last = max(acts["post"].keys())
    z = acts["post"][last]                       # (N, W)
    Wr = model.readout.weight.detach().cpu()     # (out, W)
    b = (model.readout.bias.detach().cpu() if model.readout.bias is not None
         else torch.zeros(Wr.shape[0]))
    out = acts["output"]                         # (N, out)

    terms = torch.einsum("ow,nw->now", Wr, z)    # per-term w_i z_i  (N, out, W)
    gross = terms.abs().sum(dim=2)               # E over W of |w_i z_i| -> (N, out)
    net = out.abs()                              # (N, out)
    eps = 1e-30

    Ez = z.mean(0)                               # (W,)
    align_mean = np.array([cosine(Wr[o], Ez) for o in range(Wr.shape[0])])
    out_from_mean = (Wr @ Ez + b).numpy()        # output the mean activation alone yields

    return {
        "n_last_layer": int(z.shape[1]),
        "Ez_norm": float(Ez.norm()),
        "z_rms": float(z.pow(2).mean().sqrt()),
        "readout_frob": float(Wr.norm()),
        "output_rms": float(out.pow(2).mean().sqrt()),
        "gross_mean": gross.mean(0).numpy(),          # E[ sum_i |w_i z_i| ]
        "net_mean": net.mean(0).numpy(),              # E[ |out| ]
        "cancellation_ratio": (net.mean(0) / (gross.mean(0) + eps)).numpy(),
        "readout_align_meanact": align_mean,          # cos(readout, E[z]); ~0 => orthogonal
        "output_from_mean_activation": out_from_mean,
    }
