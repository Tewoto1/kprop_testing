"""weight_structure.py -- the Q2 metrics: is W a rank-1 spike aligned to -mu?

Shared by the frozen-readout notebooks (previously each notebook re-defined
these inline). `mu = E[a_prev]` is the mean of the post-activation feeding W.
"""
from __future__ import annotations
from typing import Dict

import numpy as np
import torch


@torch.no_grad()
def mean_prev_post(model, n: int = 200_000, batch_size: int = 16_384) -> np.ndarray:
    """Monte-Carlo estimate of mu = E[a_prev]: the mean post-activation of the
    layer feeding the last hidden weight matrix, over x ~ N(0, I)."""
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    L = model.cfg.depth - 1                  # last hidden layer index (feeds readout)
    acc, done = None, 0
    while done < n:
        b = min(batch_size, n - done)
        x = torch.randn(b, model.cfg.input_dim, dtype=dtype, device=device)
        s = model.activations(x)["post"][L - 1].sum(0)
        acc = s if acc is None else acc + s
        done += b
    return (acc / done).cpu().numpy()


def weight_structure_metrics(W, mu) -> Dict[str, float]:
    """The three stories about the pre-readout matrix, as one dict:
      (a) entries more negative ........... mean_entry < 0
      (b) rows align to -mu ............... cos_neg_mu > 0, proj_sign < 0
      (c) -mu IS the low-rank structure ... align_v1_mu ~ 1, mu-energy ~ top-sigma
    """
    W = np.asarray(W, float)
    u = mu / (np.linalg.norm(mu) + 1e-30)
    proj = W @ u
    row_norms = np.linalg.norm(W, axis=1) + 1e-30
    fro2 = (W ** 2).sum()
    _, S, Vt = np.linalg.svd(W)
    v1 = Vt[0]
    return dict(
        width=W.shape[0],
        mean_entry=float(W.mean()),                       # (a) more negative?
        cos_neg_mu=float(np.mean(-proj / row_norms)),     # (b) rows align to -mu?
        proj_sign=float(proj.mean()),                     #     sign: <0 => -mu-ward
        rank1_energy_mu=float((proj ** 2).sum() / fro2),  # (c) energy in the mu direction
        top_sv_energy=float(S[0] ** 2 / (S ** 2).sum()),  #     energy in top singular dir
        align_v1_mu=float(abs(v1 @ u)),                   #     is the top sing. dir = mu?
        stable_rank=float((S ** 2).sum() / S[0] ** 2),    #     1 = rank-1, width = flat
    )


def W_last(model) -> torch.Tensor:
    """The pre-readout weight matrix (last hidden layer), on CPU."""
    return model.hidden_layers[model.cfg.depth - 1].weight.detach().cpu()


def W_first(model) -> torch.Tensor:
    """The first weight matrix (reads mean-0 input -- the control)."""
    return model.hidden_layers[0].weight.detach().cpu()
