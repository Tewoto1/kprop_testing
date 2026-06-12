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


@torch.no_grad()
def layer_stats(model, n: int = 100_000, batch_size: int = 16_384,
                dead_thresh: float = 1e-3) -> Dict[int, Dict[str, np.ndarray]]:
    """One streaming MC pass (x ~ N(0, I)) over ALL hidden layers. Per layer l:

      mu          (width,)  E[post_l]            -- the -mu direction for W_{l+1}
      active_frac (width,)  P(pre_l > 0)         -- per-unit ReLU pass rate
      dead_frac   float     fraction of units with active_frac < dead_thresh
                            (dead units transmit NO gradient: nothing props back)
      post_rms    float     RMS of post_l        -- how alive the layer's output is
    """
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    L = model.cfg.depth
    mu = {l: None for l in range(L)}
    act = {l: None for l in range(L)}
    sq = {l: 0.0 for l in range(L)}
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        x = torch.randn(b, model.cfg.input_dim, dtype=dtype, device=device)
        acts = model.activations(x)
        for l in range(L):
            post, pre = acts["post"][l], acts["pre"][l]
            s, a = post.sum(0), (pre > 0).to(dtype).sum(0)
            mu[l] = s if mu[l] is None else mu[l] + s
            act[l] = a if act[l] is None else act[l] + a
            sq[l] += float(post.pow(2).sum())
        done += b
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for l in range(L):
        af = (act[l] / done).cpu().numpy()
        out[l] = dict(mu=(mu[l] / done).cpu().numpy(), active_frac=af,
                      dead_frac=float((af < dead_thresh).mean()),
                      post_rms=float(np.sqrt(sq[l] / (done * model.cfg.hidden_dim))))
    return out


def grad_flow(model, task, batch_size: int = 4096, n_batches: int = 8) -> Dict[str, float]:
    """Per-weight-matrix gradient SCALE at the current parameters, averaged over
    fresh task batches: ||grad W|| / ||W|| (a relative per-step update scale).
    ~0 means training has stopped moving that layer ("nothing props back")."""
    p = next(model.parameters())
    device = p.device
    was_training = model.training
    model.train()
    norms = {name: 0.0 for name, _ in model.named_weights()}
    for _ in range(n_batches):
        x, y = task.sample_batch(batch_size, device)
        x, y = x.to(p.dtype), y.to(p.dtype)
        model.zero_grad(set_to_none=True)
        task.loss(model(x), y).backward()
        for name, W in model.named_weights():
            if W.grad is not None:
                norms[name] += float(W.grad.norm()) / (float(W.norm()) + 1e-30)
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return {k: v / n_batches for k, v in norms.items()}
