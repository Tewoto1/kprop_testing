"""Generates 02_analyze_to_zero.ipynb (valid nbformat-4 JSON).

Loads ONE trained-to-zero checkpoint and walks the study's questions with all plots
inline, then sweeps the checkpoint grid. Uses the unified packages: `model.MLP.load`
for the checkpoint, `analysis.trained_to_0` for the study-specific tools, and the
top-level `analysis` general circuit tools for a final task-agnostic look.

Run:  python colab_notebooks/build_analyze_notebook.py
"""
import json
import os

cells = []


def _cell_id():
    return f"cell-{len(cells):02d}"


def md(text):
    cells.append({"cell_type": "markdown", "id": _cell_id(), "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "id": _cell_id(), "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r"""# 02 — How does the MLP output 0?

Loads one checkpoint and walks every question, with all plots inline. Re-run with a
different `CKPT` to compare widths/depths (the last cells sweep the whole grid and
show the general circuit tools).""")

code(r"""
import os, sys
# Find the repo root (the folder containing the `model/` and `Mecha_preds/` packages).
CANDIDATES = [os.path.abspath(os.path.join(os.getcwd(), '..')), os.getcwd(),
              '/content/one_trained_case', '/content/drive/MyDrive/One trained case']
def _is_root(c):
    return os.path.isdir(os.path.join(c, 'model')) and os.path.isdir(os.path.join(c, 'Mecha_preds'))
REPO = next((c for c in CANDIDATES if _is_root(c)), None)
assert REPO, f'Could not find repo root (model/ + Mecha_preds/) in {CANDIDATES}. Set REPO manually.'
if REPO not in sys.path:
    sys.path.insert(0, REPO)
print('REPO =', REPO)

import numpy as np, torch
import matplotlib.pyplot as plt
from model import MLP
from analysis.common import svd_spectrum
from analysis.trained_to_0 import (gaussian_batch, collect_activations, layer_gaussian_stats,
                                   propagate_examples, gating_stats, baseline_gating,
                                   bias_stats, mask_overlap, readout_norms, output_decomposition)
""")

code(r"""
# Choose which trained model to analyse:
CKPT = os.path.join(REPO, 'checkpoints', 'zero_d3_w128_seed0_final.pt')
model, payload = MLP.load(CKPT); model.to('cpu')
DIM = model.cfg.input_dim
print(model.cfg)
print('final loss:', payload['history'][-1][1])
""")

md(r"""## Q1 — Covariance of the Gaussian after layer 1
A linear map of a Gaussian is Gaussian, so `Cov(h1) = σ²·W₁W₁ᵀ`. Empirical heatmap, the
analytic check, the eigen-spectrum, and how concentrated the variance is
(participation-ratio effective rank).""")

code(r"""
stats = layer_gaussian_stats(model, n=16384)
s0 = stats[0]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im0 = ax[0].imshow(s0['pre_cov'], cmap='RdBu_r'); ax[0].set_title('empirical Cov(h1)'); plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(s0['analytic_pre_cov'], cmap='RdBu_r'); ax[1].set_title('analytic σ²·W1 W1ᵀ'); plt.colorbar(im1, ax=ax[1])
ax[2].semilogy(s0['pre_eigvals'], marker='.'); ax[2].set_title('eigenvalues of Cov(h1)'); ax[2].set_xlabel('index')
plt.tight_layout(); plt.show()
print('max |empirical - analytic|:', float(np.abs(s0['pre_cov'] - s0['analytic_pre_cov']).max()))
print('effective rank (participation ratio):', round(s0['pre_effrank']['participation_ratio'], 2), '/', DIM)
print('top-direction variance fraction:', round(s0['top_var_fraction'], 3))
""")

md(r"""## Q2 — Are inputs aligned to a direction? Which?
Cumulative variance explained, and the leading eigenvector of `Cov(h1)` (= top
left-singular vector of W₁).""")

code(r"""
ev = s0['pre_eigvals']; cum = np.cumsum(ev) / ev.sum()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(cum, marker='.'); ax[0].axhline(0.9, ls='--', c='gray')
ax[0].set_title('cumulative variance explained'); ax[0].set_xlabel('# directions'); ax[0].set_ylim(0, 1.02)
ax[1].stem(s0['top_direction']); ax[1].set_title('leading direction of Cov(h1)'); ax[1].set_xlabel('hidden unit')
plt.tight_layout(); plt.show()
sv = svd_spectrum(model.hidden_layers[0].weight)
print('W1 singular values (top 8):', np.round(sv['S'][:8], 3))
print('W1 effective rank (PR):', round(sv['effective_rank']['participation_ratio'], 2))
""")

md(r"""## Q3 — Is the ReLU zeroing on purpose, or ~random?
Compare the trained net to a fresh random-init net of the same shape. The
**push-negative hypothesis** predicts trained `frac_zeroed` > ~0.5 and `pre_mean` < 0.
With biases OFF, layer-0 pre-activations are exactly zero-mean, so any negative-pushing
must appear in deeper layers.""")

code(r"""
g = gating_stats(model, n=16384)
base = baseline_gating(model, n=16384)
layers = sorted(g.keys())
x = gaussian_batch(DIM, 16384, seed=1); acts = collect_activations(model, x)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar([i - 0.2 for i in layers], [g[i]['frac_zeroed'] for i in layers], width=0.4, label='trained')
ax[0].bar([i + 0.2 for i in layers], [base[i]['frac_zeroed'] for i in layers], width=0.4, label='random init')
ax[0].axhline(0.5, ls='--', c='gray'); ax[0].set_xticks(layers); ax[0].set_xlabel('layer')
ax[0].set_title('fraction zeroed by ReLU'); ax[0].legend()
ax[1].hist(acts['pre'][0].flatten().numpy(), bins=120, density=True)
ax[1].axvline(0, c='r'); ax[1].set_title(f"pre-act dist L0 (mean={g[0]['pre_mean']:.3f}, skew={g[0]['pre_skew']:.2f})")
for i in layers: ax[2].hist(g[i]['per_neuron_active_rate'], bins=30, alpha=0.5, label=f'L{i}')
ax[2].axvline(0.5, ls='--', c='gray'); ax[2].set_title('per-neuron P(active)'); ax[2].set_xlabel('rate'); ax[2].legend()
plt.tight_layout(); plt.show()
for i in layers:
    print(f"L{i}: zeroed {g[i]['frac_zeroed']:.3f} (base {base[i]['frac_zeroed']:.3f}) | "
          f"pre_mean {g[i]['pre_mean']:.3f} | neurons w/ neg mean {g[i]['frac_neurons_neg_mean']:.3f} | "
          f"dead {g[i]['dead_rate']:.3f}")
""")

md(r"""## Q4 — (depth 3) what is the middle layer for?
Are consecutive gating masks **overlapping** (same coords stay on/off) or
**re-randomised** (different directions each layer)? Low Jaccard ⇒ different directions.
(Models are bias-free by default, so gating is purely the sign of W·activation.)""")

code(r"""
print('mask overlap (Jaccard of active sets between consecutive layers):')
for k, v in mask_overlap(model, n=16384).items():
    j = v['jaccard_active_sets']
    print(f"  {k}: jaccard={j:.3f}  active_rate {v['active_rate_a']:.2f} -> {v['active_rate_b']:.2f}")
bs = bias_stats(model)
if all(d is None for d in bs.values()):
    print('model is bias-free: no biases to plot')
else:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, d in bs.items():
        if d is not None:
            ax.hist(d['values'], bins=30, alpha=0.45, label=f"{name} (neg {d['frac_negative']:.2f})")
    ax.axvline(0, c='r'); ax.set_title('bias distributions'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
""")

md(r"""## Q5 — What is the last layer doing (beyond being small)?
`out = Σ wᵢzᵢ + b`. Three distinguishable mechanisms:
- **shrink** — small `z_rms` and/or `readout_frob`
- **orthogonality** — `cos(readout, E[z]) ≈ 0`
- **cancellation / mixing** — `cancellation_ratio = E|out| / E[Σ|wᵢzᵢ|] → 0`.""")

code(r"""
rn = readout_norms(model); od = output_decomposition(model, n=16384)
print('weight Frobenius norms:', {k: round(v, 3) for k, v in rn['weight_frob_norms'].items()})
print('readout Frobenius norm :', round(od['readout_frob'], 4))
print('z_rms (last hidden)    :', round(od['z_rms'], 4), '| output_rms:', round(od['output_rms'], 6))
print('gross E[sum|wz|]       :', np.round(od['gross_mean'], 4), '| net E|out|:', np.round(od['net_mean'], 6))
print('cancellation ratio     :', np.round(od['cancellation_ratio'], 4), ' (-> 0 = heavy mixing/cancel)')
print('cos(readout, E[z])     :', np.round(od['readout_align_meanact'], 4), ' (-> 0 = orthogonal to mean act)')
names = list(rn['weight_frob_norms']); vals = [rn['weight_frob_norms'][n] for n in names]
fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(names, vals)
ax.set_title('weight Frobenius norm per layer'); ax.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.show()
""")

md(r"""## Q6 — Watch example vectors deform through the net
Norms and ReLU survival of a few sampled inputs as they pass through each layer.""")

code(r"""
V = gaussian_batch(DIM, 5, seed=7)
for r in propagate_examples(model, V):
    surv = [round(l['frac_surviving_relu'], 2) for l in r['layers']]
    norms = [round(l['post_norm'], 3) for l in r['layers']]
    print(f"in-norm {r['input_norm']:.2f}  survive {surv}  post-norm {norms}  out {np.round(r['output'], 4)}")
""")

md(r"""## General circuit tools (task-agnostic)
The same checkpoint through the top-level `analysis` tools, which work on any MLP:
per-weight SVD effective rank, single-neuron knockout importance, and the PCA of the
last hidden layer's activations.""")

code(r"""
from analysis import weight_spectrum, neuron_importance, activation_pca
ws = weight_spectrum(model)
print('weight effective rank (participation ratio):',
      {name: round(s['effective_rank']['participation_ratio'], 2) for name, s in ws.items()})

xb = gaussian_batch(DIM, 4096, seed=2)
ni = neuron_importance(model, xb, layer=0)           # knockout importance in hidden layer 0
print('layer-0 top-5 neurons by knockout |Δout|:', ni['top_k'][:5])

pca = activation_pca(model, xb, layer=-1)            # PCA of the last hidden layer
print('last-hidden top explained-variance fraction:', round(pca['top_explained'], 3),
      '| effective rank:', round(pca['effective_rank']['participation_ratio'], 2))
""")

md(r"""## Sweep the whole grid
Readout norm / cancellation / activation scale across every checkpoint — how the
mechanism shifts with width and depth.""")

code(r"""
import glob
rows = []
for p in sorted(glob.glob(os.path.join(REPO, 'checkpoints', 'zero_*_final.pt'))):
    m, _ = MLP.load(p); m.to('cpu')
    od = output_decomposition(m, n=8192)
    rows.append((os.path.basename(p).replace('_final.pt',''), m.cfg.depth, m.cfg.hidden_dim,
                 od['readout_frob'], float(od['cancellation_ratio'][0]), od['z_rms']))
print(f"{'run':24s} depth width  readoutF   cancel    z_rms")
for name, d, w, rf, c, z in rows:
    print(f'{name:24s} {d:5d} {w:5d}  {rf:7.4f}  {c:7.4f}  {z:7.4f}')
""")

md(r"""### How to read it
- **Q1/Q2:** low effective rank + high top-direction fraction = the Gaussian is squashed
  toward a few directions after layer 1.
- **Q3:** trained `frac_zeroed` above the random ~0.5 baseline, with `pre_mean<0`,
  supports *pushing activations negative so the ReLU gates them*.
- **Q4:** low cross-layer Jaccard = the depth-3 middle layer re-gates a *different* set
  of coordinates (re-randomising), rather than reinforcing the same dead units.
- **Q5:** a `cancellation_ratio` near 0 means the readout keeps sizeable per-neuron terms
  that *cancel* — mixing activations to net ~0 — rather than merely shrinking them.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "02_analyze_to_zero.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
