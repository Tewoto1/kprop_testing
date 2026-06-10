# One trained case

Study **one learned case** of a small MLP and look for a **usable mechanistic
predictor** of its output — an algorithm that, given the trained weights, predicts a
statistic of the network's output (e.g. its mean over Gaussian inputs) *without*
Monte-Carlo sampling. The benchmark to beat / compare against is **cumulant
propagation** ("kprop"), so that algorithm is vendored here and runnable directly on
the models we train.

The project is laid out as a normal small ML codebase:

```
model/          the MLP under study (one class, all activations, checkpoint load/save)
checkpoints/    trained .pt files (self-describing: config + weights + history)
tasks/          task definitions + losses: train-to-zero, single half-space, distillation
training/       the training loop + a grid-runner CLI
analysis/       mechanistic-interpretability / circuit tools
  trained_to_0/   study-specific tools for the "output 0" case
Mecha_preds/    mechanistic predictors
  cumulants/      cumulant propagation as a predictor (+ exact ReLU-covariance variant)
    kprop/          the vendored kprop algorithm (from the ARC paper repo)
colab_notebooks/  the baseline scaling notebook + the analysis notebook
utils.py        device / seeding / numpy helpers
```

Everything is a plain package — **run from the repo root** (no install needed). Module
entry points are `python -m training.run ...` and
`python -m Mecha_preds.cumulants.run_comparison ...`; notebooks add the repo root to
`sys.path` in their first cell.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart — load a trained model and predict its output mean

```python
from model import MLP
from Mecha_preds.cumulants import run_cumulants, estimate_empirical_mean, compare_means

model, payload = MLP.load("checkpoints/zero_d3_w128_seed0_final.pt")
print(model.cfg, "final loss:", payload["history"][-1][1])

pred = run_cumulants(model)["mean"]                 # cumulant propagation, default k_max=3
mc, stats = estimate_empirical_mean(model=model, input_dim=model.cfg.input_dim,
                                    num_samples=200_000)   # Monte-Carlo reference
print("relative error:", compare_means(pred, mc, stats)["relative_error_mean"])
```

## Train / test

One CLI trains any task over a depth × width × seed grid; nothing about the grid,
dims, or optimizer is hard-coded (see `python -m training.run --help`):

```bash
# train to output 0 (the main study); checkpoints -> checkpoints/zero_d{D}_w{W}_seed{S}_final.pt
python -m training.run --task zero      --widths 64 128 256 --depths 2 3 --steps 8000

# separate ONE random half-space (output is a 0/1 indicator)
python -m training.run --task halfspace --widths 128 --depths 3 --offset-std 1.0

# distill a frozen random teacher MLP (student matches teacher(x))
python -m training.run --task distill   --widths 128 --depths 3 --teacher-seed 1
```

Programmatically, `tasks` are pure (data + loss) and `training` runs them:

```python
from model import ModelConfig
from tasks import ZeroTask
from training import Trainer, TrainConfig

model = ModelConfig(input_dim=128, hidden_dim=128, depth=3).build()
res = Trainer(model, ZeroTask(input_dim=128),
              TrainConfig(steps=8000, checkpoint_mode="final")).train()
print(res["final_loss"], res["final_checkpoint"])
```

Conventions: `input_dim == width` by default (a square first layer W₁), **biases OFF**
by default (ReLU half-spaces pass through the origin; layer-0 pre-activations are
exactly zero-mean), and no weight decay — so the only pressure is the task loss and the
emergent geometry is the object of study. Pass `--bias` / `ModelConfig(bias=True)` to
restore biases.

## Analysis

Each tool runs **individually** and returns a dict — there is no master "report" step.

**General circuit tools** (`analysis`, work on any `model.MLP` + any input batch):

```python
from model import MLP
from analysis import (weight_spectrum, activation_pca, neuron_importance,
                      ablate, activation_patch, direct_contributions, output_lens,
                      probe_layer)
from analysis.trained_to_0 import gaussian_batch

m, _ = MLP.load("checkpoints/zero_d3_w128_seed0_final.pt")
x = gaussian_batch(m.cfg.input_dim, 4096, seed=0)

weight_spectrum(m)["readout"]["effective_rank"]   # per-weight SVD effective rank
activation_pca(m, x, layer=-1)["top_explained"]    # PCA of last hidden activations
neuron_importance(m, x, layer=0)["top_k"]           # single-neuron knockout ranking
activation_patch(m, x, gaussian_batch(m.cfg.input_dim, 4096, seed=1), layer=1)
direct_contributions(m, x)["cancellation_ratio"]   # readout decomposition wᵢ·zᵢ
output_lens(m, x)                                   # read off each layer through the readout
```

**Study-specific tools** for the train-to-zero case (`analysis.trained_to_0`):
`layer_gaussian_stats` (Cov(h1)=σ²·W₁W₁ᵀ), `gating_stats` / `baseline_gating` /
`mask_overlap` (ReLU gating vs random init; the depth-3 middle-layer probe), and
`output_decomposition` (shrink / orthogonality / cancellation). The
`colab_notebooks/02_analyze_to_zero.ipynb` notebook walks all of these with plots.

## Mechanistic predictors — cumulant propagation

`Mecha_preds/cumulants/` wraps the real kprop algorithm (vendored, unmodified, under
`kprop/`) so it runs on a study `model.MLP`:

```python
from Mecha_preds.cumulants import run_cumulants
pred       = run_cumulants(model, config={"k_max": 3})["mean"]                       # normal kprop
pred_k2    = run_cumulants(model, config={"k_max": 2})["mean"]                       # k=2 (approx ReLU cov)
pred_exact = run_cumulants(model, config={"k_max": 2, "exact_relu_cov": True})["mean"]  # EXACT ReLU covariance
```

`exact_relu_cov=True` (ReLU, `k_max==2` only) uses the exact bivariate-Gaussian ReLU
covariance instead of the leading-order gain approximation — the "compute the second
moment exactly" variant. For any other `k_max`/activation the normal harmonic kprop runs.

Sweep cumulant-vs-Monte-Carlo error across widths (trains via the unified loop, writes
CSV + plots to `--outdir`):

```bash
python -m Mecha_preds.cumulants.run_comparison \
    --task zero --input-dim 64 --hidden-depth 3 \
    --widths 64 128 256 --seeds 0 1 2 --train-steps 5000 --k-max 3 \
    --outdir results/zero_kprop
```

## Notebooks (`colab_notebooks/`)

- **`exact_relu_k2_width_scaling_colab.ipynb`** — the **baseline** scaling experiment:
  does the exact ReLU covariance fix cumulant propagation on trained-to-zero models?
  (Answer in the notebook; compare future experiments against it.)
- **`02_analyze_to_zero.ipynb`** — the full mechanistic-interpretability walkthrough of
  one trained-to-zero checkpoint, plus the general circuit tools.

Both are generated from their `build_*.py` scripts (edit the script and re-run to
regenerate, keeping the notebooks reproducible).

## Notes

- **float64**: cumulant propagation (and its Monte-Carlo reference) run in double
  precision; `run_cumulants` builds a float64 copy of the model internally.
- Checkpoints store `model_config`, `state_dict`, `step`, `history`, and `train_config`;
  load with `MLP.load(path)` → `(model, payload)`.
- The `kprop/` library is vendored from the ARC paper *"Estimating the expected output
  of wide random MLPs more efficiently than sampling"* and is MIT-licensed (see
  `LICENSE`); only the harmonic kprop path + the exact ReLU-covariance step are kept.
