"""Generates notebooks/cumulant_experiments_colab.ipynb (valid nbformat-4 JSON).

Run:  uv run python notebooks/build_notebook.py
"""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# =============================================================================
md(r"""# Cumulant propagation vs Monte-Carlo — Colab notebook

This notebook is a **transparent wrapper** around the real cumulant-propagation
("kprop") algorithm in the `mlp_kprop` repo (the code from the paper
*"Estimating the expected output of wide random MLPs more efficiently than
sampling"*). It is built so you can run everything on Google Colab and **verify
every assumption and every variable** for yourself.

It answers three questions, each with a runnable, inspectable experiment:

1. **Q1 — Mean after training to zero.** Does kprop still predict the *output
   mean* of a simple MLP after the model is randomly initialized and then trained
   only to output `0`? (Distribution is over the input `X ~ N(0, I)`; weights are
   fixed after training.)
2. **Q2 — Width scaling at init.** Does the kprop error fall with width like the
   theory says (per-entry RMS error ~ `n^(-k_max/2)`), and does raising `k_max`
   buy accuracy?
3. **Q3 — Does training break the scaling?** Compare the width-scaling and the
   `k_max`-refinement of kprop on the *same* nets before vs after training.

> **Nothing here reimplements cumulant propagation.** We call the repo's
> `mlp_kprop` as a black box. There is a verification cell below that prints the
> source file of the function actually being called so you can confirm this.
""")

# --- Setup -------------------------------------------------------------------
md(r"""## 0. Setup — get the repo and install the *minimal* dependencies

The cumulant code only needs `torch, numpy, pandas, matplotlib, jaxtyping,
einops, tqdm, joblib, opt_einsum` (NOT the heavy `quimb`/`mpi4py` deps that the
repo lists for other components). On Colab, torch/numpy/pandas/matplotlib are
already present, so we only `pip install` the few that may be missing.

**Choose how to get the repo** by editing the cell below:
- `REPO_URL`: if you pushed this repo to GitHub, put its clone URL here and it
  will be cloned on Colab.
- `LOCAL_REPO_DIR`: if the repo is already on disk (running locally, or you
  mounted Google Drive), point to it.

The repo must contain `src/mlp_kprop/` (the real algorithm) **and** the
`cumulant_experiments/` + `experiments/` folders (the wrappers this notebook
uses).
""")

code(r"""
# ----------------------------- EDIT THIS -----------------------------------
REPO_URL       = ""   # e.g. "https://github.com/<you>/mlp_kprop.git"  (leave "" if not cloning)
LOCAL_REPO_DIR = ""   # e.g. "/content/drive/MyDrive/Cumulants" or a local path
# ---------------------------------------------------------------------------

import os, sys, subprocess

try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False

def _find_repo():
    # 1) explicit local dir
    if LOCAL_REPO_DIR and os.path.isdir(os.path.join(LOCAL_REPO_DIR, "src", "mlp_kprop")):
        return LOCAL_REPO_DIR
    # 2) clone from git
    if REPO_URL:
        dest = "/content/mlp_kprop" if IN_COLAB else os.path.abspath("./_mlp_kprop_clone")
        if not os.path.isdir(os.path.join(dest, "src", "mlp_kprop")):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
        return dest
    # 3) search upward from CWD (covers "running inside the repo/notebooks dir")
    here = os.path.abspath(".")
    for _ in range(5):
        if os.path.isdir(os.path.join(here, "src", "mlp_kprop")):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate the repo. Set REPO_URL or LOCAL_REPO_DIR above.")

REPO_DIR = _find_repo()
os.chdir(REPO_DIR)                         # so relative paths like results/ work
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)           # so `import src.mlp_kprop...` and `cumulant_experiments...` work
print("IN_COLAB:", IN_COLAB)
print("REPO_DIR:", REPO_DIR)

# Minimal deps (safe to re-run; -q quiet). torch/numpy/pandas/matplotlib assumed present.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jaxtyping", "einops", "opt_einsum", "joblib", "tqdm"], check=False)
print("deps installed")
""")

# --- Global numerics ---------------------------------------------------------
md(r"""## 1. Global numerical settings (and GPU optimization)

- **kprop runs in `float64`** (the algorithm is run in double precision in the
  repo's tests; the adapter always casts a float64 copy of the model before
  calling `mlp_kprop`).
- **GPU optimization — the forward-heavy parts (training + Monte-Carlo) run in
  `float32` on the GPU.** Consumer GPUs (Colab T4/L4) have ~16–32× *lower* float64
  throughput, and the bulk of the wall-clock here is the millions of MC forward
  passes. float32 is plenty for MC: the MC mean's accuracy is limited by sampling
  (`~1/√N ≈ 1e-3`), not by float32 rounding (`~1e-7`), and **sums are still
  accumulated in float64** to avoid cancellation. kprop's own accuracy is
  unaffected because it still gets a float64 model. So we set
  `MODEL_DTYPE = float32` on GPU, `float64` on CPU.
- **Seeded** per run (`set_seed` seeds torch/numpy/random) for reproducibility.
- **Device**: CUDA if available, else CPU. kprop and MC both run on it.
""")

code(r"""
import torch, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)   # kprop tensors are float64; adapter enforces this internally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# GPU: do training + MC in float32 (fast); CPU: float64. kprop is always float64 (adapter casts a copy).
MODEL_DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
# Bigger MC batches on GPU. (Scaling cell, where input_dim grows to 1024, uses its own smaller batch.)
MC_BATCH    = 262_144 if DEVICE == "cuda" else 8_192
print("device:", DEVICE, "| MODEL_DTYPE:", MODEL_DTYPE, "| MC_BATCH:", MC_BATCH)
""")

# --- Verify the real algorithm ----------------------------------------------
md(r"""## 2. Verify we are calling the **real** algorithm (not a reimplementation)

This prints the **source file** of the `mlp_kprop` function we will call and the
top of its definition. Confirm the path is inside `src/mlp_kprop/` of the repo.
""")

code(r"""
import inspect
import src.mlp_kprop.kprop_harmonic as kp
from src.mlp_kprop.mlp import MLP
from src.mlp_kprop.wick import WICK_COEF_D

print("mlp_kprop is defined in:\n  ", inspect.getsourcefile(kp.mlp_kprop))
print("\nMLP class is defined in:\n  ", inspect.getsourcefile(MLP))
print("\nActivations cumulant propagation supports (keys of WICK_COEF_D):")
print("  ", list(WICK_COEF_D.keys()))
print("\n--- first lines of the real mlp_kprop ---")
print("".join(inspect.getsource(kp.mlp_kprop).splitlines(keepends=True)[:18]))
""")

# --- Assumptions & config ----------------------------------------------------
md(r"""## 3. Assumptions and the **exact** setup we are testing

### What "the output mean" means here
For a fixed trained model `f` and input `X ~ N(0, I_input_dim)`, the quantity of
interest is `E_X[ f(X) ]` (a vector of length `output_dim`). We compare:
- **kprop prediction** = `K_out[1].to_tensor()`, the degree-1 cumulant returned by
  `mlp_kprop` (degree-1 cumulant = mean), and
- **Monte-Carlo truth** = sample mean of `f(X)` over many `X`.

### Assumptions (each is verifiable in this notebook)
1. **Model is the repo `MLP`.** `mlp_kprop` consumes an `MLP` instance (it reads
   `mlp.Ws`, `mlp.nonlin_names`, `mlp.init_scale`). Using the repo model means the
   trained model and the model kprop sees are identical. *Verify:* §2 prints the
   class source; §6 prints layer shapes.
2. **Input is `X ~ N(0, I_n)`**, encoded as the cumulant tower
   `K_in = {1: zeros(n), 2: eye(n)}` (mean 0, covariance I, higher cumulants 0).
   *Verify:* printed in §6.
3. **Weights are fixed after training.** The randomness kprop propagates is over
   `X`, not over weights.
4. **Activation must be in `WICK_COEF_D`** (printed in §2). Default `relu`.
   We never silently swap the activation.
5. **Weight orientation:** PyTorch `nn.Linear.weight` is `(out, in)`; kprop
   contracts over the `in` axis (`y = Wx` semantics) — **no transpose**.
   *Verify:* the single-linear sanity check in §5 (true mean is exactly the bias).
6. **`k_max`** (the budget `K`) is a free knob — **no hard cap**. Higher is more
   accurate but costs ~`O(n^k_max)` memory/time, so very large `k_max` can OOM at
   large width on a small machine (fine on a Colab GPU). `factor=True` is used only
   for `k_max ∈ {3,4}` (auto-disabled otherwise). `use_avg_metric=False` (we use
   the *exact* metric `W·metric·Wᵀ` from the actual fixed weights, not the init-time
   `E[WWᵀ]`). `output_d_max=1` (we only need the mean → big FLOP savings).
7. **Metrics.** Because training drives the mean toward 0, normalizations matter:
   - `relative_error_mean = |cp−mc| / |mc| = √NMSE` — scale-free, the headline.
   - `mc_noise_z = |cp−mc| / MC_stderr` — is the gap a real bias (z≫1) or just MC
     sampling noise on the tiny mean (z≲1)?
   - Do **not** normalize by `E‖Y‖²`: that denominator also collapses as the
     output → 0 and can make a *degrading* estimate look like it improves.

### The variables we set (edit here)
Everything below is a knob. Comments say what each does and where it's used.
""")

code(r"""
from cumulant_experiments.cumulant_adapter import (
    run_cumulant_propagation_from_model, extract_mean, run_sanity_checks, config_summary,
)
from cumulant_experiments.metrics import estimate_empirical_mean, compare_means
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, set_seed, layer_norms, output_rms

# ---- Architecture ----------------------------------------------------------
INPUT_DIM    = 64       # dimension of X ~ N(0, I_INPUT_DIM)
OUTPUT_DIM   = 1        # scalar output for Q1; the §9 scaling experiment uses a vector output
HIDDEN_DEPTH = 3        # number of hidden layers (num linear layers = HIDDEN_DEPTH + 1)
ACTIVATION   = "relu"   # MUST be a key of WICK_COEF_D (see §2)
USE_BIAS     = True     # repo MLP makes real bias params when b_var>0; exercises the bias path

# ---- Cumulant-propagation config (passed straight to mlp_kprop) ------------
CUMULANT_CONFIG = {
    "k_max": 3,            # budget K; max cumulant order tracked in full. NO hard cap now:
                           #   higher = more accurate but ~O(n^k_max) mem/time (fine on a GPU).
    "kind": "simple",      # SIMPLE = the canonical setting from the paper
    "use_avg_metric": False,  # exact metric from the fixed weights (see assumption 6)
    "factor": True,        # factorized top cumulant; auto-used only for k_max in {3,4}
    "use_pK": True,        # the real power-cumulant path (False is an ablation only)
    "output_d_max": 1,     # only output the mean (degree-1 cumulant)
}

# ---- Monte-Carlo ------------------------------------------------------------
MC_SAMPLES    = 64_000    # samples for the empirical mean (Q1 default)
MC_BATCH_SIZE = 8192      # batch size for MC (memory control)

# ---- Training-to-zero -------------------------------------------------------
TRAIN_STEPS = 3000        # AdamW steps; loss = MSE(model(x), 0) on fresh Gaussian inputs
BATCH_SIZE  = 1024
LR          = 1e-3

print("cumulant config:", config_summary(CUMULANT_CONFIG))
""")

# --- Adapter walkthrough -----------------------------------------------------
md(r"""## 4. The adapter, step by step (what actually gets fed to kprop)

`run_cumulant_propagation_from_model(model, input_dim, cumulant_config, device)`:
1. extracts each `nn.Linear`'s `(weight, bias)` from `model.Ws`,
2. verifies the first layer's `in_features == input_dim` (orientation guard),
3. builds `K_in = {1: zeros(n), 2: eye(n)}` for `X ~ N(0, I)`,
4. calls the real `mlp_kprop(...)`,
5. returns `{"raw_output": K_out, "mean": <output mean>, "metadata": {...}}`.

The cell below runs it on a tiny model **with `debug=True`** so you can see the
extracted layer shapes, the converted shapes, the kprop output keys, and the
predicted mean — i.e. exactly what crosses the boundary into the algorithm.
""")

code(r"""
import logging
logging.getLogger("cumulant_adapter").setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, force=True)

set_seed(0)
tiny = make_mlp(input_dim=8, hidden_width=16, hidden_depth=2, output_dim=1,
                activation=ACTIVATION, bias=True, device=DEVICE)
res = run_cumulant_propagation_from_model(tiny, input_dim=8,
                                          cumulant_config=CUMULANT_CONFIG, device=DEVICE, debug=True)
print("\nmetadata.layer_shapes:")
for s in res["metadata"]["layer_shapes"]:
    print("  ", s)
print("predicted mean (cp):", res["mean"])
logging.getLogger("cumulant_adapter").setLevel(logging.WARNING)
""")

# --- Sanity checks -----------------------------------------------------------
md(r"""## 5. Sanity checks (run these before trusting anything)

1. **Single linear layer** `y = Wx + b`, `x ~ N(0, I)`: the true mean is exactly
   `b`. kprop must return `b`. This validates **weight orientation** and **bias**.
2. **Small untrained MLP**: kprop mean vs a multi-million-sample MC mean — should
   agree to a small relative error (catches activation / orientation bugs).
""")

code(r"""
ok = run_sanity_checks(device=DEVICE, cumulant_config=CUMULANT_CONFIG)
assert ok, "Sanity checks failed — fix before proceeding."
""")

# --- One model end-to-end ----------------------------------------------------
md(r"""## 6. One model, end-to-end and fully visible

Build a model, print its architecture / input cumulants / layer norms, get the
kprop mean and the MC mean, and compute the metrics. This is the atomic unit the
sweeps repeat.
""")

code(r"""
set_seed(0)
# Model is built in MODEL_DTYPE (float32 on GPU for fast forward); kprop casts a float64 copy.
model = make_mlp(input_dim=INPUT_DIM, hidden_width=128, hidden_depth=HIDDEN_DEPTH,
                 output_dim=OUTPUT_DIM, activation=ACTIVATION, bias=USE_BIAS,
                 device=DEVICE, dtype=MODEL_DTYPE)
print(model)

# The exact input cumulant tower fed in (X ~ N(0, I_INPUT_DIM)):
print("\nK_in = {1: zeros(%d)  (mean=0),  2: eye(%d)  (covariance=I)}" % (INPUT_DIM, INPUT_DIM))

wn, bn = layer_norms(model)
print("weight norms per layer:", [round(x,3) for x in wn])
print("bias   norms per layer:", [None if b is None else round(b,3) for b in bn])

cp = extract_mean(run_cumulant_propagation_from_model(model, INPUT_DIM, CUMULANT_CONFIG, device=DEVICE))
mc, mc_stats = estimate_empirical_mean(model=model, input_dim=INPUT_DIM,
                                       num_samples=MC_SAMPLES, batch_size=MC_BATCH, device=DEVICE,
                                       dtype=MODEL_DTYPE)
m = compare_means(cp, mc, mc_stats)
print("\nkprop mean :", cp)
print("MC    mean :", mc, " (±", mc_stats['mc_stderr'], "stderr)")
print("relative_error_mean = |cp-mc|/|mc| =", round(m['relative_error_mean'],6))
print("mc_noise_z          = |cp-mc|/MC_stderr =", round(m['mc_noise_z'],3),
      "  (<~1: within MC noise; >>1: real bias)")
""")

# --- Q1 sweep ----------------------------------------------------------------
md(r"""## 7. Q1 — Does kprop predict the mean after training to zero?

For each (width, seed): evaluate the **initial** random model, then **train it to
zero** and evaluate again. We record the scale-free relative error and the z-score.

The sweep below is **reduced** so it runs quickly on Colab. To run the full
published sweep (widths 64/128/256/512, seeds 0–4, 5000 steps), use the CLI cell
in §8 instead. Edit `WIDTHS_Q1` / `SEEDS_Q1` as you like.
""")

code(r"""
# Self-contained imports so this cell runs even if earlier cells weren't executed.
import torch, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
from cumulant_experiments.cumulant_adapter import run_cumulant_propagation_from_model, extract_mean
from cumulant_experiments.metrics import estimate_empirical_mean, compare_means
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, set_seed

# Config falls back to sensible defaults if §1/§3 weren't run (so the cell stands alone).
G = globals()
DEVICE       = G.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE  = G.get("MODEL_DTYPE", torch.float32 if DEVICE == "cuda" else torch.float64)
MC_BATCH     = G.get("MC_BATCH", 262_144 if DEVICE == "cuda" else 8_192)
INPUT_DIM    = G.get("INPUT_DIM", 64);   OUTPUT_DIM = G.get("OUTPUT_DIM", 1)
HIDDEN_DEPTH = G.get("HIDDEN_DEPTH", 3); ACTIVATION = G.get("ACTIVATION", "relu")
USE_BIAS     = G.get("USE_BIAS", True);  BATCH_SIZE = G.get("BATCH_SIZE", 1024); LR = G.get("LR", 1e-3)
MC_SAMPLES   = G.get("MC_SAMPLES", 64_000)
TRAIN_STEPS  = G.get("TRAIN_STEPS", 3000)
CUMULANT_CONFIG = G.get("CUMULANT_CONFIG", {"k_max": 3, "kind": "simple", "use_avg_metric": False,
                                            "factor": True, "use_pK": True, "output_d_max": 1})

WIDTHS_Q1 = [64, 128, 256]      # add 512 (and 1024, slow) for the full picture
SEEDS_Q1  = [42, 43]            # different seeds => different random init; add more for tighter medians

records = []
for width in WIDTHS_Q1:
    for seed in SEEDS_Q1:
        set_seed(seed)
        model = make_mlp(input_dim=INPUT_DIM, hidden_width=width, hidden_depth=HIDDEN_DEPTH,
                         output_dim=OUTPUT_DIM, activation=ACTIVATION, bias=USE_BIAS,
                         device=DEVICE, dtype=MODEL_DTYPE)
        for phase in ["initial", "trained_to_zero"]:
            if phase == "trained_to_zero":
                train_model_to_zero(model=model, input_dim=INPUT_DIM, steps=TRAIN_STEPS,
                                    batch_size=BATCH_SIZE, lr=LR, device=DEVICE, dtype=MODEL_DTYPE)
            cp = extract_mean(run_cumulant_propagation_from_model(model, INPUT_DIM, CUMULANT_CONFIG, device=DEVICE))
            mc, st = estimate_empirical_mean(model=model, input_dim=INPUT_DIM,
                                             num_samples=MC_SAMPLES, batch_size=MC_BATCH, device=DEVICE,
                                             dtype=MODEL_DTYPE)
            m = compare_means(cp, mc, st)
            records.append(dict(width=width, seed=seed, phase=phase,
                                rel_err=m['relative_error_mean'], z=m['mc_noise_z'],
                                abs_err=m['mean_abs_error'], out_rms=st['empirical_output_rms']))
            print(f"  n={width:4d} s={seed} {phase:15s} rel_err={m['relative_error_mean']:.3e} "
                  f"z={m['mc_noise_z']:.2f} out_rms={st['empirical_output_rms']:.2e}")
dfq1 = pd.DataFrame(records)
dfq1.head(12)
""")

code(r"""
# Plot: relative error and z-score vs width, median over seeds, per phase.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, col, ttl in [(axes[0], "rel_err", "relative error |cp-mc|/|mc|"),
                     (axes[1], "z", "z = |cp-mc| / MC_stderr")]:
    for phase, c in [("initial", "tab:blue"), ("trained_to_zero", "tab:red")]:
        g = dfq1[dfq1.phase == phase].groupby("width")[col].median()
        ax.plot(g.index, g.values, marker="o", color=c, label=phase)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width"); ax.set_title(ttl); ax.grid(True, which="both", alpha=0.3); ax.legend()
axes[1].axhline(1.0, color="gray", ls="--", lw=1)
plt.tight_layout(); plt.show()
print("Reading it: at init z<~1 (kprop within MC noise). If trained z>>1, training introduced a real bias.")
""")

# --- Full CLI run ------------------------------------------------------------
md(r"""## 8. (Optional) Run the full, published Q1 sweep via the CLI

This calls `experiments/run_cumulant_train_to_zero_mean.py`, which writes an
incremental CSV, config.json, and plots. Slower (esp. width 512). Then we load
the CSV and show the saved plots inline.
""")

code(r"""
import subprocess, sys
cmd = [sys.executable, "experiments/run_cumulant_train_to_zero_mean.py",
       "--widths", "64", "128", "256",            # add 512 for the full run
       "--seeds", "0", "1", "2",
       "--mc-samples", "64000", "--train-steps", "3000",
       "--k-max", "3", "--outdir", "results/colab_q1"]
print(" ".join(cmd))
print(subprocess.run(cmd, capture_output=True, text=True).stdout[-3000:])
""")

code(r"""
from IPython.display import Image, display
import os
pdir = "results/colab_q1/plots"
if os.path.isdir(pdir):
    for f in ["relative_error_vs_width.png", "z_vs_mc_noise_vs_width.png", "output_rms_vs_width.png"]:
        p = os.path.join(pdir, f)
        if os.path.exists(p):
            print(f); display(Image(p))
""")

# --- Q2/Q3 width scaling -----------------------------------------------------
md(r"""## 9. Q2 & Q3 — Width-scaling, measured the way the law is stated

Two pitfalls make the scaling invisible if you measure a single scalar output's
relative error:
1. At `k_max=3` the kprop error is *below the Monte-Carlo noise floor* of a 64k
   (even 2M) sample mean → you measure MC noise, which is flat in width.
2. A single scalar is a high-variance estimator; the `n^(-k_max/2)` law is about
   **ensemble/many-component averages**.

Fixes used here (this is what the paper effectively does):
- **Vector output** (`OUTPUT_DIM_SCALE` components) so per-entry MSE averages over
  many components; median over seeds.
- **MC-variance debiasing:** measured per-entry MSE `E[(cp-mc)^2]` = kprop_MSE +
  Var(mc). We subtract the MC-variance floor (`mean stderr^2`) to resolve kprop
  error *below* the sampling floor.
- **`input_dim = width`** (the input dimension scales *with* the hidden width).
  The `O(n^{-k_max})` law assumes the relevant dimension → ∞; with a *fixed*
  input_dim the first layer's fan-in never grows and acts as an error floor that
  flattens the measured slope. Scaling the input dimension removes that bottleneck,
  so the slope should approach the theoretical `−k_max/2`.

We evaluate the **same** net before/after training, at `k_max ∈ {2,3}` (extend
`KMAXS_SCALE` if you like — there is no longer a hard cap), and look at (a) the
width-slope and (b) whether raising `k_max` still helps.

### The null control (`null_scaled`)
A third phase tests the obvious confound: maybe kprop "breaks" after training simply
because the **output is tiny** (the trained mean ≈ 0), not because of any
correlation structure. To rule that out we add a **random-init model whose last
layer is multiplied by `1e-5`** (`null_scaled`). This makes the output just as tiny
as the trained model, **but the weights stay random/independent** — no training.

Theoretically this should change *nothing* about kprop's accuracy: scaling the
final **linear** readout by `c` scales both the kprop mean and the MC mean by `c`
exactly (kprop is exact through linear layers), so the relative error, the
`k_max` ladder, and the width-slope are **identical to `initial`** (the debiased
RMS curve is just shifted down by `c`). If the empirics confirm that — and they
should — then a tiny output is *not* what breaks kprop; only the training-induced
correlations are (which is the `trained_to_zero` curve flattening). This is the
clean null/control for our claim.
""")

code(r"""
# Self-contained imports so this cell runs even if earlier cells weren't executed.
import torch, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt
from cumulant_experiments.cumulant_adapter import run_cumulant_propagation_from_model, extract_mean
from cumulant_experiments.metrics import estimate_empirical_mean, compare_means
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, set_seed

def debiased_rms(cp_mean, mc_mean, mc_stats):
    cp = np.asarray(cp_mean, np.float64).reshape(-1)
    mc = np.asarray(mc_mean, np.float64).reshape(-1)
    se = np.asarray(mc_stats["mc_stderr"], np.float64).reshape(-1)
    measured = float(np.mean((cp - mc) ** 2))      # = true kprop MSE + MC variance
    floor    = float(np.mean(se ** 2))             # the MC variance floor
    return math.sqrt(max(measured - floor, 0.0)), measured, floor

def loglog_slope(ws, vs):
    # Fit only the positive points (a debiased RMS can floor to 0 at large width /
    # high k_max when it dips below the MC sampling floor; drop those rather than
    # letting one zero turn the whole slope into NaN).
    ws, vs = np.asarray(ws, float), np.asarray(vs, float)
    mask = vs > 0
    if mask.sum() < 2: return float("nan")
    return float(np.polyfit(np.log(ws[mask]), np.log(vs[mask]), 1)[0])

def scale_last_layer(model, c):
    # Null control: scale the final LINEAR readout (weight + bias) by c. kprop is
    # exact through linear layers, so this should not change relative error / slope.
    with torch.no_grad():
        model.Ws[-1].weight.mul_(c)
        if model.Ws[-1].bias is not None:
            model.Ws[-1].bias.mul_(c)
    return model

# Config falls back to sensible defaults if §1/§3 weren't run (so the cell stands alone).
G = globals()
DEVICE       = G.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE  = G.get("MODEL_DTYPE", torch.float32 if DEVICE == "cuda" else torch.float64)
HIDDEN_DEPTH = G.get("HIDDEN_DEPTH", 3); ACTIVATION = G.get("ACTIVATION", "relu")
USE_BIAS     = G.get("USE_BIAS", True);  BATCH_SIZE = G.get("BATCH_SIZE", 1024); LR = G.get("LR", 1e-3)

# Defaults tuned for a Colab GPU. Width 1024 with input_dim=1024 + 2M MC takes a
# few minutes; drop it (or lower MC_SCALE) if you are on CPU.
WIDTHS_SCALE     = [64, 128, 256, 512, 1024]   # input_dim scales WITH width (see below)
SEEDS_SCALE      = [42, 43, 44]                # different inits to see the variance
KMAXS_SCALE      = [2, 3]                       # no hard cap; add 4 if you want
PHASES_SCALE     = ["initial", "null_scaled", "trained_to_zero"]
OUTPUT_DIM_SCALE = 128                          # vector output: many components to average over
NULL_SCALE       = 1e-5                         # last-layer scale for the null control
MC_SCALE         = 2_000_000                    # lower this to speed up / save memory
MC_BATCH_SCALE   = 200_000 if DEVICE == "cuda" else 50_000  # input_dim is large here
# Trained-to-zero needs ENOUGH steps for the output to actually collapse (otherwise the
# "trained" phase isn't really near zero and the effect washes out). These nets are big
# (input_dim=width up to 1024, output_dim=128), so use many steps + early-stop on loss.
TRAIN_STEPS_SCALE = 12000
TRAIN_LOSS_TOL    = 1e-8                         # early-stop once MSE-to-zero < this

def cfg_for(k):
    # factor is only implemented for k_max in {3,4}; adapter auto-handles others too.
    return {"k_max": k, "kind": "simple", "use_avg_metric": False,
            "factor": (k in (3, 4)), "use_pK": True, "output_d_max": 1}

rows = []
for width in WIDTHS_SCALE:
    in_dim = width                              # <<< KEY: input dimension scales with width
    for seed in SEEDS_SCALE:
        for phase in PHASES_SCALE:
            # Rebuild from the SAME seed each phase so initial/null/trained share one init.
            set_seed(seed)
            model = make_mlp(input_dim=in_dim, hidden_width=width, hidden_depth=HIDDEN_DEPTH,
                             output_dim=OUTPUT_DIM_SCALE, activation=ACTIVATION, bias=USE_BIAS,
                             device=DEVICE, dtype=MODEL_DTYPE)
            final_loss = float("nan")
            if phase == "null_scaled":
                scale_last_layer(model, NULL_SCALE)               # tiny output, but weights still random
            elif phase == "trained_to_zero":
                stats = train_model_to_zero(model=model, input_dim=in_dim, steps=TRAIN_STEPS_SCALE,
                                            batch_size=BATCH_SIZE, lr=LR, device=DEVICE, dtype=MODEL_DTYPE,
                                            loss_tol=TRAIN_LOSS_TOL)
                final_loss = stats["final_train_loss"]
            mc, st = estimate_empirical_mean(model=model, input_dim=in_dim, num_samples=MC_SCALE,
                                             batch_size=MC_BATCH_SCALE, device=DEVICE, dtype=MODEL_DTYPE)
            for k in KMAXS_SCALE:
                cp = extract_mean(run_cumulant_propagation_from_model(model, in_dim, cfg_for(k), device=DEVICE))
                rms, meas, floor = debiased_rms(cp, mc, st)
                # scale-free error (lets us compare phases despite very different output scales),
                # and signal-to-floor (>~3 = resolved; <~1 = below the MC sampling floor, unreliable).
                rel = rms / (st["empirical_output_rms"] + 1e-30)
                s_to_floor = (meas - floor) / (floor + 1e-300)
                rows.append(dict(phase=phase, k_max=k, width=width, input_dim=in_dim, seed=seed,
                                 debiased_rms=rms, rel_debiased_rms=rel, s_to_floor=s_to_floor,
                                 out_rms=st['empirical_output_rms'], final_train_loss=final_loss))
            # max-k s/floor at this point tells you whether the hardest curve is even resolved
            s2f_hi = next(r["s_to_floor"] for r in reversed(rows) if r["phase"]==phase and r["width"]==width and r["seed"]==seed)
            loss_str = "" if math.isnan(final_loss) else f" final_loss={final_loss:.1e}"
            print(f"  done {phase:15s} n={width} s={seed} out_rms={st['empirical_output_rms']:.2e}"
                  f"{loss_str} s/floor(k={KMAXS_SCALE[-1]})={s2f_hi:.1f}")
dfsc = pd.DataFrame(rows)
print("\nTrained-to-zero check (output should be << initial's; loss should be tiny):")
for ph in PHASES_SCALE:
    sub = dfsc[dfsc.phase==ph]
    g = sub.groupby("width").agg(out_rms=("out_rms","median"), loss=("final_train_loss","median"))
    print(f"  {ph:15s}: " + "  ".join(f"n{w}: out_rms={g.out_rms[w]:.1e}" +
          ("" if math.isnan(g.loss[w]) else f"/loss={g.loss[w]:.0e}") for w in sorted(g.index)))
print("\nSlope of debiased RMS vs width (theory ~ -k_max/2). NOTE: a curve is only meaningful")
print("where s/floor >~ 3 (printed above); k_max=3 may be below the 2M-sample MC floor.")
for phase in PHASES_SCALE:
    for k in KMAXS_SCALE:
        sub = dfsc[(dfsc.phase==phase)&(dfsc.k_max==k)]
        med = sub.groupby("width")["debiased_rms"].median()
        ws = sorted(med.index); vs = [med[w] for w in ws]
        print(f"  {phase:15s} k_max={k}: slope={loglog_slope(ws,vs):+.2f}  " +
              "  ".join(f"n{w}={med[w]:.2e}" for w in ws))
print("\nDoes raising k_max 2->3 still help? (ratio rms_k2/rms_k3, >1 = k=3 better;"
      " 'floor' = k=3 below MC floor so unresolved):")
for phase in PHASES_SCALE:
    m2 = dfsc[(dfsc.phase==phase)&(dfsc.k_max==2)].groupby("width")["debiased_rms"].median()
    m3 = dfsc[(dfsc.phase==phase)&(dfsc.k_max==3)].groupby("width")["debiased_rms"].median()
    ws = sorted(set(m2.index)&set(m3.index))
    parts = [f"n{w}={(m2[w]/m3[w]):.1f}x" if m3[w] > 0 else f"n{w}=floor" for w in ws]
    print(f"  {phase:15s}: " + "  ".join(parts))
""")

code(r"""
# Plot: debiased RMS vs width. Color = k_max; line style = phase; thin dotted = theory n^(-k/2).
# Expectation: 'initial' and 'null_scaled' have the SAME slope/k_max-ladder (null is just shifted
# down ~1e-5); 'trained_to_zero' flattens and its k_max curves collapse onto each other.
fig, ax = plt.subplots(figsize=(8, 5.4))
kc = {2: "tab:blue", 3: "tab:red", 4: "tab:green"}
ps = {"initial": dict(ls="-", marker="o"),
      "null_scaled": dict(ls="-.", marker="^"),
      "trained_to_zero": dict(ls="--", marker="s")}
for phase in PHASES_SCALE:
    for k in KMAXS_SCALE:
        sub = dfsc[(dfsc.phase==phase)&(dfsc.k_max==k)]
        if sub.empty: continue
        med = sub.groupby("width")["debiased_rms"].median()
        ws = np.array(sorted(med.index)); y = med.reindex(ws).to_numpy()
        ax.plot(ws, np.where(y<=0, np.nan, y), color=kc.get(k,"k"),
                label=f"{phase} k_max={k}", **ps.get(phase, dict(ls="-", marker="o")))
# theory guide n^(-k/2), anchored at the initial curve's smallest width
for k in KMAXS_SCALE:
    b = dfsc[(dfsc.k_max==k)&(dfsc.phase=="initial")].groupby("width")["debiased_rms"].median()
    ws = np.array(sorted(b.index), float)
    if len(b) and b.iloc[0] > 0:
        ax.plot(ws, b.iloc[0]*(ws/ws[0])**(-k/2), color=kc.get(k,"k"), ls=":", lw=1, alpha=0.5)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("width n  (= input_dim here)"); ax.set_ylabel("debiased per-entry RMS error of output mean")
ax.set_title("Width-scaling (ABSOLUTE): initial vs null-scaled vs trained-to-zero\n(thin dotted = theory n^(-k_max/2))")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8); plt.show()
""")

md(r"""**Scale-free version.** The absolute plot separates the phases by their very
different output magnitudes (`null_scaled` sits ~`1e-5×` lower, so it looks like a
different curve). Dividing the error by each model's `output_rms` removes that
scale: now **`null_scaled` should land right on top of `initial`** (proving a tiny
output changes nothing), while **`trained_to_zero` sits clearly higher** (genuinely
worse) and its `k_max=2`/`k_max=3` curves merge (the ladder collapse).
""")

code(r"""
# Plot: SCALE-FREE error (debiased RMS / output_rms) vs width -> phases are directly comparable.
fig, ax = plt.subplots(figsize=(8, 5.4))
for phase in PHASES_SCALE:
    for k in KMAXS_SCALE:
        sub = dfsc[(dfsc.phase==phase)&(dfsc.k_max==k)]
        if sub.empty: continue
        med = sub.groupby("width")["rel_debiased_rms"].median()
        ws = np.array(sorted(med.index)); y = med.reindex(ws).to_numpy()
        ax.plot(ws, np.where(y<=0, np.nan, y), color=kc.get(k,"k"),
                label=f"{phase} k_max={k}", **ps.get(phase, dict(ls="-", marker="o")))
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("width n  (= input_dim here)")
ax.set_ylabel("scale-free error = debiased RMS / output_rms")
ax.set_title("Width-scaling (SCALE-FREE): null_scaled should overlay initial;\ntrained_to_zero sits higher and its k_max curves merge")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8); plt.show()
""")

# --- Conclusions -------------------------------------------------------------
md(r"""## 10. How to read the results (cautious conclusions)

From our runs (depth 3, input_dim 64, widths 64–512), measured carefully:

- **Q1 (mean after training):** At initialization the kprop mean agrees with the
  64k-MC mean to within MC sampling noise (`z ≲ 1`) at all widths. After training
  to zero, a **statistically real bias appears** (`z` up to ~6 at width 64),
  i.e. kprop becomes **measurably less accurate** in scale-free terms. *Beware*:
  the trained *absolute* error is tiny only because the trained output is ~100×
  smaller — that is a scale illusion, not improved accuracy.

- **Q2 (scaling at init):** Qualitatively as expected — error **decreases with
  width**, the slope **steepens with `k_max`**, and raising `k_max` 2→3 buys
  6–15× accuracy. With input_dim *fixed* at 64 the measured exponents (~−0.5 for
  k=2, ~−1.0 for k=3) were *shallower* than the asymptotic `−k_max/2`, because a
  fixed input dimension is an error floor that doesn't shrink with width. The §9
  cell now uses **`input_dim = width`** (and you can raise `HIDDEN_DEPTH`) to
  remove that bottleneck — check whether the dotted `n^(-k_max/2)` guide and the
  measured slope now line up.

- **Q3 (does training break it):** The decisive signal is that **training
  disables the `k_max` refinement** — untrained, `k_max` 2→3 helps 6–15×; trained,
  it helps ~1× (no improvement). Width still helps the trained net, but at a
  `k_max`-independent rate. This is consistent with training inducing
  weight/activation correlations that violate the wide-random-MLP power-counting
  behind cumulant propagation.

- **Null control (`null_scaled`):** the random model with its last layer ×`1e-5`
  has an output as tiny as the trained model but **should match `initial`** on
  every scale-free metric (same slope, same `k_max` ladder, curve merely shifted
  down) — because scaling a linear readout is exact for kprop. If the plot shows
  `null_scaled` tracking `initial` (not `trained_to_zero`), it confirms that a
  small output is **not** what breaks kprop — only the training-induced
  correlations are. That is the clean control for the Q3 claim.

**Caveats:** few seeds and shallow depth → exponents are noisy; debiasing at
large width / high `k_max` can hit the MC floor (raise `MC_SCALE`). The robust,
scale-free findings are the **z-score bias** (Q1) and the **`k_max`-ladder
collapse** (Q3); treat exact exponents as indicative, not final.
""")

# --- Inspect source / config -------------------------------------------------
md(r"""## 11. Inspect: exactly what `compare_means` computes, and the `k_max` in use

Verify the metric definitions and the active `k_max` straight from the source —
no need to trust the prose above.
""")

code(r"""
import inspect
from cumulant_experiments.metrics import compare_means

print("===== source of compare_means (note nmse_mean vs relative_error_mean) =====\n")
print(inspect.getsource(compare_means))

print("===== k_max actually used by the algorithm =====")
try:
    print("Q1 CUMULANT_CONFIG['k_max'] =", CUMULANT_CONFIG["k_max"], " full:", CUMULANT_CONFIG)
except NameError:
    print("CUMULANT_CONFIG not defined yet (run §3).")
try:
    print("Scaling k_max values (KMAXS_SCALE) =", KMAXS_SCALE,
          "| example cfg_for(3) =", cfg_for(3))
except NameError:
    print("KMAXS_SCALE / cfg_for not defined yet (run §9).")
""")

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

out = os.path.join(os.path.dirname(__file__), "cumulant_experiments_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
