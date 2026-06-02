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
md(r"""## 1. Global numerical settings (and why)

- **`float64` everywhere.** Cumulant propagation is run in double precision in
  the repo's own tests; we match that. Monte-Carlo is also float64 for a
  consistent comparison.
- **`torch.manual_seed` / `numpy` / `random` are all seeded per run** (see
  `set_seed`) so results are reproducible.
- **Device**: uses CUDA if Colab gives you a GPU, else CPU. kprop and MC both run
  on the chosen device.
""")

code(r"""
import torch, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)   # double precision (kprop assumption)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| default dtype:", torch.get_default_dtype())
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
6. **`k_max ≤ 3`** (the budget `K`): higher OOMs at large width. `factor=True`
   for `k_max=3`. `use_avg_metric=False` (we use the *exact* metric `W·metric·Wᵀ`
   from the actual fixed weights, not the init-time `E[WWᵀ]`). `output_d_max=1`
   (we only need the mean → big FLOP savings).
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
    "k_max": 3,            # budget K; max cumulant order tracked in full. 1..3 (3 = hard max)
    "kind": "simple",      # SIMPLE = the canonical setting from the paper
    "use_avg_metric": False,  # exact metric from the fixed weights (see assumption 6)
    "factor": True,        # factorized top cumulant (needed for k_max=3 feasibility)
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
model = make_mlp(input_dim=INPUT_DIM, hidden_width=128, hidden_depth=HIDDEN_DEPTH,
                 output_dim=OUTPUT_DIM, activation=ACTIVATION, bias=USE_BIAS, device=DEVICE)
print(model)

# The exact input cumulant tower fed in (X ~ N(0, I_INPUT_DIM)):
print("\nK_in = {1: zeros(%d)  (mean=0),  2: eye(%d)  (covariance=I)}" % (INPUT_DIM, INPUT_DIM))

wn, bn = layer_norms(model)
print("weight norms per layer:", [round(x,3) for x in wn])
print("bias   norms per layer:", [None if b is None else round(b,3) for b in bn])

cp = extract_mean(run_cumulant_propagation_from_model(model, INPUT_DIM, CUMULANT_CONFIG, device=DEVICE))
mc, mc_stats = estimate_empirical_mean(model=model, input_dim=INPUT_DIM,
                                       num_samples=MC_SAMPLES, batch_size=MC_BATCH_SIZE, device=DEVICE)
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
WIDTHS_Q1 = [64, 128, 256]      # add 512 (and 1024, slow) for the full picture
SEEDS_Q1  = [0, 1]              # add more seeds for tighter medians

records = []
for width in WIDTHS_Q1:
    for seed in SEEDS_Q1:
        set_seed(seed)
        model = make_mlp(input_dim=INPUT_DIM, hidden_width=width, hidden_depth=HIDDEN_DEPTH,
                         output_dim=OUTPUT_DIM, activation=ACTIVATION, bias=USE_BIAS, device=DEVICE)
        for phase in ["initial", "trained_to_zero"]:
            if phase == "trained_to_zero":
                train_model_to_zero(model=model, input_dim=INPUT_DIM, steps=TRAIN_STEPS,
                                    batch_size=BATCH_SIZE, lr=LR, device=DEVICE)
            cp = extract_mean(run_cumulant_propagation_from_model(model, INPUT_DIM, CUMULANT_CONFIG, device=DEVICE))
            mc, st = estimate_empirical_mean(model=model, input_dim=INPUT_DIM,
                                             num_samples=MC_SAMPLES, batch_size=MC_BATCH_SIZE, device=DEVICE)
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

We evaluate the **same** net before/after training, at `k_max ∈ {2,3}`, and look
at (a) the width-slope and (b) whether raising `k_max` still helps.
""")

code(r"""
def debiased_rms(cp_mean, mc_mean, mc_stats):
    cp = np.asarray(cp_mean, np.float64).reshape(-1)
    mc = np.asarray(mc_mean, np.float64).reshape(-1)
    se = np.asarray(mc_stats["mc_stderr"], np.float64).reshape(-1)
    measured = float(np.mean((cp - mc) ** 2))      # = true kprop MSE + MC variance
    floor    = float(np.mean(se ** 2))             # the MC variance floor
    return math.sqrt(max(measured - floor, 0.0)), measured, floor

def loglog_slope(ws, vs):
    ws, vs = np.asarray(ws, float), np.asarray(vs, float)
    if np.any(vs <= 0) or len(ws) < 2: return float("nan")
    return float(np.polyfit(np.log(ws), np.log(vs), 1)[0])

# Reduced defaults for Colab. The published run used output_dim=128, seeds 0-2, 2M MC.
WIDTHS_SCALE     = [64, 128, 256]      # add 512 for the full picture
SEEDS_SCALE      = [0, 1, 2]
KMAXS_SCALE      = [2, 3]
OUTPUT_DIM_SCALE = 128                 # vector output: many components to average over
MC_SCALE         = 1_000_000           # bump to 2_000_000 to lower the debias floor
TRAIN_STEPS_SCALE = 3000

def cfg_for(k):
    return {"k_max": k, "kind": "simple", "use_avg_metric": False,
            "factor": (k >= 3), "use_pK": True, "output_d_max": 1}

rows = []
for width in WIDTHS_SCALE:
    for seed in SEEDS_SCALE:
        set_seed(seed)
        model = make_mlp(input_dim=INPUT_DIM, hidden_width=width, hidden_depth=HIDDEN_DEPTH,
                         output_dim=OUTPUT_DIM_SCALE, activation=ACTIVATION, bias=USE_BIAS, device=DEVICE)
        for phase in ["initial", "trained_to_zero"]:
            if phase == "trained_to_zero":
                train_model_to_zero(model=model, input_dim=INPUT_DIM, steps=TRAIN_STEPS_SCALE,
                                    batch_size=BATCH_SIZE, lr=LR, device=DEVICE)
            mc, st = estimate_empirical_mean(model=model, input_dim=INPUT_DIM,
                                             num_samples=MC_SCALE, batch_size=250_000, device=DEVICE)
            for k in KMAXS_SCALE:
                cp = extract_mean(run_cumulant_propagation_from_model(model, INPUT_DIM, cfg_for(k), device=DEVICE))
                rms, meas, floor = debiased_rms(cp, mc, st)
                rows.append(dict(phase=phase, k_max=k, width=width, seed=seed,
                                 debiased_rms=rms, out_rms=st['empirical_output_rms']))
        print(f"  done n={width} s={seed}")
dfsc = pd.DataFrame(rows)
print("\nSlope of debiased RMS vs width (theory ~ -k_max/2):")
for phase in ["initial", "trained_to_zero"]:
    for k in KMAXS_SCALE:
        sub = dfsc[(dfsc.phase==phase)&(dfsc.k_max==k)]
        med = sub.groupby("width")["debiased_rms"].median()
        ws = sorted(med.index); vs = [med[w] for w in ws]
        print(f"  {phase:15s} k_max={k}: slope={loglog_slope(ws,vs):+.2f}  " +
              "  ".join(f"n{w}={med[w]:.2e}" for w in ws))
print("\nDoes raising k_max 2->3 still help? (ratio rms_k2/rms_k3, >1 = k=3 better):")
for phase in ["initial","trained_to_zero"]:
    m2 = dfsc[(dfsc.phase==phase)&(dfsc.k_max==2)].groupby("width")["debiased_rms"].median()
    m3 = dfsc[(dfsc.phase==phase)&(dfsc.k_max==3)].groupby("width")["debiased_rms"].median()
    ws = sorted(set(m2.index)&set(m3.index))
    print(f"  {phase:15s}: " + "  ".join(f"n{w}={m2[w]/m3[w]:.1f}x" for w in ws))
""")

code(r"""
# Plot: debiased RMS vs width. Solid=initial, dashed=trained; color=k_max; dotted=theory n^(-k/2).
fig, ax = plt.subplots(figsize=(7.5, 5.2))
kc = {2: "tab:blue", 3: "tab:red"}
ps = {"initial": dict(ls="-", marker="o"), "trained_to_zero": dict(ls="--", marker="s")}
for phase in ["initial", "trained_to_zero"]:
    for k in KMAXS_SCALE:
        sub = dfsc[(dfsc.phase==phase)&(dfsc.k_max==k)]
        med = sub.groupby("width")["debiased_rms"].median()
        ws = np.array(sorted(med.index)); y = med.reindex(ws).to_numpy()
        ax.plot(ws, np.where(y<=0, np.nan, y), color=kc[k], label=f"{phase} k_max={k}", **ps[phase])
for k in KMAXS_SCALE:
    b = dfsc[(dfsc.k_max==k)&(dfsc.phase=="initial")].groupby("width")["debiased_rms"].median()
    ws = np.array(sorted(b.index), float)
    if len(b) and b.iloc[0] > 0:
        ax.plot(ws, b.iloc[0]*(ws/ws[0])**(-k/2), color=kc[k], ls=":", lw=1, alpha=0.6)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("width n"); ax.set_ylabel("debiased per-entry RMS error of output mean")
ax.set_title("Width-scaling: initial vs trained-to-zero (dotted = theory n^(-k_max/2))")
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
  6–15× accuracy. The measured exponents (~−0.5 for k=2, ~−1.0 for k=3) are
  *shallower* than the asymptotic `−k_max/2`, most likely because input_dim is
  fixed at 64 and depth is only 3 (not in the clean `n→∞` regime). Scaling
  input_dim with width and/or going deeper should move them toward `−k_max/2`.

- **Q3 (does training break it):** The decisive signal is that **training
  disables the `k_max` refinement** — untrained, `k_max` 2→3 helps 6–15×; trained,
  it helps ~1× (no improvement). Width still helps the trained net, but at a
  `k_max`-independent rate. This is consistent with training inducing
  weight/activation correlations that violate the wide-random-MLP power-counting
  behind cumulant propagation.

**Caveats:** few seeds and shallow depth → exponents are noisy; debiasing at
large width / high `k_max` can hit the MC floor (raise `MC_SCALE`). The robust,
scale-free findings are the **z-score bias** (Q1) and the **`k_max`-ladder
collapse** (Q3); treat exact exponents as indicative, not final.
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
