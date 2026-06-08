"""Generates notebooks/exact_relu_k2_width_scaling_colab.ipynb (valid nbformat-4 JSON).

Focused companion to build_notebook.py. The scientific question here is narrow:

    After training an MLP to output 0, how does the error of the EXACT closed-form
    K=2 ReLU cumulant propagation (exact_relu_k2=True, k_max=2) scale with width n,
    and how does it compare to the approximate harmonic "propagation" path
    (exact_relu_k2=False) at the same k_max=2?

It reuses the SAME repo library as build_notebook.py (cumulant_adapter, model_utils,
metrics) and the SAME width-scaling methodology as that notebook's section 9
(input_dim = width, vector output, MC-variance-debiased RMS, log-log slope fit).
Nothing is reimplemented; the exact path is the one added in
src/mlp_kprop/relu_k2_exact.py and wired through mlp_kprop via exact_relu_k2.

Run:  .venv/bin/python notebooks/build_exact_scaling_notebook.py
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


# =============================================================================
md(r"""# Trained-to-zero width scaling — **exact** vs **approximate** K=2 cumulant propagation

This notebook answers one question, end to end and on Colab:

> **Train an MLP so its output is ≈ 0. How does the error of the EXACT closed-form
> K=2 ReLU cumulant propagation scale with width `n`?** And how does it compare to
> the approximate ("propagation") harmonic path at the same budget `k_max = 2`?

**Two runnable versions of the same algorithm, on the same code (no copy):**
- **`propagation`** (`exact_relu_k2=False`): the general harmonic / power-cumulant
  K=2 step.
- **`exact`** (`exact_relu_k2=True`): the exact closed-form scalar Gaussian-ReLU
  mean/covariance update added in `src/mlp_kprop/relu_k2_exact.py`. The
  per-coordinate marginals (mean, diagonal variance) are *exact*; the off-diagonal
  covariance uses the first-order gain `c_i = Φ(α_i)`.

> **Key finding (verified in §4):** for ReLU these two are **numerically identical**
> (`‖exact − propagation‖ ≈ 1e-15`). The general `k_max=2` harmonic path *is* the
> exact closed-form Gaussian-ReLU propagation — so the new `relu_k2_exact.py` is a
> direct, self-contained re-derivation that validates it. The real accuracy lever is
> therefore **`k_max`**, so we scale the `exact` (`k=2`) path against the
> higher-order `k_max=3` propagation.

> **Why depth matters (read this).** For a **single hidden layer** the
> preactivation is exactly Gaussian, so the `exact` path returns the *exact*
> output mean (error = MC noise, nothing to scale). The interesting — and only
> non-trivial — regime is **≥ 2 hidden layers**, where the first-order off-diagonal
> covariance feeds the next layer's variance and the end-to-end mean becomes
> approximate. We therefore default `HIDDEN_DEPTH = 2`.

Nothing here reimplements cumulant propagation: we call the repo's `mlp_kprop` as a
black box and only flip the `exact_relu_k2` flag. Section 2 prints the source files
so you can confirm it.
""")

# --- Setup -------------------------------------------------------------------
md(r"""## 0. Setup — get the repo and install the *minimal* dependencies

Same as the main notebook. Point at the repo (clone URL or local/Drive path). The
checkout **must contain `src/mlp_kprop/relu_k2_exact.py` and the `exact_relu_k2`
flag** — i.e. push the branch with that change before running on Colab (section 2
checks this and fails loudly otherwise).
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
    if LOCAL_REPO_DIR and os.path.isdir(os.path.join(LOCAL_REPO_DIR, "src", "mlp_kprop")):
        return LOCAL_REPO_DIR
    if REPO_URL:
        dest = "/content/mlp_kprop" if IN_COLAB else os.path.abspath("./_mlp_kprop_clone")
        if not os.path.isdir(os.path.join(dest, "src", "mlp_kprop")):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
        return dest
    here = os.path.abspath(".")
    for _ in range(5):
        if os.path.isdir(os.path.join(here, "src", "mlp_kprop")):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate the repo. Set REPO_URL or LOCAL_REPO_DIR above.")

REPO_DIR = _find_repo()
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("IN_COLAB:", IN_COLAB)
print("REPO_DIR:", REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jaxtyping", "einops", "opt_einsum", "joblib", "tqdm"], check=False)
print("deps installed")
""")

# --- Global numerics ---------------------------------------------------------
md(r"""## 1. Global numerical settings

- **kprop runs in `float64`** (the adapter casts a float64 copy of the model).
- On GPU we do **training + Monte-Carlo in `float32`** (much faster; MC accuracy is
  limited by `1/√N`, not float32 rounding, and sums accumulate in float64). kprop
  is unaffected — it always gets a float64 model.
- Device = CUDA if available, else CPU.
""")

code(r"""
import torch, numpy as np, pandas as pd, math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float32 if DEVICE == "cuda" else torch.float64
MC_BATCH = 262_144 if DEVICE == "cuda" else 8_192
print("device:", DEVICE, "| MODEL_DTYPE:", MODEL_DTYPE, "| MC_BATCH:", MC_BATCH)
""")

# --- Verify the real algorithm + the exact path ------------------------------
md(r"""## 2. Verify we are calling the **real** algorithm **and** the exact path exists

Prints the source files of `mlp_kprop` and of the exact ReLU K=2 module, and
asserts the `exact_relu_k2` flag is present on `mlp_kprop`. If this assert fails,
your checkout predates the change — push/pull the branch that adds
`src/mlp_kprop/relu_k2_exact.py`.
""")

code(r"""
import inspect
import src.mlp_kprop.kprop_harmonic as kp
from src.mlp_kprop.mlp import MLP
from src.mlp_kprop.wick import WICK_COEF_D

print("mlp_kprop defined in:        ", inspect.getsourcefile(kp.mlp_kprop))
assert "exact_relu_k2" in inspect.signature(kp.mlp_kprop).parameters, (
    "This checkout has no exact_relu_k2 flag — push/pull the branch that adds "
    "src/mlp_kprop/relu_k2_exact.py and wires exact_relu_k2 through mlp_kprop."
)
import src.mlp_kprop.relu_k2_exact as rk2
print("exact ReLU K=2 step defined in:", inspect.getsourcefile(rk2.relu_k2_exact_kprop))
print("activations supported:        ", list(WICK_COEF_D.keys()))
print("\n--- closed-form moments (the exact computation) ---")
print("".join(inspect.getsource(rk2.relu_gaussian_moments).splitlines(keepends=True)[:1]))
print(inspect.getdoc(rk2.relu_k2_covariance_update))
""")

# --- The setup / knobs -------------------------------------------------------
md(r"""## 3. The exact setup and the knobs

For a fixed trained model `f` and `X ~ N(0, I_n)` we estimate `E_X[f(X)]` two ways
and compare them to a Monte-Carlo reference:
- **kprop (exact or propagation)** = degree-1 cumulant `K_out[1]` from `mlp_kprop`,
- **MC truth** = sample mean of `f(X)`.

We measure the **MC-variance-debiased per-entry RMS error** of the mean and how it
scales with width `n`. Following the main notebook's section 9 we set
**`input_dim = width`** (so the `n^(-k/2)` law is not bottlenecked by a fixed
fan-in) and use a **vector output** so the per-entry MSE averages over many
components.

Three methods (all the same algorithm; only the config differs):
- `exact (k=2)`  → `exact_relu_k2=True,  k_max=2`  (the TRUE closed-form ReLU K=2 step)
- `approx (k=2)` → `exact_relu_k2=False, k_max=2`  (the general harmonic K=2 step; ≡ exact for ReLU)
- `approx (k=3)` → `exact_relu_k2=False, k_max=3`  (higher-order; the real accuracy lever)

Phases: `initial` (untrained, the control) and `trained_to_zero` (the question).
""")

code(r"""
from cumulant_experiments.cumulant_adapter import run_cumulant_propagation_from_model, extract_mean, config_summary
from cumulant_experiments.metrics import estimate_empirical_mean, compare_means
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, set_seed, layer_norms

# ---- Architecture ----------------------------------------------------------
HIDDEN_DEPTH = 2        # >=2 so the exact path is genuinely approximate (depth 1 is exact -> no scaling)
ACTIVATION   = "relu"   # the exact path only applies to ReLU
USE_BIAS     = True
OUTPUT_DIM   = 64       # vector output: per-entry MSE averages over many components

# ---- The methods. Each is just an mlp_kprop config; nothing reimplemented. -
COMMON_CFG = {"kind": "simple", "use_avg_metric": False, "use_pK": True, "output_d_max": 1}
METHODS = {
    "exact (k=2)":  {**COMMON_CFG, "k_max": 2, "factor": False, "exact_relu_k2": True},
    "approx (k=2)": {**COMMON_CFG, "k_max": 2, "factor": False, "exact_relu_k2": False},
    "approx (k=3)": {**COMMON_CFG, "k_max": 3, "factor": True,  "exact_relu_k2": False},
}
# Methods used in the scaling sweep. "approx (k=2)" is identical to "exact (k=2)"
# for ReLU (shown in §4), so we keep just the exact k=2 path + the k=3 reference;
# add "approx (k=2)" here if you want to SEE the two K=2 curves overlap.
SWEEP_METHODS = ["exact (k=2)", "approx (k=3)"]

# ---- Sweep ranges (Colab-friendly; scale up for a sharper slope) -----------
WIDTHS = [32, 64, 128, 256]   # input_dim = width; add 512 for a longer lever arm
SEEDS  = [0, 1, 2]            # more seeds => tighter medians
PHASES = ["initial", "trained_to_zero"]

# ---- Monte-Carlo + training -------------------------------------------------
MC_SAMPLES     = 500_000      # raise to push the MC-variance floor down
TRAIN_STEPS    = 8000
TRAIN_LOSS_TOL = 1e-8         # early-stop once MSE-to-zero < this
BATCH_SIZE     = 1024
LR             = 1e-3

for name in METHODS:
    print(f"{name:13s} cfg:", config_summary(METHODS[name]))
""")

# --- Verification: exact == approx at k=2; depth caveat ----------------------
md(r"""## 4. Verify: (a) `exact (k=2)` ≡ `approx (k=2)` for ReLU, and (b) the depth caveat

**(a) The two K=2 versions coincide.** For a ReLU net the exact closed-form K=2
step and the general harmonic K=2 step produce the *same* output mean to float64
roundoff (`‖·‖ ≈ 1e-15`), while `k_max=3` differs at order `1e-2`. So `relu_k2_exact.py`
is a validated re-derivation, and the meaningful accuracy axis is `k_max`.

**(b) Depth caveat.** With **1 hidden layer** the K=2 mean is the *exact* `E[f(X)]`
(within MC noise, `z ≲ 1`); with **≥ 2 hidden layers** it is approximate — that is
the regime we scale. We default `HIDDEN_DEPTH = 2`.
""")

code(r"""
# (a) exact (k=2) vs approx (k=2) vs approx (k=3): same model, compare output means.
set_seed(0); n0 = 128
for depth in [1, 2, 3]:
    model = make_mlp(input_dim=n0, hidden_width=n0, hidden_depth=depth, output_dim=8,
                     activation=ACTIVATION, bias=USE_BIAS, device=DEVICE, dtype=MODEL_DTYPE)
    cpe = extract_mean(run_cumulant_propagation_from_model(model, n0, METHODS["exact (k=2)"],  device=DEVICE))
    cpa = extract_mean(run_cumulant_propagation_from_model(model, n0, METHODS["approx (k=2)"], device=DEVICE))
    cp3 = extract_mean(run_cumulant_propagation_from_model(model, n0, METHODS["approx (k=3)"], device=DEVICE))
    print(f"depth={depth}: ||exact_k2 - approx_k2|| = {np.linalg.norm(cpe-cpa):.2e}   "
          f"||exact_k2 - approx_k3|| = {np.linalg.norm(cpe-cp3):.2e}")
assert np.linalg.norm(cpe - cpa) < 1e-9, "exact and approx k=2 should coincide for ReLU"
print("=> exact (k=2) and approx (k=2) coincide for ReLU; k=3 is a genuinely different (higher-order) method.\n")

# (b) depth-1 exact mean is the exact E[f(X)]; depth-2 is approximate.
set_seed(0)
for depth in [1, 2]:
    model = make_mlp(input_dim=n0, hidden_width=n0, hidden_depth=depth, output_dim=OUTPUT_DIM,
                     activation=ACTIVATION, bias=USE_BIAS, device=DEVICE, dtype=MODEL_DTYPE)
    cp = extract_mean(run_cumulant_propagation_from_model(model, n0, METHODS["exact (k=2)"], device=DEVICE))
    mc, st = estimate_empirical_mean(model=model, input_dim=n0, num_samples=MC_SAMPLES,
                                     batch_size=MC_BATCH, device=DEVICE, dtype=MODEL_DTYPE)
    m = compare_means(cp, mc, st)
    verdict = "EXACT (within MC noise)" if m['mc_noise_z'] < 3 else "approximate (real error)"
    print(f"hidden_depth={depth}: exact-path  rel_err={m['relative_error_mean']:.3e}  z={m['mc_noise_z']:.2f}  -> {verdict}")
""")

# --- The main sweep ----------------------------------------------------------
md(r"""## 5. The width-scaling sweep (the experiment)

For each `(width, seed, phase)`: build the net (`input_dim = width`), optionally
train it to zero, draw one Monte-Carlo reference, then evaluate **both** methods
(`exact`, `propagation`) against that same MC. We record the MC-variance-debiased
per-entry RMS error, its scale-free version (÷ `output_rms`), and a
**signal-to-floor** ratio (`>~3` = resolved above MC noise; `<~1` = unreliable).

Reusing the section-9 estimator helpers verbatim (debiasing + slope fit).
""")

code(r"""
import torch, numpy as np, pandas as pd, math
from cumulant_experiments.cumulant_adapter import run_cumulant_propagation_from_model, extract_mean
from cumulant_experiments.metrics import estimate_empirical_mean
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, set_seed

def debiased_rms(cp_mean, mc_mean, mc_stats):
    cp = np.asarray(cp_mean, np.float64).reshape(-1)
    mc = np.asarray(mc_mean, np.float64).reshape(-1)
    se = np.asarray(mc_stats["mc_stderr"], np.float64).reshape(-1)
    measured = float(np.mean((cp - mc) ** 2))   # = true kprop MSE + MC variance
    floor    = float(np.mean(se ** 2))           # MC variance floor
    return math.sqrt(max(measured - floor, 0.0)), measured, floor

def loglog_slope(ws, vs):
    ws, vs = np.asarray(ws, float), np.asarray(vs, float)
    mask = vs > 0
    if mask.sum() < 2: return float("nan")
    return float(np.polyfit(np.log(ws[mask]), np.log(vs[mask]), 1)[0])

G = globals()
DEVICE = G.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = G.get("MODEL_DTYPE", torch.float32 if DEVICE == "cuda" else torch.float64)
MC_BATCH = G.get("MC_BATCH", 262_144 if DEVICE == "cuda" else 8_192)

rows = []
for width in WIDTHS:
    in_dim = width                                  # <<< input dimension scales WITH width
    for seed in SEEDS:
        for phase in PHASES:
            set_seed(seed)                          # same init across phases
            model = make_mlp(input_dim=in_dim, hidden_width=width, hidden_depth=HIDDEN_DEPTH,
                             output_dim=OUTPUT_DIM, activation=ACTIVATION, bias=USE_BIAS,
                             device=DEVICE, dtype=MODEL_DTYPE)
            final_loss = float("nan")
            if phase == "trained_to_zero":
                stats = train_model_to_zero(model=model, input_dim=in_dim, steps=TRAIN_STEPS,
                                            batch_size=BATCH_SIZE, lr=LR, device=DEVICE,
                                            dtype=MODEL_DTYPE, loss_tol=TRAIN_LOSS_TOL)
                final_loss = stats["final_train_loss"]
            mc, st = estimate_empirical_mean(model=model, input_dim=in_dim, num_samples=MC_SAMPLES,
                                             batch_size=MC_BATCH, device=DEVICE, dtype=MODEL_DTYPE)
            for method in SWEEP_METHODS:            # evaluate each method against the SAME MC
                cp = extract_mean(run_cumulant_propagation_from_model(model, in_dim, METHODS[method], device=DEVICE))
                rms, meas, floor = debiased_rms(cp, mc, st)
                rel = rms / (st["empirical_output_rms"] + 1e-30)
                s2f = (meas - floor) / (floor + 1e-300)
                rows.append(dict(method=method, phase=phase, width=width, input_dim=in_dim, seed=seed,
                                 debiased_rms=rms, rel_debiased_rms=rel, s_to_floor=s2f,
                                 out_rms=st["empirical_output_rms"], final_train_loss=final_loss))
            ls = "" if math.isnan(final_loss) else f" loss={final_loss:.0e}"
            print(f"  n={width:4d} s={seed} {phase:15s} out_rms={st['empirical_output_rms']:.2e}{ls}")
dfsc = pd.DataFrame(rows)
dfsc.head(8)
""")

code(r"""
# Slope summary: debiased RMS ~ n^slope, and scale-free RMS ~ n^slope, per (method, phase).
print("Scaling of the error with width  (slope of debiased RMS vs n; theory ~ -k_max/2)")
print("  s/floor < ~3 means a point is at/below the MC noise floor -> raise MC_SAMPLES.\n")
for phase in PHASES:
    for method in SWEEP_METHODS:
        sub = dfsc[(dfsc.phase == phase) & (dfsc.method == method)]
        med  = sub.groupby("width")["debiased_rms"].median()
        rmed = sub.groupby("width")["rel_debiased_rms"].median()
        s2f  = sub.groupby("width")["s_to_floor"].median()
        ws = sorted(med.index)
        slope_abs = loglog_slope(ws, [med[w] for w in ws])
        slope_rel = loglog_slope(ws, [rmed[w] for w in ws])
        print(f"  {phase:15s} {method:12s}: slope_abs={slope_abs:+.2f}  slope_scalefree={slope_rel:+.2f}  "
              + "  ".join(f"n{w}:{rmed[w]:.2e}(s/f={s2f[w]:.0f})" for w in ws))
print("\nHeadline (the question): the trained-to-zero EXACT (k=2) scale-free slope is")
sub = dfsc[(dfsc.phase=='trained_to_zero') & (dfsc.method=='exact (k=2)')]
rmed = sub.groupby('width')['rel_debiased_rms'].median(); ws = sorted(rmed.index)
print(f"    slope_scalefree = {loglog_slope(ws, [rmed[w] for w in ws]):+.2f}   "
      "(near -1 => K=2 width-scaling survives training; ~0 => training flattens it)")
""")

# --- Plots -------------------------------------------------------------------
md(r"""## 6. Plots — the scaling of the error with width

- **Left (absolute):** debiased per-entry RMS error vs `n`. Thin dotted guides =
  `n^(-1)` (k=2 theory) and `n^(-1.5)` (k=3 theory). Marker/line per method; blue =
  `initial`, red = `trained_to_zero`.
- **Right (scale-free):** the same error ÷ each model's `output_rms`, so `initial`
  and `trained_to_zero` are directly comparable despite the trained output being
  far smaller.

**What to look for:** does the **`exact (k=2)` / `trained_to_zero`** curve (red)
keep a clean `~n^-1` slope, or does training flatten it? And at init, is
`approx (k=3)` steeper than `exact (k=2)` (does higher `k_max` still buy a faster
rate)?
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
mc_color = {"initial": "tab:blue", "trained_to_zero": "tab:red"}
mt_style = {"exact (k=2)": dict(ls="-", marker="o"),
            "approx (k=2)": dict(ls=":", marker="x"),
            "approx (k=3)": dict(ls="--", marker="s")}
theory_slope = {"exact (k=2)": 1.0, "approx (k=2)": 1.0, "approx (k=3)": 1.5}

for ax, col, ttl in [(axes[0], "debiased_rms", "ABSOLUTE debiased per-entry RMS error"),
                     (axes[1], "rel_debiased_rms", "SCALE-FREE error (RMS / output_rms)")]:
    for phase in PHASES:
        for method in SWEEP_METHODS:
            sub = dfsc[(dfsc.phase == phase) & (dfsc.method == method)]
            if sub.empty: continue
            med = sub.groupby("width")[col].median()
            ws = np.array(sorted(med.index)); y = med.reindex(ws).to_numpy()
            ax.plot(ws, np.where(y <= 0, np.nan, y), color=mc_color[phase],
                    label=f"{method} / {phase}", **mt_style.get(method, dict(ls="-", marker="o")))
    # theory guides anchored at each method's initial smallest width
    for method in SWEEP_METHODS:
        base = dfsc[(dfsc.method==method)&(dfsc.phase=="initial")].groupby("width")[col].median()
        bw = np.array(sorted(base.index), float)
        if len(base) and base.iloc[0] > 0:
            ax.plot(bw, base.iloc[0]*(bw/bw[0])**(-theory_slope.get(method,1.0)),
                    color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width n  (= input_dim)"); ax.set_title(ttl)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("gray dotted = theory guides n^(-k_max/2). exact (k=2) and approx (k=2) overlap (same computation).")
""")

# --- Conclusions -------------------------------------------------------------
md(r"""## 7. How to read it

- **`exact (k=2)` ≡ `approx (k=2)`** (§4): the new closed form is a validated
  re-derivation of the general K=2 path for ReLU. Whatever scaling you see for one,
  you see for the other — they are the same computation.
- **`exact (k=2)`, `initial`:** the K=2 error falls with width; with
  `input_dim = width` and `HIDDEN_DEPTH = 2` it should track the `n^-1` guide
  (the `k_max=2` theory rate). This is the baseline scaling of the exact path.
- **`exact (k=2)`, `trained_to_zero` — the headline (answers the question):** read
  its `slope_scalefree` (printed in §5). If it stays near `-1`, training preserves
  the exact path's width-scaling; if it **flattens** toward `0`, training has
  induced weight/activation correlations the wide-random-MLP assumption misses —
  the same failure mode the main notebook documents (training *disables* the
  width/`k_max` refinement). Our broader runs lean toward the latter.
- **`exact (k=2)` vs `approx (k=3)`:** higher `k_max` is the only way to scale
  *faster* than `n^-1`. At init expect `k=3` steeper (~`n^-1.5`); after training,
  watch whether that advantage collapses (the `k_max` ladder flattening).

**Caveats:** few seeds + shallow depth ⇒ noisy exponents; at large `n` the debiased
RMS can dip to the MC floor (watch the `s/floor` column and raise `MC_SAMPLES`);
`k_max=3` is the costly method (use a GPU, or drop it from `SWEEP_METHODS`). Add
`512` to `WIDTHS` for a longer lever arm on the slope fit. To literally *see* the
two K=2 versions overlap, add `"approx (k=2)"` to `SWEEP_METHODS` and re-run §5–§6.
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

out = os.path.join(os.path.dirname(__file__), "exact_relu_k2_width_scaling_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
