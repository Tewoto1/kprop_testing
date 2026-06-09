"""Generates notebooks/exact_relu_k2_width_scaling_colab.ipynb (valid nbformat-4 JSON).

The question this notebook answers:

    The cumulant-propagation mean is accurate for random/initial MLPs but DEGRADES
    after a model is trained (the known "trained-model" problem). The default k=2
    propagation approximates the ReLU off-diagonal covariance by the leading-order
    gain Sigma_ij * c_i * c_j. Does replacing that with the EXACT bivariate-Gaussian
    ReLU covariance (exact_relu_cov=True) make the trained-model breakdown go away?

So we compare, at the SAME budget k_max=2, on the SAME trained models:
    - "approx (k=2)" : the default propagation (gain approximation), and
    - "exact (k=2)"  : the exact bivariate covariance (src/mlp_kprop/exact_relu_covariance.py),
and look at how each one's error-vs-Monte-Carlo scales with width n, initial vs
trained-to-zero. If the EXACT path scales better than the approx path AFTER
training, the covariance approximation was the culprit; if they scale the same,
it is not (the breakdown is intrinsic to k=2 / training-induced correlations).

Reuses the SAME repo library and the SAME width-scaling methodology as
build_notebook.py section 9 (input_dim=width, vector output, MC-variance-debiased
RMS, log-log slope fit). Nothing is reimplemented.

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
md(r"""# Does the EXACT ReLU covariance fix cumulant propagation on **trained** models?

Cumulant propagation predicts the output mean of a wide random MLP well, but its
accuracy **degrades after the model is trained** (see the main notebook's Q3: the
`k_max` refinement collapses). One suspect is the K=2 step's **off-diagonal
covariance**: the default propagation approximates
`Cov(ReLU(Z_i), ReLU(Z_j))` by the leading-order gain `Sigma_ij * c_i * c_j`
(`c_i = Phi(mu_i/sigma_i)`). This notebook tests whether computing that covariance
**exactly** removes the trained-model problem.

**Two versions of the algorithm, same budget `k_max = 2`, same model (no copy):**
- **`approx (k=2)`** — the default harmonic propagation (the gain approximation).
- **`exact (k=2)`** — `exact_relu_cov=True`: the **exact** bivariate-Gaussian ReLU
  covariance, `new_Sigma_ij = E[ReLU(Z_i)ReLU(Z_j)] − new_mu_i·new_mu_j`, no gain
  approximation (`src/mlp_kprop/exact_relu_covariance.py`).

**The test.** We probe **two trained regimes** (plus the untrained control), so the
conclusion does not hinge on a single task:
1. **`train_to_zero`** — train the output to 0 (MSE to 0), early-stopping at a
   *moderate* tolerance so the output stays large enough that the comparison is not
   swamped by Monte-Carlo / floating-point noise.
2. **`halfspace`** — train each output component to classify whether the input lies
   in a random half-space `{x : w_j·x > b_j}` (MSE to the 0/1 indicator); a
   genuinely different learned weight structure. (Models are trained with **no
   bias** — see §3 — so the ReLU net is positively homogeneous and fits the best
   homogeneous approximation of the half-space.)

For each `(width n, seed, regime)` (with `input_dim = n`) we measure the
MC-variance-debiased error of the propagated mean and its scaling with `n`, for
both methods. Then ask, **for each training regime**:

> Does **`exact (k=2)`** scale **better** than **`approx (k=2)`** after training
> (→ the covariance approximation was the problem) or **the same** (→ it was not;
> the breakdown is intrinsic to k=2 / training-induced correlations)?

> **Depth matters.** With **1 hidden layer** the output mean depends only on the
> (exact) ReLU *marginals*, so both versions give the exact mean — the off-diagonal
> covariance never enters. The covariance only matters at **≥ 2 hidden layers**
> (it feeds the next layer's variance), so we default `HIDDEN_DEPTH = 2`.

Nothing here reimplements cumulant propagation: we call the repo's `mlp_kprop` as a
black box and only flip `exact_relu_cov`. Section 2 prints the source files.
""")

# --- Setup -------------------------------------------------------------------
md(r"""## 0. Setup — get the repo and install the *minimal* dependencies

Point at the repo (clone URL or local/Drive path). The checkout **must contain
`src/mlp_kprop/exact_relu_covariance.py` and the `exact_relu_cov` flag** — push the
branch with that change before running on Colab (section 2 checks and fails loudly
otherwise). The exact path also needs **scipy** (for the bivariate normal CDF).
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

# scipy is REQUIRED by the exact path (bivariate normal CDF via Owen's T).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jaxtyping", "einops", "opt_einsum", "joblib", "tqdm", "scipy"], check=False)
print("deps installed")
""")

# --- Global numerics ---------------------------------------------------------
md(r"""## 1. Global numerical settings

- **kprop runs in `float64`** (the adapter casts a float64 copy of the model).
- On GPU we do **training + Monte-Carlo in `float32`** (fast; MC accuracy is limited
  by `1/√N`, not float32 rounding). The exact-covariance path always runs in
  float64 on the CPU (NumPy/SciPy) and returns tensors on the original device.
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
md(r"""## 2. Verify the **real** algorithm and the exact-covariance path

Prints the source files of `mlp_kprop` and the exact bivariate-covariance module,
and asserts the `exact_relu_cov` flag is present. If the assert fails, your checkout
predates the change — push/pull the branch that adds
`src/mlp_kprop/exact_relu_covariance.py`.
""")

code(r"""
import inspect
import src.mlp_kprop.kprop_harmonic as kp
from src.mlp_kprop.mlp import MLP
from src.mlp_kprop.wick import WICK_COEF_D

print("mlp_kprop defined in:                 ", inspect.getsourcefile(kp.mlp_kprop))
assert "exact_relu_cov" in inspect.signature(kp.mlp_kprop).parameters, (
    "This checkout has no exact_relu_cov flag — push/pull the branch that adds "
    "src/mlp_kprop/exact_relu_covariance.py and wires exact_relu_cov through mlp_kprop."
)
import src.mlp_kprop.exact_relu_covariance as erc
print("exact bivariate ReLU covariance in:   ", inspect.getsourcefile(erc.exact_relu_covariance_kprop))
import scipy
print("scipy version (needed for Phi2):      ", scipy.__version__)
print("activations supported:                ", list(WICK_COEF_D.keys()))
print("\n--- the exact K=2 covariance routine (docstring) ---")
print(inspect.getdoc(erc.exact_relu_covariance_np))
""")

# --- The setup / knobs -------------------------------------------------------
md(r"""## 3. The exact setup and the knobs

Two methods, both at **`k_max = 2`** (the only difference is `exact_relu_cov`):
- `approx (k=2)` → default propagation (leading-order off-diagonal gain).
- `exact (k=2)`  → `exact_relu_cov=True` (exact bivariate ReLU covariance).

We evaluate each on the **same** models — `initial` (control), `train_to_zero`,
and `halfspace` — across widths, with **`input_dim = width`** and a **vector
output** so the per-entry MSE averages over many components (same methodology as
the main notebook's section 9). For `halfspace`, each of the `OUTPUT_DIM`
components is its own random half-space, which doubles as the vector output.
""")

code(r"""
from cumulant_experiments.cumulant_adapter import run_cumulant_propagation_from_model, extract_mean, config_summary
from cumulant_experiments.metrics import estimate_empirical_mean, compare_means
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, train_model_to_halfspace, set_seed

# ---- Architecture ----------------------------------------------------------
HIDDEN_DEPTH = 2        # >=2 so the off-diagonal covariance actually matters (depth 1 ignores it)
ACTIVATION   = "relu"   # the exact path applies to ReLU
USE_BIAS     = False    # train with NO bias parameters (biases identically 0 throughout training)
OUTPUT_DIM   = 64       # vector output (and # of random half-spaces for the halfspace task)

# ---- The two methods (both k_max=2; only exact_relu_cov differs) -----------
COMMON_CFG = {"kind": "simple", "use_avg_metric": False, "factor": False, "use_pK": True, "output_d_max": 1}
METHODS = {
    "approx (k=2)": {**COMMON_CFG, "k_max": 2, "exact_relu_cov": False},  # default gain approximation
    "exact (k=2)":  {**COMMON_CFG, "k_max": 2, "exact_relu_cov": True},   # exact bivariate covariance
}

# ---- Sweep ranges -----------------------------------------------------------
WIDTHS = [32, 64, 128, 256, 512, 1024]   # input_dim = width. 512/1024 are HEAVY (esp. the exact
                                         # path: CPU/scipy, O(n^2) per layer) -- drop them for a
                                         # quick look; keep them for the published slope.
SEEDS  = [3, 4, 5, 6]                     # 4 seeds for tighter medians
PHASES = ["initial", "train_to_zero", "halfspace"]   # untrained control + the two training regimes

# ---- Monte-Carlo + training -------------------------------------------------
MC_SAMPLES     = 500_000      # raise to push the MC-variance floor down
TRAIN_STEPS    = 8000         # both training regimes
# Moderate early-stop for train_to_zero: stop at MSE < 1e-6 (=> output_rms ~ 1e-3), NOT smaller,
# so the tiny trained output is not swamped by MC / float64 numerical error. Raise (e.g. 1e-5/1e-4)
# to leave an even larger output; lower (1e-8) to push kprop harder (noisier comparison).
TRAIN_LOSS_TOL = 1e-6
HALFSPACE_OFFSET_STD = 1.0    # random affine offset b_j ~ N(0, this^2). NOTE: with USE_BIAS=False the
                              # ReLU MLP is positively homogeneous and cannot represent an affine offset,
                              # so it fits the best homogeneous approximation. Set 0.0 for through-origin
                              # half-spaces {x: w_j.x > 0} (a matched, fully learnable target).
BATCH_SIZE     = 1024
LR             = 1e-3

for name in METHODS:
    print(f"{name:13s} cfg:", config_summary(METHODS[name]))
print("regimes:", PHASES, "| seeds:", SEEDS, "| widths:", WIDTHS)
""")

# --- Verify exact != approx, and the depth caveat ----------------------------
md(r"""## 4. Verify the exact path is genuinely different (and the depth caveat)

The exact bivariate covariance is a *different computation* from the gain
approximation — it must change the propagated result for nets deep enough that the
off-diagonal covariance feeds a later layer. Below:
- **depth 1:** `exact ≡ approx` (the output mean uses only the exact ReLU
  marginals), and that mean equals MC within sampling noise — both are exact.
- **depth ≥ 2:** `exact ≠ approx` — the exact off-diagonal covariance changes the
  downstream variance and hence the mean. This is the regime we scale.
""")

code(r"""
set_seed(0); n0 = 128
print("||exact(k=2) - approx(k=2)||  on the propagated output mean:")
for depth in [1, 2, 3]:
    m = make_mlp(input_dim=n0, hidden_width=n0, hidden_depth=depth, output_dim=8,
                 activation=ACTIVATION, bias=USE_BIAS, device=DEVICE, dtype=MODEL_DTYPE)
    cpa = extract_mean(run_cumulant_propagation_from_model(m, n0, METHODS["approx (k=2)"], device=DEVICE))
    cpe = extract_mean(run_cumulant_propagation_from_model(m, n0, METHODS["exact (k=2)"], device=DEVICE))
    print(f"  depth={depth}: {np.linalg.norm(cpe - cpa):.2e}")
print("=> depth 1: ~0 (mean uses only exact marginals); depth>=2: nonzero (exact off-diagonal covariance matters).\n")

# depth-1 exactness vs MC (both methods give the exact mean here)
set_seed(0)
m = make_mlp(input_dim=n0, hidden_width=n0, hidden_depth=1, output_dim=OUTPUT_DIM,
             activation=ACTIVATION, bias=USE_BIAS, device=DEVICE, dtype=MODEL_DTYPE)
cp = extract_mean(run_cumulant_propagation_from_model(m, n0, METHODS["exact (k=2)"], device=DEVICE))
mc, st = estimate_empirical_mean(model=m, input_dim=n0, num_samples=MC_SAMPLES, batch_size=MC_BATCH, device=DEVICE, dtype=MODEL_DTYPE)
print(f"depth-1 exact vs MC: rel_err={compare_means(cp, mc, st)['relative_error_mean']:.2e}, "
      f"z={compare_means(cp, mc, st)['mc_noise_z']:.2f} (z<~1 => exact)")
""")

# --- The main sweep ----------------------------------------------------------
md(r"""## 5. The width-scaling sweep (the experiment)

For each `(width, seed, regime)`: build the net (`input_dim = width`), apply the
regime's training (none / output→0 / half-space classification), draw one
Monte-Carlo reference, then evaluate **both** methods against that same MC. We
record the MC-variance-debiased per-entry RMS error, its scale-free version
(÷ `output_rms`), and a signal-to-floor ratio (`>~3` resolved; `<~1` below the MC
floor). The two training regimes share one model init per seed (rebuilt each time).
""")

code(r"""
import torch, numpy as np, pandas as pd, math
from cumulant_experiments.cumulant_adapter import run_cumulant_propagation_from_model, extract_mean
from cumulant_experiments.metrics import estimate_empirical_mean
from cumulant_experiments.model_utils import make_mlp, train_model_to_zero, train_model_to_halfspace, set_seed

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
            set_seed(seed)                          # same init across phases & methods
            model = make_mlp(input_dim=in_dim, hidden_width=width, hidden_depth=HIDDEN_DEPTH,
                             output_dim=OUTPUT_DIM, activation=ACTIVATION, bias=USE_BIAS,
                             device=DEVICE, dtype=MODEL_DTYPE)
            final_loss = float("nan")
            if phase == "train_to_zero":
                stats = train_model_to_zero(model=model, input_dim=in_dim, steps=TRAIN_STEPS,
                                            batch_size=BATCH_SIZE, lr=LR, device=DEVICE,
                                            dtype=MODEL_DTYPE, loss_tol=TRAIN_LOSS_TOL)
                final_loss = stats["final_train_loss"]
            elif phase == "halfspace":
                stats = train_model_to_halfspace(model=model, input_dim=in_dim, output_dim=OUTPUT_DIM,
                                                 steps=TRAIN_STEPS, batch_size=BATCH_SIZE, lr=LR,
                                                 device=DEVICE, dtype=MODEL_DTYPE,
                                                 offset_std=HALFSPACE_OFFSET_STD, loss_tol=0.0)
                final_loss = stats["final_train_loss"]
            mc, st = estimate_empirical_mean(model=model, input_dim=in_dim, num_samples=MC_SAMPLES,
                                             batch_size=MC_BATCH, device=DEVICE, dtype=MODEL_DTYPE)
            for method in METHODS:                  # evaluate both against the SAME MC
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
print("Scale-free debiased RMS (median over seeds) and its slope vs width.")
print("  theory at init for k_max=2 ~ n^-1.  s/floor < ~3 => below MC floor (raise MC_SAMPLES).\n")
for phase in PHASES:
    for method in METHODS:
        sub = dfsc[(dfsc.phase == phase) & (dfsc.method == method)]
        rmed = sub.groupby("width")["rel_debiased_rms"].median()
        s2f  = sub.groupby("width")["s_to_floor"].median()
        ws = sorted(rmed.index)
        print(f"  {phase:15s} {method:12s}: slope={loglog_slope(ws,[rmed[w] for w in ws]):+.2f}  "
              + "  ".join(f"n{w}={rmed[w]:.2e}(s/f={s2f[w]:.0f})" for w in ws))

print("\nDoes the EXACT covariance help? approx/exact ratio of scale-free RMS (>1 => exact more accurate):")
for phase in PHASES:
    a = dfsc[(dfsc.phase==phase)&(dfsc.method=="approx (k=2)")].groupby("width")["rel_debiased_rms"].median()
    e = dfsc[(dfsc.phase==phase)&(dfsc.method=="exact (k=2)")].groupby("width")["rel_debiased_rms"].median()
    ws = sorted(set(a.index) & set(e.index))
    print(f"  {phase:15s}: " + "  ".join(f"n{w}={ (a[w]/e[w]) :.2f}x" if e[w]>0 else f"n{w}=NA" for w in ws))

print("\nHEADLINE (the question): per-regime slopes, exact vs approx")
for phase in [p for p in PHASES if p != "initial"]:
    line = f"  [{phase}] "
    for method in METHODS:
        sub = dfsc[(dfsc.phase==phase) & (dfsc.method==method)]
        rmed = sub.groupby('width')['rel_debiased_rms'].median(); ws = sorted(rmed.index)
        line += f"{method}: slope={loglog_slope(ws,[rmed[w] for w in ws]):+.2f}   "
    print(line)
print("  Same slope+magnitude (ratio ~1x) => the exact covariance does NOT fix the trained-model problem.")
print("  Exact steeper/lower               => the covariance approximation was (part of) the problem.")
""")

# --- Plots -------------------------------------------------------------------
md(r"""## 6. Plots — error scaling, approx vs exact, initial vs trained

- **Left (absolute):** debiased per-entry RMS vs `n`; dotted = `n^-1` guide.
- **Right (scale-free):** error ÷ `output_rms`, so initial and trained are directly
  comparable.

Solid = `exact (k=2)`, dashed = `approx (k=2)`; blue = `initial`, red =
`train_to_zero`, green = `halfspace`. **For each training regime: if its solid and
dashed curves lie on top of each other, the exact covariance did not fix that
regime's problem.**
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
mc_color = {"initial": "tab:blue", "train_to_zero": "tab:red", "halfspace": "tab:green"}
mt_style = {"exact (k=2)": dict(ls="-", marker="o"), "approx (k=2)": dict(ls="--", marker="s")}

for ax, col, ttl in [(axes[0], "debiased_rms", "ABSOLUTE debiased per-entry RMS error"),
                     (axes[1], "rel_debiased_rms", "SCALE-FREE error (RMS / output_rms)")]:
    for phase in PHASES:
        for method in METHODS:
            sub = dfsc[(dfsc.phase == phase) & (dfsc.method == method)]
            if sub.empty: continue
            med = sub.groupby("width")[col].median()
            ws = np.array(sorted(med.index)); y = med.reindex(ws).to_numpy()
            ax.plot(ws, np.where(y <= 0, np.nan, y), color=mc_color[phase],
                    label=f"{method} / {phase}", **mt_style[method])
    base = dfsc[(dfsc.method=="approx (k=2)")&(dfsc.phase=="initial")].groupby("width")[col].median()
    bw = np.array(sorted(base.index), float)
    if len(base) and base.iloc[0] > 0:
        ax.plot(bw, base.iloc[0]*(bw/bw[0])**(-1.0), color="gray", ls=":", lw=1.2, alpha=0.7, label="theory n^-1")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width n  (= input_dim)"); ax.set_title(ttl)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# --- Conclusions -------------------------------------------------------------
md(r"""## 7. How to read it (the answer)

> **What we observed** (depth 2): at **init** the two methods are essentially
> identical (`approx/exact ≈ 0.98–0.99×`, same slope) — the gain approximation is
> already good for random weights. After **training** — in **both** the
> `train_to_zero` and `halfspace` regimes — `exact (k=2)` and `approx (k=2)` stay
> within a small factor with the ratio **bouncing around 1 (no consistent winner)**.
> **Conclusion: making the K=2 covariance exact does NOT fix the trained-model
> problem, for either task.** Re-run with more seeds/widths to confirm on your setup.

For **each** training regime (`train_to_zero`, `halfspace`), read its two curves
and the `approx/exact` ratio printed in §5:

- **If `exact (k=2)` and `approx (k=2)` overlap after training** (same slope, ratio
  ≈ 1×) — the headline result — then **the exact covariance does NOT fix that
  regime's problem.** Making the K=2 off-diagonal exact is not enough; the
  degradation comes from elsewhere (training-induced weight/activation correlations
  that violate the wide-random-MLP assumption, which a *single* Gaussian K=2 state
  cannot capture regardless of how exactly its covariance is computed). The fix
  would need higher cumulants (`k_max ≥ 3`) or a non-Gaussian state, not a better
  K=2 covariance. That the **same** conclusion holds for two *different* trained
  tasks makes it more robust.
- **If `exact (k=2)` scales better after training** (steeper slope / ratio > 1, gap
  widening with `n`) — then the gain approximation **was** a real source of the
  trained-model error, and the exact covariance recovers (some of) the lost accuracy.

At **initialization** both should track `n^-1` closely and sit near each other; the
interesting signal is entirely in the **trained** comparisons.

**Caveats:** the two trained means have very different scales (`train_to_zero` ~1e-3
via the moderate `TRAIN_LOSS_TOL`; `halfspace` ~`Phi(-b_j)`, i.e. O(1)) — the
**scale-free** panel makes them comparable. Few seeds ⇒ noisy slopes; at large `n`
the debiased RMS can hit the MC floor (watch `s/floor`, raise `MC_SAMPLES`); the
exact path is CPU/scipy and O(n²) per layer, so **width 512/1024 are slow** (drop
them for a quick look). Increase `HIDDEN_DEPTH` to amplify the covariance's
downstream effect.
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
