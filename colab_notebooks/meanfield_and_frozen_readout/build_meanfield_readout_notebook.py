"""Generates kprop_meanfield_and_frozen_readout_colab.ipynb (valid nbformat-4 JSON).

Two hypotheses about WHY cumulant propagation ("kprop") works (or fails) on MLPs
trained to output 0, both on the unified codebase (build a `model.MLP`, train it with a
plain-SGD loop matching `training.TrainConfig`, predict its mean with
`Mecha_preds.cumulants.run_cumulants` as a black box).

H1 - mean-field / large batch. SGD's gradient is the true (population) gradient plus a
   zero-mean noise term whose size ~ 1/sqrt(batch). Take the batch HUGE (scaled with the
   width/input dim) and that noise shrinks: the optimizer follows the population gradient,
   the readout descends quickly, and the inner weights stay close to their random init
   (behave like random matrices) -- the regime kprop is built for. Prediction: kprop error
   should DROP as batch grows.

H2 - frozen readout. The readout collapses to tiny norm, gradients stop flowing back, and
   the inner weights freeze after a point. Test: REMOVE the trained last layer -- freeze
   the readout to the identity and never train it (output = last hidden activations) -- and
   train the inner layers to send those to 0. Contrast with a normal trainable readout.
   Does kprop track the frozen-readout net better, and do the inner weights still freeze?

Setup (matches the project baseline): trained-to-zero, NO bias, classical SGD lr=1e-3,
12000 steps, depths 3 and 4, widths 32..512, input_dim == width, everything float64.

Run:  python colab_notebooks/meanfield_and_frozen_readout/build_meanfield_readout_notebook.py
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
md(r"""# Does kprop work on trained-to-0 MLPs in the **mean-field** limit, and with the **readout removed**?

Cumulant propagation ("kprop") predicts the output mean of a wide *random* MLP well, but
its accuracy **degrades after training**. This notebook tests two stories for what training
does, by **manipulating the training itself** and re-checking kprop (called as a black box).

### H1 - mean-field / huge batch
SGD's update uses a minibatch-estimated gradient `g_batch = g_pop + noise`, with
`noise` zero-mean and of size `~ 1/sqrt(batch)`. Push the **batch size up** (scaled with the
input dim `n = width`, to stay consistent with theory) and that Gaussian noise term shrinks
toward 0 -- the optimizer follows the *population* gradient `g_pop = ∇ E‖f(x)‖²`.

> **Claim:** in that limit the readout descends quickly while the **inner weights barely move
> from their random init** (they stay ≈ random matrices), which is exactly the regime kprop is
> built for. **Prediction: kprop's relative error should DROP as batch → ∞.**

We measure both the kprop error *and* the actual gradient-noise scale and inner-weight drift,
so we can see the noise shrink and check whether the inner layers really stay frozen.

### H2 - frozen readout (remove the last layer)
The other story: the **readout norm collapses**, so gradients stop propagating back and the
inner weights **freeze after a point**. To probe it we **get rid of the weights past the final
ReLU** -- set the readout to the **identity and freeze it** (it never trains), so the output
*is* the last hidden activations -- and train only the inner layers to send those to 0.

> **Claim:** with a non-collapsing (identity) readout, gradients keep flowing; comparing this
> against a normal *trainable* readout (which is allowed to collapse) isolates whether the
> readout collapse is what freezes the inner weights and what kprop struggles with.

### Fixed setup (matches the project baseline)
Trained-to-0, **no bias**, **classical SGD, lr = 1e-3** (vanilla SGD at this lr moves slowly
on these nets *by design* -- the slow/frozen dynamics are the object of study), **12000
steps**, **depths 3 and 4**, **widths 32..512** with `input_dim == width`, all **float64**.
Nothing here reimplements kprop -- we call `run_cumulants` and only change the *training*.
""")

# --- §0 Setup ----------------------------------------------------------------
md(r"""## 0. Setup — locate the repo and install the minimal deps

Point at the repo root (clone URL or local/Drive path). The checkout must contain
`Mecha_preds/cumulants/kprop/`.
""")

code(r"""
# ----------------------------- EDIT THIS -----------------------------------
REPO_URL       = ""   # e.g. "https://github.com/<you>/one-trained-case.git" (leave "" if not cloning)
LOCAL_REPO_DIR = ""   # e.g. "/content/drive/MyDrive/One trained case" or a local path
# ---------------------------------------------------------------------------

import os, sys, subprocess
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False

_MARKER = os.path.join("Mecha_preds", "cumulants", "kprop")

def _find_repo():
    if LOCAL_REPO_DIR and os.path.isdir(os.path.join(LOCAL_REPO_DIR, _MARKER)):
        return LOCAL_REPO_DIR
    if REPO_URL:
        dest = "/content/one_trained_case" if IN_COLAB else os.path.abspath("./_repo_clone")
        if not os.path.isdir(os.path.join(dest, _MARKER)):
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest], check=True)
        return dest
    here = os.path.abspath(".")
    for _ in range(6):
        if os.path.isdir(os.path.join(here, _MARKER)):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate the repo. Set REPO_URL or LOCAL_REPO_DIR above.")

REPO_DIR = _find_repo()
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
print("IN_COLAB:", IN_COLAB, "| REPO_DIR:", REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jaxtyping", "einops", "opt_einsum", "joblib", "tqdm", "scipy"], check=False)
print("deps installed")
""")

# --- §1 Numerics -------------------------------------------------------------
md(r"""## 1. Global numerical settings

Everything runs in **`float64`** (kprop and its Monte-Carlo reference run in double
precision). `QUICK = True` shrinks every range so you can run the whole notebook end-to-end
in a couple of minutes to confirm it works, then flip it to `False` for the real sweep.
""")

code(r"""
import torch, numpy as np, pandas as pd, math, time
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float64
QUICK       = True     # <-- True: tiny smoke run. False: the full sweep described in §3.
print("device:", DEVICE, "| dtype:", MODEL_DTYPE, "| QUICK:", QUICK)
""")

# --- §2 Verify the real algorithm -------------------------------------------
md(r"""## 2. Verify we are calling the **real** kprop

Print the source file of the propagation entry point and confirm `run_cumulants` returns a
degree-1 (mean) cumulant on a tiny net.
""")

code(r"""
import inspect
import Mecha_preds.cumulants.kprop.kprop_harmonic as kp
from model import ModelConfig
from Mecha_preds.cumulants import run_cumulants

print("mlp_kprop defined in:", inspect.getsourcefile(kp.mlp_kprop))
_m = ModelConfig(input_dim=8, hidden_dim=8, depth=2, output_dim=1, bias=False,
                 final_bias=False, activation="relu", seed=0).build().to(DEVICE)
_out = run_cumulants(_m, 8, {"k_max": 3, "kind": "simple", "factor": True,
                             "use_pK": True, "output_d_max": 1}, device=DEVICE)
print("run_cumulants OK -> mean shape:", _out["mean"].shape, "| config:", _out["metadata"]["config"])
""")

# --- §3 Config + helpers -----------------------------------------------------
md(r"""## 3. Architecture, sweep ranges, and shared helpers

**`input_dim == width`** everywhere (square first map, consistent with the theory). The kprop
predictor is the standard `k_max = 3` harmonic propagation (`factor=True`). Training is a
plain-SGD loop that mirrors `training.TrainConfig(optimizer="sgd", weight_decay=0.0,
lr=1e-3)` but is instrumented so we can snapshot the loss, the gradient-noise scale, the
inner-weight drift from init, and the readout norm along the way.

**H1 batch sizes scale with width:** `batch = min(ratio · width, MAX_TRAIN_BATCH)`. Raise
`MAX_TRAIN_BATCH` (and add larger ratios) to push further into the mean-field limit — it is the
main memory/compute knob.

Every trained net is written to `CHECKPOINT_DIR` as a self-describing `.pt`
(`SAVE_CHECKPOINTS=True`), reloadable with `MLP.load(path) -> (model, payload)`; the payload
also carries the run's `final_loss`, `grad_noise`/`inner_drift`, the kprop error dict, and (for
Experiment B) the snapshot `history` — so any run can be re-analyzed later without retraining.
""")

code(r"""
from tasks import ZeroTask
from Mecha_preds.cumulants import estimate_empirical_mean, config_summary
from utils import set_seed

# ---- Architecture (fixed) ---------------------------------------------------
ACTIVATION = "relu"
USE_BIAS   = False          # no bias: ReLU half-spaces through the origin
LR         = 1e-3           # classical SGD, lr 1e-3 (kept 'same as before')
STEPS      = 12000

# ---- The kprop predictor (black box; standard k_max=3) ----------------------
CP_CFG = {"k_max": 3, "kind": "simple", "use_avg_metric": False, "factor": True,
          "use_pK": True, "output_d_max": 1, "exact_relu_cov": False}

# ---- Sweep ranges -----------------------------------------------------------
if QUICK:
    WIDTHS, DEPTHS, SEEDS = [32, 64], [3], [0]
    BATCH_RATIOS   = [32, 256]          # batch = ratio * width
    MAX_TRAIN_BATCH = 16_384
    STEPS_RUN      = 800
    MC_SAMPLES     = 100_000
else:
    WIDTHS, DEPTHS, SEEDS = [32, 64, 128, 256, 512], [3, 4], [0]   # 1 seed (add more for error bars)
    BATCH_RATIOS   = [32, 128, 512, 2048]   # batch = ratio * width  (1024 .. ~1.0e6, capped)
    MAX_TRAIN_BATCH = 262_144               # raise to go further into the mean-field limit
    STEPS_RUN      = STEPS                  # 12000
    MC_SAMPLES     = 1_000_000

def width_batch(width, ratio):
    return int(min(ratio * width, MAX_TRAIN_BATCH))

# ---- Checkpoints (self-describing .pt, reloadable with MLP.load) ------------
SAVE_CHECKPOINTS = True
CHECKPOINT_DIR   = "checkpoints/meanfield_and_frozen_readout"

print("widths:", WIDTHS, "| depths:", DEPTHS, "| seeds:", SEEDS)
print("batch ratios:", BATCH_RATIOS, "| MAX_TRAIN_BATCH:", MAX_TRAIN_BATCH, "| steps:", STEPS_RUN)
print("kprop:", config_summary(CP_CFG))
print("checkpoints:", CHECKPOINT_DIR if SAVE_CHECKPOINTS else "(off)")
""")

code(r"""
# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_trainable(in_dim, width, depth, out_dim, seed):
    '''Standard study MLP (readout trainable). input_dim == width.'''
    cfg = ModelConfig(input_dim=in_dim, hidden_dim=width, depth=depth, output_dim=out_dim,
                      bias=USE_BIAS, final_bias=USE_BIAS, activation=ACTIVATION, seed=seed)
    return cfg.build().to(device=DEVICE, dtype=MODEL_DTYPE)

def build_frozen_identity(in_dim, width, depth, seed):
    '''H2: 'get rid of the weights past the final ReLU' -- output_dim == width, readout set
    to the IDENTITY and frozen (requires_grad=False), so the output IS the last hidden
    (post-ReLU) activations and only the inner layers train.'''
    m = build_trainable(in_dim, width, depth, out_dim=width, seed=seed)
    with torch.no_grad():
        m.readout.weight.copy_(torch.eye(width, device=DEVICE, dtype=MODEL_DTYPE))
    m.readout.weight.requires_grad_(False)        # frozen -> SGD never updates it (grad stays None)
    return m

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def inner_weight_snapshot(model):
    '''Clone of every HIDDEN (inner) weight, in forward order, for drift measurement.'''
    return [layer.weight.detach().clone() for layer in model.hidden_layers]

def inner_drift(model, ref):
    '''Relative Frobenius drift ||W - W_init|| / ||W_init|| per inner layer (mean + list).'''
    per = []
    for layer, w0 in zip(model.hidden_layers, ref):
        per.append(float((layer.weight.detach() - w0).norm() / (w0.norm() + 1e-30)))
    return float(np.mean(per)), per

def readout_fro(model):
    return float(model.readout.weight.detach().norm())

def grad_noise_scale(model, task, batch_size, n_batches=8):
    '''Relative size of SGD's gradient noise term: sqrt(E||g_i - mean_g||^2) / ||mean_g||,
    over `n_batches` independent minibatches. Scales ~ 1/sqrt(batch); -> 0 is the mean-field
    (population-gradient) limit. Computed on the trainable params only.'''
    ps = [p for p in model.parameters() if p.requires_grad]
    gs = []
    for _ in range(n_batches):
        x, y = task.sample_batch(batch_size, DEVICE)
        loss = task.loss(model(x), y)
        g = torch.autograd.grad(loss, ps)
        gs.append(torch.cat([gi.reshape(-1) for gi in g]))
    model.zero_grad(set_to_none=True)
    G = torch.stack(gs); mean_g = G.mean(0)
    noise = (G - mean_g).pow(2).sum(1).mean().sqrt()
    signal = mean_g.pow(2).sum().sqrt()
    return float(noise / (signal + 1e-30))

# ---------------------------------------------------------------------------
# Training (mirrors training.TrainConfig: SGD, weight_decay=0, fresh Gaussian batch/step)
# ---------------------------------------------------------------------------
def train_zero(model, task, steps, batch_size, lr=LR, snap_at=None, ref_inner=None):
    '''Plain SGD train-to-0. Optionally snapshot (step, loss, inner_drift, readout_fro) at
    the iterations in `snap_at`. Returns (final_loss, history_list).'''
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    snap_at = set(snap_at or [])
    hist, last = [], float("nan")
    if ref_inner is not None and 0 in snap_at:
        d, _ = inner_drift(model, ref_inner)
        hist.append(dict(step=0, loss=float("nan"), inner_drift=d, readout_fro=readout_fro(model)))
    model.train()
    for step in range(1, steps + 1):
        x, y = task.sample_batch(batch_size, DEVICE)
        loss = task.loss(model(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        last = float(loss.detach())
        if step in snap_at:
            d = inner_drift(model, ref_inner)[0] if ref_inner is not None else float("nan")
            hist.append(dict(step=step, loss=last, inner_drift=d, readout_fro=readout_fro(model)))
    return last, hist

# ---------------------------------------------------------------------------
# kprop-vs-MC error (MC-variance-debiased, scale-free)
# ---------------------------------------------------------------------------
def kprop_error(model, in_dim, mc_samples):
    cp = run_cumulants(model, in_dim, CP_CFG, device=DEVICE)["mean"]
    mc_batch = min(262_144 if DEVICE == "cuda" else 8_192, max(4096, (1 << 26) // in_dim))
    mc, st = estimate_empirical_mean(model=model, input_dim=in_dim, num_samples=mc_samples,
                                     batch_size=mc_batch, device=DEVICE, dtype=MODEL_DTYPE)
    cp = np.asarray(cp).reshape(-1); mc = np.asarray(mc).reshape(-1)
    se = np.asarray(st["mc_stderr"]).reshape(-1)
    measured = float(np.mean((cp - mc) ** 2)); floor = float(np.mean(se ** 2))
    rms = math.sqrt(max(measured - floor, 0.0))                 # debiased per-entry RMS error
    out_rms = st["empirical_output_rms"] + 1e-30
    return dict(rel_err=rms / out_rms, abs_rms=rms, out_rms=st["empirical_output_rms"],
                s_to_floor=(measured - floor) / (floor + 1e-300))

# ---------------------------------------------------------------------------
# Checkpoints -- write the repo's self-describing .pt (reload with MLP.load).
# model.save() stores {model_config, state_dict}; `extra` adds step/history/etc.
# ---------------------------------------------------------------------------
def save_ckpt(model, name, info):
    if not SAVE_CHECKPOINTS:
        return None
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, name + ".pt")
    model.save(path, extra=info)
    return path

print("helpers ready")
""")

# --- §4 Sanity check ---------------------------------------------------------
md(r"""## 4. Sanity check (tiny width)

Confirms the moving parts before the heavy sweeps: (a) gradient noise really shrinks with
batch size, (b) the frozen-identity readout stays the identity through training while the
inner weights move, and (c) `run_cumulants` returns the right-shaped mean on the
frozen-identity net.
""")

code(r"""
set_seed(0)
w = 16
# (a) gradient noise vs batch (expect ~ /2 each time batch x4, i.e. ~1/sqrt(batch))
mt = build_trainable(w, w, depth=3, out_dim=w, seed=0)
tk = ZeroTask(input_dim=w, output_dim=w)
print("grad-noise  bs=64 :", round(grad_noise_scale(mt, tk, 64), 3),
      "| bs=1024:", round(grad_noise_scale(mt, tk, 1024), 3),
      "| bs=16384:", round(grad_noise_scale(mt, tk, 16384), 3))

# (b) frozen-identity readout: stays I, inner moves
mf = build_frozen_identity(w, w, depth=3, seed=0)
ref = inner_weight_snapshot(mf)
fl, _ = train_zero(mf, ZeroTask(input_dim=w, output_dim=w), steps=300, batch_size=512)
print("frozen readout still identity:", torch.allclose(mf.readout.weight, torch.eye(w, dtype=MODEL_DTYPE, device=DEVICE)),
      "| readout requires_grad:", mf.readout.weight.requires_grad,
      "| inner drift after 300 steps:", round(inner_drift(mf, ref)[0], 4), "| loss:", f"{fl:.2e}")

# (c) kprop on the frozen-identity net
e = kprop_error(mf, w, mc_samples=50_000)
print("kprop on frozen-identity net -> rel_err:", round(e["rel_err"], 4),
      "| out_rms:", round(e["out_rms"], 5), "| s/floor:", round(e["s_to_floor"], 1))
""")

# --- §5 Experiment A ---------------------------------------------------------
md(r"""## 5. Experiment A — H1: kprop error vs batch size (the mean-field limit)

For each `(depth, width, ratio, seed)`: build a **trainable** net (`input_dim = width`,
vector output `output_dim = width` so the per-entry error averages over many components),
record the **gradient-noise scale** at the chosen batch, train to 0 with **plain SGD** for
`STEPS_RUN` steps at `batch = min(ratio·width, MAX_TRAIN_BATCH)`, then measure the
**inner-weight drift** from init and the **kprop relative error**.

> **H1 reads off two curves:** as batch ↑ (noise ↓), does `rel_err` ↓ and does `inner_drift`
> stay small? Both ⇒ the mean-field limit makes the trained net kprop-friendly.
""")

code(r"""
rowsA, t0 = [], time.time()
for depth in DEPTHS:
    for width in WIDTHS:
        in_dim = width
        for ratio in BATCH_RATIOS:
            bs = width_batch(width, ratio)
            for seed in SEEDS:
                set_seed(seed)
                m = build_trainable(in_dim, width, depth, out_dim=width, seed=seed)
                task = ZeroTask(input_dim=in_dim, output_dim=width)
                gns = grad_noise_scale(m, task, bs)
                ref = inner_weight_snapshot(m)
                floss, _ = train_zero(m, task, steps=STEPS_RUN, batch_size=bs)
                drift, _ = inner_drift(m, ref)
                err = kprop_error(m, in_dim, MC_SAMPLES)
                save_ckpt(m, f"meanfield_d{depth}_w{width}_r{ratio}_bs{bs}_seed{seed}_final",
                          dict(step=STEPS_RUN, run_name="meanfield",
                               train_config=dict(optimizer="sgd", lr=LR, steps=STEPS_RUN,
                                                 batch_size=bs, weight_decay=0.0),
                               experiment="A_meanfield", ratio=ratio, batch=bs,
                               final_loss=floss, grad_noise=gns, inner_drift=drift, kprop=err))
                rowsA.append(dict(depth=depth, width=width, ratio=ratio, batch=bs, seed=seed,
                                  grad_noise=gns, inner_drift=drift, final_loss=floss, **err))
                print(f"  d{depth} n{width:4d} ratio{ratio:5d} bs{bs:7d} s{seed}: "
                      f"gnoise={gns:.2f} drift={drift:.3f} loss={floss:.1e} "
                      f"rel_err={err['rel_err']:.2e} (s/f={err['s_to_floor']:.0f})", flush=True)
dfA = pd.DataFrame(rowsA)
print(f"\nExperiment A done in {time.time()-t0:.0f}s, {len(dfA)} rows")
dfA.head(10)
""")

# --- §6 Experiment A results -------------------------------------------------
md(r"""## 6. Experiment A — summary and plots

Left: kprop **relative error vs batch size** (one line per width). Middle: **gradient-noise
scale vs batch** (should fall ~`1/sqrt(batch)`). Right: **inner-weight drift vs batch**
(H1 predicts it stays small / falls as batch grows). Medians over seeds.
""")

code(r"""
def med(df, by, col):
    g = df.groupby(by)[col].median()
    return g

print("H1 summary (median over seeds), per depth:")
for depth in sorted(dfA.depth.unique()):
    sub = dfA[dfA.depth == depth]
    print(f"\n depth={depth}")
    for width in sorted(sub.width.unique()):
        s = sub[sub.width == width].sort_values("batch")
        line = "  ".join(f"bs{int(r.batch)}:err={r.rel_err:.1e},gn={r.grad_noise:.2f},drift={r.inner_drift:.2f}"
                         for _, r in s.groupby("batch").median(numeric_only=True).reset_index().iterrows())
        print(f"   n={width:4d}: {line}")
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
depth_plot = sorted(dfA.depth.unique())[0]   # plot the first depth; rerun with others as needed
sub = dfA[dfA.depth == depth_plot]
widths = sorted(sub.width.unique())
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(widths)))

for ax, col, ttl, ylab in [
    (axes[0], "rel_err",     f"H1: kprop relative error vs batch (depth {depth_plot})", "rel error (debiased)"),
    (axes[1], "grad_noise",  "gradient-noise scale vs batch",                            "noise / signal"),
    (axes[2], "inner_drift", "inner-weight drift vs batch",                              "||ΔW||/||W_init||")]:
    for c, width in zip(cmap, widths):
        s = sub[sub.width == width].groupby("batch")[col].median()
        ax.plot(s.index, s.values, "o-", color=c, label=f"n={width}")
    ax.set_xscale("log", base=2)
    if col != "inner_drift":
        ax.set_yscale("log")
    ax.set_xlabel("batch size"); ax.set_ylabel(ylab); ax.set_title(ttl)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
# 1/sqrt(batch) reference on the noise panel
bb = np.array(sorted(sub.batch.unique()), float)
g0 = sub[sub.batch == bb[0]]["grad_noise"].median()
axes[1].plot(bb, g0 * np.sqrt(bb[0] / bb), "k:", lw=1, alpha=0.6, label="1/sqrt(batch)")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# --- §7 Experiment B ---------------------------------------------------------
md(r"""## 7. Experiment B — H2: frozen-identity readout vs trainable readout

For each `(depth, width, seed)` we train **two** nets to 0 at a fixed batch
(`ratio = BATCH_RATIOS[-1]`, the largest, to stay near the population gradient), capturing the
trajectory of **inner-weight drift** and **readout norm** at snapshots:

- **frozen-identity** — readout = `I`, frozen (the "remove the last layer" condition); only the
  inner layers learn. Readout norm is constant (`sqrt(width)`) by construction.
- **trainable** — a normal readout (`output_dim = width`), free to collapse.

Then kprop error on each. **H2 reads off:** does the trainable readout's norm collapse while
its inner weights freeze (drift plateaus)? Does the frozen-identity net keep its inner weights
moving and does kprop track it better/worse?
""")

code(r"""
SNAPS = sorted(set([0] + [int(STEPS_RUN * f) for f in (0.05, 0.1, 0.25, 0.5, 1.0)]))
ratioB = BATCH_RATIOS[-1]
rowsB, trajB, t0 = [], [], time.time()
for depth in DEPTHS:
    for width in WIDTHS:
        in_dim = width; bs = width_batch(width, ratioB)
        for seed in SEEDS:
            for cond, builder in [("frozen_identity", build_frozen_identity),
                                  ("trainable",       lambda i, w, d, s: build_trainable(i, w, d, w, s))]:
                set_seed(seed)
                m = builder(in_dim, width, depth, seed)
                task = ZeroTask(input_dim=in_dim, output_dim=width)
                ref = inner_weight_snapshot(m)
                floss, hist = train_zero(m, task, steps=STEPS_RUN, batch_size=bs,
                                         snap_at=SNAPS, ref_inner=ref)
                for h in hist:
                    trajB.append(dict(cond=cond, depth=depth, width=width, seed=seed, batch=bs, **h))
                err = kprop_error(m, in_dim, MC_SAMPLES)
                save_ckpt(m, f"readout-{cond}_d{depth}_w{width}_seed{seed}_final",
                          dict(step=STEPS_RUN, run_name=f"readout_{cond}",
                               train_config=dict(optimizer="sgd", lr=LR, steps=STEPS_RUN,
                                                 batch_size=bs, weight_decay=0.0),
                               experiment="B_readout", condition=cond, batch=bs,
                               final_loss=floss, final_inner_drift=inner_drift(m, ref)[0],
                               final_readout_fro=readout_fro(m), history=hist, kprop=err))
                rowsB.append(dict(cond=cond, depth=depth, width=width, seed=seed, batch=bs,
                                  final_loss=floss, final_inner_drift=inner_drift(m, ref)[0],
                                  final_readout_fro=readout_fro(m), **err))
                print(f"  d{depth} n{width:4d} s{seed} {cond:15s}: loss={floss:.1e} "
                      f"drift={rowsB[-1]['final_inner_drift']:.3f} ro_fro={rowsB[-1]['final_readout_fro']:.2e} "
                      f"rel_err={err['rel_err']:.2e}", flush=True)
dfB = pd.DataFrame(rowsB); dftrajB = pd.DataFrame(trajB)
print(f"\nExperiment B done in {time.time()-t0:.0f}s")
dfB.head(8)
""")

# --- §8 Experiment B results -------------------------------------------------
md(r"""## 8. Experiment B — summary and plots

Left: **kprop relative error**, frozen-identity vs trainable (per width). Middle:
**inner-weight drift trajectory** (does the trainable case plateau = freeze?). Right:
**readout-norm trajectory** for the trainable case (does it collapse?).
""")

code(r"""
print("H2 summary (median over seeds), per depth:")
for depth in sorted(dfB.depth.unique()):
    print(f"\n depth={depth}")
    for width in sorted(dfB[dfB.depth == depth].width.unique()):
        line = []
        for cond in ["frozen_identity", "trainable"]:
            s = dfB[(dfB.depth==depth)&(dfB.width==width)&(dfB.cond==cond)]
            if not s.empty:
                line.append(f"{cond}: err={s.rel_err.median():.1e}, drift={s.final_inner_drift.median():.2f}, "
                            f"ro_fro={s.final_readout_fro.median():.1e}, loss={s.final_loss.median():.0e}")
        print(f"   n={width:4d}: " + " | ".join(line))
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
depth_plot = sorted(dfB.depth.unique())[0]

# (left) kprop error: frozen vs trainable, vs width
for cond, mk in [("frozen_identity", "o-"), ("trainable", "s--")]:
    s = dfB[(dfB.depth==depth_plot)&(dfB.cond==cond)].groupby("width")["rel_err"].median()
    axes[0].plot(s.index, s.values, mk, label=cond)
axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
axes[0].set_xlabel("width n"); axes[0].set_ylabel("kprop rel error")
axes[0].set_title(f"H2: kprop error, frozen vs trainable (depth {depth_plot})")
axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(fontsize=9)

# (middle) inner-drift trajectory (one width)
w_traj = sorted(dftrajB.width.unique())[-1]
for cond, mk in [("frozen_identity", "o-"), ("trainable", "s--")]:
    s = dftrajB[(dftrajB.depth==depth_plot)&(dftrajB.width==w_traj)&(dftrajB.cond==cond)]
    s = s.groupby("step")["inner_drift"].median()
    axes[1].plot(s.index, s.values, mk, label=cond)
axes[1].set_xlabel("training step"); axes[1].set_ylabel("inner-weight drift")
axes[1].set_title(f"inner-weight drift trajectory (n={w_traj})")
axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

# (right) readout-norm trajectory (trainable; frozen is constant sqrt(n))
for width in sorted(dftrajB.width.unique()):
    s = dftrajB[(dftrajB.depth==depth_plot)&(dftrajB.width==width)&(dftrajB.cond=="trainable")]
    s = s.groupby("step")["readout_fro"].median()
    axes[2].plot(s.index, s.values, "o-", label=f"n={width}")
axes[2].set_xlabel("training step"); axes[2].set_ylabel("readout ||W||_F (trainable)")
axes[2].set_title("readout-norm trajectory — does it collapse?")
axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

# --- §9 How to read it -------------------------------------------------------
md(r"""## 9. How to read it (what each outcome would mean)

This is a **fresh** experiment — interpret the curves you actually get; nothing is pre-baked.

**H1 (mean-field / batch).** In §6:
- kprop `rel_err` **falls as batch grows** *and* `inner_drift` **stays small** → supports H1:
  in the population-gradient limit the inner weights stay ≈ random and kprop (built for random
  nets) works; SGD noise was what pushed the net into kprop's failure regime.
- kprop `rel_err` **flat across batch** while `grad_noise` clearly falls (check it tracks the
  `1/sqrt(batch)` reference) → H1 is *not* the story: the degradation survives in the
  population-gradient limit, so it is intrinsic to training the geometry, not an SGD-noise
  artifact. (`inner_drift` will tell you whether the inner layers moved regardless.)

**H2 (frozen readout).** In §8:
- trainable readout norm **collapses** and its inner-drift trajectory **plateaus** (freezes),
  while the **frozen-identity** net keeps inner weights moving → supports H2's mechanism
  (readout collapse stops gradient flow and freezes the inner layers).
- if kprop tracks the **frozen-identity** net much better than the trainable one → "getting rid
  of the weights past the final ReLU" removes whatever kprop struggles with (the collapsed
  readout / the correlations it induces).
- if both readout conditions freeze the inner weights similarly, or kprop is no better on the
  frozen net → readout collapse is **not** the driver.

**Cross-check.** H1 and H2 interact: a huge batch (small noise) may itself prevent the readout
collapse / inner freeze. Compare `inner_drift` and `final_readout_fro` between the two
experiments at matched width.

**Caveats.** Vanilla SGD at lr 1e-3 converges slowly — at small batch / large width the final
loss may still be well above 0 in `STEPS_RUN` steps (that *is* the slow dynamics H2 is about;
read `final_loss`). Watch `s_to_floor`: when it drops below ~3 the kprop error has hit the MC
floor — raise `MC_SAMPLES`. Few seeds ⇒ noisy medians. This is a **heavy** sweep with
`QUICK=False` (depths × widths × ratios × seeds × 12000 SGD steps × ~1M MC, plus k_max=3
kprop at width 512); use a GPU and trim ranges for a first real pass.
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

out = os.path.join(os.path.dirname(__file__), "kprop_meanfield_and_frozen_readout_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(cells), "cells")
