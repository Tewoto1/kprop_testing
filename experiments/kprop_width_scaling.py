"""Decisive width-scaling experiment for cumulant propagation: initial vs trained-to-zero.

Question
--------
Cumulant propagation ("kprop", src/mlp_kprop, real algorithm, black box) has a
theoretical accuracy that improves with width n: the per-entry MSE of an estimated
cumulant is O(n^{-k_max}), i.e. RMS error ~ n^{-k_max/2}. Does training a net to
output zero break this width-scaling (flatten the slope), or does kprop keep
improving with width even on a trained (correlated-weight) net?

Why the earlier scalar-output relative-error plot could NOT see this
-------------------------------------------------------------------
1. At k_max=3 a single net's kprop error is *below* the Monte-Carlo sampling
   noise of a 64k (even 2M) sample mean -> we'd measure MC noise, which is flat
   in width.
2. A single scalar output's relative error is a hopelessly high-variance
   estimator of the scaling law (the law is about ensemble/many-component
   averages; the relative-error denominator |mean| is a random near-zero number).

This script fixes both:
  - VECTOR output (output_dim components) so per-entry MSE averages over many
    components; plus median over seeds.
  - MC-variance DEBIASING: measured per-entry MSE E[(cp-mc)^2] = kprop_MSE +
    Var(mc). We subtract the MC variance floor (mean of per-output stderr^2) so
    we can resolve kprop error BELOW the sampling floor. (This is exactly why the
    paper uses ~2^34 samples; debiasing lets us do it with far fewer.)

Same net is evaluated before training (phase "initial") and after training to
zero (phase "trained_to_zero"), so the comparison is apples-to-apples.

Run (from repo root):
    uv run python experiments/kprop_width_scaling.py \
        --widths 64 128 256 512 --seeds 0 1 2 --k-maxs 2 3 \
        --output-dim 128 --mc-samples 2000000 --train-steps 3000 \
        --outdir results/kprop_width_scaling

Outputs: <outdir>/scaling_results.csv, <outdir>/plots/scaling_rms_vs_width.png,
and a printed slope table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import torch

from cumulant_experiments.cumulant_adapter import config_summary, extract_mean, run_cumulant_propagation_from_model
from cumulant_experiments.metrics import estimate_empirical_mean
from cumulant_experiments.model_utils import make_mlp, output_rms, set_seed, train_model_to_zero


def debiased_rms(cp_mean, mc_mean, mc_stats):
    """Per-entry RMS of (cp - true mean), MC-variance-debiased.

    measured_mse = mean_i (cp_i - mc_i)^2  =  true kprop MSE + mean_i Var(mc_i)
    floor        = mean_i stderr_i^2       =  the MC sampling-variance contribution
    debiased_mse = max(measured_mse - floor, 0)
    """
    cp = np.asarray(cp_mean, np.float64).reshape(-1)
    mc = np.asarray(mc_mean, np.float64).reshape(-1)
    stderr = np.asarray(mc_stats["mc_stderr"], np.float64).reshape(-1)
    measured_mse = float(np.mean((cp - mc) ** 2))
    floor = float(np.mean(stderr ** 2))
    debiased_mse = max(measured_mse - floor, 0.0)
    return {
        "measured_mse": measured_mse,
        "mc_floor": floor,
        "debiased_mse": debiased_mse,
        "debiased_rms": math.sqrt(debiased_mse),
        "signal_to_floor": (measured_mse - floor) / (floor + 1e-300),
    }


def loglog_slope(widths, values):
    """Least-squares slope of log(value) vs log(width); NaN if any value<=0."""
    w = np.asarray(widths, float)
    v = np.asarray(values, float)
    if np.any(v <= 0) or len(w) < 2:
        return float("nan")
    A = np.polyfit(np.log(w), np.log(v), 1)
    return float(A[0])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=128, help="vector output -> many components to average MSE over")
    p.add_argument("--hidden-depth", type=int, default=3)
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-maxs", type=int, nargs="+", default=[2, 3], choices=[1, 2, 3])
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--mc-samples", type=int, default=2_000_000)
    p.add_argument("--mc-batch-size", type=int, default=250_000)
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--outdir", type=str, default="results/kprop_width_scaling")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    torch.set_grad_enabled(True)
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "scaling_results.csv")

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({k: getattr(args, k) for k in vars(args)}, f, indent=2, default=str)

    print("=" * 88)
    print("Width-scaling of cumulant propagation: initial vs trained-to-zero (debiased vector MSE)")
    print("=" * 88)
    print(f"widths={args.widths} seeds={args.seeds} k_maxs={args.k_maxs} "
          f"output_dim={args.output_dim} mc_samples={args.mc_samples} train_steps={args.train_steps}")
    print(f"theory: per-entry RMS error ~ n^(-k_max/2)\n")

    rows = []

    def cfg_for(k):
        return {"k_max": k, "kind": "simple", "use_avg_metric": False,
                "factor": (k >= 3), "use_pK": True, "output_d_max": 1}

    def eval_all_kmax(model, phase, width, seed, extra):
        # One MC estimate (k_max-independent), reused across k_max.
        mc_mean, mc_stats = estimate_empirical_mean(
            model=model, input_dim=args.input_dim, num_samples=args.mc_samples,
            batch_size=args.mc_batch_size, device=args.device)
        for k in args.k_maxs:
            t0 = time.time()
            cp = extract_mean(run_cumulant_propagation_from_model(model, args.input_dim, cfg_for(k), device=args.device))
            d = debiased_rms(cp, mc_mean, mc_stats)
            row = {
                "phase": phase, "k_max": k, "width": width, "seed": seed,
                "output_dim": args.output_dim, "mc_samples": mc_stats["mc_samples"],
                **d, "empirical_output_rms": mc_stats["empirical_output_rms"],
                "cumulant_config": config_summary(cfg_for(k)),
                "runtime_seconds": time.time() - t0, **extra,
            }
            rows.append(row)
            print(f"  [{phase:15s} k={k} n={width:4d} s={seed}] "
                  f"debiased_rms={d['debiased_rms']:.3e}  measured_mse={d['measured_mse']:.2e}  "
                  f"floor={d['mc_floor']:.2e}  S/floor={d['signal_to_floor']:.1f}", flush=True)

    for width in args.widths:
        for seed in args.seeds:
            set_seed(seed)
            model = make_mlp(input_dim=args.input_dim, hidden_width=width, hidden_depth=args.hidden_depth,
                             output_dim=args.output_dim, activation=args.activation, bias=True,
                             device=args.device, dtype=torch.float64)
            # Phase 1: initial (untrained)
            eval_all_kmax(model, "initial", width, seed, {"final_train_loss": float("nan")})
            # Phase 2: train to zero, then re-evaluate the SAME net
            stats = train_model_to_zero(
                model=model, input_dim=args.input_dim, steps=args.train_steps,
                batch_size=args.batch_size, lr=args.lr, device=args.device, dtype=torch.float64)
            eval_all_kmax(model, "trained_to_zero", width, seed, {"final_train_loss": stats["final_train_loss"]})
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nWrote CSV: {csv_path}")

    # --- Slope table ---------------------------------------------------------
    print("\n" + "=" * 88)
    print("SLOPE OF debiased RMS vs width  (theory ~ -k_max/2). Median over seeds.")
    print("=" * 88)
    slopes = {}
    for phase in ["initial", "trained_to_zero"]:
        for k in args.k_maxs:
            sub = df[(df.phase == phase) & (df.k_max == k)]
            if sub.empty:
                continue
            med = sub.groupby("width")["debiased_rms"].median()
            widths = sorted(med.index.tolist())
            vals = [med[w] for w in widths]
            s = loglog_slope(widths, vals)
            slopes[(phase, k)] = s
            series = "  ".join(f"n{w}={med[w]:.2e}" for w in widths)
            print(f"  {phase:15s} k_max={k}: slope={s:+.2f} (theory {-k/2:+.1f})   {series}")

    print("\n" + "-" * 88)
    print("VERDICT (cautious): does training-to-zero break the width-scaling of cumulant propagation?")
    for k in args.k_maxs:
        si = slopes.get(("initial", k), float("nan"))
        st = slopes.get(("trained_to_zero", k), float("nan"))
        if math.isnan(si) or math.isnan(st):
            continue
        if st < -0.05 and st <= si * 0.5:
            msg = ("trained slope is much flatter than initial -> training APPEARS TO DEGRADE "
                   "the width-scaling (kprop improves with width more slowly after training).")
        elif st < -0.05:
            msg = ("trained slope is still clearly negative and comparable to initial -> kprop "
                   "STILL IMPROVES WITH WIDTH after training (scaling largely preserved).")
        else:
            msg = ("trained slope is ~flat -> after training, kprop error NO LONGER improves with "
                   "width at these widths (scaling appears broken).")
        print(f"  k_max={k}: initial slope {si:+.2f} vs trained slope {st:+.2f}. {msg}")
    print("  (per-entry RMS of the output mean, debiased by MC sampling variance; "
          f"{args.mc_samples}-sample MC.)")

    if not args.no_plots:
        try:
            make_plot(df, args)
        except Exception as e:
            print("Plotting failed:", e)


def make_plot(df, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    kcolors = {1: "tab:green", 2: "tab:blue", 3: "tab:red"}
    pstyle = {"initial": dict(ls="-", marker="o"), "trained_to_zero": dict(ls="--", marker="s")}
    for phase in ["initial", "trained_to_zero"]:
        for k in args.k_maxs:
            sub = df[(df.phase == phase) & (df.k_max == k)]
            if sub.empty:
                continue
            med = sub.groupby("width")["debiased_rms"].median()
            q1 = sub.groupby("width")["debiased_rms"].quantile(0.25)
            q3 = sub.groupby("width")["debiased_rms"].quantile(0.75)
            widths = np.array(sorted(med.index.tolist()))
            y = med.reindex(widths).to_numpy()
            ax.plot(widths, np.where(y <= 0, np.nan, y), color=kcolors.get(k, "k"),
                    label=f"{phase} k_max={k}", **pstyle[phase])
            ax.fill_between(widths, q1.reindex(widths).to_numpy(), q3.reindex(widths).to_numpy(),
                            color=kcolors.get(k, "k"), alpha=0.12)
    # reference slope guides n^{-k/2}
    for k in args.k_maxs:
        w = np.array(sorted(df.width.unique()), float)
        ref = (w / w[0]) ** (-k / 2)
        base = df[(df.k_max == k) & (df.phase == "initial")].groupby("width")["debiased_rms"].median()
        if not base.empty and base.iloc[0] > 0:
            ax.plot(w, base.iloc[0] * ref, color=kcolors.get(k, "k"), ls=":", lw=1, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("width n")
    ax.set_ylabel("debiased per-entry RMS error of output mean (log)")
    ax.set_title("Width-scaling of cumulant propagation: initial vs trained-to-zero\n"
                 "(dotted = theory n^(-k_max/2) guide)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(plots_dir, "scaling_rms_vs_width.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote plot:", path)


if __name__ == "__main__":
    main()
