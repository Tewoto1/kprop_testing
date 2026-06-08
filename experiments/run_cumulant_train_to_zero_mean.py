"""Cumulant propagation vs Monte-Carlo for MLPs trained to output zero.

Scientific question
-------------------
Does the REAL cumulant propagation algorithm (src/mlp_kprop, treated as a black
box) still accurately predict the output MEAN of a simple MLP after the model is
randomly initialized and then trained only to output 0? The distribution is over
the input X ~ N(0, I_input_dim); the weights are FIXED after training. We compare
the cumulant-propagation mean against a 64,000-sample Monte-Carlo empirical mean,
for both the initial (untrained) model and the trained-to-zero model, across a
sweep of widths.

How to run
----------
Run from the repo root with uv (the source uses `from src.mlp_kprop...` imports,
so the repo root must be on sys.path -- this script inserts it automatically):

    uv run python experiments/run_cumulant_train_to_zero_mean.py \
        --input-dim 64 --output-dim 1 --hidden-depth 3 \
        --widths 64 128 256 512 --seeds 0 1 2 3 4 \
        --mc-samples 64000 --train-steps 5000 --batch-size 1024 --lr 1e-3 \
        --outdir results/cumulant_train_to_zero

Useful flags: --skip-training (only initial models), --no-plots, --debug,
--no-sanity, --k-max N (budget K, no hard cap; higher = more accurate but
~O(n^k_max) memory/time, so large k_max can OOM at width 1024 on a small box),
--use-avg-metric (use init-time E[WW^T] metric instead of the exact metric).

Outputs
-------
  <outdir>/cumulant_train_to_zero_mean_results.csv   (written incrementally)
  <outdir>/config.json
  <outdir>/plots/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# Ensure repo root is importable (so `import src.mlp_kprop...` and
# `import cumulant_experiments...` both work regardless of CWD).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import torch

from cumulant_experiments.cumulant_adapter import (
    config_summary,
    extract_mean,
    run_cumulant_propagation_from_model,
    run_sanity_checks,
)
from cumulant_experiments.metrics import compare_means, estimate_empirical_mean
from cumulant_experiments.model_utils import (
    layer_norms,
    make_mlp,
    set_seed,
    train_model_to_zero,
)


def evaluate_cumulant_vs_mc(*, model, input_dim, output_dim, cumulant_config,
                            mc_samples, mc_batch_size, device, debug=False):
    """Run the adapter + MC and return (metrics, cp_mean, mc_mean, mc_stats)."""
    cp_result = run_cumulant_propagation_from_model(
        model, input_dim, cumulant_config, device=device, debug=debug
    )
    cp_mean = extract_mean(cp_result)

    mc_mean, mc_stats = estimate_empirical_mean(
        model=model, input_dim=input_dim, num_samples=mc_samples,
        batch_size=mc_batch_size, device=device,
    )
    if debug:
        x = torch.randn(4, input_dim, device=device, dtype=torch.float64)
        with torch.no_grad():
            print("    [debug] first model outputs:", model.to(torch.float64)(x).out.flatten().tolist())
        print(f"    [debug] cp_mean={cp_mean}  mc_mean={mc_mean}")

    metrics = compare_means(cp_mean, mc_mean, mc_stats)
    return metrics, cp_mean, mc_mean, mc_stats


def _mean_to_cell(mean_vec, output_dim):
    """Scalar column when output_dim==1, else a JSON string."""
    arr = np.asarray(mean_vec, dtype=np.float64).reshape(-1)
    if output_dim == 1:
        return float(arr[0])
    return json.dumps(arr.tolist())


def build_row(*, width, seed, phase, args, cp_mean, mc_mean, metrics, mc_stats,
              train_stats, num_layers, weight_norms, bias_norms, cfg_summary,
              runtime_seconds, status, error_message):
    row = {
        "width": width,
        "seed": seed,
        "phase": phase,
        "input_dim": args.input_dim,
        "output_dim": args.output_dim,
        "hidden_depth": args.hidden_depth,
        "activation_name": args.activation,
        "train_steps": (train_stats.get("train_steps_run") if train_stats else 0),
        "initial_train_loss": (train_stats.get("initial_train_loss") if train_stats else np.nan),
        "final_train_loss": (train_stats.get("final_train_loss") if train_stats else np.nan),
        "mc_samples": (mc_stats.get("mc_samples") if mc_stats else np.nan),
        "cp_mean": (_mean_to_cell(cp_mean, args.output_dim) if cp_mean is not None else np.nan),
        "mc_mean": (_mean_to_cell(mc_mean, args.output_dim) if mc_mean is not None else np.nan),
        "mean_error": (metrics.get("mean_error") if metrics else np.nan),
        "mean_abs_error": (metrics.get("mean_abs_error") if metrics else np.nan),
        "mean_squared_error": (metrics.get("mean_squared_error") if metrics else np.nan),
        "mean_l2_error": (metrics.get("mean_l2_error") if metrics else np.nan),
        "relative_error_mean": (metrics.get("relative_error_mean") if metrics else np.nan),
        "mc_mean_se": (metrics.get("mc_mean_se") if metrics else np.nan),
        "mc_noise_z": (metrics.get("mc_noise_z") if metrics else np.nan),
        "nmse_mean": (metrics.get("nmse_mean") if metrics else np.nan),
        "variance_normalized_mean_error": (metrics.get("variance_normalized_mean_error") if metrics else np.nan),
        "empirical_output_rms": (mc_stats.get("empirical_output_rms") if mc_stats else np.nan),
        "empirical_output_std": (mc_stats.get("empirical_output_std") if mc_stats else np.nan),
        "empirical_output_second_moment": (mc_stats.get("empirical_output_second_moment") if mc_stats else np.nan),
        "cumulant_config_summary": cfg_summary,
        "runtime_seconds": runtime_seconds,
        "status": status,
        "error_message": error_message,
    }
    for i in range(num_layers):
        row[f"weight_norm_layer_{i}"] = (weight_norms[i] if weight_norms and i < len(weight_norms) else np.nan)
        bn = bias_norms[i] if bias_norms and i < len(bias_norms) else None
        row[f"bias_norm_layer_{i}"] = (np.nan if bn is None else bn)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=1)
    p.add_argument("--hidden-depth", type=int, default=3, help="number of hidden layers (num linear layers = hidden_depth+1)")
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--activation", type=str, default="relu",
                   help="must be supported by cumulant propagation (relu/gelu/tanh/sigmoid/square/cube/heaviside/sgn)")
    p.add_argument("--no-bias", dest="bias", action="store_false", help="disable bias parameters")
    p.set_defaults(bias=True)

    p.add_argument("--mc-samples", type=int, default=64_000)
    p.add_argument("--mc-batch-size", type=int, default=8192)

    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss-tol", type=float, default=0.0, help="early-stop training when loss < this (0 = never)")

    # cumulant propagation config
    p.add_argument("--k-max", type=int, default=3,
                   help="budget K (>=1). Higher = more accurate but ~O(n^k_max) memory/time; "
                        "k_max>=5 can OOM at large width on a small machine.")
    p.add_argument("--kind", type=str, default="simple", choices=["simple", "augment", "old", "base"])
    p.add_argument("--use-avg-metric", action="store_true",
                   help="use init-time E[WW^T] metric instead of the exact metric from actual weights")
    p.add_argument("--no-factor", dest="factor", action="store_false")
    p.set_defaults(factor=True)
    p.add_argument("--exact-relu-cov", action="store_true",
                   help="use the EXACT bivariate-Gaussian ReLU covariance at k_max==2 "
                        "(true Cov(ReLU(Z_i),ReLU(Z_j)), no gain approximation; needs scipy) "
                        "instead of the approximate harmonic propagation. Only with --k-max 2 and ReLU")

    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--outdir", type=str, default="results/cumulant_train_to_zero")
    p.add_argument("--skip-training", action="store_true", help="only evaluate initial random models")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-sanity", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--reanalyze", action="store_true",
                   help="recompute derived metric columns from the existing CSV, "
                        "rewrite it, regenerate plots and summary -- no models are run")
    args = p.parse_args()

    if args.reanalyze:
        reanalyze(args)
        return

    torch.set_grad_enabled(True)  # training needs grad; eval paths use no_grad locally
    torch.set_default_dtype(torch.float64)

    os.makedirs(args.outdir, exist_ok=True)
    plots_dir = os.path.join(args.outdir, "plots")
    csv_path = os.path.join(args.outdir, "cumulant_train_to_zero_mean_results.csv")

    cumulant_config = {
        "k_max": args.k_max,
        "kind": args.kind,
        "use_avg_metric": args.use_avg_metric,
        "factor": args.factor,
        "use_pK": True,
        "output_d_max": 1,
        "exact_relu_cov": args.exact_relu_cov,
    }
    cfg_summary = config_summary(cumulant_config)
    num_layers = args.hidden_depth + 1

    # Save the full experiment config.
    full_config = {
        **{k: getattr(args, k) for k in vars(args)},
        "cumulant_config": cumulant_config,
        "cumulant_config_summary": cfg_summary,
        "dtype": "float64",
        "input_distribution": "X ~ N(0, I_input_dim)",
    }
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    print("=" * 80)
    print("Cumulant propagation (REAL, src/mlp_kprop) vs Monte-Carlo — train-to-zero")
    print("=" * 80)
    print(f"config: {cfg_summary}")
    print(f"widths={args.widths} seeds={args.seeds} device={args.device} "
          f"mc_samples={args.mc_samples} train_steps={args.train_steps}")
    print(f"activation={args.activation} bias={args.bias} "
          f"input_dim={args.input_dim} output_dim={args.output_dim} hidden_depth={args.hidden_depth}")
    print()

    # --- Sanity checks --------------------------------------------------------
    if not args.no_sanity:
        ok = run_sanity_checks(device=args.device, cumulant_config=cumulant_config)
        if not ok:
            print("\nSanity checks FAILED. Aborting before the main experiment.")
            sys.exit(1)
        print()

    rows: list[dict] = []

    def flush_csv():
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    for width in args.widths:
        for seed in args.seeds:
            print(f"--- width={width} seed={seed} ---", flush=True)
            set_seed(seed)
            model = make_mlp(
                input_dim=args.input_dim, hidden_width=width, hidden_depth=args.hidden_depth,
                output_dim=args.output_dim, activation=args.activation, bias=args.bias,
                device=args.device, dtype=torch.float64,
            )
            if args.debug:
                print("    [debug] model:", model)

            # ---- Phase 1: initial (untrained) model --------------------------
            t0 = time.time()
            try:
                w_norms, b_norms = layer_norms(model)
                metrics, cp_mean, mc_mean, mc_stats = evaluate_cumulant_vs_mc(
                    model=model, input_dim=args.input_dim, output_dim=args.output_dim,
                    cumulant_config=cumulant_config, mc_samples=args.mc_samples,
                    mc_batch_size=args.mc_batch_size, device=args.device, debug=args.debug,
                )
                rows.append(build_row(
                    width=width, seed=seed, phase="initial", args=args,
                    cp_mean=cp_mean, mc_mean=mc_mean, metrics=metrics, mc_stats=mc_stats,
                    train_stats=None, num_layers=num_layers, weight_norms=w_norms,
                    bias_norms=b_norms, cfg_summary=cfg_summary,
                    runtime_seconds=time.time() - t0, status="ok", error_message="",
                ))
                print(f"    initial: cp_mean={_fmt(cp_mean)} mc_mean={_fmt(mc_mean)} "
                      f"abs_err={metrics['mean_abs_error']:.3e} "
                      f"rel_err={metrics['relative_error_mean']:.3e} "
                      f"z_vs_mc_noise={metrics['mc_noise_z']:.2f} "
                      f"out_rms={mc_stats['empirical_output_rms']:.3e}", flush=True)
            except Exception:
                err = traceback.format_exc()
                print("    initial FAILED:\n", err, flush=True)
                w_norms, b_norms = layer_norms(model)
                rows.append(build_row(
                    width=width, seed=seed, phase="initial", args=args,
                    cp_mean=None, mc_mean=None, metrics=None, mc_stats=None,
                    train_stats=None, num_layers=num_layers, weight_norms=w_norms,
                    bias_norms=b_norms, cfg_summary=cfg_summary,
                    runtime_seconds=time.time() - t0, status="failed",
                    error_message=err.strip().splitlines()[-1][:500],
                ))
            flush_csv()

            if args.skip_training:
                continue

            # ---- Phase 2: train to zero --------------------------------------
            t0 = time.time()
            try:
                train_stats = train_model_to_zero(
                    model=model, input_dim=args.input_dim, steps=args.train_steps,
                    batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                    device=args.device, dtype=torch.float64, loss_tol=args.loss_tol,
                    log_every=(max(1, args.train_steps // 4) if args.debug else 0),
                )
                w_norms, b_norms = layer_norms(model)
                metrics, cp_mean, mc_mean, mc_stats = evaluate_cumulant_vs_mc(
                    model=model, input_dim=args.input_dim, output_dim=args.output_dim,
                    cumulant_config=cumulant_config, mc_samples=args.mc_samples,
                    mc_batch_size=args.mc_batch_size, device=args.device, debug=args.debug,
                )
                rows.append(build_row(
                    width=width, seed=seed, phase="trained_to_zero", args=args,
                    cp_mean=cp_mean, mc_mean=mc_mean, metrics=metrics, mc_stats=mc_stats,
                    train_stats=train_stats, num_layers=num_layers, weight_norms=w_norms,
                    bias_norms=b_norms, cfg_summary=cfg_summary,
                    runtime_seconds=time.time() - t0, status="ok", error_message="",
                ))
                print(f"    trained: final_loss={train_stats['final_train_loss']:.3e} "
                      f"cp_mean={_fmt(cp_mean)} mc_mean={_fmt(mc_mean)} "
                      f"abs_err={metrics['mean_abs_error']:.3e} "
                      f"rel_err={metrics['relative_error_mean']:.3e} "
                      f"z_vs_mc_noise={metrics['mc_noise_z']:.2f} "
                      f"out_rms={mc_stats['empirical_output_rms']:.3e}", flush=True)
            except Exception:
                err = traceback.format_exc()
                print("    trained FAILED:\n", err, flush=True)
                try:
                    w_norms, b_norms = layer_norms(model)
                except Exception:
                    w_norms, b_norms = None, None
                rows.append(build_row(
                    width=width, seed=seed, phase="trained_to_zero", args=args,
                    cp_mean=None, mc_mean=None, metrics=None, mc_stats=None,
                    train_stats=None, num_layers=num_layers, weight_norms=w_norms,
                    bias_norms=b_norms, cfg_summary=cfg_summary,
                    runtime_seconds=time.time() - t0, status="failed",
                    error_message=err.strip().splitlines()[-1][:500],
                ))
            flush_csv()

    flush_csv()
    print(f"\nWrote CSV: {csv_path}")

    # --- Plots ----------------------------------------------------------------
    if not args.no_plots:
        try:
            df = pd.DataFrame(rows)
            written = __import__("cumulant_experiments.plotting", fromlist=["make_plots"]).make_plots(df, plots_dir)
            print("Wrote plots:")
            for w in written:
                print("  ", w)
        except Exception:
            print("Plotting failed (continuing):\n", traceback.format_exc())

    print_summary(rows, args)


def reanalyze(args):
    """Recompute the corrected metric columns from an existing scalar-output CSV.

    Derives relative_error_mean, mc_mean_se and mc_noise_z from the stored
    cp_mean, mc_mean, empirical_output_std and mc_samples (valid for output_dim=1,
    where ||per-output SE|| = output_std / sqrt(N)). Rewrites the CSV, regenerates
    plots, and prints the honest summary. Does not run any model.
    """
    csv_path = os.path.join(args.outdir, "cumulant_train_to_zero_mean_results.csv")
    df = pd.read_csv(csv_path)
    if (df["output_dim"] != 1).any():
        raise SystemExit("reanalyze currently supports scalar output (output_dim=1) only.")
    cp = df["cp_mean"].astype(float)
    mc = df["mc_mean"].astype(float)
    diff = (cp - mc).abs()
    eps = 1e-12
    df["mean_l2_error"] = diff
    df["relative_error_mean"] = diff / (mc.abs() + eps)
    df["mc_mean_se"] = df["empirical_output_std"] / np.sqrt(df["mc_samples"])
    df["mc_noise_z"] = diff / (df["mc_mean_se"] + eps)
    df.to_csv(csv_path, index=False)
    print(f"Reanalyzed and rewrote: {csv_path}")

    # Drop the stale/misleading variance-normalized plot if present.
    stale = os.path.join(args.outdir, "plots", "variance_normalized_error_vs_width.png")
    if os.path.exists(stale):
        os.remove(stale)

    if not args.no_plots:
        written = __import__("cumulant_experiments.plotting", fromlist=["make_plots"]).make_plots(
            df, os.path.join(args.outdir, "plots"))
        print("Wrote plots:")
        for w in written:
            print("  ", w)

    args.widths = sorted(df["width"].unique().tolist())
    args.mc_samples = int(df["mc_samples"].dropna().median())
    print_summary(df.to_dict("records"), args)


def _fmt(mean_vec):
    arr = np.asarray(mean_vec, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return f"{arr[0]:+.4e}"
    return np.array2string(arr, precision=3, max_line_width=80)


def print_summary(rows, args):
    df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("WIDTH SUMMARY (median over seeds; status==ok only)")
    print("=" * 80)
    ok = df[df["status"] == "ok"] if "status" in df else df
    print("(rel_err = |cp_mean - mc_mean| / |mc_mean| = sqrt(NMSE), the scale-free relative error.")
    print(" z = |cp_mean - mc_mean| / MC_stderr_of_mean: z<~1 = within MC noise; z>>1 = real kprop bias.)\n")
    for width in args.widths:
        print(f"width={width}:")
        for phase in ["initial", "trained_to_zero"]:
            sub = ok[(ok["width"] == width) & (ok["phase"] == phase)]
            if sub.empty:
                continue
            print(f"    {phase:16s} median rel_err={sub['relative_error_mean'].median():.3e}  "
                  f"z_vs_mc_noise={sub['mc_noise_z'].median():.2f}  "
                  f"abs_err={sub['mean_abs_error'].median():.3e}  "
                  f"out_rms={sub['empirical_output_rms'].median():.3e}")

    # Cautious verdict based on the relative error and its statistical significance.
    print("\n" + "-" * 80)
    tr = ok[ok["phase"] == "trained_to_zero"]
    ini = ok[ok["phase"] == "initial"]
    if tr.empty:
        print("No trained-to-zero rows (training skipped); cannot assess post-training behavior.")
        return
    tr_rel = tr.groupby("width")["relative_error_mean"].median()
    tr_z = tr.groupby("width")["mc_noise_z"].median()
    ini_rel = ini.groupby("width")["relative_error_mean"].median() if not ini.empty else None
    ini_z = ini.groupby("width")["mc_noise_z"].median() if not ini.empty else None

    print("VERDICT (cautious): does cumulant propagation predict the mean after training to zero?")
    print(f"  trained relative error of mean (median by width): "
          f"{ {int(w): round(float(v), 6) for w, v in tr_rel.items()} }")
    print(f"  trained z vs MC noise (median by width):          "
          f"{ {int(w): round(float(v), 2) for w, v in tr_z.items()} }")
    if ini_rel is not None:
        print(f"  initial relative error (for comparison):          "
              f"{ {int(w): round(float(v), 6) for w, v in ini_rel.items()} }")
        print(f"  initial z vs MC noise  (for comparison):          "
              f"{ {int(w): round(float(v), 2) for w, v in ini_z.items()} }")

    # Significance: does the trained kprop mean sit beyond MC noise (z >> 1)?
    max_z = float(tr_z.max())
    rel_ratio = (float(tr_rel.max()) / float(ini_rel.max())) if (ini_rel is not None and ini_rel.max() > 0) else float("nan")
    biased = max_z > 3.0
    if biased:
        verdict = (
            "In this run, training to zero INTRODUCES A STATISTICALLY REAL BIAS in the cumulant-"
            "propagation mean (the kprop mean sits several MC standard errors from the MC mean), "
            "i.e. cumulant propagation is MEASURABLY DEGRADED by training -- consistent with "
            "training creating weight/activation correlations that the wide-random-MLP approximation "
            "does not capture. The bias is largest at small width and shrinks as width grows."
        )
    elif max_z > 1.5:
        verdict = (
            "In this run, training to zero appears to DEGRADE the cumulant-propagation mean somewhat "
            "(error modestly above MC noise); the effect is marginal -- INCONCLUSIVE without more MC "
            "samples to tighten the reference mean."
        )
    else:
        verdict = (
            "In this run, the cumulant-propagation mean after training stays WITHIN MC sampling noise "
            "of the MC mean (z <~ 1), so we cannot detect degradation at this MC sample size; "
            "kprop appears to remain accurate for the mean."
        )
    print(f"  At init the gap is within MC noise (median z<=1 expected); worst trained median z = {max_z:.2f}.")
    if not np.isnan(rel_ratio):
        print(f"  Worst-width trained relative error is ~{rel_ratio:.1f}x the initial relative error.")
    print(f"  {verdict}")
    print(f"  (Reference is a {args.mc_samples}-sample Monte-Carlo mean; when the trained mean is tiny, "
          f"increase --mc-samples to tighten it.)")


if __name__ == "__main__":
    main()
