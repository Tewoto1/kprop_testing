"""run_comparison.py -- cumulant propagation vs Monte-Carlo across a width sweep.

Builds a study ``model.MLP`` at each width/seed, evaluates the cumulant-propagation
output mean against a Monte-Carlo reference both at init and after training (via the
unified ``training`` loop on the chosen task), and writes an incremental CSV +
``config.json`` + width-scaling plots.

    python -m Mecha_preds.cumulants.run_comparison \
        --task zero --input-dim 64 --hidden-depth 3 \
        --widths 64 128 256 512 --seeds 0 1 2 3 4 \
        --train-steps 5000 --mc-samples 64000 --k-max 3 --outdir results/zero_kprop

Key flags: --task {zero,halfspace,distill}, --k-max, --kind, --exact-relu-cov
(exact bivariate ReLU covariance at k_max==2), --use-avg-metric, --no-factor,
--skip-training, --no-plots, --bias, --dtype. The model/training/MC run in float32
by default (GPU-fast; pass --dtype float64 for the old all-double behaviour); kprop
itself ALWAYS propagates in float64 internally (the adapter builds a double copy),
and the MC accumulators are float64. The headline metric is the scale-free relative
error |cp-mc|/|mc|; read it with mc_noise_z (error in MC-standard-error units).
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch

from model import ModelConfig
from training import Trainer, TrainConfig
from training.run import build_task
from utils import set_seed

from .adapter import config_summary, extract_mean, run_cumulants
from .metrics import compare_means, estimate_empirical_mean


def _evaluate(model, input_dim, cumulant_config, mc_samples, mc_batch_size, device):
    cp = extract_mean(run_cumulants(model, input_dim, cumulant_config, device=device))
    model_dtype = next(model.parameters()).dtype   # MC inputs must match the model dtype
    mc, mc_stats = estimate_empirical_mean(model=model, input_dim=input_dim,
                                           num_samples=mc_samples, batch_size=mc_batch_size,
                                           device=device, dtype=model_dtype)
    return compare_means(cp, mc, mc_stats), cp, mc, mc_stats


def _row(*, width, seed, phase, cp, mc, metrics, mc_stats, train_stats, cfg_summary,
         runtime, status, error):
    return {
        "width": width, "seed": seed, "phase": phase,
        "cp_mean": (float(cp[0]) if cp is not None and cp.size == 1
                    else (json.dumps(cp.tolist()) if cp is not None else np.nan)),
        "mc_mean": (float(mc[0]) if mc is not None and mc.size == 1
                    else (json.dumps(mc.tolist()) if mc is not None else np.nan)),
        "mean_abs_error": (metrics or {}).get("mean_abs_error", np.nan),
        "relative_error_mean": (metrics or {}).get("relative_error_mean", np.nan),
        "mc_noise_z": (metrics or {}).get("mc_noise_z", np.nan),
        "nmse_mean": (metrics or {}).get("nmse_mean", np.nan),
        "empirical_output_rms": (mc_stats or {}).get("empirical_output_rms", np.nan),
        "mc_samples": (mc_stats or {}).get("mc_samples", np.nan),
        "final_train_loss": (train_stats or {}).get("final_loss", np.nan),
        "cumulant_config": cfg_summary,
        "runtime_seconds": runtime, "status": status, "error_message": error,
    }


def main():
    p = argparse.ArgumentParser(description="Cumulant propagation vs Monte-Carlo over widths.")
    p.add_argument("--task", choices=["zero", "halfspace", "distill"], default="zero")
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=1)
    p.add_argument("--hidden-depth", type=int, default=3, help="number of hidden layers")
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--activation", default="relu")
    p.add_argument("--bias", action="store_true")
    p.add_argument("--offset-std", type=float, default=1.0)
    p.add_argument("--teacher-seed", type=int, default=None)

    p.add_argument("--mc-samples", type=int, default=64_000)
    p.add_argument("--mc-batch-size", type=int, default=8192)
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--k-max", type=int, default=3)
    p.add_argument("--kind", default="simple", choices=["simple", "augment", "old", "base"])
    p.add_argument("--use-avg-metric", action="store_true")
    p.add_argument("--no-factor", dest="factor", action="store_false")
    p.set_defaults(factor=True)
    p.add_argument("--exact-relu-cov", action="store_true",
                   help="exact bivariate-Gaussian ReLU covariance at k_max==2 (needs scipy)")

    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                   help="model/training/MC dtype (kprop is float64 internally either way)")
    p.add_argument("--outdir", default="results/comparison")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    torch.set_default_dtype(getattr(torch, args.dtype))
    if args.device == "cuda":          # TF32 matmuls on Ampere+ (fine: errors here >> 1e-3 rel)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "comparison_results.csv")

    cumulant_config = {"k_max": args.k_max, "kind": args.kind, "use_avg_metric": args.use_avg_metric,
                       "factor": args.factor, "use_pK": True, "output_d_max": 1,
                       "exact_relu_cov": args.exact_relu_cov}
    cfg_summary = config_summary(cumulant_config)
    out_dim = 1 if args.task == "halfspace" else args.output_dim

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump({**vars(args), "cumulant_config": cumulant_config},
                  f, indent=2, default=str)

    print(f"cumulant config: {cfg_summary}")
    print(f"task={args.task} widths={args.widths} seeds={args.seeds} device={args.device}\n")

    rows: list[dict] = []
    for width in args.widths:
        for seed in args.seeds:
            print(f"--- width={width} seed={seed} ---", flush=True)
            set_seed(seed)
            model_cfg = ModelConfig(input_dim=args.input_dim, hidden_dim=width,
                                    depth=args.hidden_depth, output_dim=out_dim,
                                    bias=args.bias, final_bias=args.bias,
                                    activation=args.activation, seed=seed)
            model = model_cfg.build().to(args.device)
            task = build_task(args.task, input_dim=args.input_dim, output_dim=out_dim,
                              activation=args.activation, bias=args.bias, depth=args.hidden_depth,
                              width=width, seed=seed, offset_std=args.offset_std,
                              teacher_seed=args.teacher_seed)

            for phase in (["initial"] if args.skip_training else ["initial", "trained"]):
                if phase == "trained":
                    tcfg = TrainConfig(steps=args.train_steps, batch_size=args.batch_size,
                                       lr=args.lr, weight_decay=args.weight_decay, seed=seed,
                                       checkpoint_mode="none", device=args.device,
                                       dtype=args.dtype)
                    train_stats = Trainer(model, task, tcfg, run_name=f"{args.task}_w{width}").train(
                        progress=False)
                else:
                    train_stats = None
                t0 = time.time()
                try:
                    metrics, cp, mc, mc_stats = _evaluate(model, args.input_dim, cumulant_config,
                                                          args.mc_samples, args.mc_batch_size,
                                                          args.device)
                    rows.append(_row(width=width, seed=seed, phase=phase, cp=cp, mc=mc,
                                     metrics=metrics, mc_stats=mc_stats, train_stats=train_stats,
                                     cfg_summary=cfg_summary, runtime=time.time() - t0,
                                     status="ok", error=""))
                    print(f"    {phase:8s} rel_err={metrics['relative_error_mean']:.3e} "
                          f"z={metrics['mc_noise_z']:.2f} out_rms={mc_stats['empirical_output_rms']:.3e}",
                          flush=True)
                except Exception:
                    err = traceback.format_exc()
                    print(f"    {phase} FAILED:\n", err, flush=True)
                    rows.append(_row(width=width, seed=seed, phase=phase, cp=None, mc=None,
                                     metrics=None, mc_stats=None, train_stats=train_stats,
                                     cfg_summary=cfg_summary, runtime=time.time() - t0,
                                     status="failed", error=err.strip().splitlines()[-1][:300]))
                pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nWrote CSV: {csv_path}")
    if not args.no_plots:
        from .plotting import make_plots
        try:
            for w in make_plots(pd.DataFrame(rows), os.path.join(args.outdir, "plots")):
                print("  plot:", w)
        except Exception:
            print("Plotting failed (continuing):\n", traceback.format_exc())

    _summary(pd.DataFrame(rows), args.widths)


def _summary(df, widths):
    print("\n" + "=" * 70 + "\nWIDTH SUMMARY (median over seeds; status==ok)\n" + "=" * 70)
    ok = df[df["status"] == "ok"] if "status" in df else df
    for width in widths:
        print(f"width={width}:")
        for phase in ["initial", "trained"]:
            sub = ok[(ok["width"] == width) & (ok["phase"] == phase)]
            if sub.empty:
                continue
            print(f"    {phase:8s} rel_err={sub['relative_error_mean'].median():.3e}  "
                  f"z={sub['mc_noise_z'].median():.2f}  "
                  f"out_rms={sub['empirical_output_rms'].median():.3e}")


if __name__ == "__main__":
    main()
