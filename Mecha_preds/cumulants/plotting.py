"""plotting.py -- cumulant-vs-MC error metrics vs width, median + IQR over seeds.

Consumes the DataFrame written by ``run_comparison`` (one row per width/seed/phase,
phase in {"initial", "trained"}) and writes width-scaling PNGs. matplotlib only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PHASES = ["initial", "trained"]
PHASE_STYLE = {
    "initial": dict(color="tab:blue", marker="o", label="initial (untrained)"),
    "trained": dict(color="tab:red", marker="s", label="trained"),
}


def _agg(df: pd.DataFrame, col: str):
    """median and IQR (q25, q75) of `col` by width, per phase."""
    out = {}
    ok = df[df["status"] == "ok"] if "status" in df else df
    for phase in PHASES:
        sub = ok[ok["phase"] == phase]
        if sub.empty:
            continue
        g = sub.groupby("width")[col]
        widths = np.array(sorted(sub["width"].unique()))
        out[phase] = (widths, g.median().reindex(widths).to_numpy(),
                      g.quantile(0.25).reindex(widths).to_numpy(),
                      g.quantile(0.75).reindex(widths).to_numpy())
    return out


def _logy_plot(ax, df, col, title, ylabel):
    for phase, (widths, med, q25, q75) in _agg(df, col).items():
        st = PHASE_STYLE[phase]
        ax.plot(widths, np.where(med <= 0, np.nan, med), marker=st["marker"],
                color=st["color"], label=st["label"])
        ax.fill_between(widths, np.where(q25 <= 0, np.nan, q25), np.where(q75 <= 0, np.nan, q75),
                        color=st["color"], alpha=0.18)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("width")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def make_plots(df: pd.DataFrame, outdir: str) -> list[str]:
    """Generate the width-scaling plots into ``outdir``. Returns written paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    written: list[str] = []

    def _save(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    panels = [
        ("mean_abs_error", "Mean abs error |cp - mc| vs width", "mean abs error (log)",
         "abs_error_vs_width.png"),
        ("relative_error_mean", "Relative error |cp - mc| / |mc| vs width",
         "relative error of mean (log)", "relative_error_vs_width.png"),
        ("nmse_mean", "NMSE of mean vs width (= relative error^2)", "NMSE (log)",
         "nmse_vs_width.png"),
        ("empirical_output_rms", "Empirical output RMS vs width", "RMS of model output (log)",
         "output_rms_vs_width.png"),
    ]
    for col, title, ylabel, fname in panels:
        if col not in df:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.2))
        _logy_plot(ax, df, col, title, ylabel)
        _save(fig, fname)

    # z-score: error in MC-standard-error units (separates real bias from MC noise)
    if "mc_noise_z" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        _logy_plot(ax, df, "mc_noise_z", "Error in MC-sigma units |cp - mc| / MC_stderr vs width",
                   "z = |cp - mc| / MC_stderr (log)")
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.8)
        _save(fig, "z_vs_mc_noise_vs_width.png")

    # final training loss vs width (trained only)
    ok = df[df["status"] == "ok"] if "status" in df else df
    tr = ok[ok["phase"] == "trained"]
    if not tr.empty and "final_train_loss" in tr and tr["final_train_loss"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4.2))
        g = tr.groupby("width")["final_train_loss"]
        widths = np.array(sorted(tr["width"].unique()))
        med = g.median().reindex(widths).to_numpy()
        ax.plot(widths, np.where(med <= 0, np.nan, med), marker="s", color="tab:red")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("width")
        ax.set_ylabel("final train loss (log)")
        ax.set_title("Final training loss vs width")
        ax.grid(True, which="both", alpha=0.3)
        _save(fig, "final_train_loss_vs_width.png")

    return written
