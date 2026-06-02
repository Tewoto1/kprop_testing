"""Plotting: error metrics vs width, aggregated (median + IQR band) over seeds."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PHASES = ["initial", "trained_to_zero"]
PHASE_STYLE = {
    "initial": dict(color="tab:blue", marker="o", label="initial (untrained)"),
    "trained_to_zero": dict(color="tab:red", marker="s", label="trained to zero"),
}


def _agg(df: pd.DataFrame, col: str):
    """median and IQR (q25, q75) of `col` by width, per phase."""
    out = {}
    ok = df[df["status"] == "ok"]
    for phase in PHASES:
        sub = ok[ok["phase"] == phase]
        if sub.empty:
            continue
        g = sub.groupby("width")[col]
        widths = np.array(sorted(sub["width"].unique()))
        med = g.median().reindex(widths).to_numpy()
        q25 = g.quantile(0.25).reindex(widths).to_numpy()
        q75 = g.quantile(0.75).reindex(widths).to_numpy()
        out[phase] = (widths, med, q25, q75)
    return out


def _logy_plot(ax, df, col, title, ylabel):
    data = _agg(df, col)
    for phase, (widths, med, q25, q75) in data.items():
        st = PHASE_STYLE[phase]
        med_safe = np.where(med <= 0, np.nan, med)
        ax.plot(widths, med_safe, marker=st["marker"], color=st["color"], label=st["label"])
        lo = np.where(q25 <= 0, np.nan, q25)
        hi = np.where(q75 <= 0, np.nan, q75)
        ax.fill_between(widths, lo, hi, color=st["color"], alpha=0.18)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("width")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def make_plots(df: pd.DataFrame, outdir: str) -> list[str]:
    """Generate all plots into ``outdir``. Returns list of written paths."""
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

    # 1. abs error vs width
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _logy_plot(ax, df, "mean_abs_error", "Mean abs error |cp_mean - mc_mean| vs width",
               "mean abs error (log)")
    _save(fig, "abs_error_vs_width.png")

    # 2. HEADLINE: relative error of the mean = |cp-mc|/|mc| = sqrt(NMSE)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _logy_plot(ax, df, "relative_error_mean",
               "Relative error of mean  |cp - mc| / |mc|  vs width",
               "relative error of mean (log)")
    _save(fig, "relative_error_vs_width.png")

    # 3. z-score: error in MC-standard-error units (separates real bias from MC noise)
    if "mc_noise_z" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        _logy_plot(ax, df, "mc_noise_z",
                   "Error in MC-sigma units  |cp - mc| / MC_stderr  vs width",
                   "z = |cp - mc| / MC_stderr (log)")
        ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.8)
        ax.text(ax.get_xlim()[0], 1.05, " z=1 (within MC noise)", color="gray", fontsize=8, va="bottom")
        _save(fig, "z_vs_mc_noise_vs_width.png")

    # 4. NMSE vs width (kept; note it is unstable when the trained mean -> 0)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _logy_plot(ax, df, "nmse_mean", "NMSE of mean vs width (= relative error^2)", "NMSE (log)")
    _save(fig, "nmse_vs_width.png")

    # 4. empirical output RMS vs width
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _logy_plot(ax, df, "empirical_output_rms", "Empirical output RMS vs width",
               "RMS of model output (log)")
    _save(fig, "output_rms_vs_width.png")

    # 5. final training loss vs width (trained only)
    ok = df[(df["status"] == "ok") & (df["phase"] == "trained_to_zero")]
    if not ok.empty and ok["final_train_loss"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4.2))
        g = ok.groupby("width")["final_train_loss"]
        widths = np.array(sorted(ok["width"].unique()))
        med = g.median().reindex(widths).to_numpy()
        ax.plot(widths, np.where(med <= 0, np.nan, med), marker="s", color="tab:red")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("width"); ax.set_ylabel("final train loss (log)")
        ax.set_title("Final training loss vs width"); ax.grid(True, which="both", alpha=0.3)
        _save(fig, "final_train_loss_vs_width.png")

    return written
