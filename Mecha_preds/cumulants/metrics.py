"""metrics.py -- Monte-Carlo reference mean and cumulant-vs-MC comparison.

The "ground truth" for a mechanistic predictor is a Monte-Carlo estimate of the
model's output mean over ``X ~ N(0, I)``. MC runs on the *study* ``model.MLP``
itself (the exact model cumulant propagation sees), in float64, under
``model.eval()`` / ``no_grad()``, batched to bound memory.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


@torch.no_grad()
def estimate_empirical_mean(*, model, input_dim: int, num_samples: int = 64_000,
                            batch_size: int = 8192, device: str = "cpu",
                            dtype: torch.dtype = torch.float64) -> tuple[np.ndarray, dict]:
    """Estimate ``E[Y]`` for ``Y = model(X)``, ``X ~ N(0, I_input_dim)``, via Monte Carlo.

    Returns ``(mc_mean (output_dim,), stats)`` where stats holds output_dim,
    mc_samples, the empirical second moment / rms / std, per-output second moment,
    and the standard error of the mean.
    """
    was_training = model.training
    model.eval()

    probe = model(torch.randn(2, input_dim, device=device, dtype=dtype))
    output_dim = probe.shape[-1]

    acc = torch.zeros(output_dim, dtype=torch.float64, device=device)      # sum Y
    acc_sq = torch.zeros(output_dim, dtype=torch.float64, device=device)   # sum Y^2
    n = 0
    while n < num_samples:
        b = min(batch_size, num_samples - n)
        x = torch.randn(b, input_dim, device=device, dtype=dtype)
        y = model(x).double()
        acc += y.sum(0)
        acc_sq += y.pow(2).sum(0)
        n += b

    if was_training:
        model.train()

    mc_mean = acc / n
    per_output_second_moment = acc_sq / n
    per_output_var = (per_output_second_moment - mc_mean ** 2).clamp_min(0.0)
    per_output_std = per_output_var.sqrt()
    mc_stderr = (per_output_var / n).sqrt()

    stats = {
        "output_dim": int(output_dim),
        "mc_samples": int(n),
        "empirical_output_second_moment": float(per_output_second_moment.sum().item()),
        "empirical_output_rms": float(np.sqrt(per_output_second_moment.mean().item())),
        "empirical_output_std": float(per_output_std.mean().item()),
        "per_output_second_moment": per_output_second_moment.cpu().numpy(),
        "mc_stderr": mc_stderr.cpu().numpy(),
    }
    return mc_mean.cpu().numpy(), stats


def compare_means(cp_mean: np.ndarray, mc_mean: np.ndarray, mc_stats: dict,
                  eps: float = EPS) -> dict:
    """Error metrics between the cumulant-propagation mean and the MC mean.

    HEADLINE: ``relative_error_mean = ||cp - mc|| / ||mc|| = sqrt(NMSE)`` -- scale-free,
    the right thing to read when the model is trained toward zero (it does not divide
    by a quantity that itself collapses with the output scale).

    Read it alongside ``mc_noise_z = ||cp - mc|| / mc_mean_se``: z <~ 1 means cp agrees
    with MC to within MC's own sampling noise (cannot distinguish); z >> 1 means a
    statistically real bias in the cumulant-propagation mean.
    """
    cp = np.asarray(cp_mean, dtype=np.float64).reshape(-1)
    mc = np.asarray(mc_mean, dtype=np.float64).reshape(-1)
    if cp.shape != mc.shape:
        raise ValueError(f"cp_mean shape {cp.shape} != mc_mean shape {mc.shape}")

    diff = cp - mc
    diff_norm = float(np.linalg.norm(diff))
    sq_err_sum = float(np.sum(diff ** 2))
    mc_norm = float(np.linalg.norm(mc))
    mc_norm_sq = mc_norm ** 2
    second_moment = float(mc_stats.get("empirical_output_second_moment", mc_norm_sq))

    mean_error = float(diff[0]) if cp.size == 1 else diff_norm  # signed for scalar output
    mc_stderr = np.asarray(mc_stats.get("mc_stderr", np.zeros_like(mc)), dtype=np.float64).reshape(-1)
    mc_mean_se = float(np.linalg.norm(mc_stderr))

    return {
        "mean_error": mean_error,
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "mean_squared_error": float(np.mean(diff ** 2)),
        "mean_l2_error": diff_norm,
        "relative_error_mean": diff_norm / (mc_norm + eps),          # = sqrt(NMSE); HEADLINE
        "mc_mean_se": mc_mean_se,
        "mc_noise_z": diff_norm / (mc_mean_se + eps),                # error in MC-sigma units
        "nmse_mean": sq_err_sum / (mc_norm_sq + eps),
        "variance_normalized_mean_error": sq_err_sum / (second_moment + eps),
    }
