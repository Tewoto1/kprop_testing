"""Monte-Carlo empirical mean estimation and cumulant-vs-MC comparison metrics.

All Monte-Carlo is done in float64 (consistent with cumulant propagation), under
``model.eval()`` and ``torch.no_grad()``, in batches to bound memory.
"""

from __future__ import annotations

import numpy as np
import torch

from src.mlp_kprop.mlp import MLP

EPS = 1e-12


@torch.no_grad()
def estimate_empirical_mean(
    *,
    model: MLP,
    input_dim: int,
    num_samples: int = 64_000,
    batch_size: int = 8192,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[np.ndarray, dict]:
    """Estimate E[Y] for Y = model(X), X ~ N(0, I_input_dim), via Monte Carlo.

    Returns (mc_mean (output_dim,), stats) where stats contains:
        output_dim, mc_samples,
        empirical_output_second_moment  = E[||Y||^2]
        empirical_output_rms            = sqrt(mean over samples & outputs of Y^2)
        empirical_output_std            = mean over outputs of per-output std
        per_output_second_moment        = E[Y_i^2] (vector)
        mc_stderr                       = standard error of the mean (vector)
    """
    model.eval()

    # Probe output_dim with a tiny batch.
    probe = model(torch.randn(2, input_dim, device=device, dtype=dtype)).out
    output_dim = probe.shape[-1]

    acc = torch.zeros(output_dim, dtype=torch.float64, device=device)      # sum Y
    acc_sq = torch.zeros(output_dim, dtype=torch.float64, device=device)   # sum Y^2
    n = 0
    while n < num_samples:
        b = min(batch_size, num_samples - n)
        x = torch.randn(b, input_dim, device=device, dtype=dtype)
        y = model(x).out.double()
        acc += y.sum(0)
        acc_sq += y.pow(2).sum(0)
        n += b

    mc_mean = (acc / n)                                   # (output_dim,)
    per_output_second_moment = (acc_sq / n)               # E[Y_i^2]
    per_output_var = (per_output_second_moment - mc_mean ** 2).clamp_min(0.0)
    per_output_std = per_output_var.sqrt()
    mc_stderr = (per_output_var / n).sqrt()

    second_moment = float(per_output_second_moment.sum().item())  # E[||Y||^2]
    rms = float(np.sqrt(per_output_second_moment.mean().item()))  # sqrt(mean Y^2)
    std = float(per_output_std.mean().item())

    stats = {
        "output_dim": int(output_dim),
        "mc_samples": int(n),
        "empirical_output_second_moment": second_moment,
        "empirical_output_rms": rms,
        "empirical_output_std": std,
        "per_output_second_moment": per_output_second_moment.cpu().numpy(),
        "mc_stderr": mc_stderr.cpu().numpy(),
    }
    return mc_mean.cpu().numpy(), stats


def compare_means(cp_mean: np.ndarray, mc_mean: np.ndarray, mc_stats: dict,
                  eps: float = EPS) -> dict:
    """Compute error metrics between the cumulant-propagation mean and MC mean.

    HEADLINE metric: ``relative_error_mean`` = ||cp - mc|| / ||mc|| = sqrt(NMSE).
    This is the scale-free relative error of the mean -- the right thing to look
    at when the model is trained toward zero, because it does NOT divide by a
    quantity that itself collapses (unlike dividing by E[||Y||^2], which shrinks
    with the output scale and can make a degrading estimate look like it is
    improving).

    CAVEAT (always read alongside the relative error): when the true mean is
    tiny, the 64k-sample MC estimate ``mc`` is itself only known to within its
    standard error. So we also report:
      - ``mc_mean_se``  : the MC standard error of the mean (||per-output SE||).
      - ``mc_noise_z``  : ||cp - mc|| / mc_mean_se, i.e. how many MC standard
                          errors the cumulant-propagation mean sits away from the
                          MC mean. z <~ 1 means cp agrees with MC to within MC's
                          own sampling noise (cannot distinguish -> kprop is at
                          least as good as 64k MC). z >> 1 means a statistically
                          real bias in the kprop mean.

    Also kept for completeness: raw abs/squared errors, NMSE, and the (weaker)
    variance-normalized error.
    """
    cp = np.asarray(cp_mean, dtype=np.float64).reshape(-1)
    mc = np.asarray(mc_mean, dtype=np.float64).reshape(-1)
    if cp.shape != mc.shape:
        raise ValueError(f"cp_mean shape {cp.shape} != mc_mean shape {mc.shape}")

    diff = cp - mc
    diff_norm = float(np.linalg.norm(diff))        # ||diff||
    sq_err_sum = float(np.sum(diff ** 2))          # ||diff||^2
    mean_abs_error = float(np.mean(np.abs(diff)))  # scalar => |diff|
    mean_squared_error = float(np.mean(diff ** 2)) # scalar => diff^2
    mc_norm = float(np.linalg.norm(mc))
    mc_norm_sq = mc_norm ** 2
    second_moment = float(mc_stats.get("empirical_output_second_moment", mc_norm_sq))

    if cp.size == 1:
        mean_error = float(diff[0])  # signed error for scalar output
    else:
        mean_error = diff_norm       # L2 distance for vector output

    # MC standard error of the mean (||per-output SE||).
    mc_stderr = np.asarray(mc_stats.get("mc_stderr", np.zeros_like(mc)), dtype=np.float64).reshape(-1)
    mc_mean_se = float(np.linalg.norm(mc_stderr))

    nmse_mean = sq_err_sum / (mc_norm_sq + eps)
    relative_error_mean = diff_norm / (mc_norm + eps)          # = sqrt(NMSE); HEADLINE
    mc_noise_z = diff_norm / (mc_mean_se + eps)                # error in MC-sigma units
    variance_normalized_mean_error = sq_err_sum / (second_moment + eps)  # kept, secondary

    return {
        "mean_error": mean_error,
        "mean_abs_error": mean_abs_error,
        "mean_squared_error": mean_squared_error,
        "mean_l2_error": diff_norm,
        "relative_error_mean": relative_error_mean,
        "mc_mean_se": mc_mean_se,
        "mc_noise_z": mc_noise_z,
        "nmse_mean": nmse_mean,
        "variance_normalized_mean_error": variance_normalized_mean_error,
    }
