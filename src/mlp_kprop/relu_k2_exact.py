"""Exact closed-form ReLU moment propagation for K=2 (Gaussian covariance).

This is the "true" / exact alternative to the general harmonic / power-cumulant
``nonlin_kprop`` machinery, specialised to the K=2 case (track only the mean and
the full covariance) with a ReLU nonlinearity.

For each preactivation coordinate ``Y_i ~ N(mu_i, var_i)`` with
``var_i = Sigma_ii``, ``sigma_i = sqrt(var_i)`` and ``alpha_i = mu_i / sigma_i``,
and ReLU ``phi(y) = max(y, 0)``, we use the exact scalar Gaussian moments::

    Phi = standard normal CDF        phi = standard normal PDF

    a_i   = E[ReLU(Y_i)]   = mu_i * Phi(alpha_i) + sigma_i * phi(alpha_i)
    b_i   = E[ReLU(Y_i)^2] = (mu_i^2 + var_i) * Phi(alpha_i)
                             + mu_i * sigma_i * phi(alpha_i)
    var_i = Var[ReLU(Y_i)] = b_i - a_i^2
    c_i   = Phi(alpha_i)              # 1st Hermite coef / covariance gain

The K=2 state then updates as::

    new_mu        = a
    new_Sigma     = Sigma * outer(c, c)   # off-diagonals carry the c_i c_j gain
    diag(new_Sigma) = var                 # diagonals get the EXACT ReLU variance

The per-coordinate marginals (mean and variance) are therefore exact; the
off-diagonal covariance uses the first-order gain ``c_i = Phi(alpha_i)
= E[ReLU'(Y_i)]`` -- the same leading coefficient ``relu_wick_coef(.., k=1)``
that the general algorithm uses for the degree-2 cross term.

Zero / near-zero variance is handled deterministically (``Y_i`` is a point mass
at ``mu_i``)::

    a_i   = max(mu_i, 0)
    b_i   = a_i^2
    var_i = 0
    c_i   = 1 if mu_i > 0, 0 if mu_i < 0, 0.5 if mu_i == 0  (symmetric derivative)

Everything is pure torch (no scipy, no sampling, no quadrature), preserves the
input dtype/device, and is differentiable except inside the explicit
zero-variance branch.
"""

from __future__ import annotations

import torch
from torch import Tensor

# norm_pdf/norm_cdf are pure torch helpers (norm_cdf uses torch.special.ndtr).
# Importing them keeps a single source of truth for the Gaussian pdf/cdf and is
# free of import cycles (wick.py has no intra-package imports).
from src.mlp_kprop.wick import norm_cdf, norm_pdf

# Variances at or below this (after clamping roundoff) are treated as exactly
# zero (deterministic coordinate). Keeps alpha = mu/sigma finite and matches the
# float64 scale the repo runs kprop in.
_DEFAULT_VAR_EPS = 1e-12
# A variance more negative than -_NEG_RTOL * max(1, max|var|) is considered a
# real error (not roundoff) and raises.
_NEG_RTOL = 1e-8


def relu_gaussian_moments(
    mu: Tensor,
    var: Tensor,
    *,
    var_eps: float = _DEFAULT_VAR_EPS,
    neg_rtol: float = _NEG_RTOL,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Exact scalar Gaussian-ReLU moments for ``Y ~ N(mu, var)`` (elementwise).

    Args:
        mu:  preactivation means, shape ``(n,)`` (any broadcastable shape works).
        var: preactivation variances, same shape as ``mu``.
        var_eps: variances ``<= var_eps`` are treated as deterministic.
        neg_rtol: tolerance for snapping tiny negative variances to zero; a
            variance below ``-neg_rtol * max(1, max|var|)`` raises ``ValueError``.

    Returns:
        ``(a, b, diag_var, c)`` where ``a = E[ReLU]``, ``b = E[ReLU^2]``,
        ``diag_var = Var[ReLU] = b - a^2`` and ``c = Phi(alpha)`` is the
        covariance gain. All four match the shape/dtype/device of ``mu``.
    """
    mu = torch.as_tensor(mu)
    var = torch.as_tensor(var, dtype=mu.dtype, device=mu.device)
    if mu.shape != var.shape:
        raise ValueError(f"mu and var must have the same shape, got {tuple(mu.shape)} and {tuple(var.shape)}")

    # --- Validate / clamp variance ------------------------------------------
    # Snap tiny negative roundoff to zero, but fail loudly on a meaningfully
    # negative variance (which would indicate a real upstream bug).
    var_det = var.detach()
    vmin = float(var_det.min()) if var_det.numel() else 0.0
    if vmin < 0.0:
        scale = max(1.0, float(var_det.abs().max()))
        if vmin < -neg_rtol * scale:
            raise ValueError(
                f"relu_gaussian_moments got a meaningfully negative variance: "
                f"min(var)={vmin:.3e} (scale={scale:.3e}, tol={-neg_rtol * scale:.3e}). "
                "This is not roundoff; check the upstream covariance."
            )
    var = var.clamp(min=0.0)

    zero_mask = var <= var_eps
    # Use a "safe" variance of 1 for the deterministic coordinates so the
    # stochastic branch stays finite (no div-by-zero / NaN gradients); those
    # entries are overwritten by the deterministic branch below.
    safe_var = torch.where(zero_mask, torch.ones_like(var), var)
    sigma = safe_var.sqrt()
    alpha = mu / sigma

    Phi = norm_cdf(alpha)
    phi = norm_pdf(alpha)

    # Stochastic (var > 0) closed form.
    a_stoch = mu * Phi + sigma * phi
    b_stoch = (mu * mu + var) * Phi + mu * sigma * phi
    c_stoch = Phi

    # Deterministic (var ~ 0) branch: Y is a point mass at mu.
    a_det = mu.clamp(min=0.0)  # max(mu, 0)
    b_det = a_det * a_det
    c_det = torch.where(
        mu > 0,
        torch.ones_like(mu),
        torch.where(mu < 0, torch.zeros_like(mu), torch.full_like(mu, 0.5)),
    )

    a = torch.where(zero_mask, a_det, a_stoch)
    b = torch.where(zero_mask, b_det, b_stoch)
    c = torch.where(zero_mask, c_det, c_stoch)
    # Var[ReLU] >= 0 exactly; clamp only removes roundoff (zero entries -> 0).
    diag_var = (b - a * a).clamp(min=0.0)
    return a, b, diag_var, c


def relu_k2_covariance_update(
    mu: Tensor,
    Sigma: Tensor,
    **moment_kwargs,
) -> tuple[Tensor, Tensor]:
    """Propagate ``(mu, Sigma)`` through ReLU in closed form (K=2).

    Args:
        mu:    mean vector, shape ``(n,)``.
        Sigma: covariance matrix, shape ``(n, n)``.
        **moment_kwargs: forwarded to :func:`relu_gaussian_moments`
            (``var_eps``, ``neg_rtol``).

    Returns:
        ``(new_mu, new_Sigma)`` with ``new_mu = a`` (shape ``(n,)``) and
        ``new_Sigma`` (shape ``(n, n)``) whose off-diagonals are
        ``Sigma_ij * c_i * c_j`` and whose diagonal is the exact ReLU marginal
        variance ``Var[ReLU(Y_i)]``.
    """
    mu = torch.as_tensor(mu)
    Sigma = torch.as_tensor(Sigma, dtype=mu.dtype, device=mu.device)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError(f"Sigma must be a square matrix, got shape {tuple(Sigma.shape)}")
    if mu.ndim != 1 or mu.shape[0] != Sigma.shape[0]:
        raise ValueError(
            f"mu must have shape (n,) matching Sigma; got mu {tuple(mu.shape)}, Sigma {tuple(Sigma.shape)}"
        )

    var = torch.diagonal(Sigma)  # read-only view of the diagonal
    a, _b, diag_var, c = relu_gaussian_moments(mu, var, **moment_kwargs)

    new_mu = a
    # Off-diagonals: Sigma_ij * c_i * c_j. Diagonal: exact ReLU variance.
    n = Sigma.shape[0]
    eye = torch.eye(n, dtype=Sigma.dtype, device=Sigma.device)
    new_Sigma = (Sigma * torch.outer(c, c)) * (1.0 - eye) + torch.diag(diag_var)
    return new_mu, new_Sigma


def relu_k2_exact_kprop(K_in: "dict[int, object]") -> "dict[int, object]":
    """Exact closed-form ReLU step over an ``HTower``; drop-in for ``nonlin_kprop``.

    Expects the K=2 input tower ``{1: <mean HTensor>, 2: <covariance HTensor>}``
    with radial index ``r == 0`` on both (the standard k_max=2 layout, where the
    degree-d cumulant value equals ``to_tensor()``). Returns an ``HTower`` with
    identity metric, matching the output contract of ``nonlin_kprop``.

    The output HTensors are built with ``type(K_in[1])`` rather than an imported
    ``HTensor`` class, so they live in the same module namespace as the caller's
    tower (the repo is importable as both ``src.mlp_kprop`` and ``mlp_kprop``).
    """
    K1 = K_in[1]
    K2 = K_in[2]
    if getattr(K1, "r", 0) != 0 or getattr(K2, "r", 0) != 0:
        raise AssertionError(
            "relu_k2_exact_kprop expects radial index r == 0 on the K=2 mean and "
            f"covariance HTensors (got r1={getattr(K1, 'r', None)}, r2={getattr(K2, 'r', None)})."
        )

    mu = K1.to_tensor()        # shape (n,)
    Sigma = K2.to_tensor()     # shape (n, n)
    new_mu, new_Sigma = relu_k2_covariance_update(mu, Sigma)

    HTensorCls = type(K1)
    n = K1.n
    return {
        1: HTensorCls(new_mu, r=0, n=n),
        2: HTensorCls(new_Sigma, r=0, n=n),
    }
