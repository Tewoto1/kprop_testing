"""EXACT bivariate-Gaussian ReLU covariance propagation for K=2.

This is the *true* exact K=2 ReLU activation step. The general harmonic /
power-cumulant ``nonlin_kprop`` path (the default propagation algorithm) is exact
on the ReLU marginals but uses the leading-order Hermite off-diagonal gain
``Sigma_ij <- Sigma_ij * c_i * c_j`` with ``c_i = Phi(mu_i/sigma_i)`` -- i.e. an
APPROXIMATION of the off-diagonal covariance. This module instead computes the
**exact** pairwise covariance of ``(ReLU(Z_i), ReLU(Z_j))`` under the joint
Gaussian ``Z ~ N(mu, Sigma)`` using the closed-form bivariate-normal moments.

Selectable in experiments as the "exact_relu_covariance" / "k2_exact_bivariate_relu"
/ "exact_gaussian_relu_k2" path (config flag ``exact_relu_cov``).

Math
----
Diagonal (exact univariate, ``Y ~ N(mu, var)``, ``sigma=sqrt(var)``,
``alpha=mu/sigma``)::

    E[ReLU(Y)]   = mu*Phi(alpha) + sigma*phi(alpha)
    E[ReLU(Y)^2] = (mu^2 + var)*Phi(alpha) + mu*sigma*phi(alpha)
    Var[ReLU(Y)] = E[ReLU(Y)^2] - E[ReLU(Y)]^2

Off-diagonal (i != j): with ``rho = Sigma_ij/(sigma_i sigma_j)``,
``s = sqrt(1-rho^2)``, ``phi``/``Phi`` the standard normal pdf/cdf, ``Phi2``/``phi2``
the bivariate standard normal cdf/pdf with correlation ``rho``::

    P    = Phi2(alpha_i, alpha_j; rho)
    A    = phi(alpha_i) * Phi((alpha_j - rho*alpha_i)/s)
    B    = phi(alpha_j) * Phi((alpha_i - rho*alpha_j)/s)
    D    = phi2(alpha_i, alpha_j; rho)
    M_i  = A + rho*B
    M_j  = B + rho*A
    M_ij = rho*P + (1-rho^2)*D - rho*alpha_i*A - rho*alpha_j*B

    E[ReLU(Z_i) ReLU(Z_j)] = mu_i*mu_j*P + mu_i*sigma_j*M_j
                             + mu_j*sigma_i*M_i + sigma_i*sigma_j*M_ij
    new_Sigma_ij = E[ReLU(Z_i) ReLU(Z_j)] - new_mu_i*new_mu_j

The off-diagonal is the EXACT covariance -- there is NO ``Sigma * outer(gain, gain)``
anywhere in this path.

Numerics / implementation notes
-------------------------------
* ``Phi2`` is computed exactly via Owen's T function (``scipy.special.owens_t``),
  vectorized over the whole pair matrix and validated to ~1e-16 against a
  high-precision reference. **This path depends on scipy** (allowed for this
  experiment branch).
* This is a NumPy/SciPy implementation and is **not autograd-differentiable**.
  The torch entry points detach to CPU/NumPy, compute, and return tensors on the
  original device/dtype. Run it under ``torch.no_grad()``.
* Degenerate variance (``var_i <= var_eps``): ``Z_i`` is treated as the point mass
  ``mu_i`` -- ``E[ReLU]=max(mu_i,0)``, ``Var=0`` -- and its covariance with every
  other coordinate is exactly 0 (a constant is uncorrelated with anything).
* ``|rho| == 1`` (within ``rho_tol``): the singular bivariate CDF is avoided; the
  pair reduces to a one-dimensional limit (comonotone / antimonotone) in closed
  form.
* Materially negative variances or correlations outside ``[-1, 1]`` (beyond a
  small tolerance) raise ``ValueError``; only tiny numerical roundoff is clipped.
* The output covariance is symmetrized and its diagonal is overwritten with the
  exact univariate ReLU variances.
"""

from __future__ import annotations

import math

import numpy as np

try:  # scipy is required for the exact bivariate normal CDF (Owen's T).
    from scipy.special import ndtr as _ndtr  # standard normal CDF
    from scipy.special import owens_t as _owens_t
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "exact_relu_covariance requires scipy (scipy.special.owens_t and ndtr). "
        "Install scipy, or run the default (approximate) propagation path with "
        "exact_relu_cov=False."
    ) from exc

import torch
from torch import Tensor

# Variances at or below this (after clamping roundoff) are deterministic.
_DEFAULT_VAR_EPS = 1e-12
# A variance below -_NEG_RTOL * max(1, max|var|) is a real error (not roundoff).
_NEG_RTOL = 1e-8
# |rho| within _RHO_TOL of 1 -> use the perfectly-correlated 1-D limit.
_RHO_TOL = 1e-7
# |rho| beyond 1 + _RHO_VALID_TOL is treated as an invalid input (raise).
_RHO_VALID_TOL = 1e-6

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_INV_2PI = 1.0 / (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Standard / bivariate normal helpers (NumPy, vectorized)
# ---------------------------------------------------------------------------
def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _Phi(x: np.ndarray) -> np.ndarray:
    return _ndtr(x)


def _bvn_pdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate normal PDF phi2(h, k; rho) (rho with |rho| < 1)."""
    s2 = 1.0 - rho * rho
    return np.exp(-(h * h - 2.0 * rho * h * k + k * k) / (2.0 * s2)) / (2.0 * math.pi * np.sqrt(s2))


def bvn_cdf(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard bivariate normal CDF P(X <= h, Y <= k), correlation ``rho``.

    Exact, vectorized, via Owen's T. Validated to ~1e-16 against a high-precision
    correlation-integral reference. Expects ``|rho| <= 1``; callers that may pass
    ``|rho| == 1`` should clip first (the singular case is handled by the 1-D
    limit in the covariance routine, not here).
    """
    h = np.asarray(h, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    h, k, rho = np.broadcast_arrays(h, k, rho)
    s = np.sqrt(np.clip(1.0 - rho * rho, 0.0, None))

    def _a(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        # num/den, with the directional limit sign(num)*inf when den == 0
        # (Owen's T handles +/-inf: T(h, +/-inf) = +/- 0.5*Phi(-|h|)).
        safe_den = np.where(den != 0.0, den, 1.0)
        return np.where(den != 0.0, num / safe_den, np.where(num >= 0.0, np.inf, -np.inf))

    Th = _owens_t(h, _a(k - rho * h, h * s))
    Tk = _owens_t(k, _a(h - rho * k, k * s))
    hk = h * k
    c = np.where((hk > 0.0) | ((hk == 0.0) & (h + k >= 0.0)), 0.0, 0.5)
    P = 0.5 * _Phi(h) + 0.5 * _Phi(k) - Th - Tk - c

    # Exact value at h == k == 0 (the Owen's-T a-arguments are 0/0 there):
    #   Phi2(0, 0; rho) = 1/4 + arcsin(rho) / (2 pi).
    both_zero = (h == 0.0) & (k == 0.0)
    if both_zero.any():
        P = np.where(both_zero, 0.25 + np.arcsin(np.clip(rho, -1.0, 1.0)) * _INV_2PI, P)
    return np.clip(P, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Univariate exact ReLU moments
# ---------------------------------------------------------------------------
def relu_moments_1d_np(
    mu: np.ndarray, var: np.ndarray, *, var_eps: float = _DEFAULT_VAR_EPS, neg_rtol: float = _NEG_RTOL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact univariate Gaussian-ReLU moments (mean, second moment, variance).

    Deterministic (``var <= var_eps``) coordinates collapse to the point mass
    ``mu``: ``mean = max(mu, 0)``, ``second = mean^2``, ``variance = 0``.
    """
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)

    vmin = float(var.min()) if var.size else 0.0
    if vmin < 0.0:
        scale = max(1.0, float(np.abs(var).max()))
        if vmin < -neg_rtol * scale:
            raise ValueError(
                f"exact_relu_covariance got a meaningfully negative variance: min(var)={vmin:.3e} "
                f"(scale={scale:.3e}, tol={-neg_rtol * scale:.3e}). Check the upstream covariance."
            )
    var = np.clip(var, 0.0, None)

    det = var <= var_eps
    sigma = np.sqrt(var)
    safe_sigma = np.where(det, 1.0, sigma)  # avoid 0/0 in alpha; det entries overwritten
    alpha = mu / safe_sigma

    Phi = _Phi(alpha)
    phi = _phi(alpha)
    mean_stoch = mu * Phi + sigma * phi
    second_stoch = (mu * mu + var) * Phi + mu * sigma * phi

    mean_det = np.maximum(mu, 0.0)
    mean = np.where(det, mean_det, mean_stoch)
    second = np.where(det, mean_det * mean_det, second_stoch)
    variance = np.where(det, 0.0, np.clip(second - mean * mean, 0.0, None))
    return mean, second, variance


# ---------------------------------------------------------------------------
# Exact bivariate ReLU cross-moments and the K=2 covariance update
# ---------------------------------------------------------------------------
def _relu_cross_moment_perfect_corr(
    mu: np.ndarray, sigma: np.ndarray, alpha: np.ndarray, sign: float
) -> np.ndarray:
    """E[ReLU(Z_i) ReLU(Z_j)] for the perfectly (anti)correlated limit rho = sign.

    ``sign = +1`` (comonotone): Z_j = mu_j + (sigma_j/sigma_i)(Z_i - mu_i).
    ``sign = -1`` (antimonotone). Both reduce to a 1-D Gaussian integral with a
    closed form in terms of phi/Phi. Inputs are (n, n) broadcast matrices
    (MUi/MUj, SI/SJ, Ai/Aj built by the caller); returns the (n, n) cross-moment.
    """
    MUi = mu[:, None]
    MUj = mu[None, :]
    SI = sigma[:, None]
    SJ = sigma[None, :]
    Ai = alpha[:, None]
    Aj = alpha[None, :]
    if sign > 0:
        # Both > 0 iff U > max(-Ai, -Aj) = -min(Ai, Aj). Integrate on [t, inf).
        t = -np.minimum(Ai, Aj)
        I0 = _Phi(-t)
        I1 = _phi(t)
        I2 = t * _phi(t) + _Phi(-t)
        return SI * SJ * I2 + (SI * MUj + SJ * MUi) * I1 + MUi * MUj * I0
    # antimonotone: Z_j = mu_j - (sigma_j/sigma_i)(Z_i - mu_i).
    # Both > 0 iff -Ai < U < Aj; empty if -Ai >= Aj.
    lo = -Ai
    hi = Aj
    valid = hi > lo
    Phi_hi, Phi_lo = _Phi(hi), _Phi(lo)
    phi_hi, phi_lo = _phi(hi), _phi(lo)
    J0 = Phi_hi - Phi_lo
    J1 = phi_lo - phi_hi
    J2 = (lo * phi_lo - hi * phi_hi) + (Phi_hi - Phi_lo)
    out = -SI * SJ * J2 + (SI * MUj - SJ * MUi) * J1 + MUi * MUj * J0
    return np.where(valid, out, 0.0)


def exact_relu_covariance_np(
    mu: np.ndarray,
    Sigma: np.ndarray,
    *,
    var_eps: float = _DEFAULT_VAR_EPS,
    neg_rtol: float = _NEG_RTOL,
    rho_tol: float = _RHO_TOL,
    rho_valid_tol: float = _RHO_VALID_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact K=2 ReLU propagation of ``(mu, Sigma)`` (NumPy core).

    Returns ``(new_mu, new_Sigma)`` where ``new_mu_i = E[ReLU(Z_i)]`` and
    ``new_Sigma_ij = Cov(ReLU(Z_i), ReLU(Z_j))`` under ``Z ~ N(mu, Sigma)``,
    computed with the exact bivariate-Gaussian formula (no gain approximation).
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must have shape ({n}, {n}); got {Sigma.shape}")
    
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("Sigma contains non-finite values")

    if not np.all(np.isfinite(mu)):
        raise ValueError("mu contains non-finite values")

    sym_err = float(np.max(np.abs(Sigma - Sigma.T))) if Sigma.size else 0.0
    sym_scale = max(1.0, float(np.max(np.abs(Sigma)))) if Sigma.size else 1.0
    if sym_err > 1e-8 * sym_scale:
        raise ValueError(
            f"Sigma must be symmetric for exact ReLU covariance propagation; "
            f"max |Sigma - Sigma.T| = {sym_err:.3e}, scale={sym_scale:.3e}."
        )

    # Only remove tiny asymmetry from roundoff.
    Sigma = 0.5 * (Sigma + Sigma.T)

    var = np.diag(Sigma).copy()
    new_mu, second, diag_var = relu_moments_1d_np(mu, var, var_eps=var_eps, neg_rtol=neg_rtol)

    det = var <= var_eps  # deterministic coordinates (matches relu_moments_1d_np)

    if det.any():
        det_cov = np.abs(Sigma[det, :]).copy()
        det_cov[:, det] = 0.0
        max_det_cov = float(det_cov.max()) if det_cov.size else 0.0
        cov_scale = max(1.0, float(np.max(np.abs(Sigma)))) if Sigma.size else 1.0

        # Allow small roundoff, but do not hide a materially inconsistent covariance.
        if max_det_cov > 1e-8 * cov_scale:
            raise ValueError(
                "Sigma has a near-deterministic coordinate with nonzero covariance "
                f"to another coordinate: max offending covariance={max_det_cov:.3e}, "
                f"scale={cov_scale:.3e}."
            )
    sigma = np.sqrt(np.clip(var, 0.0, None))
    safe_sigma = np.where(det, 1.0, sigma)
    alpha = mu / safe_sigma

    # Pairwise correlation matrix; det rows/cols get a harmless 0 placeholder.
    denom = np.outer(safe_sigma, safe_sigma)
    rho = Sigma / denom
    pair_stoch = (~det)[:, None] & (~det)[None, :]
    if pair_stoch.any():
        rho_off = rho[pair_stoch]
        bad = np.abs(rho_off) > 1.0 + rho_valid_tol
        if bad.any():
            worst = float(np.abs(rho_off).max())
            raise ValueError(
                f"exact_relu_covariance got an invalid correlation |rho|={worst:.6f} > 1 "
                f"(tol={rho_valid_tol:.1e}). Check the upstream covariance for non-PSD / scaling bugs."
            )
    rho = np.where(pair_stoch, np.clip(rho, -1.0, 1.0), 0.0)

    # General (|rho| < 1) branch -- clip strictly inside (-1, 1) to keep it finite;
    # near-+/-1 entries are overwritten by the 1-D limit below.
    rho_g = np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)
    s = np.sqrt(1.0 - rho_g * rho_g)

    Ai = alpha[:, None]
    Aj = alpha[None, :]
    SI = sigma[:, None]
    SJ = sigma[None, :]
    MUi = mu[:, None]
    MUj = mu[None, :]

    A = _phi(Ai) * _Phi((Aj - rho_g * Ai) / s)
    B = _phi(Aj) * _Phi((Ai - rho_g * Aj) / s)
    P = bvn_cdf(Ai, Aj, rho_g)
    D = _bvn_pdf(Ai, Aj, rho_g)
    M_i = A + rho_g * B
    M_j = B + rho_g * A
    M_ij = rho_g * P + (1.0 - rho_g * rho_g) * D - rho_g * Ai * A - rho_g * Aj * B
    Ecross = MUi * MUj * P + MUi * SJ * M_j + MUj * SI * M_i + SI * SJ * M_ij

    # Perfectly (anti)correlated limits (avoid the singular bivariate CDF).
    near_pos = rho >= 1.0 - rho_tol
    near_neg = rho <= -1.0 + rho_tol
    if near_pos.any():
        Ecross = np.where(near_pos, _relu_cross_moment_perfect_corr(mu, sigma, alpha, +1.0), Ecross)
    if near_neg.any():
        Ecross = np.where(near_neg, _relu_cross_moment_perfect_corr(mu, sigma, alpha, -1.0), Ecross)

    new_Sigma = Ecross - np.outer(new_mu, new_mu)
    # A deterministic coordinate is constant -> zero covariance with everything.
    det_involved = det[:, None] | det[None, :]
    new_Sigma = np.where(det_involved, 0.0, new_Sigma)
    # Symmetrize, then overwrite the diagonal with the exact univariate variances.
    new_Sigma = 0.5 * (new_Sigma + new_Sigma.T)
    np.fill_diagonal(new_Sigma, diag_var)
    return new_mu, new_Sigma


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------
def exact_relu_covariance_torch(mu: Tensor, Sigma: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
    """Torch entry point: detach to CPU/NumPy, compute exactly, return tensors.

    Preserves the input device/dtype. NOT autograd-differentiable (the math runs
    in NumPy/SciPy); intended to be called under ``torch.no_grad()``.
    """
    mu = torch.as_tensor(mu)
    Sigma = torch.as_tensor(Sigma)
    device, dtype = mu.device, mu.dtype
    mu_np = mu.detach().cpu().double().numpy()
    Sigma_np = Sigma.detach().cpu().double().numpy()
    new_mu_np, new_Sigma_np = exact_relu_covariance_np(mu_np, Sigma_np, **kwargs)
    new_mu = torch.as_tensor(new_mu_np, device=device, dtype=dtype)
    new_Sigma = torch.as_tensor(new_Sigma_np, device=device, dtype=dtype)
    return new_mu, new_Sigma


def exact_relu_covariance_kprop(K_in: "dict[int, object]") -> "dict[int, object]":
    """Exact bivariate ReLU step over an ``HTower``; drop-in for ``nonlin_kprop``.

    Expects the K=2 tower ``{1: <mean HTensor>, 2: <covariance HTensor>}`` with
    radial index ``r == 0`` on both. Returns an ``HTower`` with identity metric,
    matching ``nonlin_kprop``'s output contract. Output HTensors are built with
    ``type(K_in[1])`` so they share the caller's module namespace regardless of how
    this vendored package is imported.
    """
    K1 = K_in[1]
    K2 = K_in[2]
    if getattr(K1, "r", 0) != 0 or getattr(K2, "r", 0) != 0:
        raise AssertionError(
            "exact_relu_covariance_kprop expects radial index r == 0 on the K=2 mean and "
            f"covariance HTensors (got r1={getattr(K1, 'r', None)}, r2={getattr(K2, 'r', None)})."
        )
    mu = K1.to_tensor()      # (n,)
    Sigma = K2.to_tensor()   # (n, n)
    new_mu, new_Sigma = exact_relu_covariance_torch(mu, Sigma)

    HTensorCls = type(K1)
    n = K1.n
    return {1: HTensorCls(new_mu, r=0, n=n), 2: HTensorCls(new_Sigma, r=0, n=n)}
