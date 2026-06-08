"""Tests for the exact closed-form ReLU K=2 covariance propagation.

Covers the scalar Gaussian-ReLU moments, the deterministic zero-variance branch,
the full covariance update, an optional Monte-Carlo cross-check, and that the
path is (a) strictly gated to ReLU + k_max==2 inside the kprop machinery and
(b) differentiable.
"""

import math

import pytest
import torch

from mlp_kprop.relu_k2_exact import (
    relu_gaussian_moments,
    relu_k2_covariance_update,
    relu_k2_exact_kprop,
)

torch.set_default_dtype(torch.float64)
ATOL = 1e-12


# ---------------------------------------------------------------------------
# 1. Exact known values for Y ~ N(0, 1)
# ---------------------------------------------------------------------------
def test_standard_normal_known_moments():
    mu = torch.zeros(1)
    var = torch.ones(1)
    a, b, diag_var, c = relu_gaussian_moments(mu, var)

    assert torch.allclose(a, torch.tensor([1.0 / math.sqrt(2.0 * math.pi)]), atol=ATOL)
    assert torch.allclose(b, torch.tensor([0.5]), atol=ATOL)
    assert torch.allclose(diag_var, torch.tensor([0.5 - 1.0 / (2.0 * math.pi)]), atol=ATOL)
    assert torch.allclose(c, torch.tensor([0.5]), atol=ATOL)
    # diag_var must equal b - a^2 exactly.
    assert torch.allclose(diag_var, b - a * a, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. Deterministic zero-variance behaviour
# ---------------------------------------------------------------------------
def test_zero_variance_deterministic():
    mu = torch.tensor([-2.0, 0.0, 3.0])
    var = torch.zeros(3)
    a, b, diag_var, c = relu_gaussian_moments(mu, var)

    assert torch.allclose(a, torch.tensor([0.0, 0.0, 3.0]), atol=ATOL)
    assert torch.allclose(b, torch.tensor([0.0, 0.0, 9.0]), atol=ATOL)
    assert torch.allclose(diag_var, torch.tensor([0.0, 0.0, 0.0]), atol=ATOL)
    # gain: 0 if mu<0, 0.5 if mu==0 (symmetric derivative), 1 if mu>0.
    assert torch.allclose(c, torch.tensor([0.0, 0.5, 1.0]), atol=ATOL)


def test_mixed_zero_and_positive_variance():
    # A deterministic coordinate alongside a stochastic one must not corrupt the
    # stochastic one (and vice versa).
    mu = torch.tensor([0.0, 5.0])
    var = torch.tensor([1.0, 0.0])
    a, b, diag_var, c = relu_gaussian_moments(mu, var)
    # coord 0: standard normal; coord 1: deterministic point mass at 5.
    assert torch.allclose(a, torch.tensor([1.0 / math.sqrt(2.0 * math.pi), 5.0]), atol=ATOL)
    assert torch.allclose(diag_var, torch.tensor([0.5 - 1.0 / (2.0 * math.pi), 0.0]), atol=ATOL)
    assert torch.allclose(c, torch.tensor([0.5, 1.0]), atol=ATOL)


# ---------------------------------------------------------------------------
# 3. Covariance update structure
# ---------------------------------------------------------------------------
def test_covariance_update_structure():
    torch.manual_seed(0)
    n = 6
    mu = torch.randn(n)
    A = torch.randn(n, n)
    Sigma = A @ A.T + 0.5 * torch.eye(n)  # SPD covariance

    new_mu, new_Sigma = relu_k2_covariance_update(mu, Sigma)
    a, b, diag_var, c = relu_gaussian_moments(mu, torch.diagonal(Sigma))

    # mean update
    assert torch.allclose(new_mu, a, atol=ATOL)
    # symmetry preserved
    assert torch.allclose(new_Sigma, new_Sigma.T, atol=ATOL)
    # off-diagonals scaled by the c_i c_j gain
    for i in range(n):
        for j in range(n):
            if i != j:
                assert torch.allclose(new_Sigma[i, j], Sigma[i, j] * c[i] * c[j], atol=ATOL)
    # diagonals are the EXACT ReLU marginal variances
    assert torch.allclose(torch.diagonal(new_Sigma), diag_var, atol=ATOL)
    # explicit full-matrix form: off-diag(Sigma .* cc) + diag(diag_var)
    eye = torch.eye(n)
    expected = (Sigma * torch.outer(c, c)) * (1 - eye) + torch.diag(diag_var)
    assert torch.allclose(new_Sigma, expected, atol=ATOL)


def test_dtype_device_preserved():
    mu = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float32)
    Sigma = torch.eye(3, dtype=torch.float32)
    new_mu, new_Sigma = relu_k2_covariance_update(mu, Sigma)
    assert new_mu.dtype == torch.float32
    assert new_Sigma.dtype == torch.float32
    assert new_mu.device == mu.device and new_Sigma.device == Sigma.device


def test_negative_variance_roundoff_vs_real():
    mu = torch.zeros(3)
    # tiny roundoff negative -> clamped, no raise
    var_round = torch.tensor([1.0, -1e-15, 2.0])
    relu_gaussian_moments(mu, var_round)  # should not raise
    # meaningfully negative -> raise
    var_bad = torch.tensor([1.0, -0.5, 2.0])
    with pytest.raises(ValueError):
        relu_gaussian_moments(mu, var_bad)


# ---------------------------------------------------------------------------
# 4. Optional Monte-Carlo cross-check (loose; modest sample count, not slow)
# ---------------------------------------------------------------------------
def test_monte_carlo_sanity():
    # The closed form is EXACT for the marginals (mean, diagonal variance) and a
    # first-order approximation for the off-diagonal covariance (c_i c_j Sigma_ij),
    # so we use a modest equicorrelation Sigma: marginals checked tightly,
    # off-diagonals loosely. Not a slow CI default (single 1M-sample draw).
    torch.manual_seed(0)
    n = 4
    mu = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    rho = 0.2
    Sigma = torch.full((n, n), rho) + (1.0 - rho) * torch.eye(n)  # SPD, unit variances

    new_mu, new_Sigma = relu_k2_covariance_update(mu, Sigma)

    N = 1_000_000
    L = torch.linalg.cholesky(Sigma)
    relu = (mu + torch.randn(N, n) @ L.T).clamp(min=0.0)
    mc_mean = relu.mean(0)
    mc_cov = torch.cov(relu.T)

    se_mean = relu.std(0) / math.sqrt(N)
    # mean: exact, within MC noise
    assert torch.all((new_mu - mc_mean).abs() <= 6.0 * se_mean + 1e-4)
    # diagonal variance: exact, tight tolerance
    assert torch.allclose(torch.diagonal(new_Sigma), torch.diagonal(mc_cov), atol=5e-3)
    # off-diagonal: first-order approximation, looser tolerance
    eye = torch.eye(n)
    assert torch.all(((new_Sigma - mc_cov) * (1 - eye)).abs() <= 1.5e-2)


# ---------------------------------------------------------------------------
# Integration with the kprop machinery: gating + correctness
# ---------------------------------------------------------------------------
from mlp_kprop import kprop_harmonic as kprop
from mlp_kprop.kprop_harmonic import Kind, coerce_input as harmonic_coerce_input
from mlp_kprop.mlp import MLP
from mlp_kprop.wick import relu_wick_coef, norm_cdf


def _k2_input_after_linear(n=12, seed=0):
    torch.manual_seed(seed)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    K_in = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=2, kind=Kind.SIMPLE)
    K_z1 = kprop.linear_kprop(K_in, mlp.Ws[0].weight.data, k_max=2, set_metric=mlp.init_scale[0])
    return K_z1


def test_relu_kprop_exact_matches_closed_form():
    K_z1 = _k2_input_after_linear()
    mu = K_z1[1].to_tensor()
    Sigma = K_z1[2].to_tensor()

    K_out = kprop.relu_kprop(K_z1, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=True)
    a, b, diag_var, c = relu_gaussian_moments(mu, torch.diagonal(Sigma))

    # mean cumulant == a
    assert torch.allclose(K_out[1].to_tensor(), a, atol=1e-10)
    # covariance: off-diagonal gain c_i c_j, exact diagonal variance
    Sig_out = K_out[2].to_tensor()
    eye = torch.eye(Sigma.shape[0])
    expected = (Sigma * torch.outer(c, c)) * (1 - eye) + torch.diag(diag_var)
    assert torch.allclose(Sig_out, expected, atol=1e-10)
    # gain equals Phi(alpha) == relu_wick_coef k=1
    alpha = mu / torch.diagonal(Sigma).sqrt()
    assert torch.allclose(c, norm_cdf(alpha), atol=ATOL)
    assert torch.allclose(c, relu_wick_coef(mu, torch.diagonal(Sigma), k=1), atol=ATOL)
    # output towers have identity metric (nonlin_kprop contract) and r==0
    assert K_out[1].r == 0 and K_out[2].r == 0
    assert K_out[1].has_identity_metric() and K_out[2].has_identity_metric()


def test_exact_flag_noop_for_non_relu():
    # The exact flag must be a no-op for a non-ReLU activation, even at k_max==2:
    # the square-activation output is identical whether the flag is set or not.
    K_z1 = _k2_input_after_linear()
    sq_coef = kprop.WICK_COEF_D["square"]
    out_off = kprop.nonlin_kprop(K_z1, nonlin_wick_coef=sq_coef, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=False)
    out_on = kprop.nonlin_kprop(K_z1, nonlin_wick_coef=sq_coef, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=True)
    assert out_off.keys() == out_on.keys()
    for d in out_off:
        assert torch.allclose(out_off[d].to_tensor(), out_on[d].to_tensor(), atol=ATOL)


def test_exact_and_approx_relu_share_mean_at_k2():
    # The two runnable versions agree on the degree-1 (mean) cumulant at k_max==2
    # ReLU; they may differ on the covariance (that is the point of the toggle).
    K_z1 = _k2_input_after_linear()
    out_approx = kprop.relu_kprop(K_z1, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=False)
    out_exact = kprop.relu_kprop(K_z1, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=True)
    assert torch.allclose(out_approx[1].to_tensor(), out_exact[1].to_tensor(), atol=1e-9)


def test_exact_flag_noop_at_k3():
    # At k_max==3 the exact flag must NOT engage (k_max gate); output is identical
    # whether the flag is set or not.
    torch.manual_seed(1)
    n = 10
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    K_in = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE)
    K_z1 = kprop.linear_kprop(K_in, mlp.Ws[0].weight.data, k_max=3, set_metric=mlp.init_scale[0])
    off = kprop.relu_kprop(K_z1, k_max=3, kind=Kind.SIMPLE, exact_relu_k2=False)
    on = kprop.relu_kprop(K_z1, k_max=3, kind=Kind.SIMPLE, exact_relu_k2=True)
    assert off.keys() == on.keys()
    for d in off:
        assert torch.allclose(off[d].to_tensor(), on[d].to_tensor(), atol=ATOL)


def test_mlp_kprop_exact_single_hidden_layer_is_exact():
    # Single hidden layer (num_layers=2): the preactivation is EXACTLY Gaussian
    # (linear image of N(0, I)), so E[ReLU] is exact and the output mean
    # W1 @ E[ReLU] is the exact E[f(X)] -- it must match MC within sampling noise.
    torch.manual_seed(0)
    n = 32
    N = 1_000_000
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    K_out = kprop.mlp_kprop(mlp, K_in, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=True)
    cp_mean = K_out[1].to_tensor().item()

    x = torch.randn(N, n)
    z = mlp(x).out
    mc_mean = z.mean(0).item()
    stderr = z.std(0).item() / math.sqrt(N)
    assert abs(cp_mean - mc_mean) < 6.0 * stderr


def test_mlp_kprop_exact_runs_deep_and_is_finite():
    # Deeper net (2 hidden layers): the exact-K=2 path is exact only PER LAYER --
    # the first-order off-diagonal covariance gain feeds the next layer's
    # variance, so the end-to-end mean is approximate (this is the regime where
    # the "true" and "propagation" versions are expected to diverge). Here we only
    # assert it runs and returns a finite, correctly-shaped result.
    torch.manual_seed(0)
    n = 16
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=3)
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    K_out = kprop.mlp_kprop(mlp, K_in, k_max=2, kind=Kind.SIMPLE, exact_relu_k2=True)
    assert 1 in K_out
    assert torch.isfinite(K_out[1].to_tensor()).all()
    assert K_out[1].to_tensor().numel() == 1


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------
def test_differentiable_through_moments():
    # enable_grad locally: other test modules set torch.set_grad_enabled(False)
    # at import time, and that global persists across the session.
    with torch.enable_grad():
        mu = torch.tensor([0.0, 1.0, -0.5], requires_grad=True)
        var = torch.tensor([1.0, 2.0, 0.7], requires_grad=True)
        a, b, diag_var, c = relu_gaussian_moments(mu, var)
        (a.sum() + diag_var.sum() + c.sum()).backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert var.grad is not None and torch.isfinite(var.grad).all()


def test_no_nan_grad_with_zero_variance_entry():
    # Zero-variance entry must not poison gradients of the stochastic entries.
    with torch.enable_grad():
        mu = torch.tensor([0.3, 2.0], requires_grad=True)
        var = torch.tensor([1.0, 0.0], requires_grad=True)
        a, b, diag_var, c = relu_gaussian_moments(mu, var)
        a.sum().backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(var.grad).all()
