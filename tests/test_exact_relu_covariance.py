"""Tests for the EXACT bivariate-Gaussian ReLU K=2 covariance propagation.

Covers the exact univariate moments, the exact pairwise covariance (independent,
zero-mean correlated, deterministic), a regression test that fails if the code
falls back to the leading-order gain ``Sigma_ij * c_i * c_j``, a Monte-Carlo
cross-check, and the kprop integration (strictly gated to ReLU + k_max==2; K>=3
untouched; precedence over the leading-order path).
"""

import math

import numpy as np
import pytest
import torch

from mlp_kprop.exact_relu_covariance import (
    bvn_cdf,
    exact_relu_covariance_kprop,
    exact_relu_covariance_np,
    exact_relu_covariance_torch,
    relu_moments_1d_np,
)

torch.set_default_dtype(torch.float64)

INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# 1. Standard normal marginal: Y ~ N(0, 1)
# ---------------------------------------------------------------------------
def test_standard_normal_marginal():
    mean, second, var = relu_moments_1d_np(np.array([0.0]), np.array([1.0]))
    assert mean[0] == pytest.approx(INV_SQRT_2PI, abs=1e-14)
    assert second[0] == pytest.approx(0.5, abs=1e-14)
    assert var[0] == pytest.approx(0.5 - 1.0 / (2.0 * math.pi), abs=1e-14)


# ---------------------------------------------------------------------------
# 2. Independent pair: Sigma_ij = 0 => Cov(ReLU_i, ReLU_j) = 0
# ---------------------------------------------------------------------------
def test_independent_pair_zero_covariance():
    mu = np.array([0.3, -0.7])
    Sigma = np.array([[1.4, 0.0], [0.0, 0.6]])
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    # E[ReLU_i ReLU_j] = E[ReLU_i] E[ReLU_j], so the covariance is exactly 0.
    assert new_Sigma[0, 1] == pytest.approx(0.0, abs=1e-13)
    assert new_Sigma[1, 0] == pytest.approx(0.0, abs=1e-13)
    # diagonal = exact univariate variances
    m, _s, v = relu_moments_1d_np(mu, np.diag(Sigma))
    assert new_mu == pytest.approx(m, abs=1e-13)
    assert np.allclose(np.diag(new_Sigma), v, atol=1e-13)


# ---------------------------------------------------------------------------
# 3. Zero-mean correlated pair: closed form
#    E[ReLU_i ReLU_j] = (sqrt(1-rho^2) + rho*(pi - arccos(rho))) / (2 pi)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.2, 0.0, 0.3, 0.6, 0.95])
def test_zero_mean_correlated_cross_moment(rho):
    mu = np.zeros(2)
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    cross = new_Sigma[0, 1] + new_mu[0] * new_mu[1]  # E[ReLU_i ReLU_j]
    expected = (math.sqrt(1 - rho**2) + rho * (math.pi - math.acos(rho))) / (2 * math.pi)
    assert cross == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. Regression: the exact off-diagonal is NOT the leading-order gain
#    Sigma_ij * Phi(alpha_i) * Phi(alpha_j). This test FAILS if the exact path
#    secretly uses the gain approximation.
# ---------------------------------------------------------------------------
def test_not_equal_to_gain_approximation():
    rho = 0.5
    mu = np.zeros(2)
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    _new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    exact_cov = new_Sigma[0, 1]

    # The wrong approximation: Sigma_ij * c_i * c_j with c = Phi(alpha) = Phi(0) = 0.5.
    gain_cov = rho * 0.5 * 0.5  # = 0.125
    assert abs(exact_cov - gain_cov) > 1e-3, "exact path must not equal the gain approximation"

    # The exact value (from the closed form): cross - mean^2.
    cross = (math.sqrt(1 - rho**2) + rho * (math.pi - math.acos(rho))) / (2 * math.pi)
    exact_expected = cross - INV_SQRT_2PI**2
    assert exact_cov == pytest.approx(exact_expected, abs=1e-12)

    # The default propagation path's off-diagonal is the leading-order gain
    # Sigma_ij * Phi(alpha_i) * Phi(alpha_j); the exact value must differ from it.
    from scipy.special import ndtr
    alpha = mu / np.sqrt(np.diag(Sigma))
    gain_offdiag = Sigma[0, 1] * ndtr(alpha[0]) * ndtr(alpha[1])
    assert abs(exact_cov - gain_offdiag) > 1e-3


# ---------------------------------------------------------------------------
# 5. Deterministic variance: mu = [-2, 0, 3], var = [0, 0, 0]
# ---------------------------------------------------------------------------
def test_deterministic_variance():
    mu = np.array([-2.0, 0.0, 3.0])
    Sigma = np.zeros((3, 3))
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    assert np.allclose(new_mu, [0.0, 0.0, 3.0], atol=1e-13)
    assert np.allclose(new_Sigma, np.zeros((3, 3)), atol=1e-13)


def test_one_deterministic_one_random_pair():
    # If Z_i is deterministic, Cov(ReLU_i, ReLU_j) = 0 and the cross moment reduces
    # to max(mu_i, 0) * E[ReLU_j].
    mu = np.array([3.0, 0.5])
    Sigma = np.array([[0.0, 0.0], [0.0, 2.0]])  # i deterministic, j ~ N(0.5, 2)
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    mj, _s, vj = relu_moments_1d_np(np.array([0.5]), np.array([2.0]))
    assert new_mu[0] == pytest.approx(3.0, abs=1e-13)
    assert new_mu[1] == pytest.approx(mj[0], abs=1e-13)
    assert new_Sigma[0, 1] == pytest.approx(0.0, abs=1e-13)
    assert new_Sigma[1, 1] == pytest.approx(vj[0], abs=1e-13)


# ---------------------------------------------------------------------------
# 6. Phi2 sanity: matches the arcsin closed form at (0, 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rho", [-0.7, -0.3, 0.0, 0.4, 0.9])
def test_bvn_cdf_zero_zero_closed_form(rho):
    got = float(bvn_cdf(np.array(0.0), np.array(0.0), np.array(rho)))
    expected = 0.25 + math.asin(rho) / (2 * math.pi)
    assert got == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 7. Monte-Carlo cross-check (loose tolerance; modest sample count).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mu,Sigma",
    [
        (np.array([0.0, 0.0]), np.array([[1.0, 0.5], [0.5, 1.0]])),
        (np.array([0.4, -0.6]), np.array([[1.3, -0.5], [-0.5, 0.8]])),
        (np.array([-1.0, 2.0]), np.array([[0.25, 0.3], [0.3, 2.25]])),
    ],
)
def test_monte_carlo_cross_check(mu, Sigma):
    rng = np.random.default_rng(0)
    N = 4_000_000
    z = rng.multivariate_normal(mu, Sigma, size=N)
    r = np.maximum(z, 0.0)
    mc_mean = r.mean(0)
    mc_cov = np.cov(r, rowvar=False, bias=True)

    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    assert np.allclose(new_mu, mc_mean, atol=5e-3)
    assert np.allclose(new_Sigma, mc_cov, atol=5e-3)


# ---------------------------------------------------------------------------
# 8. Output shape / symmetry / diagonal contract
# ---------------------------------------------------------------------------
def test_symmetry_and_diagonal():
    rng = np.random.default_rng(1)
    n = 6
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + np.eye(n)  # SPD
    mu = rng.standard_normal(n)
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    assert np.allclose(new_Sigma, new_Sigma.T, atol=1e-14)
    _m, _s, v = relu_moments_1d_np(mu, np.diag(Sigma))
    assert np.allclose(np.diag(new_Sigma), v, atol=1e-12)


# ---------------------------------------------------------------------------
# 9. Validity guards
# ---------------------------------------------------------------------------
def test_negative_variance_raises():
    with pytest.raises(ValueError):
        exact_relu_covariance_np(np.array([0.0, 0.0]), np.array([[-0.5, 0.0], [0.0, 1.0]]))


def test_invalid_correlation_raises():
    # Sigma_01 too large for the marginal variances => |rho| > 1.
    with pytest.raises(ValueError):
        exact_relu_covariance_np(np.array([0.0, 0.0]), np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_tiny_roundoff_variance_clipped():
    # A tiny negative variance from roundoff must be clipped (treated deterministic),
    # not raised; its covariances are correspondingly tiny (PSD-consistent).
    mu = np.array([0.5, -0.3])
    Sigma = np.array([[1.0, 1e-12], [1e-12, -1e-14]])
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    assert np.isfinite(new_mu).all() and np.isfinite(new_Sigma).all()
    assert new_Sigma[1, 1] == pytest.approx(0.0, abs=1e-13)  # deterministic coord -> 0 variance
    assert new_Sigma[0, 1] == pytest.approx(0.0, abs=1e-13)  # 0 covariance with a constant


def test_asymmetric_sigma_raises():
    with pytest.raises(ValueError):
        exact_relu_covariance_np(np.array([0.0, 0.0]), np.array([[1.0, 0.3], [0.1, 1.0]]))


def test_deterministic_with_nonzero_covariance_raises():
    # var_1 ~ 0 but a sizeable covariance to coord 0 is not a valid (PSD) covariance.
    with pytest.raises(ValueError):
        exact_relu_covariance_np(np.array([0.0, 0.0]), np.array([[1.0, 0.2], [0.2, 0.0]]))


# ---------------------------------------------------------------------------
# 10. Perfect correlation limit (rho = +/-1) avoids the singular bivariate CDF.
# ---------------------------------------------------------------------------
def test_perfect_correlation_limit_matches_mc():
    mu = np.array([0.2, -0.4])
    # rho = +1 exactly (rank-1 covariance).
    si, sj = 1.1, 0.7
    Sigma = np.array([[si * si, si * sj], [si * sj, sj * sj]])
    new_mu, new_Sigma = exact_relu_covariance_np(mu, Sigma)
    assert np.isfinite(new_Sigma).all()
    rng = np.random.default_rng(2)
    N = 4_000_000
    u = rng.standard_normal(N)
    zi = mu[0] + si * u
    zj = mu[1] + sj * u
    r = np.maximum(np.stack([zi, zj], 1), 0.0)
    mc_cov = np.cov(r, rowvar=False, bias=True)
    assert new_Sigma[0, 1] == pytest.approx(mc_cov[0, 1], abs=5e-3)


# ---------------------------------------------------------------------------
# 11. Torch wrapper: preserves dtype/device, returns symmetric covariance.
# ---------------------------------------------------------------------------
def test_torch_wrapper_dtype_device():
    mu = torch.tensor([0.1, -0.2, 0.7], dtype=torch.float32)
    A = torch.randn(3, 3, dtype=torch.float32)
    Sigma = (A @ A.T) / 3 + torch.eye(3, dtype=torch.float32)
    new_mu, new_Sigma = exact_relu_covariance_torch(mu, Sigma)
    assert new_mu.dtype == torch.float32 and new_Sigma.dtype == torch.float32
    assert new_mu.device == mu.device and new_Sigma.device == Sigma.device
    assert torch.allclose(new_Sigma, new_Sigma.T, atol=1e-5)


# ===========================================================================
# Integration with the kprop machinery
# ===========================================================================
from mlp_kprop import kprop_harmonic as kprop  # noqa: E402
from mlp_kprop.kprop_harmonic import Kind, coerce_input as harmonic_coerce_input  # noqa: E402
from mlp_kprop.mlp import MLP  # noqa: E402


def _k2_input_after_linear(n=12, seed=0):
    torch.manual_seed(seed)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    K_in = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=2, kind=Kind.SIMPLE)
    return kprop.linear_kprop(K_in, mlp.Ws[0].weight.data, k_max=2, set_metric=mlp.init_scale[0])


def test_kprop_routes_to_exact_bivariate():
    K_z1 = _k2_input_after_linear()
    mu = K_z1[1].to_tensor()
    Sigma = K_z1[2].to_tensor()

    K_out = kprop.relu_kprop(K_z1, k_max=2, kind=Kind.SIMPLE, exact_relu_cov=True)
    exp_mu, exp_Sigma = exact_relu_covariance_torch(mu, Sigma)
    assert torch.allclose(K_out[1].to_tensor(), exp_mu, atol=1e-10)
    assert torch.allclose(K_out[2].to_tensor(), exp_Sigma, atol=1e-10)
    # output contract: identity metric and r == 0
    assert K_out[1].r == 0 and K_out[2].r == 0
    assert K_out[1].has_identity_metric() and K_out[2].has_identity_metric()


def test_exact_cov_differs_from_approx_k2_offdiagonal():
    # The exact-bivariate covariance must differ from the DEFAULT (approximate) k=2
    # propagation off-diagonal, while the mean and the exact diagonal variance agree.
    K_z1 = _k2_input_after_linear()
    cov = kprop.relu_kprop(K_z1, k_max=2, kind=Kind.SIMPLE, exact_relu_cov=True)
    approx = kprop.relu_kprop(K_z1, k_max=2, kind=Kind.SIMPLE)  # default = approximate k=2
    assert torch.allclose(cov[1].to_tensor(), approx[1].to_tensor(), atol=1e-9)  # means agree
    Sc, Sa = cov[2].to_tensor(), approx[2].to_tensor()
    assert torch.allclose(torch.diagonal(Sc), torch.diagonal(Sa), atol=1e-9)     # diagonals agree
    off = ~torch.eye(Sc.shape[0], dtype=torch.bool)
    assert (Sc[off] - Sa[off]).abs().max() > 1e-6                                 # off-diagonals differ


def test_exact_cov_noop_for_non_relu_and_k3():
    # Must be a no-op for a non-ReLU activation at k_max==2 ...
    K_z1 = _k2_input_after_linear()
    sq = kprop.WICK_COEF_D["square"]
    off = kprop.nonlin_kprop(K_z1, nonlin_wick_coef=sq, k_max=2, kind=Kind.SIMPLE, exact_relu_cov=False)
    on = kprop.nonlin_kprop(K_z1, nonlin_wick_coef=sq, k_max=2, kind=Kind.SIMPLE, exact_relu_cov=True)
    for d in off:
        assert torch.allclose(off[d].to_tensor(), on[d].to_tensor(), atol=1e-12)

    # ... and a no-op for ReLU at k_max==3 (K>=3 is never routed to the exact path).
    torch.manual_seed(1)
    n = 10
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    K_in = harmonic_coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE)
    K_z = kprop.linear_kprop(K_in, mlp.Ws[0].weight.data, k_max=3, set_metric=mlp.init_scale[0])
    off3 = kprop.relu_kprop(K_z, k_max=3, kind=Kind.SIMPLE, exact_relu_cov=False)
    on3 = kprop.relu_kprop(K_z, k_max=3, kind=Kind.SIMPLE, exact_relu_cov=True)
    for d in off3:
        assert torch.allclose(off3[d].to_tensor(), on3[d].to_tensor(), atol=1e-12)


def test_mlp_kprop_single_hidden_layer_is_exact():
    # One hidden layer: the output mean depends only on the (exact) marginal ReLU
    # means, so the exact-cov path matches MC within sampling noise.
    torch.manual_seed(0)
    n, N = 32, 1_000_000
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=1, num_layers=2)
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    cp = kprop.mlp_kprop(mlp, K_in, k_max=2, kind=Kind.SIMPLE, exact_relu_cov=True)[1].to_tensor().item()
    z = mlp(torch.randn(N, n)).out
    mc, stderr = z.mean(0).item(), z.std(0).item() / math.sqrt(N)
    assert abs(cp - mc) < 6.0 * stderr


def test_mlp_kprop_deep_differs_from_approx_k2():
    # Two hidden layers: the exact off-diagonal covariance feeds the next layer's
    # variance, so the end-to-end mean differs from the default (approximate) k=2.
    torch.manual_seed(0)
    n = 24
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=4, num_layers=3)
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    cp_cov = kprop.mlp_kprop(mlp, K_in, k_max=2, kind=Kind.SIMPLE, exact_relu_cov=True)[1].to_tensor()
    cp_approx = kprop.mlp_kprop(mlp, K_in, k_max=2, kind=Kind.SIMPLE)[1].to_tensor()
    assert torch.isfinite(cp_cov).all()
    assert (cp_cov - cp_approx).abs().max() > 1e-6
