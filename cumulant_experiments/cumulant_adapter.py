"""Black-box adapter from a trained PyTorch model to the REAL cumulant
propagation implementation in this repo.

The real algorithm lives in ``src.mlp_kprop.kprop_harmonic.mlp_kprop`` (the
"kprop" / cumulant propagation algorithm from the paper). We treat it as a black
box: we do NOT reimplement it, do NOT linearize, do NOT push only means through
the net, and do NOT use any Monte-Carlo samples inside the prediction.

What the adapter does:

  1. Extracts the model's linear layer weights and biases.
  2. Verifies their orientation against what cumulant propagation consumes.
  3. Builds the input cumulant tower K_in for X ~ N(0, I_input_dim).
  4. Calls the real ``mlp_kprop`` and returns its raw output plus the predicted
     output mean.

--------------------------------------------------------------------------------
Weight orientation (IMPORTANT — verified, not assumed)
--------------------------------------------------------------------------------
PyTorch ``nn.Linear`` stores ``weight`` with shape ``(out_features, in_features)``
and computes ``y = x @ W.T + b``.

The real cumulant propagation code feeds ``W = mlp.Ws[l].weight`` directly into
``linear_kprop`` -> ``HTensor.contract_W(W)``, which contracts the input cumulant
over ``W``'s dim=1 (the ``in_features`` axis). That is exactly ``W @ x`` semantics
for the mean, matching ``y = W x + b``. So **no transpose is applied or needed**.

We do not silently transpose. Sanity check #1 (single linear layer: true mean is
exactly the bias ``b``) directly confirms this orientation; if the orientation
were wrong, that check would fail loudly.

--------------------------------------------------------------------------------
Input distribution
--------------------------------------------------------------------------------
X ~ N(0, I_n) is encoded as the input cumulant tower
``K_in = {1: zeros(n), 2: eye(n)}`` (first cumulant = mean = 0, second cumulant =
covariance = I, all higher cumulants zero). This is exactly the normalization the
repo uses in its README and tests, and it matches the requested X ~ N(0, I).

--------------------------------------------------------------------------------
The ``use_avg_metric`` choice
--------------------------------------------------------------------------------
``mlp_kprop`` has a ``use_avg_metric`` flag. With ``True`` (the library's default)
it uses the *initialization-time* expected metric E[WW^T] = init_scale * I. With
``False`` it uses the exact metric W @ metric @ W^T computed from the actual
weights. Because our weights are FIXED after training (the distribution is over X,
not over weights), the exact metric is the faithful choice: ``use_avg_metric=True``
would inject stale init-time assumptions that training has invalidated. We
therefore default to ``use_avg_metric=False``. (The repo's own docstring notes
this flag "doesn't have a large effect on MSE or FLOPs".) It remains configurable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from src.mlp_kprop.harmonic import HTensor
from src.mlp_kprop.kprop_harmonic import Kind, mlp_kprop
from src.mlp_kprop.mlp import MLP
from src.mlp_kprop.wick import WICK_COEF_D

logger = logging.getLogger("cumulant_adapter")

# NOTE: there is no longer a hard cap on k_max. Larger k_max is more accurate but
# costs ~O(n^{k_max}) memory/time, so it can OOM at large width on a small machine
# (k_max=4 at width 1024 OOM'd a CPU box during development). On a GPU box (e.g.
# Colab) higher k_max is fine. `factor` is only supported for k_max in {3,4}; for
# any other k_max it is silently disabled (see _normalize_config).
SUPPORTED_FACTOR_KMAX = (3, 4)

KIND_BY_NAME = {
    "simple": Kind.SIMPLE,
    "augment": Kind.AUGMENT,
    "old": Kind.OLD,
    "base": Kind.BASE,
}


def default_cumulant_config() -> dict:
    """Default cumulant-propagation configuration."""
    return {
        "k_max": 3,            # budget parameter K (max cumulant order tracked in full)
        "kind": "simple",      # SIMPLE = canonical setting from the paper
        "use_avg_metric": False,  # exact metric from the actual fixed weights (see module docstring)
        "factor": True,        # factorized top cumulant; needed for k_max=3 feasibility
        "use_pK": True,        # power-cumulant path (the real algorithm; False is an ablation)
        "output_d_max": 1,     # we only need the mean (degree-1 cumulant) -> huge FLOP savings
        "exact_relu_k2": False,  # leading-order closed-form ReLU at k_max==2 (see below)
        "exact_relu_cov": False, # EXACT bivariate Gaussian ReLU covariance at k_max==2 (see below)
    }


# Selectable ReLU K=2 activation steps, all on the SAME code/model (no fork, no
# copy). They only engage for ReLU AND k_max==2 (K>=3 is never affected):
#   - exact_relu_cov=False, exact_relu_k2=False (default): the general harmonic
#     "cumulant PROPAGATION" algorithm.
#   - exact_relu_k2=True: exact closed-form marginals, but the off-diagonal uses
#     the leading-order Hermite gain Sigma_ij*c_i*c_j (src.mlp_kprop.relu_k2_exact).
#   - exact_relu_cov=True: the EXACT bivariate Gaussian ReLU covariance -- the
#     off-diagonal is the true Cov(ReLU(Z_i), ReLU(Z_j)), no gain approximation
#     (src.mlp_kprop.exact_relu_covariance; depends on scipy, non-autograd). Takes
#     precedence over exact_relu_k2. This is the "exact_relu_covariance" /
#     "k2_exact_bivariate_relu" / "exact_gaussian_relu_k2" path.


def _normalize_config(cumulant_config: Optional[dict]) -> dict:
    cfg = default_cumulant_config()
    if cumulant_config:
        cfg.update(cumulant_config)
    if cfg["k_max"] < 1:
        raise ValueError("k_max must be >= 1")
    if isinstance(cfg["kind"], str):
        cfg["kind"] = cfg["kind"].lower()
        if cfg["kind"] not in KIND_BY_NAME:
            raise ValueError(f"Unknown kind {cfg['kind']!r}; choose from {list(KIND_BY_NAME)}")
    # `factor` is only implemented for k_max in {3,4}; disable it (with a heads-up)
    # for any other k_max so the real mlp_kprop doesn't raise NotImplementedError.
    if cfg["factor"] and cfg["k_max"] not in SUPPORTED_FACTOR_KMAX:
        cfg = dict(cfg)
        cfg["factor"] = False
        if cfg["k_max"] > 4:
            logger.warning("factor=True is only supported for k_max in {3,4}; disabling factor "
                           "for k_max=%d (expect higher memory/time).", cfg["k_max"])
    # The exact closed-form ReLU path only engages for ReLU at k_max==2; warn if
    # requested with a different k_max so the user isn't silently running the
    # approximate path.
    if (cfg.get("exact_relu_k2") or cfg.get("exact_relu_cov")) and cfg["k_max"] != 2:
        flag = "exact_relu_cov" if cfg.get("exact_relu_cov") else "exact_relu_k2"
        logger.warning("%s=True only takes effect at k_max==2 (got k_max=%d); "
                       "the general (approximate) propagation path will run instead.", flag, cfg["k_max"])
    return cfg


def config_summary(cumulant_config: dict) -> str:
    cfg = _normalize_config(cumulant_config)
    kind = cfg["kind"] if isinstance(cfg["kind"], str) else cfg["kind"].name.lower()
    return (
        f"k_max={cfg['k_max']},kind={kind},use_avg_metric={cfg['use_avg_metric']},"
        f"factor={cfg['factor']},use_pK={cfg['use_pK']},output_d_max={cfg['output_d_max']},"
        f"exact_relu_k2={cfg.get('exact_relu_k2', False)},"
        f"exact_relu_cov={cfg.get('exact_relu_cov', False)}"
    )


def _extract_linear_layers(model: MLP) -> list[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Pull (weight, bias) tensors out of the model in forward order.

    We read directly from ``model.Ws`` (the repo MLP's ModuleList of nn.Linear).
    These are exactly the tensors cumulant propagation will consume, so this both
    documents the conversion and lets us log/verify shapes.
    """
    layers = []
    for W in model.Ws:
        weight = W.weight.detach()
        bias = None if W.bias is None else W.bias.detach()
        layers.append((weight, bias))
    return layers


@torch.no_grad()
def run_cumulant_propagation_from_model(
    model: MLP,
    input_dim: int,
    cumulant_config: Optional[dict] = None,
    device: str = "cpu",
    *,
    debug: bool = False,
) -> dict:
    """Feed a trained model's weights into the real cumulant propagation algorithm.

    Returns a dict:
        {
            "raw_output": <K_out, the raw HTower returned by mlp_kprop>,
            "mean": np.ndarray of shape (output_dim,)   # the propagated output mean
            "metadata": {...}                           # shapes, config, etc.
        }
    """
    cfg = _normalize_config(cumulant_config)
    if not isinstance(model, MLP):
        raise TypeError(
            "run_cumulant_propagation_from_model expects a repo MLP instance "
            f"(src.mlp_kprop.mlp.MLP), got {type(model)!r}. The real mlp_kprop "
            "consumes an MLP (it needs nonlin_names / init_scale / Ws)."
        )
    if model.layernorm:
        raise NotImplementedError("cumulant propagation does not support layernorm")

    # Cumulant propagation runs in double precision. We operate on a DEEP COPY cast
    # to float64 so the caller's model is left untouched -- important for the GPU
    # path, where training/Monte-Carlo keep the model in float32 (much faster) while
    # kprop still gets a faithful float64 copy. (A bare model.to(float64) would
    # mutate the caller's params in place.)
    import copy
    needs_copy = (next(model.parameters()).dtype != torch.float64) or \
                 (str(next(model.parameters()).device) != str(torch.device(device)))
    model = (copy.deepcopy(model) if needs_copy else model).to(device=device, dtype=torch.float64)
    model.eval()

    # --- 1. Extract weights/biases and verify orientation ---------------------
    layers = _extract_linear_layers(model)
    layer_shapes = []
    for li, (weight, bias) in enumerate(layers):
        out_f, in_f = weight.shape  # nn.Linear weight is (out_features, in_features)
        layer_shapes.append({
            "layer": li,
            "weight_shape": tuple(weight.shape),  # (out, in) -- consumed as-is, no transpose
            "bias_shape": None if bias is None else tuple(bias.shape),
        })
        # Orientation check: the layer that reads the input must contract over input_dim.
        if li == 0 and in_f != input_dim:
            raise ValueError(
                f"First layer in_features={in_f} != input_dim={input_dim}. "
                "Weight orientation or input_dim is wrong."
            )
        if debug:
            logger.info(
                "extracted layer %d: weight (out=%d, in=%d) [fed to kprop as-is, no transpose], bias=%s",
                li, out_f, in_f, None if bias is None else tuple(bias.shape),
            )

    # Activation sanity: every activation must be supported by cumulant propagation.
    for name in model.nonlin_names:
        if name not in WICK_COEF_D:
            raise ValueError(
                f"Activation {name!r} is not supported by cumulant propagation "
                f"(supported: {list(WICK_COEF_D)})."
            )

    # --- 2. Build input cumulant tower for X ~ N(0, I_input_dim) --------------
    K_in = {
        1: torch.zeros(input_dim, device=device, dtype=torch.float64),
        2: torch.eye(input_dim, device=device, dtype=torch.float64),
    }

    kind = cfg["kind"] if isinstance(cfg["kind"], Kind) else KIND_BY_NAME[cfg["kind"]]

    # --- 3. Call the REAL cumulant propagation algorithm ----------------------
    K_out = mlp_kprop(
        model,
        K_in,
        k_max=cfg["k_max"],
        kind=kind,
        use_avg_metric=cfg["use_avg_metric"],
        factor=cfg["factor"],
        use_pK=cfg["use_pK"],
        output_d_max=cfg["output_d_max"],
        exact_relu_k2=cfg.get("exact_relu_k2", False),
        exact_relu_cov=cfg.get("exact_relu_cov", False),
    )

    # --- 4. Extract the predicted output mean (degree-1 cumulant = mean) ------
    if 1 not in K_out:
        raise RuntimeError("cumulant propagation returned no degree-1 (mean) cumulant")
    mean_tensor = K_out[1].to_tensor() if isinstance(K_out[1], HTensor) else K_out[1].to_tensor()
    mean = mean_tensor.detach().cpu().double().numpy().reshape(-1)

    if debug:
        logger.info("cumulant propagation output keys: %s", list(K_out.keys()))
        logger.info("cp_mean shape=%s value=%s", mean.shape, mean)

    return {
        "raw_output": K_out,
        "mean": mean,
        "metadata": {
            "config": config_summary(cfg),
            "config_dict": cfg,
            "layer_shapes": layer_shapes,
            "nonlin_names": list(model.nonlin_names),
            "output_dim": int(mean.shape[0]),
            "input_dim": int(input_dim),
        },
    }


def extract_mean(cp_result: dict) -> np.ndarray:
    """Pull the predicted output mean (np.ndarray, shape (output_dim,))."""
    return np.asarray(cp_result["mean"], dtype=np.float64).reshape(-1)


# =============================================================================
# Sanity checks
# =============================================================================

def sanity_check_single_linear(
    *, input_dim: int = 8, output_dim: int = 3, device: str = "cpu",
    cumulant_config: Optional[dict] = None, atol: float = 1e-8,
) -> dict:
    """Sanity #1: a single linear layer y = W x + b with x ~ N(0, I).

    The true mean is EXACTLY b. Cumulant propagation must reproduce b (up to
    numerical error). This also directly validates weight orientation + bias.
    """
    set_seed_local(0)
    # num_layers=1 -> a single linear layer, no activation. b_var>0 -> bias exists.
    mlp = MLP(input_dim=input_dim, hidden_dim=output_dim, output_dim=output_dim,
              num_layers=1, b_var=1.0).to(device=device, dtype=torch.float64)
    # Set known weights/bias.
    with torch.no_grad():
        mlp.Ws[0].weight.copy_(torch.randn(output_dim, input_dim, dtype=torch.float64))
        b = torch.randn(output_dim, dtype=torch.float64)
        mlp.Ws[0].bias.copy_(b)

    cfg = dict(cumulant_config or {})
    cfg.setdefault("k_max", 2)
    res = run_cumulant_propagation_from_model(mlp, input_dim, cfg, device=device)
    cp_mean = extract_mean(res)
    true_mean = b.cpu().numpy()
    err = float(np.max(np.abs(cp_mean - true_mean)))
    passed = err < atol
    print(f"  [sanity:single-linear] max|cp_mean - b| = {err:.3e}  -> {'PASS' if passed else 'FAIL'}")
    return {"passed": passed, "max_abs_error": err, "cp_mean": cp_mean, "true_mean": true_mean}


def sanity_check_small_mlp(
    *, input_dim: int = 8, hidden_width: int = 16, hidden_depth: int = 2,
    output_dim: int = 1, activation: str = "relu", device: str = "cpu",
    cumulant_config: Optional[dict] = None, mc_samples: int = 4_000_000,
    rel_tol: float = 0.05, n_sigma: float = 6.0,
) -> dict:
    """Sanity #2: a small untrained MLP, cumulant mean vs a large Monte-Carlo mean.

    Not expected to be exact (finite k_max), but should be close. Passes if the
    cumulant mean is within n_sigma MC standard errors OR within rel_tol relative
    error of the MC mean. Catches gross orientation/activation bugs.
    """
    from cumulant_experiments.model_utils import make_mlp

    set_seed_local(0)
    mlp = make_mlp(input_dim=input_dim, hidden_width=hidden_width, hidden_depth=hidden_depth,
                   output_dim=output_dim, activation=activation, bias=True,
                   device=device, dtype=torch.float64)

    cfg = dict(cumulant_config or {})
    res = run_cumulant_propagation_from_model(mlp, input_dim, cfg, device=device)
    cp_mean = extract_mean(res)

    # Large Monte-Carlo reference.
    with torch.no_grad():
        mlp.eval()
        n, batch = 0, 200_000
        acc = torch.zeros(output_dim, dtype=torch.float64, device=device)
        acc_sq = torch.zeros(output_dim, dtype=torch.float64, device=device)
        while n < mc_samples:
            b = min(batch, mc_samples - n)
            x = torch.randn(b, input_dim, device=device, dtype=torch.float64)
            y = mlp(x).out
            acc += y.sum(0)
            acc_sq += y.pow(2).sum(0)
            n += b
        mc_mean = (acc / n).cpu().numpy()
        mc_var = (acc_sq / n).cpu().numpy() - mc_mean ** 2
        mc_stderr = np.sqrt(np.maximum(mc_var, 0.0) / n)

    abs_err = np.abs(cp_mean - mc_mean)
    within_sigma = bool(np.all(abs_err <= n_sigma * (mc_stderr + 1e-12)))
    rel_err = float(np.linalg.norm(cp_mean - mc_mean) / (np.linalg.norm(mc_mean) + 1e-12))
    passed = within_sigma or rel_err < rel_tol
    print(
        f"  [sanity:small-mlp] cp_mean={cp_mean}  mc_mean={mc_mean}  "
        f"max|err|={float(abs_err.max()):.3e}  rel_err={rel_err:.3e}  "
        f"(<= {n_sigma}sigma: {within_sigma}) -> {'PASS' if passed else 'FAIL'}"
    )
    return {
        "passed": passed, "cp_mean": cp_mean, "mc_mean": mc_mean,
        "max_abs_error": float(abs_err.max()), "rel_error": rel_err,
        "within_sigma": within_sigma,
    }


def set_seed_local(seed: int) -> None:
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)


def run_sanity_checks(*, device: str = "cpu", cumulant_config: Optional[dict] = None) -> bool:
    """Run all adapter sanity checks. Returns True iff all pass."""
    print("Running cumulant-propagation adapter sanity checks...")
    ok = True
    r1 = sanity_check_single_linear(device=device, cumulant_config=cumulant_config)
    ok = ok and r1["passed"]
    r2 = sanity_check_small_mlp(device=device, cumulant_config=cumulant_config)
    ok = ok and r2["passed"]
    print(f"Sanity checks {'PASSED' if ok else 'FAILED'}.")
    return ok
