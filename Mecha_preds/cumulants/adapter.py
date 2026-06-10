"""adapter.py -- run cumulant propagation on a study ``model.MLP``.

This is the bridge the rest of the project uses: hand it a trained ``model.MLP``
and it predicts the model's output mean over ``X ~ N(0, I)`` with the REAL
cumulant-propagation algorithm in ``Mecha_preds.cumulants.kprop`` (treated as a
black box -- no linearization, no mean-only push-through, no Monte-Carlo inside
the prediction).

How it works
------------
``mlp_kprop`` consumes the kprop library's own ``MLP`` (it reads ``Ws`` /
``nonlin_names`` / ``init_scale`` / ``layernorm``), not the study model. So
``model_to_kprop`` builds a kprop ``MLP`` of the same shape/activation and copies
the study model's weights into it. Both are ``nn.Linear`` stacks with the same
``(out, in)`` weight orientation, so weights are copied verbatim (no transpose);
the first-layer orientation is checked against ``input_dim``.

Two ReLU K=2 covariance paths, selectable per call, on the SAME model:
  * ``exact_relu_cov=False`` (default): the general harmonic propagation -- its K=2
    ReLU off-diagonal uses the leading-order gain approximation. Works for any
    ``k_max`` and any supported activation.
  * ``exact_relu_cov=True`` (ReLU + ``k_max==2`` only): the EXACT bivariate-Gaussian
    ReLU covariance (``kprop.exact_relu_covariance``; depends on scipy). No effect
    for other activations or ``k_max``, so all k>=3 behavior is preserved.

Input distribution ``X ~ N(0, I_n)`` is encoded as ``K_in = {1: zeros(n), 2: eye(n)}``.
Cumulant propagation runs in float64.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from model import MLP as StudyMLP
from .kprop import Kind, MLP as KpropMLP, HTensor, WICK_COEF_D, mlp_kprop

logger = logging.getLogger("cumulant_adapter")

# `factor` is only implemented for k_max in {3,4}; auto-disabled otherwise.
SUPPORTED_FACTOR_KMAX = (3, 4)

KIND_BY_NAME = {"simple": Kind.SIMPLE, "augment": Kind.AUGMENT, "old": Kind.OLD, "base": Kind.BASE}


def default_cumulant_config() -> dict:
    """Default cumulant-propagation configuration."""
    return {
        "k_max": 3,               # budget K (max cumulant order tracked in full)
        "kind": "simple",         # SIMPLE = canonical setting from the paper
        "use_avg_metric": False,  # exact metric from the actual (fixed) weights
        "factor": True,           # factorized top cumulant; needed for k_max=3 feasibility
        "use_pK": True,           # power-cumulant path (the real algorithm)
        "output_d_max": 1,        # we only need the mean (degree-1 cumulant)
        "exact_relu_cov": False,  # EXACT bivariate-Gaussian ReLU covariance at k_max==2
    }


def _normalize_config(config: Optional[dict]) -> dict:
    cfg = default_cumulant_config()
    if config:
        cfg.update(config)
    if cfg["k_max"] < 1:
        raise ValueError("k_max must be >= 1")
    if isinstance(cfg["kind"], str):
        cfg["kind"] = cfg["kind"].lower()
        if cfg["kind"] not in KIND_BY_NAME:
            raise ValueError(f"Unknown kind {cfg['kind']!r}; choose from {list(KIND_BY_NAME)}")
    if cfg["factor"] and cfg["k_max"] not in SUPPORTED_FACTOR_KMAX:
        cfg["factor"] = False
        if cfg["k_max"] > 4:
            logger.warning("factor=True only supported for k_max in {3,4}; disabling for "
                           "k_max=%d (expect higher memory/time).", cfg["k_max"])
    if cfg.get("exact_relu_cov") and cfg["k_max"] != 2:
        logger.warning("exact_relu_cov=True only takes effect at k_max==2 (got k_max=%d); "
                       "the general (approximate) propagation path will run instead.", cfg["k_max"])
    return cfg


def config_summary(config: dict) -> str:
    cfg = _normalize_config(config)
    kind = cfg["kind"] if isinstance(cfg["kind"], str) else cfg["kind"].name.lower()
    return (f"k_max={cfg['k_max']},kind={kind},use_avg_metric={cfg['use_avg_metric']},"
            f"factor={cfg['factor']},use_pK={cfg['use_pK']},output_d_max={cfg['output_d_max']},"
            f"exact_relu_cov={cfg.get('exact_relu_cov', False)}")


def model_to_kprop(model: StudyMLP, *, device: str = "cpu",
                   dtype: torch.dtype = torch.float64) -> KpropMLP:
    """Build a kprop ``MLP`` matching ``model`` (a ``model.MLP``) and copy its weights.

    The study MLP is ``depth`` hidden ``Linear+activation`` blocks plus a readout,
    i.e. ``depth + 1`` linear layers -> kprop ``num_layers = depth + 1``. Bias
    parameters are created per layer iff the corresponding study layer has a bias
    (``cfg.bias`` for hidden, ``cfg.final_bias`` for the readout). Init values are
    irrelevant -- every weight/bias is overwritten by a copy from ``model``.
    """
    if not isinstance(model, StudyMLP):
        raise TypeError(f"run_cumulants expects a model.MLP, got {type(model)!r}")
    cfg = model.cfg
    if cfg.activation not in WICK_COEF_D:
        raise ValueError(f"activation {cfg.activation!r} is not supported by cumulant "
                         f"propagation (supported: {sorted(WICK_COEF_D)}).")
    num_layers = cfg.depth + 1
    b_var = [1.0 if cfg.bias else 0.0] * cfg.depth + [1.0 if cfg.final_bias else 0.0]
    kmlp = KpropMLP(input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim, output_dim=cfg.output_dim,
                    num_layers=num_layers, nonlin=cfg.activation, init_kind="he", b_var=b_var)
    kmlp = kmlp.to(device=device, dtype=dtype)

    src_layers = list(model.hidden_layers) + [model.readout]
    if len(src_layers) != len(kmlp.Ws):
        raise RuntimeError(f"layer-count mismatch: study={len(src_layers)} kprop={len(kmlp.Ws)}")
    with torch.no_grad():
        for src, dst in zip(src_layers, kmlp.Ws):
            if tuple(src.weight.shape) != tuple(dst.weight.shape):
                raise RuntimeError(f"weight shape mismatch {tuple(src.weight.shape)} vs "
                                   f"{tuple(dst.weight.shape)}")
            dst.weight.copy_(src.weight.to(dtype))
            if (dst.bias is None) != (src.bias is None):
                raise RuntimeError("bias presence mismatch between study and kprop layer")
            if dst.bias is not None:
                dst.bias.copy_(src.bias.to(dtype))
    return kmlp.eval()


@torch.no_grad()
def run_cumulants(model: StudyMLP, input_dim: Optional[int] = None,
                  config: Optional[dict] = None, *, device: str = "cpu",
                  debug: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I_input_dim)`` via cumulant propagation.

    `input_dim` defaults to ``model.cfg.input_dim``. Returns
    ``{"raw_output": K_out, "mean": np.ndarray (output_dim,), "metadata": {...}}``.
    """
    cfg = _normalize_config(config)
    if input_dim is None:
        input_dim = model.cfg.input_dim

    kmlp = model_to_kprop(model, device=device)
    in_f = kmlp.Ws[0].weight.shape[1]
    if in_f != input_dim:
        raise ValueError(f"first layer in_features={in_f} != input_dim={input_dim}")

    K_in = {1: torch.zeros(input_dim, device=device, dtype=torch.float64),
            2: torch.eye(input_dim, device=device, dtype=torch.float64)}
    kind = cfg["kind"] if isinstance(cfg["kind"], Kind) else KIND_BY_NAME[cfg["kind"]]

    K_out = mlp_kprop(kmlp, K_in, k_max=cfg["k_max"], kind=kind,
                      use_avg_metric=cfg["use_avg_metric"], factor=cfg["factor"],
                      use_pK=cfg["use_pK"], output_d_max=cfg["output_d_max"],
                      exact_relu_cov=cfg.get("exact_relu_cov", False))

    if 1 not in K_out:
        raise RuntimeError("cumulant propagation returned no degree-1 (mean) cumulant")
    mean = K_out[1].to_tensor().detach().cpu().double().numpy().reshape(-1)
    if debug:
        logger.info("cp_mean shape=%s value=%s", mean.shape, mean)

    return {
        "raw_output": K_out,
        "mean": mean,
        "metadata": {
            "config": config_summary(cfg),
            "config_dict": cfg,
            "nonlin_names": list(kmlp.nonlin_names),
            "input_dim": int(input_dim),
            "output_dim": int(mean.shape[0]),
        },
    }


def extract_mean(cp_result: dict) -> np.ndarray:
    """Pull the predicted output mean (np.ndarray, shape (output_dim,))."""
    return np.asarray(cp_result["mean"], dtype=np.float64).reshape(-1)
