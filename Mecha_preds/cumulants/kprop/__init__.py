"""Vendored cumulant-propagation ("kprop") library.

This is the implementation of the cumulant-propagation algorithm from
"Estimating the expected output of wide random MLPs more efficiently than
sampling" (https://arxiv.org/abs/2605.05179), copied from the paper's repo and
re-homed here as a self-contained package (internal imports are relative, so the
package is relocatable). Only the import closure of the harmonic kprop path plus
the exact bivariate-Gaussian ReLU covariance step is vendored; the symbolic and
deprecated diagonal-slice variants are not included.

Do not call this directly with a study model -- use
``Mecha_preds.cumulants.run_cumulants`` (which converts a ``model.MLP`` into the
kprop ``MLP`` this library consumes). The entry point ``mlp_kprop`` reads
``mlp.Ws`` / ``mlp.nonlin_names`` / ``mlp.init_scale`` / ``mlp.layernorm`` off the
kprop ``MLP`` below.
"""
from .kprop_harmonic import Kind, mlp_kprop, nonlin_kprop, linear_kprop, coerce_input
from .harmonic import HTensor
from .mlp import MLP
from .wick import WICK_COEF_D
from .exact_relu_covariance import (
    exact_relu_covariance_kprop,
    exact_relu_covariance_np,
    exact_relu_covariance_torch,
)

__all__ = [
    "Kind", "mlp_kprop", "nonlin_kprop", "linear_kprop", "coerce_input",
    "HTensor", "MLP", "WICK_COEF_D",
    "exact_relu_covariance_kprop", "exact_relu_covariance_np", "exact_relu_covariance_torch",
]
