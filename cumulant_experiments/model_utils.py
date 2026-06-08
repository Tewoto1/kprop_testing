"""Model construction, seeding, and training-to-zero utilities.

The model used everywhere in this experiment is the *repo's own* ``MLP`` class
(``src.mlp_kprop.mlp.MLP``). That is deliberate: the real cumulant propagation
entry point ``mlp_kprop`` consumes an ``MLP`` instance (it reads ``mlp.Ws``,
``mlp.nonlin_names``, ``mlp.init_scale``, ``mlp.layernorm``). Using the repo's
own model means there is exactly one source of truth for the architecture and no
risk of an activation/weight-orientation mismatch between "the model we train"
and "the model cumulant propagation sees". We still extract and verify weights
explicitly in the adapter (see ``cumulant_adapter.py``).

Everything runs in float64 for consistency with cumulant propagation, which the
repo runs in double precision in its tests.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from src.mlp_kprop.mlp import MLP


def set_seed(seed: int) -> None:
    """Seed torch / numpy / python RNGs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_mlp(
    *,
    input_dim: int,
    hidden_width: int,
    hidden_depth: int,
    output_dim: int,
    activation: str,
    bias: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> MLP:
    """Build a fresh randomly-initialized repo MLP.

    The repo convention is that ``num_layers`` is the number of *linear* layers,
    and the number of hidden layers is ``num_layers - 1`` (no activation after
    the final linear layer). So ``hidden_depth`` hidden layers => ``num_layers =
    hidden_depth + 1``.

    ``activation`` must be one of the activations that cumulant propagation
    supports (keys of ``WICK_COEF_D``): relu, gelu, tanh, sigmoid, square, cube,
    heaviside, sgn. We do NOT silently substitute a different activation.

    Biases: the repo MLP only creates bias parameters when ``b_var > 0`` or
    ``b_mean != 0``. We set ``b_var = 1.0`` when ``bias=True`` so that genuine,
    trainable bias parameters exist and the cumulant-propagation bias path is
    exercised end-to-end.
    """
    num_layers = hidden_depth + 1
    mlp = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_width,
        output_dim=output_dim,
        num_layers=num_layers,
        nonlin=activation,
        init_kind="he",
        b_var=(1.0 if bias else None),
    )
    return mlp.to(device=device, dtype=dtype)


def layer_norms(model: MLP) -> tuple[list[float], list[Optional[float]]]:
    """Return (weight_norms, bias_norms) per linear layer (Frobenius / L2)."""
    weight_norms: list[float] = []
    bias_norms: list[Optional[float]] = []
    for W in model.Ws:
        weight_norms.append(float(W.weight.detach().norm().item()))
        bias_norms.append(None if W.bias is None else float(W.bias.detach().norm().item()))
    return weight_norms, bias_norms


@torch.no_grad()
def output_rms(model: MLP, input_dim: int, *, batch: int = 8192,
               device: str = "cpu", dtype: torch.dtype = torch.float64) -> float:
    """RMS of the model output on a held-out standard-Gaussian batch."""
    model.eval()
    x = torch.randn(batch, input_dim, device=device, dtype=dtype)
    y = model(x).out
    return float(y.pow(2).mean().sqrt().item())


def train_model_to_zero(
    *,
    model: MLP,
    input_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    loss_tol: float = 0.0,
    log_every: int = 0,
) -> dict:
    """Train ``model`` so its output is pushed toward 0 under x ~ N(0, I).

    Loss is MSE(model(x), 0) on freshly-sampled Gaussian inputs each step
    (so there is no fixed dataset to overfit; the target is identically zero).

    Stops early if the running loss falls below ``loss_tol`` (when > 0).

    Returns a stats dict: initial_train_loss, final_train_loss, train_steps_run.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    initial_loss: Optional[float] = None
    final_loss = float("nan")
    steps_run = 0

    with torch.enable_grad():
        for step in range(steps):
            x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
            out = model(x).out
            loss = out.pow(2).mean()  # MSE against the all-zeros target
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().item())
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val
            steps_run = step + 1

            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"    [train] step {step:5d}/{steps}  loss={loss_val:.6e}", flush=True)

            if loss_tol > 0.0 and loss_val < loss_tol:
                break

    model.eval()
    return {
        "initial_train_loss": float("nan") if initial_loss is None else initial_loss,
        "final_train_loss": final_loss,
        "train_steps_run": steps_run,
    }


def train_model_to_halfspace(
    *,
    model: MLP,
    input_dim: int,
    output_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.0,
    offset_std: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    loss_tol: float = 0.0,
    log_every: int = 0,
) -> dict:
    """Train ``model`` to classify whether x ~ N(0, I) lies in random half-spaces.

    Each of the ``output_dim`` output components gets its own FIXED random affine
    half-space: a unit normal ``w_j`` and offset ``b_j ~ N(0, offset_std^2)``,
    drawn once. The per-component target is the 0/1 indicator
    ``y_j = 1[x . w_j > b_j]``; the loss is MSE(model(x), y) on freshly-sampled
    Gaussian inputs each step (so there is no fixed dataset to overfit). This gives
    a genuinely different trained weight structure from ``train_model_to_zero``
    (the per-component output mean tends to ``E[y_j] = Phi(-b_j)``, not 0), so it is
    a second probe of how training affects cumulant propagation.

    Stops early if the running loss falls below ``loss_tol`` (when > 0). MSE against
    a hard {0,1} boundary plateaus above 0, so usually leave ``loss_tol = 0``.

    Returns a stats dict: initial_train_loss, final_train_loss, train_steps_run.
    """
    model.train()
    # Fixed random affine half-spaces, one per output component.
    Wn = torch.randn(output_dim, input_dim, device=device, dtype=dtype)
    Wn = Wn / Wn.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = offset_std * torch.randn(output_dim, device=device, dtype=dtype)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    initial_loss: Optional[float] = None
    final_loss = float("nan")
    steps_run = 0

    with torch.enable_grad():
        for step in range(steps):
            x = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
            y = (x @ Wn.T - b > 0).to(dtype)  # (batch, output_dim) 0/1 labels
            out = model(x).out
            loss = (out - y).pow(2).mean()  # MSE to the half-space indicators
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().item())
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val
            steps_run = step + 1

            if log_every and (step % log_every == 0 or step == steps - 1):
                print(f"    [halfspace] step {step:5d}/{steps}  loss={loss_val:.6e}", flush=True)

            if loss_tol > 0.0 and loss_val < loss_tol:
                break

    model.eval()
    return {
        "initial_train_loss": float("nan") if initial_loss is None else initial_loss,
        "final_train_loss": final_loss,
        "train_steps_run": steps_run,
    }
