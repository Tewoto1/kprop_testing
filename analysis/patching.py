"""analysis/patching.py -- activation patching (the causal-intervention staple).

Take a "clean" input and a "corrupt" input, then substitute the corrupt run's
post-activations at one hidden layer into the clean run and watch the output move.
This isolates how much the information carried at that layer determines the output.

`activation_patch(model, x_clean, x_corrupt, layer, neurons=None)` patches the whole
layer (or only `neurons`) per-sample, so `x_clean` and `x_corrupt` must have the same
batch size. Returns clean/corrupt/patched outputs and the patching effect
(patched - clean).
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional
import torch

from .common import collect_activations, run_with_intervention


def activation_patch(model, x_clean, x_corrupt, layer: int,
                     neurons: Optional[Iterable[int]] = None) -> Dict:
    """Patch `x_corrupt`'s post-activations at `layer` into the `x_clean` forward.

    If `neurons` is None the whole layer is patched, else only those indices.
    `effect = patched_output - clean_output`; `mean_abs_effect` summarizes it.
    """
    if x_clean.shape[0] != x_corrupt.shape[0]:
        raise ValueError("x_clean and x_corrupt must have the same batch size for per-sample patching")
    corrupt = collect_activations(model, x_corrupt)
    clean_out = collect_activations(model, x_clean)["output"]
    corrupt_post = corrupt["post"][layer]
    idx = None if neurons is None else torch.as_tensor(list(neurons), dtype=torch.long)

    def fn(a: torch.Tensor) -> torch.Tensor:
        a = a.clone()
        src = corrupt_post.to(a.device, a.dtype)
        if idx is None:
            a[:] = src
        else:
            a[:, idx] = src[:, idx]
        return a

    patched = run_with_intervention(model, x_clean, {layer: fn}).detach().cpu()
    effect = patched - clean_out
    return {
        "layer": layer,
        "neurons": (None if idx is None else idx.tolist()),
        "clean_output": clean_out.numpy(),
        "corrupt_output": corrupt["output"].numpy(),
        "patched_output": patched.numpy(),
        "effect": effect.numpy(),
        "mean_abs_effect": float(effect.abs().mean()),
    }
