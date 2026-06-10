"""model.py -- MLP definitions for the mech-interp experiments.

Every model here is a plain fully-connected MLP. The single class `MLP`
covers all the variants we need: vary `hidden_dim` for width and `depth` for
the number of hidden layers. It exposes *every* intermediate activation in one
forward pass, so the analysis code can read pre-/post-ReLU tensors directly
(no hooks needed -- though analysis/activations.py offers a hook path too).

Conventions
-----------
`depth` = number of hidden (Linear -> activation) blocks.

    depth=2:  in -> h1 -> h2 -> out          (3 weight matrices)
    depth=3:  in -> h1 -> h2 -> h3 -> out     (4 weight matrices)

The depth=3 "middle layer" we interrogate is hidden block index 1 (h2).
For the train-to-zero experiments `input_dim == hidden_dim` (a square first
map), but the class accepts any dimensions.

Biases default to OFF (bias == final_bias == False): every layer is then a
pure linear map, so all ReLU half-space boundaries pass through the origin and
layer-0 pre-activations are exactly zero-mean (W1 . Gaussian). Set bias=True /
final_bias=True to restore them.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "identity": nn.Identity,
}


@dataclass
class ModelConfig:
    """All knobs for one MLP. `.build()` returns the `MLP`."""
    input_dim: int
    hidden_dim: int
    depth: int                      # number of hidden Linear+activation blocks
    output_dim: int = 1
    bias: bool = False              # hidden-layer bias (OFF by default: half-spaces through origin)
    final_bias: bool = False        # readout bias (OFF by default)
    activation: str = "relu"
    seed: Optional[int] = None      # seed applied just before weight init

    def build(self) -> "MLP":
        return MLP(self)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {cfg.activation!r}; choose from {list(_ACTIVATIONS)}")
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        self.cfg = cfg
        dims = [cfg.input_dim] + [cfg.hidden_dim] * cfg.depth
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=cfg.bias) for i in range(cfg.depth)
        )
        self.act = _ACTIVATIONS[cfg.activation]()
        self.readout = nn.Linear(cfg.hidden_dim, cfg.output_dim, bias=cfg.final_bias)

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """If return_activations, also return a dict with every pre-/post-ReLU
        tensor keyed by hidden-layer index (0-based), plus input and output."""
        pre: Dict[int, torch.Tensor] = {}
        post: Dict[int, torch.Tensor] = {}
        h = x
        for i, layer in enumerate(self.hidden_layers):
            z = layer(h)            # pre-activation  (B, hidden_dim)
            a = self.act(z)         # post-activation (B, hidden_dim)
            pre[i], post[i] = z, a
            h = a
        out = self.readout(h)
        if return_activations:
            return out, {"input": x, "pre": pre, "post": post, "output": out}
        return out

    @torch.no_grad()
    def activations(self, x: torch.Tensor) -> Dict:
        """Convenience: eval/no-grad forward returning just the activation dict."""
        was_training = self.training
        self.eval()
        _, acts = self.forward(x, return_activations=True)
        if was_training:
            self.train()
        return acts

    # -- introspection ---------------------------------------------------
    def named_weights(self) -> List[Tuple[str, torch.Tensor]]:
        """Weight matrices in forward order: hidden0 (W1) ... readout."""
        out = [(f"hidden{i}", layer.weight) for i, layer in enumerate(self.hidden_layers)]
        out.append(("readout", self.readout.weight))
        return out

    def named_biases(self) -> List[Tuple[str, Optional[torch.Tensor]]]:
        out = [(f"hidden{i}", layer.bias) for i, layer in enumerate(self.hidden_layers)]
        out.append(("readout", self.readout.bias))
        return out

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- (de)serialization ----------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {"model_config": asdict(self.cfg), "state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> Tuple["MLP", dict]:
        """Returns (model, payload). `payload` also holds step/history/configs
        when the checkpoint was written by the Trainer."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**payload["model_config"])
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload
