"""model -- the MLP under study.

One class (``MLP``) covers every variant: vary ``hidden_dim`` for width and
``depth`` for the number of hidden layers. ``ModelConfig(...).build()`` returns an
``MLP``; ``MLP.load(path)`` reconstructs one (plus its training payload) from a
checkpoint. The forward pass can return every intermediate activation, which is
what the ``analysis`` package reads.
"""
from .mlp import MLP, ModelConfig

__all__ = ["MLP", "ModelConfig"]
