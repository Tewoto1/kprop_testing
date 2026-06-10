"""analysis.trained_to_0 -- mech-interp tools specific to the train-to-zero study.

These encode the project's original questions about a ReLU MLP trained to output 0
on a Gaussian ball: the first-layer covariance Cov(h1)=W1 W1^T, whether the ReLU
deliberately zeros ~half the mass (vs a random-init baseline), the depth-3 middle
layer's role (cross-layer gating overlap), and what the readout does beyond being
small (shrink / orthogonality / cancellation). Each function runs on its own and
returns a dict -- there is no orchestrator.

For task-agnostic circuit tools (PCA, ablation, patching, attribution, probing),
use the top-level `analysis` package instead.
"""
from .activations import gaussian_batch, collect_activations
from .geometry import layer_gaussian_stats, propagate_examples
from .relu_gating import gating_stats, baseline_gating, bias_stats, mask_overlap
from .output_layer import readout_norms, output_decomposition

__all__ = [
    "gaussian_batch", "collect_activations",
    "layer_gaussian_stats", "propagate_examples",
    "gating_stats", "baseline_gating", "bias_stats", "mask_overlap",
    "readout_norms", "output_decomposition",
]
