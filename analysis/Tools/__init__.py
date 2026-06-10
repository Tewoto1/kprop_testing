"""analysis -- mechanistic-interpretability / circuit tools for the study MLP.

Two layers, and each tool runs individually (no master "report" step):

  * General, task-agnostic circuit tools at the top level -- work on any
    `model.MLP` and any input batch:
        common.collect_activations / run_with_intervention   (primitives)
        pca.activation_pca / pca.weight_spectrum             (PCA, weight SVD)
        ablation.ablate / ablation.neuron_importance         (causal knockout)
        patching.activation_patch                            (activation patching)
        attribution.direct_contributions / output_lens       (direct attribution / logit-lens)
        probing.linear_probe / probe_layer                   (linear probes)

  * `analysis.trained_to_0` -- the study-specific toolkit for the "train an MLP to
    output 0" questions (first-layer Gaussian covariance, ReLU gating vs baseline,
    readout shrink/orthogonality/cancellation).

Example:
    from model import MLP
    from analysis import weight_spectrum, neuron_importance, activation_pca
    from analysis.trained_to_0 import gaussian_batch, layer_gaussian_stats
    m, _ = MLP.load("checkpoints/zero_d3_w128_seed0_final.pt")
    print(weight_spectrum(m)["readout"]["effective_rank"])
    x = gaussian_batch(m.cfg.input_dim, 4096, seed=0)
    print(activation_pca(m, x, layer=-1)["top_explained"])
"""
from .common import collect_activations, run_with_intervention
from .pca import activation_pca, weight_spectrum
from .ablation import ablate, neuron_importance
from .patching import activation_patch
from .attribution import direct_contributions, output_lens

__all__ = [
    "collect_activations", "run_with_intervention",
    "activation_pca", "weight_spectrum",
    "ablate", "neuron_importance",
    "activation_patch",
    "direct_contributions", "output_lens"
]
