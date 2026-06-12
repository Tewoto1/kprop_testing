"""experiments.py -- THE one place experiment settings and checkpoint recycling live.

Every notebook / script imports its knobs from here instead of re-defining them.
Full guide: EXPERIMENTS.md. Quick map of where each kind of setting is defined:

    architecture (width/depth/bias/activation) .. model.ModelConfig   (model/mlp.py)
    optimization (steps/batch/lr/optimizer) ..... training.TrainConfig (training/trainer.py)
    task / data (what the net is trained on) .... tasks/               (tasks/)
    grids, naming, checkpoint reuse ............. THIS FILE

Typical notebook cell:

    import experiments as E
    model, payload, loaded = E.get_or_train(
        E.ckpt_path("noiseless", E.run_name("readout-frozen_identity", depth=2, width=64)),
        build=lambda: E.build_frozen_identity(64, depth=2, seed=0),
        task=ZeroTask(input_dim=64, output_dim=64),
        train_cfg=E.default_train_cfg(64),
    )

`get_or_train` is the recycling rule of this repo: if the checkpoint already
exists it is loaded, otherwise it is trained and saved under that exact name --
so re-running a notebook never re-trains what is already on disk.
"""
from __future__ import annotations
import glob
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import torch

from model import MLP, ModelConfig
from tasks import Task
from training import TrainConfig, Trainer

# ---------------------------------------------------------------------------
# Standard sweep grids -- edit here and every notebook picks it up
# ---------------------------------------------------------------------------
WIDTHS: List[int] = [16, 32, 64, 128, 256, 512]   # the standard width sweep
QUICK_WIDTHS: List[int] = [32, 64, 128]      # smoke-test sweep (QUICK=True in notebooks)
DEPTHS: List[int] = [2, 3, 4]
SEEDS: List[int] = [0]
LR: float = 1e-3

def batch_steps(width: int) -> Tuple[int, int]:
    """Per-width (batch_size, steps): scaled down for wide nets so a step never
    blows up; Adam reaches ~1e-7 train-to-zero loss well inside these budgets."""
    if width <= 64:
        return 4096, 3000
    if width <= 128:
        return 2048, 1500
    return 1024, 1000

def default_train_cfg(width: int, seed: int = 0, **overrides) -> TrainConfig:
    """The standard Adam config for this repo, sized to `width`."""
    batch, steps = batch_steps(width)
    kw = dict(steps=steps, batch_size=batch, lr=LR, seed=seed)
    kw.update(overrides)
    return TrainConfig(**kw)

# ---------------------------------------------------------------------------
# Checkpoint locations & naming -- one root, one folder per study
# ---------------------------------------------------------------------------
CHECKPOINT_ROOT = "checkpoints"
STUDY_DIRS: Dict[str, str] = {
    # frozen/trainable readout + meanfield width grids ("noiseless" study)
    "noiseless": "noiseless_Layerless",
    # halfspace / max / zerobias weight-dissection checkpoints
    "weight_analysis": "weight_analysis_checkpoints",
}

def ckpt_dir(study: str) -> str:
    """checkpoints/<study folder>; created on demand."""
    d = os.path.join(CHECKPOINT_ROOT, STUDY_DIRS.get(study, study))
    os.makedirs(d, exist_ok=True)
    return d

def run_name(prefix: str, *, depth: int, width: int, seed: int = 0,
             **extras) -> str:
    """Canonical run name: <prefix>_d<depth>_w<width>[_k<v>...]_seed<seed>.

    `extras` (e.g. r=32, bs=4096) are appended in order as `_<k><v>` to match
    the existing meanfield names like meanfield_d3_w128_r32_bs4096_seed0.
    """
    parts = [f"{prefix}_d{depth}_w{width}"]
    parts += [f"{k}{v}" for k, v in extras.items()]
    parts.append(f"seed{seed}")
    return "_".join(parts)

def ckpt_path(study: str, name: str, tag: str = "final") -> str:
    return os.path.join(ckpt_dir(study), f"{name}_{tag}.pt")

_NAME_RE = re.compile(
    r"(?P<prefix>.+?)_d(?P<depth>\d+)_w(?P<width>\d+)(?P<extras>(_[a-z]+\d+)*)"
    r"_seed(?P<seed>\d+)_(?P<tag>\w+)\.pt$")

def parse_ckpt_name(path: str) -> Optional[dict]:
    """Parse a canonical checkpoint filename back into its fields (or None)."""
    m = _NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    d = dict(prefix=m["prefix"], depth=int(m["depth"]), width=int(m["width"]),
             seed=int(m["seed"]), tag=m["tag"], path=path)
    for extra in (m["extras"] or "").strip("_").split("_"):
        em = re.match(r"([a-z]+)(\d+)$", extra)
        if em:
            d[em.group(1)] = int(em.group(2))
    return d

def list_checkpoints(study: Optional[str] = None, pattern: str = "*.pt") -> List[dict]:
    """Scan checkpoint folders. ALWAYS call this (or get_or_train) before
    training: if a run already exists on disk, load it instead of re-training."""
    dirs = [ckpt_dir(study)] if study else \
           [os.path.join(CHECKPOINT_ROOT, d) for d in STUDY_DIRS.values()]
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, pattern))):
            out.append(parse_ckpt_name(p) or {"path": p})
    return out

# ---------------------------------------------------------------------------
# Model zoo -- the variants this project studies
# ---------------------------------------------------------------------------
def build_mlp(width: int, depth: int, *, output_dim: int = 1, seed: int = 0,
              input_dim: Optional[int] = None, device=None, dtype=None,
              **cfg_overrides) -> MLP:
    """Standard study MLP: square first layer (input_dim == width), no biases."""
    cfg = ModelConfig(input_dim=input_dim or width, hidden_dim=width, depth=depth,
                      output_dim=output_dim, seed=seed, **cfg_overrides)
    m = cfg.build()
    return m.to(device=device, dtype=dtype) if (device or dtype) else m

def build_frozen_identity(width: int, depth: int, seed: int = 0,
                          device=None, dtype=None) -> MLP:
    """Readout frozen to the identity -> the output IS the last hidden
    post-ReLU activations (the Q2 'frozen readout' condition)."""
    m = build_mlp(width, depth, output_dim=width, seed=seed, device=device, dtype=dtype)
    with torch.no_grad():
        m.readout.weight.copy_(torch.eye(width, dtype=m.readout.weight.dtype,
                                         device=m.readout.weight.device))
    m.readout.weight.requires_grad_(False)
    return m

# ---------------------------------------------------------------------------
# Checkpoint recycling -- the load-before-train rule, as one function
# ---------------------------------------------------------------------------
def get_or_train(path: str, build: Callable[[], MLP], task: Task,
                 train_cfg: TrainConfig, *, extra_meta: Optional[dict] = None,
                 load_existing: bool = True, map_location="cpu",
                 progress: bool = True) -> Tuple[MLP, dict, bool]:
    """Load the checkpoint at `path` if it exists, else build+train+save it there.

    Returns (model, payload, was_loaded). On load, `payload` is the checkpoint
    dict (history/train_config/...); on train, it is the Trainer result dict.
    `path` must follow the `ckpt_path(...)` convention (ends in `_<tag>.pt`).
    """
    if load_existing and os.path.exists(path):
        model, payload = MLP.load(path, map_location=map_location)
        return model, payload, True
    ckdir, base = os.path.split(path)
    name = re.sub(r"_\w+\.pt$", "", base)        # strip the trailing _<tag>.pt
    model = build()
    trainer = Trainer(model, task, train_cfg, checkpoint_dir=ckdir or ".",
                      run_name=name, extra_meta=extra_meta)
    result = trainer.train(progress=progress)
    return model, result, False

def final_loss(payload: dict) -> float:
    """Final training loss from either a loaded checkpoint payload or a Trainer
    result (handles both shapes)."""
    if payload.get("final_loss") is not None:
        return float(payload["final_loss"])
    hist = payload.get("history") or []
    return float(hist[-1][1]) if hist else float("nan")
