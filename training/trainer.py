"""training/trainer.py -- the optimization loop and checkpointing.

A `Trainer` runs a `tasks.Task` against a `model.MLP`: it draws fresh batches,
applies the task's loss, steps the optimizer, records the loss history, and writes
self-describing checkpoints per `TrainConfig.checkpoint_mode`:

    "none"      no checkpoints
    "final"     one checkpoint at the end                 (default)
    "periodic"  every `checkpoint_every` steps + a final
    "all"       alias of "periodic"

Checkpoints are .pt files holding the model config, state dict, step, full loss
history and the train config -- everything the analysis tools need to reconstruct
a run (load with `model.MLP.load(path)`).

Numerics: training runs in `TrainConfig.dtype` (float32 by default -- the repo
policy; float64 is opt-in for accuracy studies). The model and every batch are cast
to that dtype, TF32 matmuls are enabled on CUDA, Adam/AdamW use the fused CUDA
kernel, and the loss is synced to host only on log/`tol_check_every` steps (a
per-step `.cpu()` would serialize the GPU). To train many same-architecture models
at once (seeds of one width) see `training.parallel.train_ensemble`.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import os
import time
import torch

from model import MLP, ModelConfig
from tasks import Task
from utils import pick_device, pick_dtype, enable_fast_matmul, set_seed


@dataclass
class TrainConfig:
    steps: int = 8000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0          # 0.0 = pure MSE objective (the default study)
    optimizer: str = "adam"            # "adam" | "adamw" | "sgd"
    grad_clip: Optional[float] = None
    loss_tol: float = 0.0              # >0: early-stop once the step loss drops below this
    tol_check_every: int = 50          # how often (in steps) the loss_tol test syncs to host
    tol_patience: int = 1              # stop only after this many CONSECUTIVE sub-tol checks
                                       # (a single above-tol check resets the count) -- guards
                                       # against stopping on one lucky batch
    log_every: int = 200
    checkpoint_mode: str = "final"     # none | final | periodic | all
    checkpoint_every: int = 1000
    seed: int = 0
    device: str = "auto"
    dtype: str = "float32"             # "float32" (repo policy; GPU-fast) | "float64" | "auto"


def _make_optimizer(model, cfg: TrainConfig, device: torch.device):
    params = [p for p in model.parameters() if p.requires_grad]   # skip frozen weights
    name = cfg.optimizer.lower()
    # fused=True runs the whole Adam update as one kernel on CUDA (fewer launches)
    fused = {"fused": True} if (device.type == "cuda" and name in ("adam", "adamw")) else {}
    try:
        if name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay, **fused)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, **fused)
    except (RuntimeError, TypeError):   # fused unavailable on this build -> plain version
        if name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


class Trainer:
    def __init__(self, model: MLP, task: Task, cfg: TrainConfig,
                 checkpoint_dir: str = "checkpoints", run_name: str = "run",
                 extra_meta: Optional[dict] = None):
        self.model = model
        self.task = task
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.dtype = pick_dtype(getattr(cfg, "dtype", "auto"))
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.extra_meta = extra_meta or {}
        self.history: List[Tuple[int, float]] = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save(self, step: int, tag: str) -> str:
        path = os.path.join(self.checkpoint_dir, f"{self.run_name}_{tag}.pt")
        self.model.save(path, extra={
            "step": step,
            "history": self.history,
            "final_loss": self.history[-1][1] if self.history else None,
            "train_config": asdict(self.cfg),
            "run_name": self.run_name,
            **self.extra_meta,
        })
        return path

    def train(self, progress: bool = True) -> Dict:
        set_seed(self.cfg.seed)
        enable_fast_matmul(self.device)                     # TF32 on CUDA (no-op elsewhere)
        self.model.to(device=self.device, dtype=self.dtype).train()
        opt = _make_optimizer(self.model, self.cfg, self.device)
        cfg = self.cfg
        # Syncing loss to host every step serializes the GPU (one .cpu() per step
        # blocks the launch queue). Only sync when we actually need the number:
        tol_every = max(1, cfg.tol_check_every) if cfg.loss_tol > 0.0 else 0
        below = 0                                           # consecutive sub-tol checks
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            x, y = self.task.sample_batch(cfg.batch_size, self.device)
            if x.dtype != self.dtype:
                x, y = x.to(self.dtype), y.to(self.dtype)
            out = self.model(x)
            loss = self.task.loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            opt.step()

            need_log = step == 1 or step % cfg.log_every == 0 or step == cfg.steps
            need_tol = tol_every and step % tol_every == 0
            if need_log or need_tol:
                loss_val = float(loss.detach().cpu())       # the only host sync
                if need_log:
                    self.history.append((step, loss_val))
                    if progress:
                        print(f"[{self.run_name}] step {step:>6}/{cfg.steps}  "
                              f"loss {loss_val:.3e}", flush=True)
                if cfg.loss_tol > 0.0:
                    below = below + 1 if loss_val < cfg.loss_tol else 0
                    if below >= max(1, cfg.tol_patience):   # stably below tol -> stop
                        if not self.history or self.history[-1][0] != step:
                            self.history.append((step, loss_val))
                        break
            if cfg.checkpoint_mode in ("periodic", "all") and step % cfg.checkpoint_every == 0:
                self._save(step, tag=f"step{step}")

        final_path = None
        last_step = self.history[-1][0] if self.history else cfg.steps
        if cfg.checkpoint_mode != "none":
            final_path = self._save(last_step, tag="final")
        return {
            "history": self.history,
            "final_loss": self.history[-1][1] if self.history else None,
            "steps_run": last_step,
            "final_checkpoint": final_path,
            "seconds": time.time() - t0,
        }


def train_model(model_cfg: ModelConfig, task: Task, train_cfg: TrainConfig,
                checkpoint_dir: str = "checkpoints", run_name: str = "run",
                progress: bool = True):
    """Build a model from `model_cfg`, train it on `task`, return (model, result)."""
    model = model_cfg.build()
    trainer = Trainer(model, task, train_cfg, checkpoint_dir, run_name)
    result = trainer.train(progress=progress)
    return model, result
