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
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict
import os
import time
import torch

from model import MLP, ModelConfig
from tasks import Task
from utils import pick_device, set_seed


@dataclass
class TrainConfig:
    steps: int = 8000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0          # 0.0 = pure MSE objective (the default study)
    optimizer: str = "adam"            # "adam" | "adamw" | "sgd"
    grad_clip: Optional[float] = None
    loss_tol: float = 0.0              # >0: early-stop once the step loss drops below this
    log_every: int = 200
    checkpoint_mode: str = "final"     # none | final | periodic | all
    checkpoint_every: int = 1000
    seed: int = 0
    device: str = "auto"


def _make_optimizer(model, cfg: TrainConfig):
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


class Trainer:
    def __init__(self, model: MLP, task: Task, cfg: TrainConfig,
                 checkpoint_dir: str = "checkpoints", run_name: str = "run"):
        self.model = model
        self.task = task
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.history: List[Tuple[int, float]] = []
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save(self, step: int, tag: str) -> str:
        path = os.path.join(self.checkpoint_dir, f"{self.run_name}_{tag}.pt")
        self.model.save(path, extra={
            "step": step,
            "history": self.history,
            "train_config": asdict(self.cfg),
            "run_name": self.run_name,
        })
        return path

    def train(self, progress: bool = True) -> Dict:
        set_seed(self.cfg.seed)
        self.model.to(self.device).train()
        opt = _make_optimizer(self.model, self.cfg)
        cfg = self.cfg
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            x, y = self.task.sample_batch(cfg.batch_size, self.device)
            out = self.model(x)
            loss = self.task.loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            opt.step()

            loss_val = float(loss.detach().cpu())
            if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
                self.history.append((step, loss_val))
                if progress:
                    print(f"[{self.run_name}] step {step:>6}/{cfg.steps}  "
                          f"loss {loss_val:.3e}", flush=True)
            if cfg.checkpoint_mode in ("periodic", "all") and step % cfg.checkpoint_every == 0:
                self._save(step, tag=f"step{step}")
            if cfg.loss_tol > 0.0 and loss_val < cfg.loss_tol:
                if self.history[-1][0] != step:
                    self.history.append((step, loss_val))
                break

        final_path = None
        if cfg.checkpoint_mode != "none":
            final_path = self._save(cfg.steps, tag="final")
        return {
            "history": self.history,
            "final_loss": self.history[-1][1] if self.history else None,
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
