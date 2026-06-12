"""training/parallel.py -- train MANY same-architecture models in ONE loop (vmap).

The sequential pattern "for seed in SEEDS: Trainer(...).train()" wastes the GPU:
each small MLP underfills it, and N runs pay N x the kernel-launch overhead. Here
the N models' parameters are stacked into single tensors (leading dim = model
index) and one `torch.vmap`-ed forward/backward drives them all at once, so a
4-seed sweep costs roughly ONE training run of wall time.

Why this is exactly equivalent to N independent runs (not an approximation):
  * the loss is `sum_i loss_i(params_i)`, so `d(total)/d(params_i) = d(loss_i)/
    d(params_i)` -- gradients never mix across models;
  * Adam/AdamW/SGD updates are elementwise, so one optimizer on the stacked
    tensors performs N independent per-model updates.
Only the data stream differs from a sequential run (batches are drawn from one
generator for all models), which is irrelevant for fresh-sampled (infinite-data)
tasks like ZeroTask.

Early stopping is per model: every `cfg.tol_check_every` steps the per-model
losses are synced once; a model that first drops below `cfg.loss_tol` has its
parameters snapshotted at that step (training continues for the stragglers, but
the returned/saved state of a converged model is its at-tol snapshot -- it is NOT
trained further). The loop ends when all models converged or `cfg.steps` is hit.

Use `train_many` (falls back to sequential `Trainer`s if torch.func/vmap is
unavailable on the device) or `train_ensemble` directly.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import os
import time
import warnings

import torch

from model import MLP
from tasks import Task
from utils import pick_device, pick_dtype, enable_fast_matmul, set_seed
from .trainer import TrainConfig, Trainer, _make_optimizer


def _slice_state(params: Dict[str, torch.Tensor], buffers: Dict[str, torch.Tensor],
                 i: int) -> Dict[str, torch.Tensor]:
    state = {k: v[i].detach().clone() for k, v in params.items()}
    state.update({k: v[i].detach().clone() for k, v in buffers.items()})
    return state


def train_ensemble(models: Sequence[MLP], task: Task, cfg: TrainConfig, *,
                   run_names: Optional[Sequence[str]] = None,
                   checkpoint_dir: Optional[str] = None,
                   extra_meta: Optional[dict] = None,
                   progress: bool = True) -> List[Tuple[MLP, Dict]]:
    """Train N same-architecture models simultaneously. Returns [(model, result)].

    `result` mirrors `Trainer.train`: history / final_loss / steps_run /
    final_checkpoint / seconds, plus `converged` when `cfg.loss_tol > 0`.
    Checkpoints are written iff `checkpoint_dir` and `run_names` are given (one
    `<run_name>_final.pt` per model, same payload shape as `Trainer`).
    Note: `grad_clip` is applied to the stacked ensemble jointly, not per model.
    """
    from torch.func import stack_module_state, functional_call

    n_models = len(models)
    assert n_models > 0
    device, dtype = pick_device(cfg.device), pick_dtype(getattr(cfg, "dtype", "auto"))
    enable_fast_matmul(device)
    set_seed(cfg.seed)
    models = [m.to(device=device, dtype=dtype) for m in models]
    for m in models:
        m.train()

    base = copy.deepcopy(models[0]).to("meta")          # structure only, no storage
    params, buffers = stack_module_state(models)        # copies; originals untouched
    trainable = [p for p in params.values() if p.requires_grad]   # frozen layers excluded

    class _P:                                            # adapter for _make_optimizer
        def parameters(self):
            return trainable
    opt = _make_optimizer(_P(), cfg, device)

    def one_model_loss(p, b, x, y):
        return task.loss(functional_call(base, (p, b), (x,)), y)

    loss_fn = torch.vmap(one_model_loss, in_dims=(0, 0, 0, 0))

    histories: List[List[Tuple[int, float]]] = [[] for _ in range(n_models)]
    done_step: List[Optional[int]] = [None] * n_models
    snapshots: List[Optional[dict]] = [None] * n_models
    tol_every = max(1, cfg.tol_check_every) if cfg.loss_tol > 0.0 else 0
    t0, last_step = time.time(), cfg.steps

    for step in range(1, cfg.steps + 1):
        x, y = task.sample_batch(n_models * cfg.batch_size, device)
        x = x.to(dtype).reshape(n_models, cfg.batch_size, *x.shape[1:])
        y = y.to(dtype).reshape(n_models, cfg.batch_size, *y.shape[1:])
        losses = loss_fn(params, buffers, x, y)          # (n_models,)
        opt.zero_grad(set_to_none=True)
        losses.sum().backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()

        need_log = step == 1 or step % cfg.log_every == 0 or step == cfg.steps
        need_tol = tol_every and step % tol_every == 0
        if need_log or need_tol:
            lv = losses.detach().cpu().tolist()          # the only host sync
            if need_log:
                for i in range(n_models):
                    if done_step[i] is None:
                        histories[i].append((step, lv[i]))
                if progress:
                    live = [v for i, v in enumerate(lv) if done_step[i] is None]
                    print(f"[ensemble x{n_models}] step {step:>6}/{cfg.steps}  "
                          f"active {len(live)}  loss min {min(live):.3e} "
                          f"max {max(live):.3e}" if live else
                          f"[ensemble x{n_models}] step {step}: all converged", flush=True)
            if cfg.loss_tol > 0.0:
                for i in range(n_models):
                    if done_step[i] is None and lv[i] < cfg.loss_tol:
                        done_step[i] = step
                        snapshots[i] = _slice_state(params, buffers, i)
                        if not histories[i] or histories[i][-1][0] != step:
                            histories[i].append((step, lv[i]))
                if all(s is not None for s in done_step):
                    last_step = step
                    break

    results: List[Tuple[MLP, Dict]] = []
    seconds = time.time() - t0
    for i, m in enumerate(models):
        state = snapshots[i] if snapshots[i] is not None else _slice_state(params, buffers, i)
        m.load_state_dict(state)
        m.eval()
        steps_run = done_step[i] if done_step[i] is not None else last_step
        final_loss = histories[i][-1][1] if histories[i] else None
        ckpt = None
        if checkpoint_dir and run_names and cfg.checkpoint_mode != "none":
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = os.path.join(checkpoint_dir, f"{run_names[i]}_final.pt")
            m.save(ckpt, extra={
                "step": steps_run, "history": histories[i], "final_loss": final_loss,
                "train_config": asdict(cfg), "run_name": run_names[i],
                "ensemble_size": n_models, **(extra_meta or {}),
            })
        results.append((m, {
            "history": histories[i], "final_loss": final_loss, "steps_run": steps_run,
            "final_checkpoint": ckpt, "seconds": seconds,
            "converged": (done_step[i] is not None) if cfg.loss_tol > 0.0 else None,
        }))
    return results


def train_many(models: Sequence[MLP], task: Task, cfg: TrainConfig, *,
               run_names: Optional[Sequence[str]] = None,
               checkpoint_dir: Optional[str] = None,
               extra_meta: Optional[dict] = None,
               progress: bool = True) -> List[Tuple[MLP, Dict]]:
    """`train_ensemble` with a sequential-`Trainer` fallback (vmap unsupported,
    old torch, exotic device). Same return shape either way."""
    if len(models) > 1:
        try:
            return train_ensemble(models, task, cfg, run_names=run_names,
                                  checkpoint_dir=checkpoint_dir, extra_meta=extra_meta,
                                  progress=progress)
        except (ImportError, RuntimeError, NotImplementedError) as e:
            warnings.warn(f"ensemble training unavailable ({e!r}); falling back to "
                          f"sequential training.")
    out = []
    for i, m in enumerate(models):
        name = run_names[i] if run_names else f"run{i}"
        tr = Trainer(m, task, cfg, checkpoint_dir=checkpoint_dir or "checkpoints",
                     run_name=name, extra_meta=extra_meta)
        if checkpoint_dir is None:                      # honor "no checkpoints" intent
            tr.cfg = TrainConfig(**{**asdict(cfg), "checkpoint_mode": "none"})
        out.append((m, tr.train(progress=progress)))
    return out
