"""training/run.py -- one CLI to train any task over a depth x width x seed grid.

    python -m training.run --task zero      --widths 64 128 256 --depths 2 3 --steps 8000
    python -m training.run --task halfspace --widths 128 --depths 3 --offset-std 1.0
    python -m training.run --task distill   --widths 128 --depths 3 --teacher-seed 1

Everything is a flag; nothing about the grid, dims, or optimizer is hard-coded.
By design `input_dim` defaults to `width` (a square first layer W1, matching the
study), and biases are OFF by default (half-space boundaries through the origin).
Each run is checkpointed as `{task}_d{depth}_w{width}_seed{seed}_final.pt`.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Optional, Sequence, Dict
import argparse

from model import ModelConfig
from tasks import Task, ZeroTask, HalfspaceTask, DistillTask, random_teacher
from .trainer import TrainConfig, train_model


def build_task(task: str, *, input_dim: int, output_dim: int, activation: str, bias: bool,
               depth: int, width: int, seed: int, offset_std: float,
               teacher_seed: Optional[int]) -> Task:
    """Construct the requested task, sized to `input_dim`/`output_dim`."""
    if task == "zero":
        return ZeroTask(input_dim=input_dim, output_dim=output_dim)
    if task == "halfspace":
        return HalfspaceTask(input_dim=input_dim, offset_std=offset_std, seed=seed)
    if task == "distill":
        teacher = random_teacher(input_dim=input_dim, hidden_dim=width, depth=depth,
                                 output_dim=output_dim, activation=activation, bias=bias,
                                 seed=(teacher_seed if teacher_seed is not None else seed + 10_000))
        return DistillTask(teacher)
    raise ValueError(f"unknown task {task!r}")


def train_grid(task: str, *, widths: Sequence[int], depths: Sequence[int],
               seeds: Sequence[int] = (0,), input_dim: Optional[int] = None,
               output_dim: int = 1, activation: str = "relu", bias: bool = False,
               offset_std: float = 1.0, teacher_seed: Optional[int] = None,
               train_cfg: Optional[TrainConfig] = None, checkpoint_dir: str = "checkpoints",
               checkpoint_mode: str = "final", progress: bool = True) -> Dict[str, dict]:
    """Train every (depth, width, seed) for `task`. Returns {run_name: result}.

    `input_dim` defaults to `width` per run (square first layer). HalfspaceTask
    forces output_dim=1.
    """
    base_cfg = train_cfg or TrainConfig()
    results: Dict[str, dict] = {}
    for depth in depths:
        for width in widths:
            for seed in seeds:
                in_dim = width if input_dim is None else input_dim
                out_dim = 1 if task == "halfspace" else output_dim
                task_obj = build_task(task, input_dim=in_dim, output_dim=out_dim,
                                      activation=activation, bias=bias, depth=depth,
                                      width=width, seed=seed, offset_std=offset_std,
                                      teacher_seed=teacher_seed)
                model_cfg = ModelConfig(input_dim=in_dim, hidden_dim=width, depth=depth,
                                        output_dim=out_dim, bias=bias, final_bias=bias,
                                        activation=activation, seed=seed)
                cfg = replace(base_cfg, seed=seed, checkpoint_mode=checkpoint_mode)
                run_name = f"{task}_d{depth}_w{width}_seed{seed}"
                if progress:
                    print(f"=== training {run_name} ===", flush=True)
                _, res = train_model(model_cfg, task_obj, cfg, checkpoint_dir=checkpoint_dir,
                                     run_name=run_name, progress=progress)
                results[run_name] = res
                if progress and res["final_loss"] is not None:
                    print(f"  -> final loss {res['final_loss']:.3e} ({res['seconds']:.1f}s)\n",
                          flush=True)
    return results


def main():
    # optimization defaults come straight from TrainConfig -- ONE source of truth
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train MLPs on a task over a depth x width grid.")
    p.add_argument("--task", choices=["zero", "halfspace", "distill"], default="zero")
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--input-dim", type=int, default=None, help="defaults to width (square W1)")
    p.add_argument("--output-dim", type=int, default=1)
    p.add_argument("--activation", default="relu")
    p.add_argument("--bias", action="store_true", help="enable biases (off by default)")
    p.add_argument("--offset-std", type=float, default=1.0, help="halfspace: offset b ~ N(0, this^2)")
    p.add_argument("--teacher-seed", type=int, default=None, help="distill: teacher init seed")
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--optimizer", default=d.optimizer, choices=["adam", "adamw", "sgd"])
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--checkpoint-mode", default=d.checkpoint_mode, choices=["none", "final", "periodic", "all"])
    p.add_argument("--checkpoint-every", type=int, default=d.checkpoint_every)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--device", default=d.device)
    args = p.parse_args()

    train_cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                            weight_decay=args.weight_decay, optimizer=args.optimizer,
                            grad_clip=args.grad_clip, checkpoint_every=args.checkpoint_every,
                            device=args.device)
    train_grid(args.task, widths=args.widths, depths=args.depths, seeds=args.seeds,
               input_dim=args.input_dim, output_dim=args.output_dim, activation=args.activation,
               bias=args.bias, offset_std=args.offset_std, teacher_seed=args.teacher_seed,
               train_cfg=train_cfg, checkpoint_dir=args.checkpoint_dir,
               checkpoint_mode=args.checkpoint_mode)


if __name__ == "__main__":
    main()
