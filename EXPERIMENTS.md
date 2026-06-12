# EXPERIMENTS.md — how to run, change, and add experiments

The working manual for this repo. If you are about to probe a training parameter,
start here.

## 1. Where every setting is defined

| Setting | Lives in | Examples |
|---|---|---|
| **Sweep grids & defaults** (widths, depths, seeds, lr, per-width batch/steps) | `experiments.py` (top of file) | `WIDTHS`, `DEPTHS`, `SEEDS`, `LR`, `batch_steps(w)` |
| **Architecture** | `ModelConfig` in `model/mlp.py` | `hidden_dim`, `depth`, `bias`, `activation`, `seed` |
| **Optimization** | `TrainConfig` in `training/trainer.py` | `steps`, `batch_size`, `lr`, `optimizer`, `loss_tol`, `checkpoint_mode` |
| **Task / data** | one file per task in `tasks/` | `ZeroTask`, `HalfspaceTask`, `DistillTask` |
| **Checkpoint folders & naming** | `experiments.py` (`STUDY_DIRS`, `run_name`, `ckpt_path`) | `"noiseless"` → `checkpoints/noiseless_Layerless/` |
| **Predictor (kprop) knobs** | `Mecha_preds/cumulants/adapter.py` | `k_max`, `exact_relu_cov` |
| **Analysis tools** | `analysis/Tools/` | `weight_spectrum`, `weight_structure_metrics`, … |

Rule of thumb: **notebooks define no knobs of their own** — they `import experiments as E`
and read everything from there. Change a default once in `experiments.py` and every
notebook picks it up on the next run.

`training/run.py` (the CLI) takes its argparse defaults directly from `TrainConfig()`,
so changing a default in `trainer.py` changes the CLI default too — there is no second
copy to keep in sync.

## 2. Changing a setting — real examples

**Change the learning rate for everything:** edit `LR = 1e-3` in `experiments.py`.
For one run only: `python -m training.run --task zero --lr 3e-4 ...` or
`E.default_train_cfg(width, lr=3e-4)` in a notebook.

**Change the width sweep:** edit `WIDTHS = [16, 32, 64, 128, 256]` in `experiments.py`
(`QUICK_WIDTHS` is the smoke-test sweep notebooks use when `QUICK=True`).

**Change steps/batch for wide nets:** edit `batch_steps(width)` in `experiments.py` —
it returns `(batch_size, steps)` per width and is the single per-width budget rule.

**Turn biases on:** `--bias` on the CLI, or `ModelConfig(..., bias=True, final_bias=True)`.
They are OFF by default on purpose (half-space boundaries through the origin).

**Train to an exact loss instead of a step count:** `TrainConfig(loss_tol=1e-8)` —
early-stops once the step loss drops below it.

## 3. Running experiments

**CLI** (grid over depth × width × seed, one checkpoint per cell):

```bash
python -m training.run --task zero      --widths 64 128 256 --depths 2 3 --steps 8000
python -m training.run --task halfspace --widths 128 --depths 3 --offset-std 1.0
python -m training.run --task distill   --widths 128 --depths 3 --teacher-seed 1
```

**Programmatic / notebook** (this is the pattern every notebook uses):

```python
import experiments as E
from tasks import ZeroTask

w, d, seed = 64, 2, 0
model, payload, loaded = E.get_or_train(
    E.ckpt_path("noiseless", E.run_name("readout-trainable", depth=d, width=w, seed=seed)),
    build=lambda: E.build_mlp(w, d, output_dim=w, seed=seed),
    task=ZeroTask(input_dim=w, output_dim=w),
    train_cfg=E.default_train_cfg(w, seed=seed),
)
print("loaded from disk:" if loaded else "trained fresh:", E.final_loss(payload))
```

**kprop comparison sweep** (trains + predicts + writes CSV/plots):

```bash
python -m Mecha_preds.cumulants.run_comparison --task zero --input-dim 64 \
    --hidden-depth 3 --widths 64 128 256 --seeds 0 1 2 --k-max 3 --outdir results/zero_kprop
```

## 4. Checkpoint recycling — the rule of this repo

**Never retrain what is already on disk.** Before any training, check the checkpoint
folders; `E.get_or_train(path, ...)` does this for you (loads if `path` exists, trains
and saves there otherwise). To see what exists:

```python
import experiments as E
for c in E.list_checkpoints("noiseless"):       # or list_checkpoints() for all studies
    print(c)   # {'prefix': 'readout-frozen_identity', 'depth': 2, 'width': 64, 'seed': 0, ...}
```

**Naming convention** (parsed by `E.parse_ckpt_name`):

```
<prefix>_d<depth>_w<width>[_<extra><val>...]_seed<seed>_<tag>.pt
readout-frozen_identity_d2_w64_seed0_final.pt
meanfield_d3_w128_r32_bs4096_seed0_final.pt
```

**Current studies** (`STUDY_DIRS` in `experiments.py`):

| Study key | Folder | Contents |
|---|---|---|
| `noiseless` | `checkpoints/noiseless_Layerless/` | frozen/trainable readout (d2 width sweep) + meanfield grids |
| `weight_analysis` | `checkpoints/weight_analysis_checkpoints/` | halfspace / max / zerobias dissection models |

Checkpoints are self-describing `.pt` files: `model_config`, `state_dict`, `step`,
`history`, `final_loss`, `train_config` (+ any `extra_meta`). Load with
`MLP.load(path)` → `(model, payload)`.

⚠️ The depth-3 frozen/trainable checkpoints in `noiseless_Layerless` stalled at loss
~1e-3 (undertrained — no weight structure). Use the depth-2 set for Q2 conclusions;
see §6 of the frozen-readout notebook.

## 5. Adding a new experiment

1. **New task?** Add one file in `tasks/` subclassing `tasks.base.Task`
   (implement `sample_batch`; override `MSE_loss` only for a non-MSE objective),
   export it in `tasks/__init__.py`, and add a branch in `training/run.py::build_task`
   if you want it on the CLI.
2. **New checkpoint family?** Add a `STUDY_DIRS` entry in `experiments.py` and pick a
   `run_name` prefix. Don't invent ad-hoc folder names in notebooks.
3. **New notebook?** Notebooks are *generated* from `build_*.py` scripts (edit the
   script, re-run it — keeps them reproducible & diffable). Start from
   `colab_notebooks/noiseless_and_frozen_readout/build_frozen_readout_structure_notebook.py`,
   the reference example. Builders share `colab_notebooks/_nb.py`
   (`NotebookBuilder` + the standard `BOOTSTRAP_CELL` repo-locator); don't re-define
   cell helpers or bootstrap code.
4. **New reusable metric?** Put it in `analysis/Tools/` and export it from
   `analysis/__init__.py` — never define a metric inline in a notebook if a second
   notebook might want it.

## 6. Gotchas

- **float64 everywhere in analysis/kprop**: notebooks call
  `torch.set_default_dtype(torch.float64)`; `run_cumulants` builds a float64 copy
  internally.
- **kprop needs Python ≥ 3.12** (the vendored code uses newer syntax). Training and
  analysis work on older versions.
- **Run from the repo root** — everything is a plain package, no install. Generated
  notebooks locate the root via their bootstrap cell.
- **`results/` is git-ignored** (regenerable); `checkpoints/*.pt` ARE tracked on
  purpose — they are the "one learned case".
- The Trainer skips frozen params (`requires_grad=False`) automatically — freezing a
  layer requires no custom loop.
