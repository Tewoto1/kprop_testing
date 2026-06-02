# Cumulant propagation vs Monte-Carlo for MLPs trained to output zero

This experiment answers one question:

> Does the **real** cumulant propagation algorithm in this repo (`src/mlp_kprop`,
> the "kprop" algorithm from the paper) still accurately predict the **output
> mean** of a simple MLP after the model is randomly initialized and then trained
> only to output `0`?

The distribution is over the **input** `X ~ N(0, I_input_dim)`. The model weights
are **fixed** after training. We never average over weights.

## What is (and is not) used

- The real algorithm is `src.mlp_kprop.kprop_harmonic.mlp_kprop`. It is treated as
  a **black box** — we do not reimplement, linearize, push only means through the
  net, assume Gaussianity per layer, or use Monte-Carlo samples inside the
  prediction.
- The model is the repo's own `src.mlp_kprop.mlp.MLP`, because `mlp_kprop`
  consumes an `MLP` instance (it reads `mlp.Ws`, `mlp.nonlin_names`,
  `mlp.init_scale`). Using the repo model guarantees the trained model and the
  model cumulant propagation sees are byte-for-byte identical.

## Files

- `cumulant_adapter.py` — `run_cumulant_propagation_from_model(model, input_dim,
  cumulant_config, device)`: extracts weights/biases, **verifies orientation**
  (PyTorch `nn.Linear` is `(out, in)`; kprop contracts over `in`, so **no
  transpose**), builds `K_in = {1: zeros(n), 2: eye(n)}` for `X ~ N(0, I)`, calls
  `mlp_kprop`, and returns `{"raw_output", "mean", "metadata"}`. Also contains the
  two sanity checks.
- `model_utils.py` — `make_mlp`, `train_model_to_zero` (MSE to the all-zeros
  target on fresh Gaussian inputs each step), seeding, layer-norm logging.
- `metrics.py` — `estimate_empirical_mean` (64,000-sample MC, batched, float64,
  `no_grad`) and `compare_means`. **Headline metric: relative error of the mean**
  `|cp − mc| / |mc| = √NMSE` (scale-free; does not divide by anything that itself
  collapses as the output → 0). Plus `mc_noise_z = |cp − mc| / MC_stderr`, which
  says whether the cp-vs-MC gap is a real bias (z ≫ 1) or just MC sampling noise
  on the tiny mean (z ≲ 1). Raw abs error and NMSE are also recorded.
- `plotting.py` — median + IQR-band curves vs width.
- `../experiments/run_cumulant_train_to_zero_mean.py` — the CLI runner.

## Sanity checks (run before the sweep unless `--no-sanity`)

1. **Single linear layer** `y = Wx + b`: the true mean is exactly `b`. Cumulant
   propagation must reproduce `b` (observed error: `0.0`). This validates weight
   orientation and bias handling.
2. **Small untrained MLP**: cumulant mean vs a multi-million-sample MC mean
   (observed relative error ≈ 0.7%).

## Numerical details

- Everything runs in **float64** (cumulant propagation is run in double precision
  in the repo's own tests).
- `k_max` (the budget `K`) defaults to **3** and is **hard-capped at 3** —
  `k_max=4` OOMs at width 1024 on CPU. `factor=True` is used for `k_max=3`.
- `use_avg_metric` defaults to **False**: because the weights are fixed after
  training, the exact metric `W @ metric @ Wᵀ` is the faithful choice (the
  init-time metric `E[WWᵀ]` would inject stale assumptions). Configurable via
  `--use-avg-metric`.

## How to run

From the **repo root**, with `uv`:

```bash
uv run python experiments/run_cumulant_train_to_zero_mean.py \
    --input-dim 64 --output-dim 1 --hidden-depth 3 \
    --widths 64 128 256 512 --seeds 0 1 2 3 4 \
    --mc-samples 64000 --train-steps 5000 --batch-size 1024 --lr 1e-3 \
    --k-max 3 --outdir results/cumulant_train_to_zero
```

Width 1024 is optional (slower): add it to `--widths`.

Useful flags: `--skip-training` (only initial models), `--no-plots`, `--debug`
(prints architecture, layer/converted shapes, kprop output keys, first outputs),
`--no-sanity`, `--no-bias`, `--use-avg-metric`.

## Outputs

- `results/cumulant_train_to_zero/cumulant_train_to_zero_mean_results.csv`
  (written incrementally after every model, so partial progress is never lost).
  One row per (width, seed, phase∈{initial, trained_to_zero}).
- `results/cumulant_train_to_zero/config.json` — full experiment settings.
- `results/cumulant_train_to_zero/plots/*.png` — error vs width (abs error, NMSE,
  variance-normalized error, output RMS, final train loss).

## Reading the metrics

The model is trained toward zero, so its mean is tiny. The correct scale-free
measure of "does cumulant propagation predict the mean" is the **relative error**
`|cp − mc| / |mc|` (= √NMSE). Do **not** normalize by `E‖Y‖²` — that denominator
*also* collapses as the output → 0, which can make a *degrading* estimate look
like it is improving (an earlier version of this experiment made that mistake).

One more subtlety: when the true mean is tiny, the 64k-sample MC estimate of it
is itself only good to a few ×0.1%. So we report `mc_noise_z = |cp − mc| /
MC_stderr`: `z ≲ 1` means the cp-vs-MC gap is within MC's own sampling noise
(cumulant propagation is at least as good as 64k MC); `z ≫ 1` means a real bias.

To make the relative error itself less MC-noise-limited at the widths where the
trained mean is smallest, increase `--mc-samples`.

### Result of the default run (widths 64/128/256/512, seeds 0–4)

With the correct scale, training to zero **degrades** cumulant propagation's mean
prediction. At initialization the cp-vs-MC gap is within MC noise at every width
(median z ≈ 0.6–1). After training, a statistically real bias appears — median
z ≈ 6 at width 64, 3.8 at 128 — i.e. relative error ~3–6× larger than at init.
The bias shrinks with width (z: 6 → 3.8 → 1.1 → 0.9), so wider trained nets
recover toward the random-limit accuracy; at width 512 the test is MC-noise-
limited (z ≈ 1) and cannot resolve residual bias without more samples. This is
consistent with training inducing weight/activation correlations that the
wide-random-MLP approximation behind cumulant propagation does not capture.
