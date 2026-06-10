"""Mecha_preds -- mechanistic predictors of a trained MLP's output.

A "mechanistic predictor" takes a trained ``model.MLP`` and predicts a statistic
of its output (e.g. the output mean over Gaussian inputs) without Monte-Carlo
sampling. For now the only predictor shipped here is cumulant propagation
(``Mecha_preds.cumulants``), which also includes a variant that computes the
second moment (the K=2 ReLU covariance) exactly.
"""
