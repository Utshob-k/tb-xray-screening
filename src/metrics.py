"""Metrics used for threshold selection and evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_at_specificity(y_true, y_prob, target_spec: float = 0.90) -> float:
    """Find the lowest threshold that meets the requested specificity."""
    _validate_binary_inputs(y_true, y_prob)
    if not 0 < target_spec <= 1:
        raise ValueError("target_spec must be in (0, 1]")
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(1 - fpr >= target_spec)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def sens_spec(y_true, y_prob, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    pred = np.asarray(y_prob) >= threshold
    tp = np.sum(pred & (y_true == 1))
    fn = np.sum(~pred & (y_true == 1))
    tn = np.sum(~pred & (y_true == 0))
    fp = np.sum(pred & (y_true == 0))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def _validate_binary_inputs(y_true, y_prob) -> None:
    """Catch invalid labels before asking sklearn for an AUROC."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be one-dimensional arrays of equal length")
    if not len(y_true):
        raise ValueError("metrics need at least one sample")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUROC and threshold selection need both classes")


def bootstrap_ci(y_true, y_prob, fn, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Return a percentile bootstrap interval for a metric."""
    _validate_binary_inputs(y_true, y_prob)
    if n < 1:
        raise ValueError("n must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    # Each draw needs both classes; small datasets can lose one by chance.
    class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in class_indices])
        stats.append(fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(y_true, y_prob, threshold: float) -> dict:
    _validate_binary_inputs(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))
    sens, spec = sens_spec(y_true, y_prob, threshold)
    return {
        "n": int(len(y_true)),
        "n_tb": int(np.sum(y_true)),
        "auroc": auc,
        "auroc_ci95": bootstrap_ci(y_true, y_prob, roc_auc_score),
        "sensitivity": sens,
        "sensitivity_ci95": bootstrap_ci(y_true, y_prob, lambda y, p: sens_spec(y, p, threshold)[0]),
        "specificity": spec,
        "specificity_ci95": bootstrap_ci(y_true, y_prob, lambda y, p: sens_spec(y, p, threshold)[1]),
        "threshold": threshold,
    }
