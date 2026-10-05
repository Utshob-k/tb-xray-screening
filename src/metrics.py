"""Screening metrics. Accuracy alone is misleading here, so report these instead."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def threshold_at_specificity(y_true, y_prob, target_spec: float = 0.90) -> float:
    """Pick the decision threshold on VALIDATION data, never on the test set."""
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


def bootstrap_ci(y_true, y_prob, fn, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for any metric fn(y_true, y_prob). Small test sets make this important."""
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:  # resample landed on one class only
            continue
        stats.append(fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(y_true, y_prob, threshold: float) -> dict:
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
