"""Task-specific performance measures used in the empirical comparison."""

from __future__ import annotations

import numpy as np

from src.network import MLP


def classification_scores(net: MLP, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    yhat, _ = net.forward(x, w)
    pred = yhat.argmax(axis=1)
    true = y.argmax(axis=1)
    accuracy = float(np.mean(pred == true))
    n_classes = y.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0.0 else 2 * prec * rec / (prec + rec))
    return {
        "error": net.error(x, y, w),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
    }


def regression_scores(net: MLP, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    yhat, _ = net.forward(x, w)
    mse = float(np.mean((yhat - y) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"error": mse, "mse": mse, "rmse": rmse, "r2": r2}


def evaluate(net: MLP, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    if net.task == "classification":
        return classification_scores(net, x, y, w)
    return regression_scores(net, x, y, w)
