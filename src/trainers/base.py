"""Shared training result and stopping logic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.network import MLP


@dataclass
class TrainResult:
    weights: np.ndarray
    best_weights: np.ndarray
    train_errors: list[float] = field(default_factory=list)
    val_errors: list[float] = field(default_factory=list)
    grad_evals_trace: list[int] = field(default_factory=list)
    grad_evals: int = 0
    fn_evals: int = 0
    iterations: int = 0
    best_val: float = float("inf")
    stopped_reason: str = ""
    wall_time: float = 0.0

    def as_dict(self) -> dict:
        return {
            "grad_evals": self.grad_evals,
            "fn_evals": self.fn_evals,
            "iterations": self.iterations,
            "best_val": self.best_val,
            "stopped_reason": self.stopped_reason,
            "wall_time": self.wall_time,
        }


class EarlyStopper:
    """Restore the weights with the lowest validation error after a plateau."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.best_w: np.ndarray | None = None
        self.bad = 0

    def update(self, val_error: float, w: np.ndarray) -> bool:
        if val_error < self.best - 1e-12:
            self.best = val_error
            self.best_w = w.copy()
            self.bad = 0
            return False
        self.bad += 1
        return self.patience > 0 and self.bad >= self.patience


def record(
    net: MLP,
    w: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    result: TrainResult,
) -> float:
    train_e = net.error(x_train, y_train, w)
    result.fn_evals += 1
    result.train_errors.append(train_e)
    result.grad_evals_trace.append(result.grad_evals)
    if x_val is None or y_val is None:
        result.val_errors.append(float("nan"))
        monitored = train_e
    else:
        val_e = net.error(x_val, y_val, w)
        result.fn_evals += 1
        result.val_errors.append(val_e)
        monitored = val_e
    if monitored < result.best_val:
        result.best_val = monitored
        result.best_weights = w.copy()
    return monitored
