"""Mini-batch stochastic gradient descent with optional momentum."""

from __future__ import annotations

import time

import numpy as np

from src.network import MLP
from src.trainers.base import EarlyStopper, TrainResult, record


class SGD:
    name = "SGD"

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        batch_size: int | None = 32,
        max_epochs: int = 2000,
        patience: int = 50,
        grad_tol: float = 1e-8,
        max_grad_evals: int = 50_000,
        log_every: int = 1,
    ) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_tol = grad_tol
        self.max_grad_evals = max_grad_evals
        self.log_every = log_every

    def fit(
        self,
        net: MLP,
        x_train: np.ndarray,
        y_train: np.ndarray,
        w0: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> TrainResult:
        rng = rng or np.random.default_rng()
        w = w0.copy()
        v = np.zeros_like(w)
        n = x_train.shape[0]
        batch = n if self.batch_size is None else min(self.batch_size, n)
        result = TrainResult(weights=w, best_weights=w.copy())
        stopper = EarlyStopper(self.patience)
        t0 = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            perm = rng.permutation(n)
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                g = net.gradient(x_train[idx], y_train[idx], w)
                result.grad_evals += 1
                v = self.momentum * v - self.learning_rate * g
                w = w + v
                if result.grad_evals >= self.max_grad_evals:
                    record(net, w, x_train, y_train, x_val, y_val, result)
                    result.iterations = epoch
                    result.weights = stopper.best_w if stopper.best_w is not None else w
                    result.best_weights = result.weights.copy()
                    result.stopped_reason = "max_grad_evals"
                    result.wall_time = time.perf_counter() - t0
                    return result

            if epoch % self.log_every == 0:
                val_e = record(net, w, x_train, y_train, x_val, y_val, result)
                g_full = net.gradient(x_train, y_train, w)
                result.grad_evals += 1
                if np.linalg.norm(g_full) < self.grad_tol:
                    result.iterations = epoch
                    result.weights = stopper.best_w if stopper.best_w is not None else w
                    result.best_weights = result.weights.copy()
                    result.stopped_reason = "grad_tol"
                    result.wall_time = time.perf_counter() - t0
                    return result
                if stopper.update(val_e, w):
                    result.iterations = epoch
                    result.weights = stopper.best_w if stopper.best_w is not None else w
                    result.best_weights = result.weights.copy()
                    result.stopped_reason = "early_stop"
                    result.wall_time = time.perf_counter() - t0
                    return result

        record(net, w, x_train, y_train, x_val, y_val, result)
        result.iterations = self.max_epochs
        result.weights = stopper.best_w if stopper.best_w is not None else w
        result.best_weights = result.weights.copy()
        result.stopped_reason = "max_epochs"
        result.wall_time = time.perf_counter() - t0
        return result
