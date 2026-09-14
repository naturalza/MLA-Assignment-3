"""Scaled conjugate gradient of Moller (1993), implemented from SCG.pdf."""

from __future__ import annotations

import time

import numpy as np

from src.network import MLP
from src.trainers.base import EarlyStopper, TrainResult, record


class ScaledConjugateGradient:
    """Offline SCG. Control parameters follow Moller: sigma <= 1e-4, lambda <= 1e-6."""

    name = "SCG"

    def __init__(
        self,
        sigma: float = 1e-4,
        lambd: float = 1e-6,
        max_iterations: int = 2000,
        patience: int = 50,
        grad_tol: float = 1e-8,
        max_grad_evals: int = 50_000,
    ) -> None:
        self.sigma = sigma
        self.lambd = lambd
        self.max_iterations = max_iterations
        self.patience = patience
        self.grad_tol = grad_tol
        self.max_grad_evals = max_grad_evals

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
        del rng
        w = w0.copy()
        n_w = w.size
        result = TrainResult(weights=w, best_weights=w.copy())
        stopper = EarlyStopper(self.patience)
        t0 = time.perf_counter()

        def err(vec: np.ndarray) -> float:
            result.fn_evals += 1
            return net.error(x_train, y_train, vec)

        def grad(vec: np.ndarray) -> np.ndarray:
            result.grad_evals += 1
            return net.gradient(x_train, y_train, vec)

        g = grad(w)
        r = -g
        p = r.copy()
        success = True
        lambd = self.lambd
        lambd_bar = 0.0
        k = 1
        e_w = err(w)
        s = np.zeros_like(w)

        while k <= self.max_iterations:
            if result.grad_evals >= self.max_grad_evals:
                result.stopped_reason = "max_grad_evals"
                break

            p_norm = float(np.linalg.norm(p))
            if p_norm == 0.0:
                result.stopped_reason = "zero_direction"
                break

            # Step 2 of Moller: second-order information on a successful iteration.
            if success:
                sigma_k = self.sigma / p_norm
                s = (grad(w + sigma_k * p) - g) / sigma_k
                delta = float(p @ s)
            else:
                delta = float(p @ s)

            # Step 3: scale the curvature estimate with the Levenberg-Marquardt parameter.
            delta = delta + (lambd - lambd_bar) * (p_norm ** 2)

            # Step 4: force a positive-definite Hessian approximation.
            if delta <= 0.0:
                lambd = 2.0 * (lambd - delta / (p_norm ** 2))
                delta = -delta + lambd * (p_norm ** 2)
                lambd_bar = lambd

            mu = float(p @ r)
            if delta == 0.0:
                result.stopped_reason = "zero_curvature"
                break
            alpha = mu / delta

            e_new = err(w + alpha * p)
            # Equation (26): comparison of actual versus quadratic reduction.
            denom = mu * mu
            delta_k = (2.0 * delta * (e_w - e_new) / denom) if denom != 0.0 else 0.0

            if delta_k >= 0.0:
                w = w + alpha * p
                e_w = e_new
                g = grad(w)
                r_next = -g
                lambd_bar = 0.0
                success = True
                if k % n_w == 0:
                    p = r_next.copy()
                else:
                    beta = float((r_next @ r_next - r_next @ r) / mu) if mu != 0.0 else 0.0
                    p = r_next + beta * p
                r = r_next
                # Restart conjugacy when the new direction is not a descent direction.
                if p @ r <= 0.0:
                    p = r.copy()
                if delta_k >= 0.75:
                    lambd *= 0.25
            else:
                lambd_bar = lambd
                success = False

            if delta_k < 0.25:
                lambd = lambd + delta * (1.0 - delta_k) / (p_norm ** 2)
                lambd = min(lambd, 1e6)

            if not success:
                # Failed conjugate step: raise the scale and resume from steepest descent.
                p = r.copy()
                success = True
                lambd_bar = 0.0

            result.iterations = k
            if success:
                val_e = record(net, w, x_train, y_train, x_val, y_val, result)
                if np.linalg.norm(r) < self.grad_tol:
                    result.stopped_reason = "grad_tol"
                    break
                if stopper.update(val_e, w):
                    w = stopper.best_w if stopper.best_w is not None else w
                    result.stopped_reason = "early_stop"
                    break
            k += 1
        else:
            result.stopped_reason = "max_iterations"

        result.weights = w
        if stopper.best_w is not None:
            result.best_weights = stopper.best_w
        result.wall_time = time.perf_counter() - t0
        return result
