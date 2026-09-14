"""LeapFrog LFOP1(b) of Snyman (1982, 1983), implemented from LFROGa.pdf and LFROGb.pdf."""

from __future__ import annotations

import time

import numpy as np

from src.network import MLP
from src.trainers.base import EarlyStopper, TrainResult, record


class LeapFrog:
    """Dynamic leap-frog minimiser LFOP1(b).

    Core trajectory and kinetic-energy interference follow algorithm (23) in
    Snyman 1982. Adaptive time-step growth and the m-step obtuse-angle reduction
    follow Snyman 1983 (LFOP1(b)). Default control values are those used in the
    1983 numerical tests: initial dt = 0.5, m = 3, delta1 = 0.001.
    """

    name = "LeapFrog"

    def __init__(
        self,
        dt: float = 0.5,
        delta: float = 1.0,
        m: int = 3,
        delta1: float = 0.001,
        max_iterations: int = 5000,
        patience: int = 50,
        grad_tol: float = 1e-8,
        max_grad_evals: int = 50_000,
        j_restart: int = 2,
    ) -> None:
        self.dt = dt
        self.delta = delta
        self.m = m
        self.delta1 = delta1
        self.max_iterations = max_iterations
        self.patience = patience
        self.grad_tol = grad_tol
        self.max_grad_evals = max_grad_evals
        self.j_restart = j_restart

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
        x = w0.copy()
        result = TrainResult(weights=x, best_weights=x.copy())
        stopper = EarlyStopper(self.patience)
        t0 = time.perf_counter()

        def grad(vec: np.ndarray) -> np.ndarray:
            result.grad_evals += 1
            return net.gradient(x_train, y_train, vec)

        dt = float(self.dt)
        delta = float(self.delta)
        i_count = 0
        j_lim = self.j_restart
        s_obtuse = 0
        n_success = 0

        g = grad(x)
        a = -g
        v = 0.5 * a * dt
        x_prev = x.copy()
        v_prev = v.copy()
        a_prev = a.copy()

        k = 0
        while k < self.max_iterations:
            if result.grad_evals >= self.max_grad_evals:
                result.stopped_reason = "max_grad_evals"
                break

            # Steps 3-4 of algorithm (23): cap the displacement by delta.
            v_norm = float(np.linalg.norm(v))
            step = v_norm * dt
            if step >= delta and v_norm > 0.0:
                v = (delta / (dt * v_norm)) * v
                v_norm = float(np.linalg.norm(v))
                step = v_norm * dt
            elif np.dot(a, a_prev) > 0.0:
                # LFOP1(b) time-step increment after consecutive successful steps.
                n_success += 1
                dt = min(dt * (1.0 + n_success * self.delta1), 10.0)

            # LFOP1(b) time-step reduction after m consecutive obtuse gradient angles.
            if s_obtuse >= self.m:
                dt = max(dt * 0.5, 1e-8)
                x = 0.5 * (x + x_prev)
                v = 0.25 * (v + v_prev)
                s_obtuse = 0
                n_success = 0
                g = grad(x)
                a = -g

            x_prev = x.copy()
            v_prev = v.copy()
            a_prev = a.copy()

            # Steps 5-6: leap-frog integration.
            x = x + v * dt
            g = grad(x)
            a = -g
            v_new = v + a * dt

            if np.dot(a, a_prev) > 0.0:
                s_obtuse = 0
            else:
                s_obtuse += 1
                n_success = 0

            if np.linalg.norm(a) < self.grad_tol:
                val_e = record(net, x, x_train, y_train, x_val, y_val, result)
                result.stopped_reason = "grad_tol"
                result.iterations = k + 1
                stopper.update(val_e, x)
                break

            v_new_norm = float(np.linalg.norm(v_new))
            v_norm = float(np.linalg.norm(v))
            if v_new_norm > v_norm:
                # Kinetic energy increased: continue the trajectory.
                i_count = 0
                v = v_new
            else:
                # Kinetic-energy interference of algorithm (23) steps 8-9.
                x = 0.5 * (x + x_prev)
                i_count += 1
                if i_count < j_lim:
                    v = 0.25 * (v_new + v)
                else:
                    v = np.zeros_like(v)
                    j_lim = 1
                g = grad(x)
                a = -g
                v = v + a * dt

            k += 1
            val_e = record(net, x, x_train, y_train, x_val, y_val, result)
            result.iterations = k
            if stopper.update(val_e, x):
                x = stopper.best_w if stopper.best_w is not None else x
                result.stopped_reason = "early_stop"
                break
        else:
            result.stopped_reason = "max_iterations"

        result.weights = x
        if stopper.best_w is not None:
            result.best_weights = stopper.best_w
        result.wall_time = time.perf_counter() - t0
        return result
