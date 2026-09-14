"""One-hidden-layer multilayer perceptron with packed weights and analytic gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Task = Literal["classification", "regression"]


def xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """Uniform Xavier initialisation matched to hyperbolic tangent hidden units."""
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class MLP:
    """Fully connected network with one hidden layer of hyperbolic tangent units.

    Classification uses a softmax output and mean categorical cross-entropy.
    Regression uses a linear output and mean squared error.
    Weights and biases are stored packed as a single vector so that SCG and
    LeapFrog operate on an unconstrained problem of dimension n_params.
    """

    n_in: int
    n_hidden: int
    n_out: int
    task: Task

    def n_params(self) -> int:
        return (
            self.n_in * self.n_hidden
            + self.n_hidden
            + self.n_hidden * self.n_out
            + self.n_out
        )

    def init_weights(self, rng: np.random.Generator) -> np.ndarray:
        w1 = xavier_uniform(rng, self.n_in, self.n_hidden)
        b1 = np.zeros(self.n_hidden)
        w2 = xavier_uniform(rng, self.n_hidden, self.n_out)
        b2 = np.zeros(self.n_out)
        return self.pack(w1, b1, w2, b2)

    def pack(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()])

    def unpack(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i = 0
        n_w1 = self.n_in * self.n_hidden
        w1 = w[i : i + n_w1].reshape(self.n_in, self.n_hidden)
        i += n_w1
        b1 = w[i : i + self.n_hidden]
        i += self.n_hidden
        n_w2 = self.n_hidden * self.n_out
        w2 = w[i : i + n_w2].reshape(self.n_hidden, self.n_out)
        i += n_w2
        b2 = w[i : i + self.n_out]
        return w1, b1, w2, b2

    def forward(self, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, dict]:
        w1, b1, w2, b2 = self.unpack(w)
        z1 = x @ w1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ w2 + b2
        if self.task == "classification":
            yhat = softmax(z2)
        else:
            yhat = z2
        cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "yhat": yhat, "w1": w1, "w2": w2}
        return yhat, cache

    def error(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        yhat, _ = self.forward(x, w)
        if self.task == "classification":
            return float(-np.mean(np.sum(y * np.log(yhat + 1e-12), axis=1)))
        return float(np.mean((yhat - y) ** 2))

    def gradient(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        yhat, cache = self.forward(x, w)
        n = x.shape[0]
        a1 = cache["a1"]
        w2 = cache["w2"]
        if self.task == "classification":
            dz2 = (yhat - y) / n
        else:
            dz2 = (2.0 / (n * y.shape[1])) * (yhat - y)
        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ w2.T
        dz1 = da1 * (1.0 - a1 ** 2)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        return self.pack(dw1, db1, dw2, db2)

    def predict(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        yhat, _ = self.forward(x, w)
        if self.task == "classification":
            return yhat.argmax(axis=1)
        return yhat


def finite_difference_gradient(
    net: MLP,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference gradient used only to verify backpropagation."""
    g = np.zeros_like(w)
    for i in range(w.size):
        e = np.zeros_like(w)
        e[i] = eps
        plus = net.error(x, y, w + e)
        minus = net.error(x, y, w - e)
        g[i] = (plus - minus) / (2.0 * eps)
    return g
