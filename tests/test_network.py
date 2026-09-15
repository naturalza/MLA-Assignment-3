"""Gradient check and XOR smoke tests."""

from __future__ import annotations

import numpy as np

from src.data import xor_dataset
from src.network import MLP, finite_difference_gradient
from src.trainers import SGD, LeapFrog, ScaledConjugateGradient


def test_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(0)
    x, y = xor_dataset()
    net = MLP(2, 3, 2, "classification")
    w = net.init_weights(rng)
    analytic = net.gradient(x, y, w)
    numeric = finite_difference_gradient(net, x, y, w, eps=1e-5)
    rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1e-12)
    assert rel < 1e-4, rel


def test_sgd_learns_xor() -> None:
    x, y = xor_dataset()
    net = MLP(2, 4, 2, "classification")
    w0 = net.init_weights(np.random.default_rng(1))
    res = SGD(
        learning_rate=0.5,
        momentum=0.9,
        batch_size=None,
        max_epochs=2000,
        patience=0,
        max_grad_evals=20000,
    ).fit(net, x, y, w0)
    pred = net.predict(x, res.best_weights)
    true = y.argmax(axis=1)
    assert float(np.mean(pred == true)) == 1.0


def test_scg_decreases_xor_error() -> None:
    x, y = xor_dataset()
    net = MLP(2, 4, 2, "classification")
    w0 = net.init_weights(np.random.default_rng(1))
    start = net.error(x, y, w0)
    res = ScaledConjugateGradient(max_iterations=200, patience=0).fit(net, x, y, w0)
    assert res.train_errors[-1] < start
    pred = net.predict(x, res.best_weights)
    assert float(np.mean(pred == y.argmax(axis=1))) >= 0.75


def test_leapfrog_decreases_xor_error() -> None:
    x, y = xor_dataset()
    net = MLP(2, 4, 2, "classification")
    w0 = net.init_weights(np.random.default_rng(1))
    start = net.error(x, y, w0)
    res = LeapFrog(max_iterations=1500, patience=0, delta=1.0).fit(net, x, y, w0)
    assert res.train_errors[-1] < start
