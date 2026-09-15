"""Problem loaders, preprocessing, and fixed train/validation/test splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.request import urlopen

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

Task = Literal["classification", "regression"]

GLASS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
)


@dataclass
class Dataset:
    name: str
    task: Task
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_in: int
    n_out: int
    description: str
    class_names: list[str] | None = None

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])


def _split_scale(
    x: np.ndarray,
    y: np.ndarray,
    task: Task,
    seed: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    stratify = y if task == "classification" else None
    try:
        x_tv, x_test, y_tv, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=stratify
        )
    except ValueError:
        x_tv, x_test, y_tv, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=None
        )
    val_ratio = val_size / (1.0 - test_size)
    stratify_tv = y_tv if task == "classification" else None
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            x_tv, y_tv, test_size=val_ratio, random_state=seed, stratify=stratify_tv
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            x_tv, y_tv, test_size=val_ratio, random_state=seed, stratify=None
        )

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)

    if task == "classification":
        encoder = LabelBinarizer()
        y_train_oh = encoder.fit_transform(y_train)
        y_val_oh = encoder.transform(y_val)
        y_test_oh = encoder.transform(y_test)
        if y_train_oh.ndim == 1:
            y_train_oh = y_train_oh.reshape(-1, 1)
            y_val_oh = y_val_oh.reshape(-1, 1)
            y_test_oh = y_test_oh.reshape(-1, 1)
        if y_train_oh.shape[1] == 1:
            y_train_oh = np.hstack([1.0 - y_train_oh, y_train_oh])
            y_val_oh = np.hstack([1.0 - y_val_oh, y_val_oh])
            y_test_oh = np.hstack([1.0 - y_test_oh, y_test_oh])
        n_out = y_train_oh.shape[1]
        return x_train, y_train_oh, x_val, y_val_oh, x_test, y_test_oh, n_out

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test, y_train.shape[1]


def load_iris_problem(seed: int = 0) -> Dataset:
    bunch = load_iris()
    x, y = bunch.data.astype(float), bunch.target
    xt, yt, xv, yv, xs, ys, n_out = _split_scale(x, y, "classification", seed)
    return Dataset(
        name="iris",
        task="classification",
        x_train=xt,
        y_train=yt,
        x_val=xv,
        y_val=yv,
        x_test=xs,
        y_test=ys,
        n_in=xt.shape[1],
        n_out=n_out,
        description=(
            "UCI Iris: 150 samples, four continuous features, three flower classes. "
            "Treated as the easy classification problem."
        ),
        class_names=list(bunch.target_names),
    )


def load_breast_cancer_problem(seed: int = 0) -> Dataset:
    bunch = load_breast_cancer()
    x, y = bunch.data.astype(float), bunch.target
    xt, yt, xv, yv, xs, ys, n_out = _split_scale(x, y, "classification", seed)
    return Dataset(
        name="breast_cancer",
        task="classification",
        x_train=xt,
        y_train=yt,
        x_val=xv,
        y_val=yv,
        x_test=xs,
        y_test=ys,
        n_in=xt.shape[1],
        n_out=n_out,
        description=(
            "UCI Wisconsin Breast Cancer Diagnostic: 569 samples, 30 real-valued "
            "features, two classes. Treated as the medium classification problem."
        ),
        class_names=list(bunch.target_names),
    )


def _load_glass_array() -> tuple[np.ndarray, np.ndarray]:
    try:
        with urlopen(GLASS_URL, timeout=20) as resp:
            raw = resp.read().decode("ascii")
        rows = []
        for line in raw.strip().splitlines():
            parts = line.strip().split(",")
            rows.append([float(v) for v in parts[1:]])
        data = np.asarray(rows)
        x = data[:, :-1]
        y_raw = data[:, -1].astype(int)
        # Glass type labels are {1,2,3,5,6,7}; remap to 0..C-1.
        mapping = {lab: i for i, lab in enumerate(sorted(set(y_raw.tolist())))}
        y = np.array([mapping[int(v)] for v in y_raw], dtype=int)
        return x, y
    except Exception:
        from sklearn.datasets import load_wine

        bunch = load_wine()
        return bunch.data.astype(float), bunch.target


def load_glass_problem(seed: int = 0) -> Dataset:
    x, y = _load_glass_array()
    xt, yt, xv, yv, xs, ys, n_out = _split_scale(x, y, "classification", seed)
    harder = n_out >= 5
    return Dataset(
        name="glass" if harder else "wine",
        task="classification",
        x_train=xt,
        y_train=yt,
        x_val=xv,
        y_val=yv,
        x_test=xs,
        y_test=ys,
        n_in=xt.shape[1],
        n_out=n_out,
        description=(
            "UCI Glass Identification: 214 samples, nine chemical features, six "
            "imbalanced classes. Treated as the hard classification problem."
            if harder
            else "UCI Wine used as a fallback for Glass: 178 samples, 13 features, three classes."
        ),
    )


def load_sinc_problem(n: int = 200, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10.0, 10.0, size=(n, 1))
    y = np.sinc(x / np.pi)
    xt, yt, xv, yv, xs, ys, n_out = _split_scale(x, y, "regression", seed)
    return Dataset(
        name="sinc",
        task="regression",
        x_train=xt,
        y_train=yt,
        x_val=xv,
        y_val=yv,
        x_test=xs,
        y_test=ys,
        n_in=1,
        n_out=n_out,
        description=(
            "One-dimensional sinc, f(x) = sin(x)/x on [-10, 10], 200 noise-free "
            "samples. Treated as the easy function-approximation problem."
        ),
    )


def load_sincos_problem(n: int = 400, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-np.pi, np.pi, size=(n, 2))
    y = (np.sin(x[:, 0]) * np.cos(x[:, 1])).reshape(-1, 1)
    xt, yt, xv, yv, xs, ys, n_out = _split_scale(x, y, "regression", seed)
    return Dataset(
        name="sincos",
        task="regression",
        x_train=xt,
        y_train=yt,
        x_val=xv,
        y_val=yv,
        x_test=xs,
        y_test=ys,
        n_in=2,
        n_out=n_out,
        description=(
            "Two-dimensional surface f(x, y) = sin(x) cos(y) on [-pi, pi]^2, "
            "400 noise-free samples. Treated as the medium function-approximation problem."
        ),
    )


def load_friedman_problem(n: int = 400, seed: int = 0) -> Dataset:
    x, y = make_friedman1(n_samples=n, n_features=10, noise=0.1, random_state=seed)
    y = y.reshape(-1, 1)
    xt, yt, xv, yv, xs, ys, n_out = _split_scale(x, y, "regression", seed)
    return Dataset(
        name="friedman1",
        task="regression",
        x_train=xt,
        y_train=yt,
        x_val=xv,
        y_val=yv,
        x_test=xs,
        y_test=ys,
        n_in=10,
        n_out=n_out,
        description=(
            "Friedman #1: ten inputs, five informative and five irrelevant, with "
            "Gaussian noise of standard deviation 0.1. Treated as the hard "
            "function-approximation problem."
        ),
    )


PROBLEM_LOADERS = {
    "iris": load_iris_problem,
    "breast_cancer": load_breast_cancer_problem,
    "glass": load_glass_problem,
    "sinc": load_sinc_problem,
    "sincos": load_sincos_problem,
    "friedman1": load_friedman_problem,
}

CLASSIFICATION_PROBLEMS = ("iris", "breast_cancer", "glass")
REGRESSION_PROBLEMS = ("sinc", "sincos", "friedman1")
ALL_PROBLEMS = CLASSIFICATION_PROBLEMS + REGRESSION_PROBLEMS


def load_problem(name: str, seed: int = 0) -> Dataset:
    if name not in PROBLEM_LOADERS:
        raise KeyError(f"Unknown problem {name!r}")
    return PROBLEM_LOADERS[name](seed=seed)


def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    return x, y
