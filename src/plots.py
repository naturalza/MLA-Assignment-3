"""Plot helpers for the report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

DISPLAY_NAMES = {
    "iris": "Iris",
    "breast_cancer": "Breast Cancer",
    "glass": "Glass",
    "sinc": "sinc",
    "sincos": "sincos",
    "friedman1": "Friedman #1",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "axes.linewidth": 1.1,
    }
)

REPORT_FIGURES = ROOT / "report" / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for folder in (FIGURES, REPORT_FIGURES):
        fig.savefig(folder / f"{name}.pdf")
        fig.savefig(folder / f"{name}.png", dpi=200)
    plt.close(fig)


def _errorbar(ax, df: pd.DataFrame, problem: str) -> None:
    sub = df[df["problem"] == problem]
    grouped = sub.groupby("hidden")["val_error"]
    means = grouped.mean()
    stds = grouped.std(ddof=1).fillna(0.0)
    ax.errorbar(
        means.index,
        means.values,
        yerr=stds.values,
        marker="o",
        markersize=9,
        linewidth=2.4,
        capsize=4,
        elinewidth=1.6,
    )
    ax.set_xlabel("Hidden units")
    ax.set_ylabel("Validation error")
    ax.set_title(DISPLAY_NAMES.get(problem, problem))
    ax.grid(True, alpha=0.3)


def plot_hidden_units(csv_path: Path | None = None) -> None:
    path = csv_path or (RESULTS / "hidden_units.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    for problem in ["iris", "breast_cancer", "glass", "sinc", "sincos", "friedman1"]:
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        _errorbar(ax, df, problem)
        _save(fig, f"hidden_{problem}")


def plot_final_bars(csv_path: Path | None = None) -> None:
    path = csv_path or (RESULTS / "final.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)

    def _one(task: str, metric: str, ylabel: str, filename: str, problem_order: list[str]) -> None:
        sub = df[df["task"] == task].copy()
        if sub.empty:
            return
        sub["problem"] = sub["problem"].map(lambda p: DISPLAY_NAMES.get(p, p))
        order_display = [DISPLAY_NAMES.get(p, p) for p in problem_order]
        means = sub.groupby(["problem", "algorithm"])[metric].mean().unstack()
        means = means.reindex([n for n in order_display if n in means.index])
        fig, ax = plt.subplots(figsize=(3.45, 2.8))
        means.plot(kind="bar", ax=ax, rot=0, width=0.75)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, filename)

    _one(
        "classification",
        "test_accuracy",
        "Test accuracy",
        "final_classification",
        ["iris", "breast_cancer", "glass"],
    )
    _one(
        "regression",
        "test_mse",
        "Test mean squared error",
        "final_regression",
        ["sinc", "sincos", "friedman1"],
    )


def plot_learning_curves(histories: dict[str, tuple[np.ndarray, np.ndarray]], name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for label, (xs, ys) in histories.items():
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("Gradient evaluations")
    ax.set_ylabel("Training error")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, name)
