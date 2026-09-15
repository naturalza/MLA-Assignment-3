"""Statistical tests for the three-algorithm comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def holm_correction(p_values: list[tuple[str, float]]) -> list[tuple[str, float, float, bool]]:
    """Holm-Bonferroni adjusted p-values. Returns (name, raw_p, adj_p, reject_at_0.05)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i][1])
    adj: list[float] = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx][1]
        scaled = min(1.0, raw * (m - rank))
        running = max(running, scaled)
        adj[idx] = running
    return [
        (name, p, adj[i], adj[i] < 0.05)
        for i, (name, p) in enumerate(p_values)
    ]


def compare_three(scores: dict[str, np.ndarray], alpha: float = 0.05) -> dict:
    """scores maps algorithm name -> 1-D array of test scores (higher is better)."""
    names = list(scores.keys())
    arrays = [np.asarray(scores[n], dtype=float) for n in names]
    shapiro = {n: float(stats.shapiro(a).pvalue) for n, a in zip(names, arrays)}
    all_normal = all(p > alpha for p in shapiro.values())
    if all_normal:
        overall = stats.f_oneway(*arrays)
        overall_name = "anova"
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = stats.ttest_ind(arrays[i], arrays[j], equal_var=False).pvalue
                pairs.append((f"{names[i]} vs {names[j]}", float(p)))
    else:
        overall = stats.kruskal(*arrays)
        overall_name = "kruskal"
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = stats.mannwhitneyu(arrays[i], arrays[j], alternative="two-sided").pvalue
                pairs.append((f"{names[i]} vs {names[j]}", float(p)))
    return {
        "shapiro": shapiro,
        "all_normal": all_normal,
        "overall_test": overall_name,
        "overall_p": float(overall.pvalue),
        "overall_stat": float(overall.statistic),
        "pairwise": [
            {"pair": n, "p": p, "p_holm": ph, "significant": sig}
            for n, p, ph, sig in holm_correction(pairs)
        ],
        "means": {n: float(np.mean(a)) for n, a in zip(names, arrays)},
        "stds": {n: float(np.std(a, ddof=1)) for n, a in zip(names, arrays)},
    }


def friedman_ranks(mean_scores: np.ndarray) -> dict:
    """mean_scores: (n_problems, n_algorithms), higher is better."""
    n_problems, n_algs = mean_scores.shape
    ranks = np.zeros_like(mean_scores)
    for i in range(n_problems):
        ranks[i] = stats.rankdata(-mean_scores[i], method="average")
    stat, p = stats.friedmanchisquare(*[mean_scores[:, j] for j in range(n_algs)])
    avg_ranks = ranks.mean(axis=0)
    return {
        "ranks": ranks.tolist(),
        "average_ranks": avg_ranks.tolist(),
        "statistic": float(stat),
        "p": float(p),
    }
