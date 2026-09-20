"""Experiment runners: sanity checks, hidden-unit search, parameter search, final comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data import ALL_PROBLEMS, load_problem, xor_dataset
from src.metrics import evaluate
from src.network import MLP, finite_difference_gradient
from src.stats import compare_three, friedman_ranks
from src.trainers import LeapFrog, SGD, ScaledConjugateGradient

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"

HIDDEN_GRIDS = {
    "iris": [2, 4, 6, 8, 12, 16],
    "breast_cancer": [4, 8, 12, 16, 24, 32],
    "glass": [4, 8, 12, 16, 24, 32],
    "wine": [4, 8, 12, 16, 24, 32],
    "sinc": [2, 4, 6, 8, 12, 16],
    "sincos": [4, 8, 12, 16, 24],
    "friedman1": [4, 8, 12, 16, 24, 32],
}

SGD_GRID = {
    "learning_rate": [0.001, 0.01, 0.1],
    "momentum": [0.0, 0.9],
    "batch_size": [16, 32, None],
}
SCG_GRID = {
    "sigma": [1e-4, 1e-5],
    "lambd": [1e-6, 1e-4],
}
LEAP_GRID = {
    "dt": [0.1, 0.5, 1.0],
    "delta": [1.0, 10.0, 100.0],
}


def make_trainer(name: str, **kwargs):
    if name == "SGD":
        return SGD(**kwargs)
    if name == "SCG":
        return ScaledConjugateGradient(**kwargs)
    if name == "LeapFrog":
        return LeapFrog(**kwargs)
    raise KeyError(name)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_sanity() -> None:
    rng = np.random.default_rng(0)
    x, y = xor_dataset()
    net = MLP(n_in=2, n_hidden=4, n_out=2, task="classification")
    w = net.init_weights(rng)
    analytic = net.gradient(x, y, w)
    numeric = finite_difference_gradient(net, x, y, w, eps=1e-5)
    rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(numeric) + 1e-12)
    print(f"XOR gradient relative error: {rel:.3e}")
    if rel > 1e-4:
        raise SystemExit("Gradient check failed")

    for name, trainer in [
        ("SGD", SGD(learning_rate=0.5, momentum=0.9, batch_size=None, max_epochs=2000, patience=0, max_grad_evals=20000)),
        ("SCG", ScaledConjugateGradient(max_iterations=200, patience=0, max_grad_evals=2000)),
        ("LeapFrog", LeapFrog(dt=0.5, delta=1.0, max_iterations=2000, patience=0, max_grad_evals=5000)),
    ]:
        w0 = net.init_weights(np.random.default_rng(1))
        res = trainer.fit(net, x, y, w0)
        acc = evaluate(net, x, y, res.best_weights)["accuracy"]
        print(
            f"{name:8s} XOR accuracy={acc:.3f}  train_err={res.train_errors[-1]:.4f}  "
            f"iters={res.iterations}  reason={res.stopped_reason}"
        )

    # Rosenbrock via a fake 2-parameter "network" is not needed; sinc is the regression check.
    data = load_problem("sinc", seed=0)
    net_r = MLP(n_in=data.n_in, n_hidden=8, n_out=data.n_out, task="regression")
    for name, trainer in [
        ("SGD", SGD(learning_rate=0.05, momentum=0.9, batch_size=32, max_epochs=400, patience=40)),
        ("SCG", ScaledConjugateGradient(max_iterations=300, patience=40)),
        ("LeapFrog", LeapFrog(max_iterations=800, patience=40)),
    ]:
        w0 = net_r.init_weights(np.random.default_rng(2))
        res = trainer.fit(
            net_r, data.x_train, data.y_train, w0, data.x_val, data.y_val
        )
        scores = evaluate(net_r, data.x_test, data.y_test, res.best_weights)
        print(
            f"{name:8s} sinc test MSE={scores['mse']:.4f}  r2={scores['r2']:.3f}  "
            f"reason={res.stopped_reason}"
        )


def run_hidden_units(problems: list[str], n_runs: int, seed: int) -> None:
    rows = []
    for problem in problems:
        data = load_problem(problem, seed=seed)
        grid = HIDDEN_GRIDS[data.name]
        print(f"=== hidden-unit search: {data.name} {grid} ===")
        for h in grid:
            net = MLP(data.n_in, h, data.n_out, data.task)
            val_errors = []
            train_errors = []
            for run in range(n_runs):
                rng = np.random.default_rng(seed + 1000 * h + run)
                trainer = ScaledConjugateGradient(
                    max_iterations=400, patience=30, max_grad_evals=4000
                )
                w0 = net.init_weights(rng)
                res = trainer.fit(
                    net, data.x_train, data.y_train, w0, data.x_val, data.y_val, rng
                )
                train_errors.append(res.train_errors[-1] if res.train_errors else float("nan"))
                val_errors.append(res.best_val)
                rows.append(
                    {
                        "problem": data.name,
                        "hidden": h,
                        "run": run,
                        "train_error": train_errors[-1],
                        "val_error": res.best_val,
                        "grad_evals": res.grad_evals,
                        "reason": res.stopped_reason,
                    }
                )
            print(
                f"  h={h:3d}  val={np.mean(val_errors):.4f}±{np.std(val_errors):.4f}  "
                f"train={np.mean(train_errors):.4f}"
            )
    _write_csv(RESULTS / "hidden_units.csv", rows)
    _select_hidden(rows)


def _select_hidden(rows: list[dict]) -> dict[str, int]:
    """Smallest hidden size whose mean val error is within 5% of the best mean."""
    from collections import defaultdict

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["problem"], r["hidden"])].append(r["val_error"])
    selected: dict[str, int] = {}
    problems = sorted({p for p, _ in grouped})
    for problem in problems:
        means = {
            h: float(np.mean(grouped[(problem, h)]))
            for p, h in grouped
            if p == problem
        }
        best = min(means.values())
        threshold = best * 1.05 if best > 0 else best + 1e-4
        candidates = sorted(h for h, m in means.items() if m <= threshold)
        selected[problem] = candidates[0]
        print(f"selected hidden units for {problem}: {selected[problem]} (best mean={best:.4f})")
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "hidden_units.json").write_text(json.dumps(selected, indent=2))
    return selected


def _product(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    combos = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in grid[k]]
    return combos


def run_param_search(problems: list[str], n_runs: int, seed: int) -> None:
    hidden_path = CONFIGS / "hidden_units.json"
    if not hidden_path.exists():
        raise SystemExit("Run hidden-units first")
    hidden = json.loads(hidden_path.read_text())
    rows = []
    trainers = {
        "SGD": (SGD, SGD_GRID),
        "SCG": (ScaledConjugateGradient, SCG_GRID),
        "LeapFrog": (LeapFrog, LEAP_GRID),
    }
    for problem in problems:
        data = load_problem(problem, seed=seed)
        h = hidden[data.name]
        net = MLP(data.n_in, h, data.n_out, data.task)
        print(f"=== param search: {data.name} h={h} ===")
        for alg_name, (cls, grid) in trainers.items():
            for params in _product(grid):
                vals = []
                for run in range(n_runs):
                    rng = np.random.default_rng(seed + run + 100 * {"SGD": 1, "SCG": 2, "LeapFrog": 3}[alg_name])
                    fit_kwargs = dict(params)
                    if alg_name == "SGD":
                        fit_kwargs.update(max_epochs=400, patience=30, max_grad_evals=4000)
                    elif alg_name == "SCG":
                        fit_kwargs.update(max_iterations=400, patience=30, max_grad_evals=4000)
                    else:
                        fit_kwargs.update(max_iterations=800, patience=30, max_grad_evals=4000)
                    trainer = cls(**fit_kwargs)
                    w0 = net.init_weights(rng)
                    res = trainer.fit(
                        net, data.x_train, data.y_train, w0, data.x_val, data.y_val, rng
                    )
                    vals.append(res.best_val)
                    rows.append(
                        {
                            "problem": data.name,
                            "algorithm": alg_name,
                            "params": json.dumps(params),
                            "run": run,
                            "val_error": res.best_val,
                            "grad_evals": res.grad_evals,
                            "reason": res.stopped_reason,
                        }
                    )
                print(
                    f"  {alg_name:8s} {params}  val={np.mean(vals):.4f}±{np.std(vals):.4f}"
                )
    _write_csv(RESULTS / "param_search.csv", rows)
    _select_params(rows)


def _select_params(rows: list[dict]) -> None:
    from collections import defaultdict

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(r["problem"], r["algorithm"], r["params"])].append(r["val_error"])
    selected: dict[str, dict[str, dict]] = {}
    for (problem, alg, params), vals in grouped.items():
        mean = float(np.mean(vals))
        selected.setdefault(problem, {})
        cur = selected[problem].get(alg)
        if cur is None or mean < cur["mean_val"]:
            selected[problem][alg] = {"params": json.loads(params), "mean_val": mean}
    CONFIGS.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "params.json").write_text(json.dumps(selected, indent=2))
    for problem, algs in selected.items():
        print(f"best params {problem}:")
        for alg, info in algs.items():
            print(f"  {alg}: {info['params']} (val={info['mean_val']:.4f})")


def run_final(problems: list[str], n_runs: int, seed: int) -> None:
    hidden = json.loads((CONFIGS / "hidden_units.json").read_text())
    params = json.loads((CONFIGS / "params.json").read_text())
    rows = []
    for problem in problems:
        data = load_problem(problem, seed=seed)
        h = hidden[data.name]
        net = MLP(data.n_in, h, data.n_out, data.task)
        print(f"=== final: {data.name} h={h} ===")
        for alg_name in ("SGD", "SCG", "LeapFrog"):
            p = params[data.name][alg_name]["params"]
            scores_primary = []
            for run in range(n_runs):
                rng = np.random.default_rng(10_000 + seed + run)
                fit_kwargs = dict(p)
                if alg_name == "SGD":
                    fit_kwargs.update(max_epochs=1500, patience=60, max_grad_evals=20000)
                elif alg_name == "SCG":
                    fit_kwargs.update(max_iterations=1500, patience=60, max_grad_evals=20000)
                else:
                    fit_kwargs.update(max_iterations=3000, patience=60, max_grad_evals=20000)
                trainer = make_trainer(alg_name, **fit_kwargs)
                w0 = net.init_weights(rng)
                res = trainer.fit(
                    net, data.x_train, data.y_train, w0, data.x_val, data.y_val, rng
                )
                test = evaluate(net, data.x_test, data.y_test, res.best_weights)
                train = evaluate(net, data.x_train, data.y_train, res.best_weights)
                primary = test["accuracy"] if data.task == "classification" else -test["mse"]
                scores_primary.append(primary)
                row = {
                    "problem": data.name,
                    "task": data.task,
                    "algorithm": alg_name,
                    "run": run,
                    "hidden": h,
                    "grad_evals": res.grad_evals,
                    "iterations": res.iterations,
                    "wall_time": res.wall_time,
                    "reason": res.stopped_reason,
                    "train_error": train["error"],
                    "test_error": test["error"],
                }
                row.update({f"test_{k}": v for k, v in test.items()})
                rows.append(row)
            print(
                f"  {alg_name:8s} primary={np.mean(scores_primary):.4f}±{np.std(scores_primary):.4f}"
            )
    _write_csv(RESULTS / "final.csv", rows)
    _summarise_final(rows)


def _summarise_final(rows: list[dict]) -> None:
    from collections import defaultdict

    by_problem: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    task_of: dict[str, str] = {}
    for r in rows:
        task_of[r["problem"]] = r["task"]
        key = "test_accuracy" if r["task"] == "classification" else "test_mse"
        # For MSE, comparison uses negated values (higher better).
        val = r[key]
        by_problem[r["problem"]][r["algorithm"]].append(
            val if r["task"] == "classification" else -val
        )
    summaries = {}
    mean_table = []
    alg_order = ["SGD", "SCG", "LeapFrog"]
    problems = sorted(by_problem)
    for problem in problems:
        scores = {alg: np.array(by_problem[problem][alg]) for alg in alg_order}
        summaries[problem] = compare_three(scores)
        print(f"--- {problem} ---")
        print(json.dumps(summaries[problem], indent=2))
        mean_table.append([float(np.mean(scores[a])) for a in alg_order])
    friedman = friedman_ranks(np.array(mean_table))
    print("Friedman:", friedman)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "final_stats.json").write_text(
        json.dumps({"per_problem": summaries, "friedman": friedman}, indent=2)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Option 2 experiment runner")
    p.add_argument(
        "stage",
        choices=["sanity", "hidden-units", "params", "final"],
    )
    p.add_argument("--problems", nargs="*", default=list(ALL_PROBLEMS))
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.stage == "sanity":
        run_sanity()
        return
    if args.stage == "hidden-units":
        run_hidden_units(args.problems, args.runs or 10, args.seed)
        return
    if args.stage == "params":
        run_param_search(args.problems, args.runs or 8, args.seed)
        return
    run_final(args.problems, args.runs or 30, args.seed)


if __name__ == "__main__":
    main()
