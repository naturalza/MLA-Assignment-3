# RW741 Assignment 3 — Option 2

Comparison of stochastic gradient descent, scaled conjugate gradient (Møller, 1993), and LeapFrog LFOP1(b) (Snyman, 1982; 1983) as training algorithms for a one-hidden-layer feedforward neural network.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root so that `src` is importable:

```bash
PYTHONPATH=. python -m src.experiments sanity
PYTHONPATH=. python -m src.experiments hidden-units --runs 10
PYTHONPATH=. python -m src.experiments params --runs 8
PYTHONPATH=. python -m src.experiments final --runs 30
```

Each stage writes CSV or JSON under `results/` and locks architecture or parameter choices under `configs/`. Stages must be run in that order.

## Problems

| Name | Type | Role |
|---|---|---|
| `iris` | classification | easy |
| `breast_cancer` | classification | medium |
| `glass` (Wine fallback if UCI is unreachable) | classification | hard |
| `sinc` | regression | easy |
| `sincos` | regression | medium |
| `friedman1` | regression | hard |

## Report

IEEE conference, two-column, 10pt, in `report/`. The submitted PDF filename must be `????????RW741assignment3.pdf`.
