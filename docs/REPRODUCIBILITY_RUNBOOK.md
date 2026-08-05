# Reproducibility runbook

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Smoke test

```bash
python experiments/run_tsplib_suite.py --fast --runs 1 --instances berlin52 --out-dir results_smoke/berlin52
```

## Standard experiment

```bash
python experiments/run_tsplib_suite.py --runs 30
python experiments/run_ablation.py --runs 20
python experiments/run_sensitivity.py --runs 10
python experiments/make_figures.py
python experiments/make_tables.py
```

The standard experiment writes the main outputs to `results/`, `figures/`, and
`tables/`. The ablation outputs are written to `results/ablation/`; the paired
beta/gamma grid is written to `results/sensitivity/`.

The standard benchmark seeds each algorithm-instance-run combination
deterministically. The ablation uses paired seeds: all variants receive the same
seed for an instance-run pair. Statistical tests first average repeated runs
within each instance and then treat the 11 instances as the independent blocks.

## Targeted checks

```bash
python -m pytest -q
```

The tests cover positive-affine invariance of the objective weights, simplex
normalisation, the row-stochastic zero-diagonal interaction kernel, its expected
directionality, and the range of edge-disagreement diversity.
