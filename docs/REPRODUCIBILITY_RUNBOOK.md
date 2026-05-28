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
python experiments/make_figures.py
python experiments/make_tables.py
```

The standard experiment writes the main outputs to `results/`, `figures/`, and `tables/`.
