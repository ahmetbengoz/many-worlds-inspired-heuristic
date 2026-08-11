# MWI-H: A Multi-Trajectory Weighting-and-Interaction Heuristic

This repository contains the reproducibility package for the manuscript:

**Do Multi-Trajectory Weights Preserve Search Diversity? A Controlled Component Study of MWI-H on the Symmetric Traveling Salesman Problem**

MWI-H is a classical population-based heuristic for symmetric TSP instances. Its
name denotes **multi-trajectory weighting and interaction**: candidate tours are
assigned normalized objective-dependent weights, sources exchange information
through a directed row-stochastic kernel, personal-best memory provides
trajectory persistence, and bounded 2-opt search intensifies accepted tours.

The method is defined by these components rather than by an analogy to a natural
or physical process. Weight entropy is reported as a diagnostic of influence
balance; solution-space diversity is measured separately by pairwise undirected
edge disagreement.

## Repository contents

```text
data/                     TSPLIB input instances used in the experiments
src/                      MWI-H and baseline algorithm implementations
experiments/              Experiment, figure, and table generation scripts
results/                  Reported fixed-configuration and ablation outputs
figures/                  Generated manuscript figures in PNG/PDF
tables/                   Generated manuscript tables in CSV/Markdown
docs/                     Reproducibility and release notes
requirements.txt          Python dependencies
run_smoke_test.ps1        Quick one-instance validation script
run_standard_experiment.ps1  Full standard experiment script
```

## Benchmark protocol

The reported experiment uses 11 symmetric Euclidean TSPLIB instances and 30
independent runs per algorithm-instance pair. MWI-H is compared with an
operator-controlled iterated local search (ILS), simulated annealing (SA),
genetic algorithm (GA), ant colony optimization (ACO), and artificial bee
colony / bee colony optimization (ABC/BCO). The ILS control uses the same
double-bridge perturbation and bounded 2-opt primitives as MWI-H. A paired
20-run ablation study tests four component removals on the same instances.

The independent block in the Friedman and paired Wilcoxon analyses is the TSP
instance (11 blocks). Repeated runs estimate each algorithm-instance mean and
are not treated as independent problem instances.

The main reported outputs are stored in `results/`:

- `dataset_summary.csv`
- `per_run_results.csv`
- `performance_summary.csv`
- `convergence_curves.csv.gz`
- `entropy_curves.csv.gz`
- `statistical_ranks.csv`
- `statistical_tests.csv`
- `parameter_settings.json`
- `ablation/ablation_per_run.csv`
- `ablation/ablation_summary.csv`
- `ablation/ablation_tests.csv`
- `sensitivity/sensitivity_per_run.csv`
- `sensitivity/sensitivity_summary.csv`

## Quick start

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Run a quick smoke test:

```bash
python experiments/run_tsplib_suite.py --fast --runs 1 --instances berlin52 --out-dir results_smoke/berlin52
```

Reproduce the standard experiment:

```bash
python experiments/run_tsplib_suite.py --runs 30
python experiments/run_ablation.py --runs 20
python experiments/run_sensitivity.py --runs 10
python experiments/make_figures.py
python experiments/make_tables.py
```

On Windows PowerShell, the same workflow can be launched with:

```powershell
powershell -ExecutionPolicy Bypass -File ./run_standard_experiment.ps1
```

## Interpretation boundary

This package supports a controlled component study on a modest symmetric-TSP
benchmark. It is not a claim that MWI-H supersedes specialised TSP solvers such
as LKH or EAX, and it should not be read as evidence of universal superiority
over other metaheuristics. The paper reports performance, uncertainty,
instance-level non-parametric tests, ablations, runtime, weight balance, and
edge diversity within the stated fixed-configuration protocol.

## Archival release

The reproducibility package is archived on Zenodo:

https://doi.org/10.5281/zenodo.21888355.

## Citation

Please cite the associated manuscript and archived release. `CITATION.cff`
contains the software citation metadata.
