# MWI-H: An Entropy-Aware Trajectory-Persistence Metaheuristic

This repository contains the reproducibility package for the manuscript:

**MWI-H: An Entropy-Aware Trajectory-Persistence Metaheuristic for Mitigating Premature Commitment in Heuristic Search**

MWI-H is a classical metaheuristic framework that operationalizes measurement-inspired trajectory persistence through normalized objective values, unit-mass influence weights, entropy-based diversity monitoring, a directed transition kernel, and a population-level update operator. It does not use quantum hardware and does not simulate quantum mechanics.

## Repository contents

```text
data/tsplib/              TSPLIB input instances used in the experiments
src/                      MWI-H and baseline algorithm implementations
experiments/              Experiment, figure, and table generation scripts
results/                  Reported fixed-configuration experiment outputs
figures/                  Generated manuscript figures in PNG/PDF
tables/                   Generated manuscript tables in CSV/Markdown
docs/                     Reproducibility and release notes
requirements.txt          Python dependencies
run_smoke_test.ps1        Quick one-instance validation script
run_standard_experiment.ps1  Full standard experiment script
```

## Benchmark protocol

The reported experiment uses 11 symmetric Euclidean TSPLIB instances and 30 independent runs per algorithm-instance pair. MWI-H is compared with simulated annealing (SA), genetic algorithm (GA), ant colony optimization (ACO), and artificial bee colony / bee colony optimization (ABC/BCO) baselines under a fixed-configuration reproducibility protocol.

The main reported outputs are stored in `results/`:

- `dataset_summary.csv`
- `per_run_results.csv`
- `performance_summary.csv`
- `convergence_curves.csv`
- `entropy_curves.csv`
- `statistical_ranks.csv`
- `statistical_tests.csv`
- `parameter_settings.json`

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
python experiments/make_figures.py
python experiments/make_tables.py
```

On Windows PowerShell, the same workflow can be launched with:

```powershell
powershell -ExecutionPolicy Bypass -File ./run_standard_experiment.ps1
```

## Reported headline result

Under the fixed-configuration benchmark protocol, MWI-H obtains the best average rank among the tested methods and the best mean result on 10 of 11 TSPLIB instances. The average mean optimality gap of MWI-H is 5.47% in the reported run set. These results should be interpreted within the stated fixed-configuration protocol, not as a universal claim of solver superiority.

## Archival release

This repository is intended to be archived through the GitHub-Zenodo integration. After creating the final GitHub release, update the manuscript Data and Code Availability statement with the newly generated Zenodo version DOI.

## Citation

Please cite the associated manuscript and the archived Zenodo release once available. A preliminary citation file is provided in `CITATION.cff` and should be updated with the final DOI after the Zenodo release is generated.
