# Many-Worlds–Inspired Heuristic Optimization (MWI-H)

This repository provides a reproducible Python implementation accompanying the manuscript:

**Optimization as Measurement: A Many-Worlds–Inspired Framework for Escaping Premature Commitment in Heuristic Search**

The code reproduces the convergence comparison figure (Figure 3) on a 50-city Euclidean TSP instance.

---

## Repository structure

- `data/` : dataset used in experiments (coordinates + distance matrix)
- `src/` : implementations of MWI-H and baseline heuristics
- `experiments/` : scripts to reproduce results and figures
- `figures/` : generated plots (output directory)

---

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt

## Reproducing Figure 3 (TSP convergence comparison)

-From the repository root:

   python experiments/run_tsp_experiments.py

 -This will:

   Load data/TSP_50_Cities_Data.xlsx

 - Run Simulated Annealing (SA), Genetic Algorithm (GA), and MWI-H

- Generate the convergence plot and save it to figures/Figure_3_TSP_Convergence.png

##  Citation

If you use this code, please cite the accompanying manuscript:

Ahmet Bengöz,
Optimization as Measurement: A Many-Worlds–Inspired Framework for Escaping Premature Commitment in Heuristic Search,
under review.

## License

MIT License (see LICENSE).
