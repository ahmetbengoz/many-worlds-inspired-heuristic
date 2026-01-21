# Many-Worlds–Inspired Heuristic Optimization (MWI-H)

This repository provides a reproducible Python implementation accompanying the manuscript:

**“Optimization as Measurement: A Many-Worlds–Inspired Framework for Escaping Premature Commitment in Heuristic Search”**

The code reproduces the convergence comparison figure (Figure 3) on a 50-city Euclidean TSP instance.

## Repository structure

- `data/` : dataset used in experiments (coordinates + distance matrix)
- `src/` : implementations of MWI-H and baseline heuristics
- `experiments/` : scripts to reproduce results and figures
- `figures/` : generated plots (output directory)

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
