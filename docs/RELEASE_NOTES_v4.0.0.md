# Release notes: v4.0.0

This release rebuilds the MWI-H study as a component-defined symmetric-TSP
heuristic and reproducibility package.

Changes include:

- redefinition of MWI-H as multi-trajectory weighting and interaction, with no
  physical or biological design metaphor;
- explicit positive-affine invariance, simplex, row-stochasticity,
  directionality, and elitist-monotonicity properties;
- zero-diagonal interaction probabilities, preventing self-sampling;
- separate influence-balance and tour-edge-diversity diagnostics;
- an operator-controlled ILS baseline using the same double-bridge and bounded
  2-opt primitives;
- a paired component ablation over all 11 benchmark instances;
- a paired 4-by-4 beta/gamma sensitivity grid on three representative instances;
- corrected instance-level Friedman and Wilcoxon/Holm analyses, avoiding
  pseudoreplication of stochastic runs;
- runtime reporting, deterministic seeds, tests, and an expanded runbook; and
- updated manuscript figures, tables, and machine-readable result files.

The empirical claims are limited to the stated fixed-configuration protocol and
do not position MWI-H against specialised TSP solvers.
