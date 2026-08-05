| symbol | meaning |
| --- | --- |
| $X^t$ | Population of candidate trajectories at iteration $t$ |
| $x_i^t$ | The $i$th candidate solution / trajectory at iteration $t$ |
| $N$ | Population size / number of maintained trajectories |
| $f(x_i^t)$ | Objective value of trajectory $x_i^t$ |
| $\hat f_i^t$ | Min-max normalized objective value at iteration $t$ |
| $\beta$ | Influence-weight selection sharpness |
| $p_i^t$ | Normalized influence weight of trajectory $i$ |
| $H^t$ | Entropy of the influence-weight distribution (weight-balance diagnostic) |
| $N_{\mathrm{eff}}^t$ | Effective influence count, $\exp(H^t)$ |
| $D_E^t$ | Mean pairwise undirected edge disagreement among tours |
| $\gamma$ | Directed transition selectivity parameter |
| $\tau_{ij}^t$ | Directed transition probability from trajectory $i$ to trajectory $j$ |
| $\mathcal{N}(.)$ | Neighborhood operator such as 2-opt, swap, or insertion |
| $\mathcal{U}(.)$ | Population-level update operator |
| $I_{\max}$ | Maximum number of iterations |
| $\epsilon$ | Small numerical stability constant |
