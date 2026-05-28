from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
fig_dir = ROOT / 'figures'
fig_dir.mkdir(exist_ok=True)
res = ROOT / 'results'

# Figure 1: architecture diagram
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
box = dict(boxstyle='round,pad=0.45', fc='white', ec='black', lw=1.2)
boxes = [
    (0.12, 0.72, '1. Population\ntrajectories\n$X^t=\\{x_1^t,\\ldots,x_N^t\\}$\n$f(x_i^t)$'),
    (0.36, 0.72, '2. Objective\nnormalization\n$\\hat f_i^t$'),
    (0.60, 0.72, '3. Influence\nweights\n$p_i^t$ , $\\sum_i p_i^t=1$'),
    (0.84, 0.72, '4. Directed\ntransition kernel\n$\\tau_{ij}^t \\ne \\tau_{ji}^t$'),
    (0.36, 0.28, '5. Population\nupdate operator\n$X^{t+1}=\\mathcal{U}(X^t,P^t,\\tau^t,\\mathcal{N})$'),
    (0.66, 0.28, '6. Entropy / persistence\nmetrics\n$H^t$, $N_{\\mathrm{eff}}^t$'),
]
for x, y, text in boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=10, bbox=box)

def arrow(a,b):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', lw=1.4))
arrow((0.22,0.72),(0.29,0.72)); arrow((0.46,0.72),(0.53,0.72)); arrow((0.70,0.72),(0.77,0.72))
arrow((0.12,0.62),(0.31,0.38)); arrow((0.60,0.62),(0.42,0.38)); arrow((0.84,0.62),(0.43,0.38)); arrow((0.60,0.62),(0.66,0.39))
fig.tight_layout()
fig.savefig(fig_dir / 'figure1_architecture.png', dpi=300)
fig.savefig(fig_dir / 'figure1_architecture.pdf')
plt.close(fig)

# Figure 2: corrected normalized weights (illustrative, sum=1)
rng = np.random.default_rng(20260520)
iterations = np.arange(1, 31)
base = np.vstack([np.linspace(0.15+i*0.03, 0.75-i*0.02, len(iterations)) for i in range(5)]).T
noise = rng.normal(0, 0.03, size=base.shape)
fhat = np.clip(base + noise, 0, 1)
beta = 5.0
w = np.exp(-beta * fhat)
p = w / w.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 5))
for k in range(p.shape[1]):
    ax.plot(iterations, p[:, k], label=f'Trajectory {k+1}')
ax.plot(iterations, p.sum(axis=1), linestyle='--', label='Weight sum')
ax.set_xlabel('Iteration')
ax.set_ylabel('Influence weight')
ax.legend(loc='best', fontsize=8)
fig.tight_layout()
fig.savefig(fig_dir / 'figure2_normalized_weights.png', dpi=300)
fig.savefig(fig_dir / 'figure2_normalized_weights.pdf')
plt.close(fig)

# Figure 3: effective trajectories for selected instances
ent = pd.read_csv(res / 'entropy_curves.csv')
fig, ax = plt.subplots(figsize=(9, 5))
for inst in ['berlin52', 'ch150', 'lin318']:
    sub = ent[ent['instance'] == inst]
    if sub.empty:
        continue
    mean = sub.groupby('iteration')['N_eff'].mean().reset_index()
    ax.plot(mean['iteration'], mean['N_eff'], label=inst)
ax.set_xlabel('Iteration')
ax.set_ylabel('Effective number of active trajectories')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(fig_dir / 'figure3_neff_curves.png', dpi=300)
fig.savefig(fig_dir / 'figure3_neff_curves.pdf')
plt.close(fig)

# Figure 4: convergence curves for a selected instance
conv = pd.read_csv(res / 'convergence_curves.csv')
sel_inst = 'ch150'
fig, ax = plt.subplots(figsize=(9, 5))
for algo, sub in conv[conv['instance'] == sel_inst].groupby('algorithm'):
    mean = sub.groupby('iteration_index')['gap_percent'].mean().reset_index()
    ax.plot(mean['iteration_index'], mean['gap_percent'], label=algo)
ax.set_xlabel('Recorded iteration index')
ax.set_ylabel('Mean best-so-far gap (%)')
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(fig_dir / 'figure4_convergence_ch150.png', dpi=300)
fig.savefig(fig_dir / 'figure4_convergence_ch150.pdf')
plt.close(fig)

# Figure 5: boxplot of final gaps by algorithm
per = pd.read_csv(res / 'per_run_results.csv')
algos = [a for a in ['MWI-H', 'ACO', 'ABC/BCO', 'GA', 'SA'] if a in set(per['algorithm'])]
data = [per.loc[per['algorithm'] == a, 'gap_percent'].dropna().values for a in algos]
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(data, tick_labels=algos, showmeans=True)
ax.set_ylabel('Final best gap (%)')
fig.tight_layout()
fig.savefig(fig_dir / 'figure5_gap_boxplot.png', dpi=300)
fig.savefig(fig_dir / 'figure5_gap_boxplot.pdf')
plt.close(fig)

# Figure 6: average ranks
ranks = pd.read_csv(res / 'statistical_ranks.csv')
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(ranks['algorithm'], ranks['average_rank'])
ax.set_ylabel('Average rank (lower is better)')
fig.tight_layout()
fig.savefig(fig_dir / 'figure6_average_ranks.png', dpi=300)
fig.savefig(fig_dir / 'figure6_average_ranks.pdf')
plt.close(fig)

print('Figures written to', fig_dir)
