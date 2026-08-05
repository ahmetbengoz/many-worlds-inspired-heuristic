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
    (0.10, 0.70, '1. Tour population\n$X^t=\\{x_1^t,\\ldots,x_N^t\\}$\n$f(x_i^t)$'),
    (0.34, 0.70, '2. Objective\nnormalisation\n$\\hat f_i^t$'),
    (0.58, 0.70, '3. Influence weights\n$p_i^t$, $\\sum_i p_i^t=1$'),
    (0.82, 0.70, '4. Directed interaction\n$\\tau_{ij}^t \\ne \\tau_{ji}^t$'),
    (0.58, 0.27, '5. Candidate generation\nand bounded 2-opt\n$X^{t+1}=\\mathcal{U}(X^t,P^t,\\tau^t,\\mathcal{N})$'),
    (0.86, 0.27, '6. Separate diagnostics\nweight balance: $N_{\\mathrm{eff}}^t$\nedge diversity: $D_E^t$'),
]
for x, y, text in boxes:
    ax.text(x, y, text, ha='center', va='center', fontsize=10, bbox=box)

def arrow(a,b):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', lw=1.4))
arrow((0.19,0.70),(0.27,0.70)); arrow((0.42,0.70),(0.49,0.70)); arrow((0.69,0.70),(0.73,0.70))
arrow((0.82,0.61),(0.62,0.37)); arrow((0.68,0.27),(0.76,0.27))
fig.tight_layout()
fig.savefig(fig_dir / 'figure1_architecture.png', dpi=300)
fig.savefig(fig_dir / 'figure1_architecture.pdf')
plt.close(fig)

# Figure 2: normalized weights (illustrative, sum=1)
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

# Figure 3: weight balance and edge diversity for selected instances
ent = pd.read_csv(res / 'entropy_curves.csv.gz')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for inst in ['berlin52', 'ch150', 'lin318']:
    sub = ent[ent['instance'] == inst]
    if sub.empty:
        continue
    mean = sub.groupby('iteration')['N_eff'].mean().reset_index()
    axes[0].plot(mean['iteration'], mean['N_eff'], label=inst)
    div = sub.groupby('iteration')['edge_diversity'].mean().reset_index()
    axes[1].plot(div['iteration'], div['edge_diversity'], label=inst)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Effective influence count')
axes[0].legend(loc='best')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Mean pairwise edge disagreement')
axes[1].set_ylim(0, 1)
axes[1].legend(loc='best')
fig.tight_layout()
fig.savefig(fig_dir / 'figure3_diagnostics.png', dpi=300)
fig.savefig(fig_dir / 'figure3_diagnostics.pdf')
plt.close(fig)

# Figure 4: component-ablation gaps by variant
abl = pd.read_csv(res / 'ablation' / 'ablation_summary.csv')
variant_order = ['full', 'uniform_weights', 'no_directional_penalty', 'no_persistence', 'no_local_search']
abl_rank = (abl.groupby('variant', as_index=False)
            .agg(mean_gap=('mean_gap_percent', 'mean'),
                 mean_rank=('rank_within_instance', 'mean')))
abl_rank['order'] = abl_rank['variant'].map({v: i for i, v in enumerate(variant_order)})
abl_rank = abl_rank.sort_values('order')
display_labels = {
    'full': 'Full', 'uniform_weights': 'Uniform\nweights',
    'no_directional_penalty': 'No directional\npenalty',
    'no_persistence': 'No\npersistence', 'no_local_search': 'No local\nsearch',
}
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([display_labels[v] for v in abl_rank['variant']], abl_rank['mean_rank'])
ax.set_ylabel('Mean within-instance rank (lower is better)')
fig.tight_layout()
fig.savefig(fig_dir / 'figure4_component_ablation.png', dpi=300)
fig.savefig(fig_dir / 'figure4_component_ablation.pdf')
plt.close(fig)

# Figure 5: beta/gamma sensitivity averaged over the three representative instances
sens = pd.read_csv(res / 'sensitivity' / 'sensitivity_summary.csv')
sens_mean = sens.groupby(['beta', 'gamma'], as_index=False)['mean_gap_percent'].mean()
heat = sens_mean.pivot(index='beta', columns='gamma', values='mean_gap_percent').sort_index(ascending=False)
fig, ax = plt.subplots(figsize=(7.6, 5.6))
im = ax.imshow(heat.values, aspect='auto', cmap='viridis_r')
ax.set_xticks(np.arange(len(heat.columns)), labels=[f'{x:g}' for x in heat.columns])
ax.set_yticks(np.arange(len(heat.index)), labels=[f'{x:g}' for x in heat.index])
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\beta$')
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        value = heat.iloc[i, j]
        color = 'white' if value > np.nanmedian(heat.values) else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=9)
fig.colorbar(im, ax=ax, label='Mean optimality gap (%)')
fig.tight_layout()
fig.savefig(fig_dir / 'figure5_sensitivity_heatmap.png', dpi=300)
fig.savefig(fig_dir / 'figure5_sensitivity_heatmap.pdf')
plt.close(fig)

# Additional diagnostic: boxplot of final gaps by algorithm
per = pd.read_csv(res / 'per_run_results.csv')
algos = [a for a in ['MWI-H', 'ILS', 'ACO', 'ABC/BCO', 'GA', 'SA'] if a in set(per['algorithm'])]
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
