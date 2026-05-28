from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
tables=ROOT/'tables'; tables.mkdir(exist_ok=True)
res=ROOT/'results'
notation = pd.DataFrame([
    {'symbol':'$X^t$','meaning':'Population of candidate trajectories at iteration $t$'},
    {'symbol':'$x_i^t$','meaning':'The $i$th candidate solution / trajectory at iteration $t$'},
    {'symbol':'$N$','meaning':'Population size / number of maintained trajectories'},
    {'symbol':'$f(x_i^t)$','meaning':'Objective value of trajectory $x_i^t$'},
    {'symbol':'$\\hat f_i^t$','meaning':'Min-max normalized objective value at iteration $t$'},
    {'symbol':'$\\beta$','meaning':'Influence-weight selection sharpness'},
    {'symbol':'$p_i^t$','meaning':'Normalized influence weight of trajectory $i$'},
    {'symbol':'$H^t$','meaning':'Entropy of the influence-weight distribution'},
    {'symbol':'$N_{\\mathrm{eff}}^t$','meaning':'Effective number of active trajectories, $\\exp(H^t)$'},
    {'symbol':'$\\gamma$','meaning':'Directed transition selectivity parameter'},
    {'symbol':'$\\tau_{ij}^t$','meaning':'Directed transition probability from trajectory $i$ to trajectory $j$'},
    {'symbol':'$\\mathcal{N}(.)$','meaning':'Neighborhood operator such as 2-opt, swap, or insertion'},
    {'symbol':'$\\mathcal{U}(.)$','meaning':'Population-level update operator'},
    {'symbol':'$I_{\\max}$','meaning':'Maximum number of iterations'},
    {'symbol':'$\\epsilon$','meaning':'Small numerical stability constant'},
])
notation.to_csv(tables/'table1_notation.csv',index=False)

diff = pd.DataFrame([
    {'method':'Simulated Annealing (SA)','population_structure':'Single trajectory','diversity_mechanism':'Temperature-based acceptance','transition_or_exchange':'No inter-trajectory transition','persistence_model':'Current trajectory may move to worse states; no weighted coexistence','difference_from_MWIH':'MWI-H maintains multiple weighted trajectories and tracks entropy.'},
    {'method':'Genetic Algorithm (GA)','population_structure':'Population of chromosomes','diversity_mechanism':'Crossover, mutation, selection pressure','transition_or_exchange':'Genetic recombination','persistence_model':'Low-fitness individuals are commonly eliminated by selection','difference_from_MWIH':'MWI-H avoids full collapse by preserving influence-weighted trajectories.'},
    {'method':'Island models','population_structure':'Multiple subpopulations','diversity_mechanism':'Migration topology and isolated evolution','transition_or_exchange':'Periodic migration among islands','persistence_model':'Subpopulations persist; individuals still subject to selection','difference_from_MWIH':'MWI-H operates at trajectory-influence level with a directed transition kernel, not island migration.'},
    {'method':'ABC/BCO','population_structure':'Food sources / bee roles','diversity_mechanism':'Employed, onlooker, and scout search phases','transition_or_exchange':'Recruitment around food sources','persistence_model':'Unproductive sources are abandoned after a limit','difference_from_MWIH':'MWI-H does not model bee roles and uses normalized influence and entropy to quantify non-collapse.'},
    {'method':'ACO','population_structure':'Constructive ant agents','diversity_mechanism':'Pheromone evaporation and probabilistic construction','transition_or_exchange':'Indirect pheromone-mediated communication','persistence_model':'Search memory stored in edges/pheromone, not full trajectory weights','difference_from_MWIH':'MWI-H stores influence over full candidate trajectories and directed trajectory transitions.'},
    {'method':'QEA/QPSO','population_structure':'Quantum-inspired representation or particle model','diversity_mechanism':'Amplitude/probability representation or quantum-behaved update','transition_or_exchange':'Representation-specific update equations','persistence_model':'Usually representation-level probability, not explicit TSP trajectory persistence','difference_from_MWIH':'MWI-H uses no quantum hardware and no quantum state; MWI is only a design metaphor.'},
])
diff.to_csv(tables/'table2_method_differences.csv',index=False)

pd.read_csv(res/'dataset_summary.csv').to_csv(tables/'table3_tsplib_instances.csv',index=False)
params=json.load(open(res/'parameter_settings.json'))
rows=[]
for alg, ps in params.items():
    for k,v in ps.items(): rows.append({'algorithm':alg,'parameter':k,'value':v})
pd.DataFrame(rows).to_csv(tables/'table4_parameter_settings.csv',index=False)
summary=pd.read_csv(res/'performance_summary.csv')
summary_cols=['instance','dimension','algorithm','runs','best_known','best','mean','std','mean_gap_percent','best_gap_percent','rank_within_instance','budget_mode']
summary[summary_cols].to_csv(tables/'table5_performance_results.csv',index=False)
pd.read_csv(res/'statistical_ranks.csv').to_csv(tables/'table6a_statistical_ranks.csv',index=False)
pd.read_csv(res/'statistical_tests.csv').to_csv(tables/'table6b_statistical_tests.csv',index=False)
# Also create markdown copies.
for csv in tables.glob('table*.csv'):
    df=pd.read_csv(csv)
    csv.with_suffix('.md').write_text(df.to_markdown(index=False), encoding='utf-8')
print('Tables written to', tables)
