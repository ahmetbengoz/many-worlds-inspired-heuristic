# Influence redundancy and structural diversity in population based traveling salesman search

## Version 5 source code

This branch contains the source code, tests, and experiment generator for the
v5 controlled component study. It crosses objective guidance and structural
survival with common initialization and two bounded ILS controls. It is a
measurement study, not a claim of state-of-the-art TSP accuracy.

Complete experimental records, saved tours, input hashes, and supplementary
results accompany the manuscript as Online Resource 3. They are not currently
published on this GitHub branch. The code generates those files locally.

### Reproduce

```bash
python -m pip install -r requirements-v5.txt
python -m pytest tests -q
python experiments/run_component_study.py --prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_component_study.py --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_component_study.py --references --workers 2
python experiments/analyse_component_study.py
python experiments/plot_component_study.py
```

The preparation step downloads nine additional public EUC_2D TSPLIB instances
and creates eight seeded synthetic instances, supplementing the eleven public
instances already present. All six conditions share the same initialization
for a paired seed. Main seeds are 10001 through 10030; LKH seeds are 1 through 5.
Synthetic reference lengths are feasible solver bounds, not certified optima.
The analysis treats instance means as blocks and corrects five primary tests
with Holm. Run-level data are not independent problem instances.

The v5 module uses a tested true four-edge double bridge. The legacy routine
is retained with its archived v4 implementation. The official tsp225 optimum
is 3916. V5 uses this reference; old metadata carried 3919. The old experiment
and new experiment must not be pooled.

The class-level influence decomposition applies the standard Shannon chain
rule. Exact clone classes respect undirected edge equivalence, including tour
rotation and reversal. Distinct tours can still be near-clones, so edge
disagreement and relative-to-incumbent quality bands are also recorded.

The quality-band audit was added after two smoke outputs had been inspected
and is secondary. It changes no search decisions. The --enrich option checks
replayed best tours, lengths, work counts and original traces before adding
these fields. Original timing excludes that later measurement extension.

The earlier Zenodo DOI 10.5281/zenodo.21888355 identifies v4, not v5. Use the
exact computational source commit for this version and Online Resource 3 for
its associated data. No new archival DOI is claimed.

## Preserved v4 material

The original source, data and results outside the v5 module remain for historical
traceability. They describe the earlier experiment and its limitations.
