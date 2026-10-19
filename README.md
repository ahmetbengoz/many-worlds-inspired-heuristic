# Influence redundancy and structural diversity in population based traveling salesman search

## Version 6 source code

This branch contains the source code, tests and experiment generators for a
measurement and control study of population-based symmetric TSP search. The
first experiment crosses objective guidance and structural survival on 28
Euclidean instances. The extension tests an incumbent-relative uphill gate and
class- versus slot-based survival control on 56 instances, with longer search
and a stronger ILS control. All search parameters are fixed in each protocol.

Complete experimental records, saved tours, input hashes, and supplementary
results accompany the manuscript as Online Resource 3. They are not currently
published on this GitHub branch. The code generates those files locally.

### First experiment

```bash
python -m pip install -r requirements-v6.txt
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

The earlier Zenodo DOI 10.5281/zenodo.21888355 identifies v4, not v5 or v6. Use the
exact computational source commit for this version and Online Resource 3 for
its associated data. No new archival DOI is claimed.

### Measurement and control extension

After reproducing the first experiment, run:

```bash
python experiments/run_adaptive_study.py --prepare
python experiments/validate_extended_inputs.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_adaptive_study.py --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_adaptive_study.py --references --workers 2
python experiments/analyse_adaptive_study.py
python experiments/plot_adaptive_study.py
```

There are 32 TSPLIB and 24 generated inputs, spanning 48 to 3,038 cities.
The added inputs include ATT, GEO and EXPLICIT distances. New paired seeds are
20001 through 20030. The original 28 inputs receive 50 and 250 iterations; all
56 receive 250 iterations, giving 15,120 new search runs over six conditions.
Five LKH seeds per added input extend the reference set to 280 unique tours.

The uphill gate admits an uphill crowding proposal only within 5% of the
incumbent at iteration start. Improving replacements remain eligible. This is
not a global population quality constraint. Class control activates gated
crowding below four clone-resolved effective trajectories; slot control uses
the same threshold on slot influence. ILS-descent repeats candidate-list scans
to convergence and matches the paired Guided delta-trial budget exactly.
Other operations and elapsed time are not equated by this primitive count.

This extension was designed after the first experiment was inspected. It is
exploratory. Five paired instance-mean contrasts at each horizon form one
ten-test Holm family; suite-specific analyses are separately corrected.
The protocol, search-module hashes, input hashes, environments, all best tours
and traces accompany the manuscript in Online Resource 3. Completed blocks
can resume from cache after interruption. See
[the v6 runbook](docs/REPRODUCIBILITY_v6.md) for definitions and verification.

The tests include regression against the original indexed/crowding policies,
gate boundaries, exact ILS work, restricted-neighborhood convergence, clone
equivalence and independent checks of the added distance formats. The study
examines measurement and intervention effects, rather than specialist-solver
superiority or a new entropy theorem.

## Preserved v4 material

The original source, data and results outside the v5 module remain for historical
traceability. They describe the earlier experiment and its limitations.
