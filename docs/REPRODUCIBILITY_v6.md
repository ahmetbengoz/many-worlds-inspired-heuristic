# Reproducing the measurement and control extension

The first experiment remains in `src/component_study.py` and `results/v5`.
The extension uses `src/adaptive_study.py`, `src/tsplib_extended.py` and
`experiments/run_adaptive_study.py`. Search-module hashes were recorded before
the extension runs. The extension is exploratory, designed after inspecting
the first experiment, not a preregistered confirmatory study.

## Design and interventions

The benchmark has 56 symmetric instances: 32 TSPLIB inputs and 24 generated
inputs. The 12 added TSPLIB cases include ATT, GEO and EXPLICIT distances and
the 3,038-city pcb3038 instance. Each uniform/clustered geometry and size cell
has three fixed generated inputs. New paired seeds are 20001 through 20030.
All 56 inputs receive 250 population iterations; the original 28 also receive
50 iterations. Six conditions give 15,120 new search runs.

All population policies use objective guidance beta=6, gamma=4. Guided uses
indexed survival; Crowding uses nearest-tour survival with clone-aware elites.
Gated adds a 5% incumbent-relative eligibility condition to uphill crowding
proposals. Improving replacements remain eligible, even outside this band.
The gate is not a population-wide quality constraint. Adaptive-class activates
gated crowding when clone-resolved influence is below 4; Adaptive-slot uses
slot influence below 4. Otherwise both use indexed survival. Elite handling
follows the active survival mode. These thresholds were not tuned on the new
outputs.

Every population run makes 16*T main proposals. The common stagnation rule can
add restart proposals, recorded separately. `gate_rejections` is the number of
gate-ineligible uphill proposals, including proposals that already fail the
random acceptance test. It does not count accepted moves prevented by the gate.

ILS-descent starts from the same common initial best. It repeats candidate-list
2-opt scans until no improvement remains in that restricted scan or its work
budget is exhausted. Its final descent is truncated to match the paired Guided
delta-trial total exactly, including initialization and restarts. It uses the
shared perturbation and swap/insertion probabilities. Other operations and
runtime are not matched by this one-primitive control.

## Commands

For a source-only checkout, reproduce the first experiment first, using the
README commands. Then run:

```bash
python -m pip install -r requirements-v6.txt
python -m pytest tests -q
python experiments/run_adaptive_study.py --prepare
python experiments/validate_extended_inputs.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_adaptive_study.py --workers 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_adaptive_study.py --references --workers 2
python experiments/analyse_adaptive_study.py
python experiments/plot_adaptive_study.py
```

Completed instance/seed/horizon blocks are cached atomically. Interrupted runs
can resume with the same command. Cached blocks retain their original timing;
new blocks may be executed in a later session. The archive records execution
environments. Reported CPU seconds are per process. Wall seconds can include
contention and are not an equal-time solver comparison.

`--prepare` generates inputs and locks the protocol if none exists. Do not use
old cached outputs with modified search modules or settings. Preserve the
deposited protocol and verify its module hashes before reproducing this version.

## References and analysis

Each new instance receives five LKH runs with seeds 1 through 5 and RUNS=1.
The original 140 references are reused, giving 280 unique reference tours in
total. EXPLICIT triangular files are expanded losslessly to FULL_MATRIX for
elkai's input bridge. Generated references are feasible bounds, not optima.

The ten primary extension tests comprise five fixed contrasts at each horizon,
with one Holm family. Instance means over 30 seeds are statistical blocks.
The 15 long-horizon stratum tests have a separate exploratory Holm family.
Median intervals use 10,000 paired instance bootstrap samples. Wilcoxon tests
use exact untied ranks, or Pratt ranks and 50,000 seeded sign permutations.

Analysis checks all 15,120 new tours and all 280 references independently,
input hashes, paired initialization, seed coverage and exact ILS budgets.
It verifies that the first 50 population iterations are identical in both
horizons for the original inputs. A same-state audit compares both trigger
criteria on the Guided populations, without treating outcomes from different
policies as ground-truth signal labels.

Complete experimental records accompany the manuscript as Online Resource 3,
including compressed JSONL best tours and traces, CSV results, the protocol,
environment records and matching code. Those records are not published on this
GitHub branch. With the archive, analysis and plots can be rebuilt without
rerunning search. The older v4 Zenodo DOI does not identify this extension.
