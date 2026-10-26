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

The v6.0.0 distribution includes all fixed inputs and complete compressed
records. To regenerate validation, tables and figures without rerunning search,
install the dependencies, run the tests, then run both analysis scripts and
both plotting scripts as shown in the README. The analyses read compressed
JSONL records when per-block cache directories are absent. The extension's
15,120 main records are split across four ordered
`results/v6/run_records.part-*.jsonl.gz` files. The loader reads all four as one
record sequence. Their decompressed concatenation is byte-identical to the
verified combined archive. Analysis may also read or regenerate that combined
`run_records.jsonl.gz` file locally.

To rerun search from scratch, use an isolated copy with empty results folders.
Reproduce the first experiment first, using the README commands. Then run:

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

Complete experimental records are included in the repository and accompany the
manuscript as Online Resource 3: compressed JSONL best tours and traces, CSV
results, protocols, environment records and matching code. Large CSV tables
are compressed as `.csv.gz`; the analyses regenerate uncompressed tables.
`docs/RELEASE_MANIFEST_v6.0.0.json` hashes all supplied experimental files.
The complete v6.0.0 archive is available at
https://doi.org/10.5281/zenodo.23669935. Its release tag identifies commit
`2f18dbd1d6994d2f6f337c5f3b8a0b7dc41038ca`.
