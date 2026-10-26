# MWI-H v6.0.0

This release supplies the complete computational record for Influence
redundancy and structural diversity in population based traveling salesman
search. It combines two separately documented experiments.

The guidance-by-survival factorial study has 5,040 main runs on 28 Euclidean
instances. The exploratory measurement-control extension adds 15,120 runs on
56 symmetric instances, including ATT, GEO and EXPLICIT distances, up to 3,038
cities. The original 28 inputs are examined at 50 and 250 iterations; all 56
are examined at 250 iterations. Each condition uses 30 paired seeds.

The release includes all 20,160 main records and 280 unique LKH reference
tours, saved best tours and traces, 56 hashed inputs, fixed protocols,
environment records, independent validation, instance-level statistical tables,
six publication figures and 18 regression and property tests.

Search modules are unchanged from the verified source state
5967a2e47d35b731e9079cf6972e67e39aeee190. The distribution adds complete
scientific records, byte-checked atomic CSV exports and release metadata.
The extension's main JSONL archive is distributed in four independently
readable gzip parts to accommodate transfer limits; the loader combines them
automatically. Decompressed records are unchanged.
The calculations and search decisions are unchanged. Protocol execution snapshots retain
the exact runner used at launch.

The principal extension result is a modest long-horizon gain for class-based
against slot-based activation of the same gated survival rule. The difference
is not significant at the shorter horizon. Gating alone worsens short-horizon
quality, and the stronger ILS control changes the short-horizon assessment.
The data support the stated controlled comparison on symmetric TSP.

Archival DOI: https://doi.org/10.5281/zenodo.23669935

Release tag: `v6.0.0`

Release commit: `2f18dbd1d6994d2f6f337c5f3b8a0b7dc41038ca`

The published Zenodo archive and the release contain the same 215 files.
DOI metadata were finalized after archival publication; search code and
experimental records are unchanged.
