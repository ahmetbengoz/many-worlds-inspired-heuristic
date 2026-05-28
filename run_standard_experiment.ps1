# Run from repository root
$ErrorActionPreference = "Stop"
python experiments\run_tsplib_suite.py --runs 30
python experiments\make_figures.py
python experiments\make_tables.py
Write-Host "Standard experiment completed. Outputs written to results/, figures/, and tables/."
