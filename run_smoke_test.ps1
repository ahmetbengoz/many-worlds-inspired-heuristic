# Quick validation run from repository root
$ErrorActionPreference = "Stop"
python experiments\run_tsplib_suite.py --fast --runs 1 --instances berlin52 --out-dir results_smoke\berlin52
Write-Host "Smoke test completed. Outputs written to results_smoke\berlin52."
