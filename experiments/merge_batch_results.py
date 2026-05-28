from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats_tests import friedman_and_wilcoxon

CSV_NAMES = [
    "dataset_summary.csv",
    "per_run_results.csv",
    "convergence_curves.csv",
    "entropy_curves.csv",
    "performance_summary.csv",
]


def read_many(batch_root: Path, name: str) -> pd.DataFrame:
    parts = []
    for sub in sorted(batch_root.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / name
        if p.exists():
            df = pd.read_csv(p)
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if name == "dataset_summary.csv":
        out = out.drop_duplicates(subset=["instance"], keep="first")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default=str(ROOT / "results_standard"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    batch_root = Path(args.batch_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    frames = {name: read_many(batch_root, name) for name in CSV_NAMES}

    for name, df in frames.items():
        if df.empty:
            print(f"WARNING: no rows found for {name}")
            continue
        if name == "performance_summary.csv" and "mean_gap_percent" in df.columns:
            df["rank_within_instance"] = df.groupby("instance")["mean_gap_percent"].rank(method="average")
        df.to_csv(out_dir / name, index=False)
        print(f"Wrote {out_dir / name}: {len(df)} rows")

    per_run = frames.get("per_run_results.csv", pd.DataFrame())
    if not per_run.empty:
        ranks_df, stats_df = friedman_and_wilcoxon(per_run)
        ranks_df.to_csv(out_dir / "statistical_ranks.csv", index=False)
        stats_df.to_csv(out_dir / "statistical_tests.csv", index=False)
        print(f"Wrote statistical_ranks.csv and statistical_tests.csv")

    # Copy parameter settings from the first completed batch.
    for sub in sorted(batch_root.iterdir()):
        p = sub / "parameter_settings.json"
        if p.exists():
            shutil.copy2(p, out_dir / "parameter_settings.json")
            print(f"Copied parameter_settings.json from {sub.name}")
            break

    print("Merged batch results into", out_dir)


if __name__ == "__main__":
    main()
