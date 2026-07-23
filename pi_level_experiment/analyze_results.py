#!/usr/bin/env python
"""
Run locally after pulling chain_experiment_results.csv back from the Kaggle kernel.
States the pre-registered pass/fail explicitly (real >> shuffled/substituted/
attention_scrambled) rather than leaving it to eyeballing a table -- see
pi_level_experiment/README.md for what each condition means.
"""
import sys

import numpy as np
import pandas as pd


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("\n=== Per-condition summary ===")
    summary = df.groupby("condition")["iou"].agg(["mean", "std", "median", "min", "max", "count"])
    print(summary.round(4))

    print("\n=== Per-condition, per-n (subject count) breakdown ===")
    print(df.groupby(["condition", "n"])["iou"].mean().unstack(0).round(4))

    if "real" not in summary.index:
        print("\nNo 'real' condition rows -- cannot evaluate the money result.")
        return

    real = df.loc[df.condition == "real", "iou"].to_numpy()
    print(f"\n=== Money-result check (real mean = {real.mean():.4f}, n={len(real)}) ===")
    all_pass = True
    for cond in ("shuffled", "substituted", "attention_scrambled"):
        if cond not in summary.index:
            print(f"  {cond}: NO DATA")
            continue
        other = df.loc[df.condition == cond, "iou"].to_numpy()
        d = cohens_d(real, other)
        gap = real.mean() - other.mean()
        verdict = "PASS (real clearly higher)" if gap > 0.1 and d > 0.5 else "FAIL / WEAK (no clear separation)"
        if "PASS" not in verdict:
            all_pass = False
        print(f"  real vs {cond}: gap={gap:+.4f}, Cohen's d={d:.2f} -> {verdict}")

    print("\n=== Overall verdict ===")
    if all_pass:
        print("REAL is meaningfully separated from all three control conditions.")
        print("This replicates -- on real generated images, real captured attention -- the")
        print("discrimination the CR pilot's persistence check (0.83 vs 0.83) could not show.")
    else:
        print("At least one control condition did NOT separate cleanly from REAL.")
        print("This does not confirm the metric's structural lock-immunity claim as strongly")
        print("as hoped -- report this honestly, do not round up to a pass.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "chain_experiment_results.csv"
    main(path)
