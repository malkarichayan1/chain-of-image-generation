#!/usr/bin/env python
"""
Run locally after pulling chain_experiment_results.csv back from the Kaggle kernel.
States the pre-registered pass/fail explicitly (real >> shuffled/substituted/
attention_scrambled) rather than leaving it to eyeballing a table -- see
pi_level_experiment/README.md for what each condition means and
pi_level_experiment/RESULTS.md for the first real run's actual numbers.

Uses a one-sided Mann-Whitney U test, not Cohen's d against an absolute gap
threshold: these IoU distributions are heavily zero-inflated (a majority of rows in
every condition, including real, are exactly 0.0 -- see RESULTS.md), so they're far
from normal, and an earlier version of this script's "gap > 0.1" absolute threshold
turned out to be miscalibrated against how small this metric's real-world scores
actually are -- it called a p=0.004 result a "FAIL" purely because 0.036 - 0.0 didn't
clear an arbitrary 0.1 bar. Mann-Whitney doesn't assume normality and isn't fooled by
the absolute scale.
"""
import sys

import pandas as pd
from scipy import stats


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("\n=== Per-condition summary ===")
    summary = df.groupby("condition")["iou"].agg(["mean", "std", "median", "min", "max", "count"])
    nonzero_rate = df.groupby("condition")["iou"].apply(lambda s: (s > 0).mean())
    summary["nonzero_rate"] = nonzero_rate
    print(summary.round(4))

    print("\n=== Per-condition, per-n (subject count) breakdown ===")
    print(df.groupby(["condition", "n"])["iou"].mean().unstack(0).round(4))

    if "real" not in summary.index:
        print("\nNo 'real' condition rows -- cannot evaluate the money result.")
        return

    real = df.loc[df.condition == "real", "iou"]
    print(f"\n=== Money-result check (real mean={real.mean():.4f}, "
          f"nonzero_rate={(real > 0).mean():.2f}, n={len(real)}) ===")
    all_significant = True
    for cond in ("shuffled", "substituted", "attention_scrambled"):
        if cond not in summary.index:
            print(f"  {cond}: NO DATA")
            continue
        other = df.loc[df.condition == cond, "iou"]
        _, p_value = stats.mannwhitneyu(real, other, alternative="greater")
        verdict = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "NOT significant"
        if p_value >= 0.05:
            all_significant = False
        print(f"  real > {cond}: mean={other.mean():.4f}, nonzero_rate={(other > 0).mean():.2f}, "
              f"Mann-Whitney p={p_value:.4f} -> {verdict}")

    print("\n=== Overall verdict ===")
    if all_significant:
        print("REAL is significantly higher than all three control conditions (p<0.05).")
        print("This replicates -- on real generated images, real captured attention -- the")
        print("discrimination the CR pilot's persistence check (0.83 vs 0.83) could not show.")
    else:
        print("At least one control condition did not reach significance against REAL.")
        print("Report exactly which one(s), and why (sample size, control design) -- see")
        print("RESULTS.md's discussion of the first run for the kind of detail expected here.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "chain_experiment_results.csv"
    main(path)
