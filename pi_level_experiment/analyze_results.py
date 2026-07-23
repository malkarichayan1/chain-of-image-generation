#!/usr/bin/env python
"""
Run locally after pulling chain_experiment_results.csv back from Kaggle (v4 schema) or
after running score_chains.py locally against a pulled-back segmentation cache (Step 4
schema, with chain_key/prompt_id/seed). States the pre-registered pass/fail explicitly
(real >> every control condition) rather than leaving it to eyeballing a table.

Extended per docs/part-b-strengthening-design.md's Step 4:
  - Reports BOTH a pooled test (Mann-Whitney, one-sided -- directly comparable to the v4
    numbers already published in RESULTS.md) and a clustered test (Wilcoxon signed-rank
    on per-prompt paired differences, n=9 prompts). Multi-seed rows are clustered within
    prompt, so the pooled test over ~81 rows is anticonservative; if the two diverge, the
    clustered test governs and that is stated plainly, not buried.
  - Handles any set of control conditions present (the new attn_scrambled_samechain /
    _crosschain / _sameattr controls, not just the old single attention_scrambled), so
    this script works unchanged against both the v4 CSV and Step 4's confirmatory run.
  - Optional threshold-sweep robustness curve (p vs T per contrast) when a manifest path
    and segmentation cache dir are also given -- free, since Stage 3 is pure numpy.

Uses a one-sided Mann-Whitney U test, not Cohen's d against an absolute gap threshold:
these IoU distributions are heavily zero-inflated (a majority of rows in every condition,
including real, are exactly 0.0 -- see RESULTS.md), so they're far from normal, and an
earlier version of this script's "gap > 0.1" absolute threshold turned out to be
miscalibrated against how small this metric's real-world scores actually are -- it called
a p=0.004 result a "FAIL" purely because 0.036 - 0.0 didn't clear an arbitrary 0.1 bar.
Mann-Whitney doesn't assume normality and isn't fooled by the absolute scale.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

REAL = "real"


def _prompt_group_column(df: pd.DataFrame) -> str:
    """Step 4's schema clusters within prompt_id (stable across seeds); the older v4
    schema has no prompt_id/seed split, so chain_id is the closest available grouping."""
    return "prompt_id" if "prompt_id" in df.columns else "chain_id"


def control_conditions(df: pd.DataFrame) -> List[str]:
    return sorted(c for c in df["condition"].unique() if c != REAL)


def pooled_check(df: pd.DataFrame) -> Dict[str, dict]:
    """One-sided Mann-Whitney U, real vs each control condition, pooled over every row --
    directly comparable to the v4 numbers in RESULTS.md. Anticonservative under multi-seed
    (rows within a prompt aren't independent draws), which is exactly why the clustered
    check below exists alongside it."""
    real = df.loc[df.condition == REAL, "iou"]
    results = {}
    for cond in control_conditions(df):
        other = df.loc[df.condition == cond, "iou"]
        if len(other) == 0:
            results[cond] = dict(p_value=float("nan"), mean=float("nan"), nonzero_rate=float("nan"),
                                  n=0, significant=False)
            continue
        _, p_value = stats.mannwhitneyu(real, other, alternative="greater")
        results[cond] = dict(p_value=p_value, mean=float(other.mean()),
                              nonzero_rate=float((other > 0).mean()), n=len(other),
                              significant=p_value < 0.05)
    return results


def clustered_check(df: pd.DataFrame) -> Dict[str, dict]:
    """Wilcoxon signed-rank on per-prompt paired differences (real_mean - control_mean),
    one pair per prompt (n=9 prompts, however many chains/seeds fed into each). This is
    the conservative test to lead with: it doesn't inflate significance from multiple
    correlated seeds/steps within the same prompt the way the pooled test can."""
    group_col = _prompt_group_column(df)
    real_by_group = df.loc[df.condition == REAL].groupby(group_col)["iou"].mean()
    results = {}
    for cond in control_conditions(df):
        other_by_group = df.loc[df.condition == cond].groupby(group_col)["iou"].mean()
        paired = real_by_group.align(other_by_group, join="inner")
        diffs = (paired[0] - paired[1]).to_numpy()
        n_groups = len(diffs)
        if n_groups < 2 or np.all(diffs == diffs[0]):
            results[cond] = dict(p_value=float("nan"), n_groups=n_groups, significant=False,
                                  note="insufficient variation across prompts for Wilcoxon")
            continue
        try:
            _, p_value = stats.wilcoxon(diffs, alternative="greater")
        except ValueError as e:
            results[cond] = dict(p_value=float("nan"), n_groups=n_groups, significant=False, note=str(e))
            continue
        results[cond] = dict(p_value=p_value, n_groups=n_groups, significant=p_value < 0.05,
                              mean_diff=float(diffs.mean()))
    return results


def threshold_sweep_curve(manifest: dict, cache_dir: Path, t_values=None) -> pd.DataFrame:
    """p-value (pooled Mann-Whitney, real vs each control) as a function of threshold T --
    free once Stage 2's cache exists, since score_chains.py is pure numpy. Robustness
    check, not a re-calibration: T is not re-selected here."""
    from score_chains import score_chains

    if t_values is None:
        t_values = tuple(round(t, 2) for t in np.arange(0.05, 0.96, 0.05))
    rows = []
    for t in t_values:
        df_t = score_chains(manifest, cache_dir, threshold=t)
        for cond, res in pooled_check(df_t).items():
            rows.append(dict(threshold=t, condition=cond, p_value=res["p_value"]))
    return pd.DataFrame(rows)


def main(csv_path: str, manifest_path: Optional[str] = None, cache_dir: Optional[str] = None):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print("\n=== Per-condition summary ===")
    summary = df.groupby("condition")["iou"].agg(["mean", "std", "median", "min", "max", "count"])
    summary["nonzero_rate"] = df.groupby("condition")["iou"].apply(lambda s: (s > 0).mean())
    print(summary.round(4))

    print("\n=== Per-condition, per-n (subject count) breakdown ===")
    print(df.groupby(["condition", "n"])["iou"].mean().unstack(0).round(4))

    if REAL not in summary.index:
        print(f"\nNo '{REAL}' condition rows -- cannot evaluate the money result.")
        return

    real = df.loc[df.condition == REAL, "iou"]
    print(f"\n=== Pooled check (Mann-Whitney, real mean={real.mean():.4f}, "
          f"nonzero_rate={(real > 0).mean():.2f}, n={len(real)}) ===")
    pooled = pooled_check(df)
    for cond, res in pooled.items():
        verdict = "SIGNIFICANT (p<0.05)" if res["significant"] else "NOT significant"
        print(f"  real > {cond}: mean={res['mean']:.4f}, nonzero_rate={res['nonzero_rate']:.2f}, "
              f"p={res['p_value']:.4f} -> {verdict}")

    group_col = _prompt_group_column(df)
    n_prompts = df[group_col].nunique()
    print(f"\n=== Clustered check (Wilcoxon signed-rank, per-{group_col}, n={n_prompts}) ===")
    clustered = clustered_check(df)
    for cond, res in clustered.items():
        if "note" in res:
            print(f"  real > {cond}: {res['note']} (n_groups={res['n_groups']})")
            continue
        verdict = "SIGNIFICANT (p<0.05)" if res["significant"] else "NOT significant"
        print(f"  real > {cond}: mean_diff={res['mean_diff']:.4f}, n_groups={res['n_groups']}, "
              f"p={res['p_value']:.4f} -> {verdict}")

    print("\n=== Overall verdict (clustered test governs when it diverges from pooled) ===")
    all_pooled_significant = all(r["significant"] for r in pooled.values())
    all_clustered_significant = all(r.get("significant", False) for r in clustered.values())
    if all_clustered_significant:
        print("REAL is significantly higher than every control condition under the clustered "
              "(conservative) test. This replicates -- on real generated images, real captured "
              "attention -- the discrimination the CR pilot's persistence check (0.83 vs 0.83) "
              "could not show.")
    elif all_pooled_significant:
        print("Pooled test is significant for every control, but the clustered per-prompt test "
              "is not for at least one. Reported honestly: the effect looks real per-row but the "
              "sample of *prompts* is small -- report this divergence, do not lead with pooled alone.")
    else:
        print("At least one control condition did not reach significance under the pooled test. "
              "Report exactly which one(s), and why (sample size, control design) -- see "
              "RESULTS.md's discussion of the first run for the kind of detail expected here.")

    if manifest_path and cache_dir:
        print("\n=== Threshold-sweep robustness curve ===")
        with open(manifest_path) as f:
            manifest = json.load(f)
        curve = threshold_sweep_curve(manifest, Path(cache_dir))
        print(curve.pivot(index="threshold", columns="condition", values="p_value").round(4))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "chain_experiment_results.csv"
    manifest_arg = sys.argv[2] if len(sys.argv) > 2 else None
    cache_dir_arg = sys.argv[3] if len(sys.argv) > 3 else None
    main(path, manifest_arg, cache_dir_arg)
