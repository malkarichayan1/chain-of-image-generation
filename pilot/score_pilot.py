#!/usr/bin/env python3
"""Aggregate appears-at-step / persists-to-final rates by condition and
report statistical tests (means, std, paired t-tests, Wilcoxon signed-rank)
for the CoIG Causal Relevance audit.
"""

import argparse
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Causal Relevance results with paired statistical tests")
    parser.add_argument("--results_csv", type=str, default="causal_relevance_results.csv")
    return parser.parse_args()


def run_paired_tests(item_df: pd.DataFrame, metric: str, cond1: str, cond2: str):
    """Run paired t-test and Wilcoxon signed-rank test between cond1 and cond2 for metric."""
    col1 = (metric, cond1)
    col2 = (metric, cond2)
    
    valid = item_df[[col1, col2]].dropna()
    x = valid[col1]
    y = valid[col2]
    
    n = len(valid)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "t_stat": float("nan"), "t_p": float("nan"), "w_stat": float("nan"), "w_p": float("nan")}
    
    mean_diff = (x - y).mean()
    
    # Paired t-test
    if (x - y).std() == 0:
        t_stat, t_p = 0.0, 1.0
    else:
        t_stat, t_p = stats.ttest_rel(x, y)
        
    # Wilcoxon signed-rank test
    diff = x - y
    if (diff == 0).all():
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            w_stat, w_p = stats.wilcoxon(x, y)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
            
    return {
        "n": n,
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "t_p": t_p,
        "w_stat": w_stat,
        "w_p": w_p,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    df = df.dropna(subset=["appears_at_step", "persists_to_final"])

    # Overall Summary Across Claims
    print("=== OVERALL SUMMARY (ALL CLAIMS) ===")
    summary = df.groupby("condition")[["appears_at_step", "persists_to_final"]].agg(["mean", "std", "count"])
    print(summary)
    print("\n")

    # Item-Level Aggregation for Paired Tests
    item_df = df.groupby(["item_index", "condition"])[["appears_at_step", "persists_to_final"]].mean().unstack(level="condition")

    metrics = ["appears_at_step", "persists_to_final"]
    pairs = [("real", "shuffled"), ("real", "substituted"), ("shuffled", "substituted")]

    print("=== PAIRED STATISTICAL TESTS (ITEM LEVEL AGGREGATION) ===")
    test_results = []
    for metric in metrics:
        for cond1, cond2 in pairs:
            if (metric, cond1) in item_df.columns and (metric, cond2) in item_df.columns:
                res = run_paired_tests(item_df, metric, cond1, cond2)
                res["metric"] = metric
                res["comparison"] = f"{cond1} vs {cond2}"
                test_results.append(res)

    res_df = pd.DataFrame(test_results)
    if not res_df.empty:
        cols = ["metric", "comparison", "n", "mean_diff", "t_stat", "t_p", "w_stat", "w_p"]
        print(res_df[cols].to_string(index=False))
    print("\n")

    def persist_mean(condition: str) -> float:
        return df.loc[df["condition"] == condition, "persists_to_final"].mean()

    real_persist = persist_mean("real") if "real" in df["condition"].values else 0
    shuffled_persist = persist_mean("shuffled") if "shuffled" in df["condition"].values else 0
    substituted_persist = persist_mean("substituted") if "substituted" in df["condition"].values else 0

    print("Go/no-go read:")
    print(
        f"  persists_to_final -- real={real_persist:.2f} "
        f"shuffled={shuffled_persist:.2f} substituted={substituted_persist:.2f}"
    )

    if shuffled_persist > 0.5 and substituted_persist < 0.3:
        print(
            "  Shuffled persistence stays high while Substituted stays low -> "
            "the lock confound is visible. Continue to the full study."
        )
    elif (real_persist - substituted_persist) > 0.5:
        print(
            "  Real clearly beats Substituted -> Causal Relevance tracks real "
            "semantics. Consider pivoting to Track 2."
        )
    else:
        print("  Mixed pattern -- inspect the per-chain results before deciding.")


if __name__ == "__main__":
    main()

