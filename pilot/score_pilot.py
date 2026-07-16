#!/usr/bin/env python3
"""Aggregate appears-at-step / persists-to-final rates by condition and
report the go/no-go signal for the CoIG Causal Relevance audit.

n=10 chains per condition -- this reads effect sizes (the gap between
conditions), not p-values. See docs/pilot-design.md for the go/no-go
criteria this echoes.
"""

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pilot Causal Relevance results")
    parser.add_argument("--results_csv", type=str, default="causal_relevance_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    df = df.dropna(subset=["appears_at_step", "persists_to_final"])

    summary = df.groupby("condition")[["appears_at_step", "persists_to_final"]].agg(["mean", "std", "count"])
    print(summary)

    def persist_mean(condition: str) -> float:
        return df.loc[df["condition"] == condition, "persists_to_final"].mean()

    real_persist = persist_mean("real")
    shuffled_persist = persist_mean("shuffled")
    substituted_persist = persist_mean("substituted")

    print("\nGo/no-go read:")
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
