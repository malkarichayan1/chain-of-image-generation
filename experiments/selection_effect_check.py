#!/usr/bin/env python
"""
Part D, check 1: does the person-detection filter (Mask R-CNN finding fewer people than
a prompt's subject count) drop chains at random, or systematically by subject count/prompt?

10 of 36 attempted chains were never scored because detection failed (RESULTS.md). A
non-random filter -- e.g. one that disproportionately drops harder (higher-n) prompts --
would mean the surviving 26/36 sample is not representative of the full attribute space the
paper claims to cover. Pure pandas against `artifacts/manifest_combined.json`; no GPU.
"""
import json
from pathlib import Path
from typing import List, TypedDict

import pandas as pd


class ChainSummary(TypedDict):
    chain_key: str
    prompt_id: int
    seed: int
    n: int
    detected: bool
    num_people_detected: int


def summarize_chains(manifest: dict) -> List[ChainSummary]:
    return [
        ChainSummary(
            chain_key=c["chain_key"], prompt_id=c["prompt_id"], seed=c["seed"], n=c["n"],
            detected=c["detected"], num_people_detected=c["num_people_detected"],
        )
        for c in manifest["chains"]
    ]


def detection_rate_by_prompt(chains: List[ChainSummary]) -> pd.DataFrame:
    df = pd.DataFrame(chains)
    out = df.groupby("prompt_id").agg(
        n_subjects=("n", "first"), attempted=("detected", "size"), detected=("detected", "sum"),
    )
    out["rate"] = out["detected"] / out["attempted"]
    return out.reset_index()


def detection_rate_by_subject_count(chains: List[ChainSummary]) -> pd.DataFrame:
    df = pd.DataFrame(chains)
    out = df.groupby("n").agg(attempted=("detected", "size"), detected=("detected", "sum"))
    out["rate"] = out["detected"] / out["attempted"]
    return out.reset_index()


def undercount_failure_share(chains: List[ChainSummary]) -> float:
    """Share of undetected chains whose failure is explained by Mask R-CNN finding
    strictly fewer people than the prompt required (the disclosed failure mode) --
    as opposed to some other, uncharacterized cause."""
    undetected = [c for c in chains if not c["detected"]]
    if not undetected:
        return 1.0
    explained = sum(1 for c in undetected if c["num_people_detected"] < c["n"])
    return explained / len(undetected)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Part D1: is the person-detection filter systematic?")
    ap.add_argument("--manifest", type=str, default="artifacts/manifest_combined.json")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    chains = summarize_chains(manifest)

    print("=== Detection rate by prompt_id ===")
    print(detection_rate_by_prompt(chains).to_string(index=False))
    print()
    print("=== Detection rate by subject count (n) ===")
    print(detection_rate_by_subject_count(chains).to_string(index=False))
    print()
    print(f"Undercount-explained share of failures: {undercount_failure_share(chains):.1%}")
