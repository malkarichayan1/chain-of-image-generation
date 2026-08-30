#!/usr/bin/env python
"""
Part C, Step 2: score_chains.py's four RNG-dependent
conditions (substituted, attn_scrambled_samechain/_crosschain/_sameattr) each draw one random
partner via a single scoring seed. The growth run's published p-values for these four
contrasts are one realization of that draw. This module re-scores across many seeds and
reports the resulting clustered-test p-value distribution per contrast, so "significant" can
be judged against seed stability rather than a single draw.

real vs shuffled is intentionally excluded: `shuffled` iterates every wrong step
deterministically (score_chains.py's shuffled loop uses no rng.choice), so it has no RNG
dependence to sweep.
"""
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from analyze_results import clustered_check
from config import Config
from score_chains import DEFAULT_THRESHOLD, score_chains

RNG_DEPENDENT_CONDITIONS = (
    "substituted",
    "attn_scrambled_samechain",
    "attn_scrambled_crosschain",
    "attn_scrambled_sameattr",
)

_ANALYSIS_DEFAULTS = Config().analysis
# Pre-registered per the design doc's Step 2: a contrast counts as "robust" only if
# significant in at least this fraction of swept seeds.
ROBUSTNESS_THRESHOLD = _ANALYSIS_DEFAULTS.rng_sweep_robustness_threshold


def rng_sweep(manifest: dict, cache_dir: Path, threshold: float = DEFAULT_THRESHOLD,
              n_seeds: int = _ANALYSIS_DEFAULTS.rng_sweep_n_seeds,
              conditions: Iterable[str] = RNG_DEPENDENT_CONDITIONS) -> pd.DataFrame:
    rows: List[dict] = []
    for seed in range(n_seeds):
        df = score_chains(manifest, cache_dir, threshold=threshold, seed=seed)
        clustered = clustered_check(df)
        for cond in conditions:
            res = clustered.get(cond)
            if res is None:
                continue
            rows.append(dict(seed=seed, condition=cond,
                              p_value=res.get("p_value", float("nan")),
                              significant=bool(res.get("significant", False))))
    return pd.DataFrame(rows)


def summarize_rng_sweep(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("condition")
    summary = grouped["p_value"].agg(
        median_p="median",
        iqr_p=lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25)),
    )
    summary["frac_significant"] = grouped["significant"].mean()
    summary["n_seeds"] = grouped.size()
    summary["robust"] = summary["frac_significant"] >= ROBUSTNESS_THRESHOLD
    return summary


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Part C Step 2: RNG sweep over score_chains.py's stochastic conditions")
    ap.add_argument("--manifest", type=str, default="artifacts/manifest_combined.json")
    ap.add_argument("--cache-dir", type=str, default="artifacts/segmentation_cache")
    ap.add_argument("--n-seeds", type=int, default=_ANALYSIS_DEFAULTS.rng_sweep_n_seeds)
    args = ap.parse_args()

    manifest_path = args.manifest
    cache_dir_arg = args.cache_dir
    n_seeds_arg = args.n_seeds

    with open(manifest_path) as f:
        manifest_data = json.load(f)
    sweep_df = rng_sweep(manifest_data, Path(cache_dir_arg), n_seeds=n_seeds_arg)
    summary_df = summarize_rng_sweep(sweep_df)
    print(f"RNG sweep over {n_seeds_arg} seeds ({len(sweep_df)} (seed, condition) rows):")
    print(summary_df.round(4))
