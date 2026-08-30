#!/usr/bin/env python
"""
Step 3 of the pipeline: pre-registered threshold calibration. Sweeps score_chains.py's threshold T over the
calibration set ONLY (the four prompts skipped in the v4 run: 3, 4, 6, 8) and selects T
by a criterion that never references Real-vs-Shuffled separation:

    maximize Real nonzero rate, subject to Substituted nonzero rate remaining exactly 0.

This leans only on Phase A's structural guarantee (a never-rendered attribute must
produce an empty delta), not on the effect the calibrated threshold will later be used to
measure -- a threshold chosen this way cannot have been tuned toward the p-value it
computes. Prompts 0, 1, 2, 5, 7 are held out and unopened by this module; it refuses to
run if a manifest contains no calibration-set chains, rather than silently falling back to
scoring held-out data.

This script only *finds* the constant. Freezing it into score_chains.py's
DEFAULT_THRESHOLD (with the sweep range, selected value, criterion, and calibration chains
recorded in a comment) is a manual step, committed separately from Step 4's confirmatory
scoring run -- see the design doc's Step 3 acceptance criteria for why that ordering
matters.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from score_chains import CALIBRATION_PROMPT_IDS, HELD_OUT_PROMPT_IDS, score_chains

_sweep_cfg = Config().scoring
DEFAULT_SWEEP = tuple(round(t, 2) for t in np.arange(
    _sweep_cfg.calibration_sweep_start, _sweep_cfg.calibration_sweep_stop, _sweep_cfg.calibration_sweep_step
))


def restrict_to_calibration(manifest: dict) -> dict:
    """The refusal guard: raises if the manifest has no calibration-set chains, rather
    than silently calibrating against held-out prompts 0/1/2/5/7."""
    chains = [c for c in manifest["chains"] if c["prompt_id"] in CALIBRATION_PROMPT_IDS]
    if not chains:
        raise ValueError(
            f"no calibration-set chains found (prompt_id in {sorted(CALIBRATION_PROMPT_IDS)}) "
            "-- refusing to calibrate on held-out data"
        )
    leaked = [c["prompt_id"] for c in chains if c["prompt_id"] in HELD_OUT_PROMPT_IDS]
    assert not leaked, f"calibration and held-out prompt sets must be disjoint, got {leaked} in both"
    return {"chains": chains}


def sweep_threshold(manifest: dict, cache_dir: Path, t_values=DEFAULT_SWEEP, seed: int = 0) -> pd.DataFrame:
    calibration_manifest = restrict_to_calibration(manifest)
    rows = []
    for t in t_values:
        df = score_chains(calibration_manifest, cache_dir, threshold=t, seed=seed)
        real = df[df.condition == "real"]["iou"]
        substituted = df[df.condition == "substituted"]["iou"]
        rows.append(dict(
            threshold=t,
            real_nonzero_rate=float((real > 0).mean()) if len(real) else float("nan"),
            substituted_nonzero_rate=float((substituted > 0).mean()) if len(substituted) else float("nan"),
            n_real=len(real), n_substituted=len(substituted),
        ))
    return pd.DataFrame(rows)


def select_calibrated_threshold(sweep: pd.DataFrame) -> Optional[float]:
    """maximize real_nonzero_rate subject to substituted_nonzero_rate == 0 exactly. Ties
    broken toward the smallest T, for a deterministic, reproducible choice. Returns None
    -- an informative negative, not a forced pick -- if no T satisfies the constraint."""
    valid = sweep[sweep["substituted_nonzero_rate"] == 0.0]
    if valid.empty:
        return None
    best_rate = valid["real_nonzero_rate"].max()
    candidates = valid[valid["real_nonzero_rate"] == best_rate]
    return float(candidates["threshold"].min())


def calibrate(manifest: dict, cache_dir: Path, t_values=DEFAULT_SWEEP, seed: int = 0) -> dict:
    sweep = sweep_threshold(manifest, cache_dir, t_values=t_values, seed=seed)
    selected = select_calibrated_threshold(sweep)
    return dict(
        selected_threshold=selected,
        sweep=sweep.to_dict(orient="records"),
        calibration_prompt_ids=sorted(CALIBRATION_PROMPT_IDS),
        criterion="maximize real_nonzero_rate subject to substituted_nonzero_rate == 0",
    )


if __name__ == "__main__":
    import argparse

    from config import DEFAULT_CONFIG_PATH

    ap = argparse.ArgumentParser(description="Step 3: pre-registered threshold calibration")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    manifest_path = args.manifest or str(cfg.paths.manifest_path)
    cache_dir_arg = args.cache_dir or str(cfg.paths.segmentation_cache_dir)

    with open(manifest_path) as f:
        manifest_data = json.load(f)
    result = calibrate(manifest_data, Path(cache_dir_arg))

    print(pd.DataFrame(result["sweep"]).round(4))
    if result["selected_threshold"] is None:
        print("\nNo threshold satisfies the calibration criterion (substituted never stays "
              "at 0 while real's hit rate improves). This is an informative negative -- see "
              "the design doc's Risks section. Do not relax the criterion to rescue this step.")
    else:
        print(f"\nSelected threshold: {result['selected_threshold']}")
        print("Freeze this into score_chains.py's DEFAULT_THRESHOLD, with this sweep, "
              "criterion, and calibration_prompt_ids recorded in a comment, as its own commit.")
