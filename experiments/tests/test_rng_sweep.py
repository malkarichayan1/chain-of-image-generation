"""
Tests for rng_sweep.py (Part C, Step 2): re-scores across many
RNG seeds and summarizes each RNG-dependent control condition's clustered-test p-value
distribution. score_chains.py's substituted/attn_scrambled_* conditions each draw one random
partner via a single scoring seed -- the growth run's published p-values for these four
contrasts are one realization of that draw, untested for seed stability until this.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rng_sweep import RNG_DEPENDENT_CONDITIONS, rng_sweep, summarize_rng_sweep
from segment_cache import save_cached_map


def _three_prompt_manifest(tmp_path: Path):
    """3 prompts x 2 seeds, one step each, distinct attributes per prompt -- enough
    structure for substituted/attn_scrambled_crosschain/_sameattr to each find a valid
    partner (attn_scrambled_samechain has none, since each chain has only one step;
    the sweep must handle a condition being partially or fully absent gracefully)."""
    cache_dir = tmp_path / "cache"
    img_dir = tmp_path / "images"
    attn_dir = tmp_path / "attn"

    def img(name: str) -> str:
        return str(img_dir / name)

    def attn(name: str) -> str:
        peaked = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        p = attn_dir / f"{name}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, peaked)
        return str(p)

    def seg(image_path: str, attribute: str, present: bool):
        arr = np.ones((4, 4), dtype=np.float32) if present else np.zeros((4, 4), dtype=np.float32)
        save_cached_map(cache_dir, image_path, attribute, arr)

    attrs = ["red apron", "yellow helmet", "blue gloves"]
    chains = []
    for prompt_id in range(3):
        attribute = attrs[prompt_id]
        for seed in (0, 1):
            base = img(f"p{prompt_id}_s{seed}_step0.png")
            s1 = img(f"p{prompt_id}_s{seed}_step1.png")
            seg(base, attribute, present=False)
            seg(s1, attribute, present=True)
            for other in attrs:
                if other != attribute:
                    seg(base, other, present=False)
                    seg(s1, other, present=False)
            chains.append({
                "chain_key": f"p{prompt_id}_s{seed}", "prompt_id": prompt_id, "seed": seed,
                "n": 1, "prompt": f"a photo with {attribute}", "detected": True,
                "base_image_path": base, "subject_boxes": {},
                "steps": [{"step_idx": 1, "subject": "person", "attribute": attribute,
                           "image_path": s1, "attention_path": attn(f"p{prompt_id}_s{seed}_step1")}],
            })
    return {"chains": chains}, cache_dir


def test_rng_sweep_returns_one_row_per_seed_per_available_condition(tmp_path: Path):
    manifest, cache_dir = _three_prompt_manifest(tmp_path)
    df = rng_sweep(manifest, cache_dir, threshold=0.5, n_seeds=4)
    assert set(df["condition"].unique()) <= set(RNG_DEPENDENT_CONDITIONS)
    assert set(df["seed"].unique()) == {0, 1, 2, 3}


def test_real_and_shuffled_are_excluded_from_the_sweep(tmp_path: Path):
    manifest, cache_dir = _three_prompt_manifest(tmp_path)
    df = rng_sweep(manifest, cache_dir, threshold=0.5, n_seeds=3)
    assert "real" not in df["condition"].unique()
    assert "shuffled" not in df["condition"].unique()


def test_summarize_rng_sweep_reports_median_iqr_and_significant_fraction(tmp_path: Path):
    manifest, cache_dir = _three_prompt_manifest(tmp_path)
    df = rng_sweep(manifest, cache_dir, threshold=0.5, n_seeds=10)
    summary = summarize_rng_sweep(df)
    assert {"median_p", "iqr_p", "frac_significant", "n_seeds", "robust"} <= set(summary.columns)
    assert (summary["frac_significant"] >= 0).all() and (summary["frac_significant"] <= 1).all()


def test_summarize_rng_sweep_flags_seed_fragile_vs_robust_contrasts():
    df = pd.DataFrame(
        [dict(seed=s, condition="fragile", p_value=0.01 if s % 2 == 0 else 0.5,
              significant=(s % 2 == 0)) for s in range(10)]
        + [dict(seed=s, condition="robust", p_value=0.001, significant=True) for s in range(10)]
    )
    summary = summarize_rng_sweep(df)
    assert summary.loc["fragile", "frac_significant"] == pytest.approx(0.5)
    assert summary.loc["robust", "frac_significant"] == pytest.approx(1.0)
    # Decision rule (design doc Step 2): robust only at >=95% of seeds significant.
    assert bool(summary.loc["fragile", "robust"]) is False
    assert bool(summary.loc["robust", "robust"]) is True
