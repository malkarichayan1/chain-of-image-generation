"""
Tests for analyze_results.py's Part C additions (docs/part-c-validation-design.md):
Step 3 (Holm-Bonferroni correction across the 5 control contrasts) and Step 4
(leave-one-prompt-out sensitivity, reporting the worst-case p-value per contrast).
"""
import numpy as np
import pandas as pd
import pytest

from analyze_results import (
    clustered_check,
    holm_correction,
    joint_threshold_topk_sweep,
    leave_one_out_check,
)
from segment_cache import save_cached_map


def test_holm_correction_matches_hand_computed_worked_example():
    # p=[0.01,0.02,0.03], m=3: adjusted = max-running of (m-i)*p sorted ascending
    # -> [3*0.01, max(0.03,2*0.02), max(0.04,1*0.03)] = [0.03, 0.04, 0.04].
    p_values = {"a": 0.01, "b": 0.02, "c": 0.03}
    result = holm_correction(p_values)
    assert result["a"]["adjusted_p"] == pytest.approx(0.03)
    assert result["b"]["adjusted_p"] == pytest.approx(0.04)
    assert result["c"]["adjusted_p"] == pytest.approx(0.04)
    assert all(r["significant"] for r in result.values())


def test_holm_correction_is_monotonic_and_never_below_raw_p():
    p_values = {
        "shuffled": 0.0039, "substituted": 0.0039, "attn_scrambled_samechain": 0.0039,
        "attn_scrambled_crosschain": 0.0195, "attn_scrambled_sameattr": 0.0156,
    }
    result = holm_correction(p_values)
    assert all(result[cond]["adjusted_p"] >= result[cond]["p_value"] for cond in p_values)
    sorted_by_raw = sorted(result.items(), key=lambda kv: kv[1]["p_value"])
    adjusted_seq = [r["adjusted_p"] for _, r in sorted_by_raw]
    assert all(adjusted_seq[i] <= adjusted_seq[i + 1] for i in range(len(adjusted_seq) - 1))


def test_holm_correction_growth_run_pvalues_all_survive():
    """Recomputation of already-published data (RESULTS.md's growth-run clustered
    p-values), not a rescoring -- confirms the "beats every control" headline still
    holds after correcting for testing 5 contrasts against the same real sample."""
    p_values = {
        "shuffled": 0.0039, "substituted": 0.0039, "attn_scrambled_samechain": 0.0039,
        "attn_scrambled_crosschain": 0.0195, "attn_scrambled_sameattr": 0.0156,
    }
    result = holm_correction(p_values)
    assert all(r["significant"] for r in result.values())


def _synthetic_prompt_df(n_prompts: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for prompt_id in range(n_prompts):
        real_val = rng.uniform(0.3, 0.6)
        ctrl_val = rng.uniform(0.0, 0.3)
        for _ in range(3):
            rows.append(dict(prompt_id=prompt_id, condition="real",
                              iou=real_val + rng.normal(0, 0.01)))
            rows.append(dict(prompt_id=prompt_id, condition="ctrl",
                              iou=ctrl_val + rng.normal(0, 0.01)))
    return pd.DataFrame(rows)


def test_leave_one_out_check_matches_manually_computed_worst_case():
    df = _synthetic_prompt_df(n_prompts=6, seed=0)
    result = leave_one_out_check(df)

    manual_ps = []
    for prompt in df["prompt_id"].unique():
        subset = df[df["prompt_id"] != prompt]
        manual_ps.append(clustered_check(subset)["ctrl"]["p_value"])

    assert result["ctrl"]["p_value"] == pytest.approx(max(manual_ps))
    assert result["ctrl"]["dropped_prompt"] in set(df["prompt_id"].unique())


def test_leave_one_out_check_significant_flag_matches_worst_case_p():
    df = _synthetic_prompt_df(n_prompts=6, seed=1)
    result = leave_one_out_check(df)
    for cond, res in result.items():
        if np.isnan(res["p_value"]):
            continue
        assert res["significant"] == (res["p_value"] < 0.05)


def _two_prompt_manifest_for_sweep(tmp_path):
    """Minimal manifest with 2 prompts x 2 steps, enough for shuffled (a same-chain
    wrong step exists) and substituted (a disjoint foreign attribute exists) to score
    at every (T, threshold_pct) grid point -- exercises joint_threshold_topk_sweep's
    plumbing, not a statistical claim."""
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

    attrs = ["red apron", "yellow helmet"]
    chains = []
    for prompt_id, attribute in enumerate(attrs):
        base = img(f"p{prompt_id}_step0.png")
        s1, s2 = img(f"p{prompt_id}_step1.png"), img(f"p{prompt_id}_step2.png")
        for image_path, present in ((base, False), (s1, True), (s2, True)):
            seg(image_path, attribute, present)
            other = attrs[1 - prompt_id]
            seg(image_path, other, present=False)
        chains.append({
            "chain_key": f"p{prompt_id}_s42", "prompt_id": prompt_id, "seed": 42, "n": 1,
            "prompt": f"a photo with {attribute}", "detected": True,
            "base_image_path": base, "subject_boxes": {},
            "steps": [
                {"step_idx": 1, "subject": "person", "attribute": attribute,
                 "image_path": s1, "attention_path": attn(f"p{prompt_id}_step1")},
                {"step_idx": 2, "subject": "person", "attribute": attribute,
                 "image_path": s2, "attention_path": attn(f"p{prompt_id}_step2")},
            ],
        })
    return {"chains": chains}, cache_dir


def test_joint_threshold_topk_sweep_covers_full_grid_for_shuffled_and_substituted(tmp_path):
    manifest, cache_dir = _two_prompt_manifest_for_sweep(tmp_path)
    t_values = (0.3, 0.5, 0.7)
    topk_values = (0.1, 0.2)
    curve = joint_threshold_topk_sweep(manifest, cache_dir, t_values=t_values, topk_values=topk_values)
    assert set(curve["condition"].unique()) <= {"shuffled", "substituted"}
    assert {"threshold", "threshold_pct", "condition", "pooled_p", "clustered_p"} <= set(curve.columns)
    n_grid_points = len(t_values) * len(topk_values)
    assert len(curve[curve.condition == "shuffled"]) == n_grid_points
    assert len(curve[curve.condition == "substituted"]) == n_grid_points
