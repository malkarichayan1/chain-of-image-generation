#!/usr/bin/env python
"""
Part D, check 2: is `iou` just tracking subject bounding-box size, rather than anything
attention-specific? Part C Step 6 already showed `delta_area` alone reproduces two of the
five contrasts -- one step short of "the metric measures how much new pixel area appeared."
This asks the adjacent question with a variable Step 6 never touched: the *subject's own*
bounding-box area (from Mask R-CNN, cached in the manifest), a classic nuisance regressor
for attention-map studies (do big objects trivially win cross-attention regardless of
whether the attribute is correctly bound to them?).

Pure pandas/numpy against already-generated artifacts (manifest + scored CSV); no GPU,
no new model calls. Partial correlation is computed by hand (residualize both variables
against delta_area via OLS, correlate the residuals) rather than pulling in statsmodels,
matching this project's existing no-new-dependencies convention.
"""
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bbox_area_lookup(manifest: dict, img_size: int) -> Dict[Tuple[str, str], float]:
    """(chain_key, subject) -> bbox area as a fraction of the full image area, so it's
    on the same [0, 1]-ish scale as `delta_area`/`curr_mask_area` in the scored CSV."""
    img_area = float(img_size) ** 2
    out = {}
    for c in manifest["chains"]:
        for subject, box in c.get("subject_boxes", {}).items():
            x1, y1, x2, y2 = box
            out[(c["chain_key"], subject)] = ((x2 - x1) * (y2 - y1)) / img_area
    return out


def join_bbox_area(df: pd.DataFrame, lookup: Dict[Tuple[str, str], float]) -> pd.DataFrame:
    out = df.copy()
    out["bbox_area_frac"] = [
        lookup.get((row.chain_key, row.subject)) for row in out.itertuples()
    ]
    return out


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y residuals after regressing out x via simple OLS (with intercept)."""
    design = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coeffs


def confound_report(df: pd.DataFrame, condition: str = "real") -> dict:
    """Correlates `iou` against `bbox_area_frac` for rows of `condition`, both directly
    and after partialling out `delta_area` -- the direct correlation conflates "bigger
    subject" with "bigger delta mask" (subjects with more pixels tend to have more pixels
    change too); the partial correlation isolates whether bbox size adds anything on top
    of what delta_area (already validated in Part C Step 6) explains."""
    sub = df[df["condition"] == condition].dropna(subset=["bbox_area_frac"])
    iou = sub["iou"].to_numpy()
    bbox = sub["bbox_area_frac"].to_numpy()
    delta = sub["delta_area"].to_numpy()

    pearson_r, pearson_p = stats.pearsonr(iou, bbox)
    spearman_r, spearman_p = stats.spearmanr(iou, bbox)

    iou_resid = _residualize(iou, delta)
    bbox_resid = _residualize(bbox, delta)
    partial_r, partial_p = stats.pearsonr(iou_resid, bbox_resid)

    return dict(
        n=len(sub),
        pearson_r=float(pearson_r), pearson_p=float(pearson_p),
        spearman_r=float(spearman_r), spearman_p=float(spearman_p),
        partial_r_controlling_delta_area=float(partial_r),
        partial_p_controlling_delta_area=float(partial_p),
    )


if __name__ == "__main__":
    manifest = json.loads(Path("artifacts/manifest_combined.json").read_text())
    lookup = bbox_area_lookup(manifest, manifest["img_size"])

    df = pd.read_csv("artifacts/chain_experiment_results_v7.csv")
    df = join_bbox_area(df, lookup)
    missing = df["bbox_area_frac"].isna().sum()
    print(f"rows missing a bbox match: {missing}/{len(df)}")

    for condition in ["real", "attn_scrambled_samechain", "attn_scrambled_crosschain", "attn_scrambled_sameattr"]:
        report = confound_report(df, condition=condition)
        print(f"\n=== {condition} ===")
        for k, v in report.items():
            print(f"  {k}: {v}")
