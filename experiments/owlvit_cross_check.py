#!/usr/bin/env python
"""
Part D, check 4: does an independent detector (OWL-ViT, zero-shot open-vocabulary object
detection) corroborate CLIPSeg's "never detected in curr at all" call for the 52 real
zero-IoU rows (zero_inflation_recheck.py), or does it find the attribute where CLIPSeg
missed it -- which would point at CLIPSeg specifically being the weak link, rather than the
attribute genuinely never rendering?

Runs entirely on CPU locally (no Kaggle/GPU needed -- OWL-ViT is ~0.6s/inference on this
machine, so a full pass over all 74 real-condition rows' curr images is a couple of
minutes, not a sample). Scores are cached in `artifacts/owlvit_scores.json` keyed the same
way as segment_cache.py's `cache_key`, so re-runs after the first are instant.
"""
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from PIL import Image

from config import Config
from score_chains import chain_image_path, detected_chains, parse_manifest
from segment_cache import cache_key

SCORE_CACHE_PATH = Path("artifacts/owlvit_scores.json")


def _load_score_cache(path: Path = SCORE_CACHE_PATH) -> Dict[str, float]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_score_cache(cache: Dict[str, float], path: Path = SCORE_CACHE_PATH) -> None:
    path.write_text(json.dumps(cache))


def _owlvit_detect_fn():
    """Lazily loads OWL-ViT once and returns a closure (image, attribute) -> max detection
    confidence for that text query in that image. Mirrors segment_cache.py's
    `_default_segment_fn` pattern (lazy import, untested here -- the cache/comparison logic
    below is the tested surface, same split as that module)."""
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    owlvit_model = Config().models.owlvit_model
    processor = OwlViTProcessor.from_pretrained(owlvit_model)
    model = OwlViTForObjectDetection.from_pretrained(owlvit_model)
    model.eval()

    def _detect(image: Image.Image, attribute: str) -> float:
        import torch

        inputs = processor(text=[[attribute]], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = [(image.size[1], image.size[0])]
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )[0]
        scores = results["scores"]
        return float(scores.max().item()) if len(scores) else 0.0

    return _detect


def collect_owlvit_scores(pairs: list, image_dir: Path = Path(".")) -> Dict[str, float]:
    """pairs: list of (image_path, attribute). Returns {cache_key: score}, using and
    updating the on-disk cache so repeat runs skip already-scored pairs."""
    cache = _load_score_cache()
    detect = None
    for image_path, attribute in pairs:
        key = cache_key(image_path, attribute)
        if key in cache:
            continue
        if detect is None:
            detect = _owlvit_detect_fn()
        image = Image.open(image_path).convert("RGB")
        cache[key] = detect(image, attribute)
        _save_score_cache(cache)
    return cache


def build_comparison_table(
    df: pd.DataFrame, manifest: dict, owlvit_scores: Dict[str, float], condition: str = "real",
) -> pd.DataFrame:
    """Joins each row's OWL-ViT curr-image confidence next to CLIPSeg's own call
    (delta_area == 0 and iou == 0, the "never detected in curr" set per
    zero_inflation_recheck.py) so the two detectors' agreement can be read off directly."""
    chains_by_key = {c.chain_key: c for c in detected_chains(parse_manifest(manifest))}
    sub = df[df["condition"] == condition].copy()

    def curr_path_for(row) -> str:
        chain = chains_by_key[row.chain_key]
        return chain_image_path(chain, int(row.true_step))

    rows = []
    for row in sub.itertuples():
        curr_path = curr_path_for(row)
        key = cache_key(curr_path, row.attribute)
        rows.append(dict(
            chain_key=row.chain_key, attribute=row.attribute, true_step=row.true_step,
            clipseg_never_detected_in_curr=(row.iou == 0 and row.delta_area == 0),
            owlvit_max_confidence=owlvit_scores.get(key),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Part D4: does OWL-ViT corroborate CLIPSeg's misses?")
    ap.add_argument("--manifest", type=str, default="artifacts/manifest_combined.json")
    ap.add_argument("--results-csv", type=str, default="artifacts/chain_experiment_results_v7.csv")
    ap.add_argument("--out-csv", type=str, default="artifacts/owlvit_cross_check.csv")
    ap.add_argument("--condition", type=str, default="real")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    df = pd.read_csv(args.results_csv)
    chains_by_key = {c.chain_key: c for c in detected_chains(parse_manifest(manifest))}

    real = df[df["condition"] == args.condition]
    pairs = [
        (chain_image_path(chains_by_key[r.chain_key], int(r.true_step)), r.attribute)
        for r in real.itertuples()
    ]
    print(f"Scoring {len(set(pairs))} unique (image, attribute) pairs with OWL-ViT...")
    scores = collect_owlvit_scores(pairs)

    table = build_comparison_table(df, manifest, scores, condition=args.condition)
    print(table.groupby("clipseg_never_detected_in_curr")["owlvit_max_confidence"]
          .agg(["count", "mean", "median", "max"]).round(4))
    table.to_csv(args.out_csv, index=False)
