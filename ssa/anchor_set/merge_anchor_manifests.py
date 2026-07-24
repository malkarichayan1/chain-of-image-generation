#!/usr/bin/env python
"""
Combines an anchor-set manifest.json with a later growth run's manifest (new prompt_ids
only) so the growth batch can be added without regenerating the original prompts on GPU.
Pure dict/JSON merge -- no models, no GPU. Adapted from
pi_level_experiment/merge_manifests.py's approach (chain_key -> prompt_id).

Used to combine artifacts_sdxl/manifest.json (ids 0-17, kernel v1) with the 2026-07-24
growth run's manifest (ids 18-23, GROWTH_PROMPT_IDS-scoped kernel push) -- see
generate_anchor_images_sdxl.py's GROWTH_PROMPT_IDS docstring for why the new prompts were
generated separately instead of regenerating everything.

    py -3 merge_anchor_manifests.py OUT.json IN1.json [IN2.json ...]
"""
import json
import sys
from typing import List


def merge_anchor_manifests(manifests: List[dict]) -> dict:
    if not manifests:
        raise ValueError("need at least one manifest to merge")
    base = manifests[0]
    for m in manifests[1:]:
        if m["img_size"] != base["img_size"]:
            raise ValueError(f"img_size mismatch: {base['img_size']} vs {m['img_size']}")

    images = [img for m in manifests for img in m["images"]]
    ids = [img["prompt_id"] for img in images]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate prompt_id(s) across manifests, refusing to merge: {sorted(dupes)}")

    seeds = sorted({s for m in manifests for s in m["candidate_seeds"]})
    return {"model": base["model"], "candidate_seeds": seeds, "img_size": base["img_size"],
            "num_inference_steps": base["num_inference_steps"],
            "early_window_fraction": base["early_window_fraction"], "images": images}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: merge_anchor_manifests.py OUT.json IN1.json [IN2.json ...]")
        sys.exit(1)

    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    loaded = []
    for p in in_paths:
        with open(p) as f:
            loaded.append(json.load(f))

    merged = merge_anchor_manifests(loaded)
    detected = sum(1 for img in merged["images"] if img["detected"])
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(in_paths)} manifests -> {out_path}: "
          f"{detected}/{len(merged['images'])} images detected")
