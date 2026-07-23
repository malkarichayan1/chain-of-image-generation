#!/usr/bin/env python
"""
Combines two Stage-1 manifest.json files (docs/part-b-strengthening-design.md) into one,
so a growth run's new seed can be added to Stage 4's confirmatory sample without
regenerating the original seeds on GPU. Pure dict/JSON merge -- no models, no GPU.

Used to combine the Step 4 confirmatory run's manifest (SEEDS=[42,7,1234], kernel v1) with
the growth extension's manifest (SEEDS=[2024], kernel v2) -- see generate_chains.py's
"Growth extension" docstring for why the new seed was generated separately instead of
regenerating everything.
"""
import json
import sys
from typing import List


def merge_manifests(manifests: List[dict]) -> dict:
    if not manifests:
        raise ValueError("need at least one manifest to merge")
    img_size = manifests[0]["img_size"]
    for m in manifests[1:]:
        if m["img_size"] != img_size:
            raise ValueError(f"img_size mismatch: {img_size} vs {m['img_size']}")

    seeds = sorted({s for m in manifests for s in m["seeds"]})
    chains = [c for m in manifests for c in m["chains"]]

    keys = [c["chain_key"] for c in chains]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate chain_key(s) across manifests, refusing to merge: {sorted(dupes)}")

    return {"seeds": seeds, "img_size": img_size, "chains": chains}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: merge_manifests.py OUT.json IN1.json [IN2.json ...]")
        sys.exit(1)

    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    loaded = []
    for p in in_paths:
        with open(p) as f:
            loaded.append(json.load(f))

    merged = merge_manifests(loaded)
    detected = sum(1 for c in merged["chains"] if c["detected"])
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(in_paths)} manifests -> {out_path}: "
          f"{detected}/{len(merged['chains'])} chains detected, seeds={merged['seeds']}")
