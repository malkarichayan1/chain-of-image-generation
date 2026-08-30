#!/usr/bin/env python
"""
Part A, Experiment 3b: the sharper falsification control briefing §5.4 flags as missing.

exp3_attention_scramble.py scrambles CROSS-item -- it borrows a DIFFERENT image's model_scores,
with the VALUES shuffled, so the donor's own attribute identity is destroyed twice over (wrong
image, then re-permuted). That conflates two different questions: "does attention distinguish
subjects at all" and "does THIS attribute's specific attention map matter, versus any attention
map from THIS SAME image." This script isolates the second, harder question: within one image,
swap which attribute's own captured model_scores feeds each attribute's ownership call -- a
derangement of the image's n attribute slots, so every attribute's prediction is remade from a
DIFFERENT attribute's real (not reshuffled) map, but never its own. If accuracy survives this,
the metric doesn't need the SPECIFIC attribute-token's map to land on the right subject -- some
other per-image regularity (box size, position, generic salience) is carrying it. If accuracy
collapses toward chance, attribute-specific attention content is doing real work.

Restricted to n >= 3: a derangement at n=2 has exactly one possibility (the forced swap of
the two attributes' maps), which makes permuted accuracy 1 - real_accuracy by arithmetic, not
by any property of attention -- the same degenerate case already rejected for Experiment 3's
within-item scrambling. n=2 images are skipped entirely here, not partially scored.

Pure re-analysis of already-captured `model_scores` -- no GPU, no re-capture, no new labels.

Run as a many-seed sweep (one derangement draw is a lucky/unlucky-draw risk, same reasoning as
Experiment 3) plus one paired McNemar (real vs. permuted correctness) at a fixed seed.

Run from inside ssa/anchor_set/:
    py -3 exp3b_within_item_permutation.py --artifacts-dir artifacts_flux --annotator annotator1
    py -3 exp3b_within_item_permutation.py --artifacts-dir artifacts_flux_hard --annotator consensus
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from scipy import stats

from anchor_common import NON_SUBJECT_LABELS, chance_baseline, label_key, load_labels
from config import Config

_SEED_DEFAULTS = Config().seeds
MCNEMAR_SEED = _SEED_DEFAULTS.mcnemar_seed  # matches exp3_attention_scramble.py's convention
SWEEP_SEEDS = range(_SEED_DEFAULTS.sweep_n_seeds)
MIN_N = 3


def random_derangement(n: int, rng: random.Random) -> List[int]:
    """A permutation of range(n) with no fixed point, via rejection sampling. Requires n >= 2
    (no derangement exists for n < 2). Acceptance probability approaches 1/e as n grows, so
    this stays fast at the anchor set's n = 3..6 range -- no need for a constructive algorithm."""
    if n < 2:
        raise ValueError(f"no derangement exists for n={n}")
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def permuted_predict(model_scores: Dict[str, float]) -> str:
    """Argmax of a borrowed (not this attribute's own) model_scores dict -- same rule the
    metric itself uses, just fed someone else's real map."""
    return max(model_scores, key=lambda s: model_scores[s])


def _eligible_images(manifest: dict, min_n: int) -> List[dict]:
    """Detected images with exactly n attributes (one per subject, the anchor-set invariant)
    and n >= min_n -- the only images a within-item derangement is defined and non-degenerate
    for."""
    return [img for img in manifest["images"]
            if img.get("detected") and len(img["attributes"]) == img["n"] >= min_n]


def within_item_permutation_accuracy(manifest: dict, labels: Dict[str, str], seed: int,
                                     min_n: int = MIN_N) -> Dict[int, dict]:
    """One derangement draw per eligible image: each labeled attribute's prediction is remade
    from a DIFFERENT attribute's own model_scores within the SAME image. Returns per-stratum
    {n_scored, n_correct, accuracy, chance}."""
    rng = random.Random(seed)
    totals: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n_scored": 0, "n_correct": 0})
    for img in _eligible_images(manifest, min_n):
        n = img["n"]
        attrs = img["attributes"]
        derangement = random_derangement(n, rng)
        for i, attr in enumerate(attrs):
            key = label_key(img["prompt_id"], attr["attribute"])
            if key not in labels or labels[key] in NON_SUBJECT_LABELS:
                continue
            pred = permuted_predict(attrs[derangement[i]]["model_scores"])
            totals[n]["n_scored"] += 1
            totals[n]["n_correct"] += int(pred == labels[key])

    return {
        n: dict(n_scored=t["n_scored"], n_correct=t["n_correct"],
                accuracy=(t["n_correct"] / t["n_scored"]) if t["n_scored"] else None,
                chance=chance_baseline(n))
        for n, t in totals.items()
    }


def permutation_sweep(manifest: dict, labels: Dict[str, str],
                      seeds: Iterable[int] = SWEEP_SEEDS, min_n: int = MIN_N) -> Dict[int, dict]:
    """Runs within_item_permutation_accuracy across many seeds. Per stratum: seeds run, median
    accuracy, chance, and the fraction of seeds where a two-sided exact binomial test against
    chance does NOT reject the null (p >= 0.05) -- mirrors exp3_attention_scramble.py's
    falsification-clean fraction, same >= 0.95 decision threshold."""
    per_seed = [within_item_permutation_accuracy(manifest, labels, seed, min_n) for seed in seeds]
    strata = sorted({n for d in per_seed for n in d})
    out = {}
    for n in strata:
        accs = [d[n]["accuracy"] for d in per_seed if d[n]["accuracy"] is not None]
        p_values = []
        for d in per_seed:
            s = d[n]
            if s["n_scored"] == 0:
                continue
            p_values.append(stats.binomtest(
                s["n_correct"], s["n_scored"], s["chance"], alternative="two-sided").pvalue)
        frac_clean = (sum(1 for p in p_values if p >= 0.05) / len(p_values)) if p_values else None
        out[n] = dict(
            n_seeds=len(accs),
            median_accuracy=statistics.median(accs) if accs else None,
            chance=chance_baseline(n),
            frac_not_significantly_different_from_chance=frac_clean,
        )
    return out


def real_vs_permuted_mcnemar(manifest: dict, labels: Dict[str, str], seed: int = MCNEMAR_SEED,
                             min_n: int = MIN_N) -> dict:
    """Paired McNemar between the metric's real prediction (its own attribute's model_scores,
    read straight from the manifest's `predicted_owner`) and one within-item-permuted draw's
    prediction, on the exact same scored rows. Real should beat permuted if the attribute's OWN
    attention content -- not just being somewhere in this image -- is doing the work."""
    rng = random.Random(seed)
    real_only = permuted_only = 0
    n_total = 0
    for img in _eligible_images(manifest, min_n):
        attrs = img["attributes"]
        derangement = random_derangement(img["n"], rng)
        for i, attr in enumerate(attrs):
            key = label_key(img["prompt_id"], attr["attribute"])
            if key not in labels or labels[key] in NON_SUBJECT_LABELS:
                continue
            human = labels[key]
            real_correct = attr["predicted_owner"] == human
            permuted_correct = permuted_predict(attrs[derangement[i]]["model_scores"]) == human
            n_total += 1
            if real_correct and not permuted_correct:
                real_only += 1
            elif permuted_correct and not real_correct:
                permuted_only += 1

    n_discordant = real_only + permuted_only
    p = (stats.binomtest(real_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=n_total, real_only_correct=real_only, permuted_only_correct=permuted_only,
                n_discordant=n_discordant, p_value=p)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 3b: within-item token-permutation falsification control (n >= 3)")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--n-seeds", type=int, default=200)
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")

    print(f"Experiment 3b -- within-item token-permutation control "
          f"({artifacts_dir}, annotator={args.annotator}, {args.n_seeds} seeds, n>={MIN_N})\n")
    sweep = permutation_sweep(manifest, labels, seeds=range(args.n_seeds))
    for n, s in sorted(sweep.items()):
        print(f"  n={n}: median_accuracy={s['median_accuracy']:.3f} chance={s['chance']:.3f} "
              f"falsification_clean_fraction={s['frac_not_significantly_different_from_chance']:.3f} "
              f"(n_seeds={s['n_seeds']})")

    print(f"\nPaired McNemar, real vs. within-item-permuted (seed={MCNEMAR_SEED}):")
    print(f"  {real_vs_permuted_mcnemar(manifest, labels)}")


if __name__ == "__main__":
    main()
