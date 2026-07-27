#!/usr/bin/env python
"""
Part A, Experiment 3: does the signal survive attention randomization? The falsification test
the whole approach depends on. If accuracy stays high under scrambled attention, the metric isn't
actually using attention -- it's picking up on something else (subject size, position, prompt
structure). If accuracy drops to chance, attention is genuinely doing the work.

Scrambles CROSS-ITEM, within the same n-stratum -- NOT by permuting a single item's own
model_scores across its own subjects. A within-item permutation degenerates at n=2: two values
have exactly one non-identity permutation (a forced swap), so scrambled accuracy would equal
`1 - real accuracy` by arithmetic, not by any property of attention. Instead, each scored
attribute's prediction is remade using a DIFFERENT image's (same n) model_scores, randomly
permuted onto this image's own subjects -- i.e., "would the metric have done just as well seeing
a completely unrelated item's attention?" Mirrors the attn_scrambled_crosschain condition metric
B already uses (pi_level_experiment/).

Run as a many-seed sweep (a single shuffle is exactly the "lucky/unlucky draw" problem this
repo has flagged before), plus one paired McNemar (real vs. scrambled correctness) at a single
fixed, pre-registered seed for reporting alongside the sweep.

Design doc: docs/part-a-five-experiment-battery-design.md (Experiment 3).

NOT YET RUN ON REAL DATA -- see docs/part-a-five-experiment-battery-design.md. Run against the
dummy fixture today:
    py -3 exp3_attention_scramble.py --artifacts-dir artifacts_dummy --annotator dummy
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from scipy import stats

from anchor_common import build_agreement_rows, chance_baseline, load_labels

MCNEMAR_SEED = 42  # matches this repo's existing DEFAULT_SEED convention (pi_level_experiment/)
SWEEP_SEEDS = range(200)


def scramble_predict(subjects: Sequence[str], donor_scores: Dict[str, float],
                     rng: random.Random) -> str:
    """Randomly permute `donor_scores`' VALUES onto `subjects` (a different image's own subject
    slots -- the donor's subject names are never used), then return the argmax. The donor's
    identity is irrelevant to the returned name; only its numeric magnitudes are borrowed."""
    values = list(donor_scores.values())
    rng.shuffle(values)
    assigned = dict(zip(subjects, values))
    return max(assigned, key=lambda s: assigned[s])


def _score_pool_by_stratum(manifest: dict) -> Dict[int, List[Tuple[int, Dict[str, float]]]]:
    """n -> [(prompt_id, model_scores), ...] for every detected image's every attribute."""
    pool: Dict[int, List[Tuple[int, Dict[str, float]]]] = defaultdict(list)
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            pool[img["n"]].append((img["prompt_id"], attr["model_scores"]))
    return pool


def cross_item_scramble_accuracy(manifest: dict, labels: Dict[str, str], seed: int) -> Dict[int, dict]:
    """One scrambled draw: every scored row's prediction is remade from a randomly permuted,
    DIFFERENT-image donor at the same stratum. Returns per-stratum {n_scored, n_correct,
    accuracy, chance}. Raises ValueError if a stratum has no cross-item donor available (needs
    >= 2 distinct detected images at that n)."""
    rows = build_agreement_rows(manifest, labels)
    subjects_by_id = {img["prompt_id"]: img["subjects"]
                      for img in manifest["images"] if img.get("detected")}
    pool = _score_pool_by_stratum(manifest)
    rng = random.Random(seed)

    totals: Dict[int, Dict[str, int]] = defaultdict(lambda: {"n_scored": 0, "n_correct": 0})
    for r in rows:
        if not r["scored"]:
            continue
        n = r["n"]
        candidates = [(pid, ms) for pid, ms in pool[n] if pid != r["prompt_id"]]
        if not candidates:
            raise ValueError(
                f"stratum n={n} has no cross-item donor (need >= 2 distinct detected images "
                f"at that n; found only prompt_id={r['prompt_id']} itself)")
        _, donor_scores = rng.choice(candidates)
        pred = scramble_predict(subjects_by_id[r["prompt_id"]], donor_scores, rng)
        totals[n]["n_scored"] += 1
        totals[n]["n_correct"] += int(pred == r["human_label"])

    return {
        n: dict(n_scored=t["n_scored"], n_correct=t["n_correct"],
                accuracy=(t["n_correct"] / t["n_scored"]) if t["n_scored"] else None,
                chance=chance_baseline(n))
        for n, t in totals.items()
    }


def scramble_sweep(manifest: dict, labels: Dict[str, str],
                   seeds: Iterable[int] = SWEEP_SEEDS) -> Dict[int, dict]:
    """Runs cross_item_scramble_accuracy across many seeds. Per stratum: the number of seeds
    run, the median accuracy across seeds, that stratum's chance, and the fraction of seeds
    where a two-sided exact binomial test against chance does NOT reject the null (p >= 0.05) --
    "falsification-clean" requires this fraction to be >= 0.95 (see design doc's decision
    rule)."""
    per_seed = [cross_item_scramble_accuracy(manifest, labels, seed) for seed in seeds]
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


def real_vs_scrambled_mcnemar(manifest: dict, labels: Dict[str, str], seed: int = MCNEMAR_SEED) -> dict:
    """Paired McNemar between real correctness and one scrambled draw's correctness, on the
    exact same scored rows -- real should beat scrambled if attention content is doing the
    work."""
    rows = build_agreement_rows(manifest, labels)
    subjects_by_id = {img["prompt_id"]: img["subjects"]
                      for img in manifest["images"] if img.get("detected")}
    pool = _score_pool_by_stratum(manifest)
    rng = random.Random(seed)

    real_only = scrambled_only = 0
    n = 0
    for r in rows:
        if not r["scored"]:
            continue
        candidates = [(pid, ms) for pid, ms in pool[r["n"]] if pid != r["prompt_id"]]
        if not candidates:
            raise ValueError(f"stratum n={r['n']} has no cross-item donor")
        _, donor_scores = rng.choice(candidates)
        scrambled_pred = scramble_predict(subjects_by_id[r["prompt_id"]], donor_scores, rng)
        real_correct = r["correct"]
        scrambled_correct = scrambled_pred == r["human_label"]
        n += 1
        if real_correct and not scrambled_correct:
            real_only += 1
        elif scrambled_correct and not real_correct:
            scrambled_only += 1

    n_discordant = real_only + scrambled_only
    p = (stats.binomtest(real_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=n, real_only_correct=real_only, scrambled_only_correct=scrambled_only,
                n_discordant=n_discordant, p_value=p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 3: attention-randomization falsification")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--n-seeds", type=int, default=200)
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")

    print(f"Experiment 3 -- attention-randomization falsification "
          f"({artifacts_dir}, annotator={args.annotator}, {args.n_seeds} seeds)\n")
    sweep = scramble_sweep(manifest, labels, seeds=range(args.n_seeds))
    for n, s in sorted(sweep.items()):
        print(f"  n={n}: median_accuracy={s['median_accuracy']:.3f} chance={s['chance']:.3f} "
              f"falsification_clean_fraction={s['frac_not_significantly_different_from_chance']:.3f} "
              f"(n_seeds={s['n_seeds']})")

    print(f"\nPaired McNemar, real vs. scrambled (seed={MCNEMAR_SEED}):")
    print(f"  {real_vs_scrambled_mcnemar(manifest, labels)}")


if __name__ == "__main__":
    main()
