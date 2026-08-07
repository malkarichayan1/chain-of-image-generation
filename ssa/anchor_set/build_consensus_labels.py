#!/usr/bin/env python
"""
Builds a majority-vote consensus label set (and consensus count-clean judgment) across
several annotators' label files -- briefing §9.2: "we have three annotators at kappa ~= 0.95
[0.89-0.91 on the FLUX-hard set] and currently analyze them separately. A consensus label set
would tighten every interval slightly."

Policy: majority vote over exactly the annotators the caller supplies. A key is UNANIMOUS
when every voting annotator agrees, MAJORITY when a strict majority (more than half of the
annotators who voted on that key) agrees but not all, and NO_CONSENSUS when no value clears a
strict majority -- those rows are dropped from the consensus output rather than guessed at.
With exactly 3 raters, "none" only happens when all three pick different values (rare for a
closed-vocabulary label set, but real -- see label_key's subject/none/unclear/shared alphabet).

The consensus key set is the UNION of every annotator's keys: an annotator who hasn't reached
a given row simply doesn't vote on it, so a row only two of three annotators reached still
gets a two-way consensus. This mirrors how the underlying labels_<annotator>.json files can
each be partial, and is what lets build_consensus_labels.py run productively even before every
annotator has finished the full pass.

Run from inside ssa/anchor_set/:
    py -3 build_consensus_labels.py --artifacts-dir artifacts_flux_hard \
        --annotators akhil grace pranav
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from anchor_common import load_labels, save_labels

AGREEMENT_KINDS = ("unanimous", "majority", "none")


def majority_vote(values: Sequence[str]) -> Tuple[Optional[str], str]:
    """Returns (winner, agreement) for one row's annotator values. `agreement` is one of
    AGREEMENT_KINDS; `winner` is None exactly when agreement == "none" (no value to report)."""
    if not values:
        raise ValueError("no values to vote on")
    counts = Counter(values)
    winner, top_count = counts.most_common(1)[0]
    if top_count == len(values):
        return winner, "unanimous"
    if top_count > len(values) / 2:
        return winner, "majority"
    return None, "none"


def build_consensus(label_files: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, str], dict]:
    """label_files: {annotator: {key: value}}. Returns (consensus_labels, stats), where stats
    is {"n_keys", "unanimous", "majority", "none"} -- counts of how each key in the union
    resolved. Rows that resolve to "none" are counted but excluded from consensus_labels."""
    all_keys = sorted({k for labels in label_files.values() for k in labels})
    consensus: Dict[str, str] = {}
    stats = {kind: 0 for kind in AGREEMENT_KINDS}
    stats["n_keys"] = len(all_keys)
    for key in all_keys:
        values = [labels[key] for labels in label_files.values() if key in labels]
        winner, agreement = majority_vote(values)
        stats[agreement] += 1
        if winner is not None:
            consensus[key] = winner
    return consensus, stats


def format_stats(name: str, annotators: Sequence[str], stats: dict, n_written: int) -> str:
    return (f"{name}: {stats['n_keys']} keys voted on across {list(annotators)} -> "
            f"unanimous={stats['unanimous']} majority={stats['majority']} "
            f"no_consensus={stats['none']} (wrote {n_written})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a majority-vote consensus label/count set across annotators")
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--annotators", nargs="+", required=True,
                    help="annotator ids to vote across, e.g. --annotators akhil grace pranav")
    ap.add_argument("--out-annotator", default="consensus",
                    help="writes labels_<out-annotator>.json / counts_<out-annotator>.json")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    label_files = {a: load_labels(artifacts_dir / f"labels_{a}.json") for a in args.annotators}
    for a, labels in label_files.items():
        if not labels:
            raise SystemExit(f"No labels found for annotator {a!r} in {artifacts_dir}")
    count_files = {a: load_labels(artifacts_dir / f"counts_{a}.json") for a in args.annotators}

    consensus_labels, label_stats = build_consensus(label_files)
    consensus_counts, count_stats = build_consensus(count_files)

    out_labels = artifacts_dir / f"labels_{args.out_annotator}.json"
    out_counts = artifacts_dir / f"counts_{args.out_annotator}.json"
    save_labels(out_labels, consensus_labels)
    save_labels(out_counts, consensus_counts)

    print(format_stats("Labels", args.annotators, label_stats, len(consensus_labels)))
    print(f"  -> {out_labels}")
    print(format_stats("Counts", args.annotators, count_stats, len(consensus_counts)))
    print(f"  -> {out_counts}")


if __name__ == "__main__":
    main()
