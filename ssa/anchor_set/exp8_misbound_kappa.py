#!/usr/bin/env python
"""
Inter-annotator agreement restricted to the DISOBEYED rows only (experiment #13 in the
CPGA experiment set).

Why this exists: claim C3 -- "on rows where the model disobeyed the prompt, attention is
no better than chance" -- rests entirely on the subset where `human_label !=
intended_subject`. The headline agreement numbers (kappa 0.889-0.912 on artifacts_flux_hard)
are dominated by the easy OBEYED rows, where all three annotators trivially agree because
the image shows exactly what the prompt asked for. If agreement collapses on the disobeyed
subset, then C3's ground truth is annotator noise rather than rendering failure, and the
claim has to be weakened. This is pure re-analysis of existing labels -- no GPU, no new
data.

SELECTION BIAS, and why the subset is defined by the CONSENSUS label:
    Defining "disobeyed" using one member of the pair being tested (e.g. selecting rows
    where annotator3 said something other than intended_subject, then computing kappa(annotator3,
    annotator2)) conditions the subset on one rater's value. That mechanically depresses kappa:
    it over-samples rows where that rater is an outlier, including their own label noise.
    Selecting on the consensus label instead is symmetric with respect to any pair -- it
    treats both members identically.

    This is NOT fully independent of the pair (the consensus is a majority vote over the
    same three annotators, so each rater contributes to it), and that residual dependence
    is stated rather than papered over. The `--selection either` mode reports the
    asymmetric alternative as a sensitivity check; expect it to give a LOWER kappa, and
    read that as the bias described above, not as a separate finding.

Interpretation guide, decided before running (this is the pre-registered read):
    kappa >= 0.6 on the disobeyed subset  -> C3's ground truth holds; the disobeyed rows
                                             are real rendering failures the annotators
                                             independently see.
    kappa 0.4-0.6                         -> report C3 with the agreement caveat attached
                                             to the number.
    kappa < 0.4                           -> C3's foundation is annotator disagreement;
                                             the claim cannot be made as written.

`cohens_kappa` is anchor_common's canonical implementation, called on RAW label dicts
(including the none/unclear/shared categories) exactly as analyze_agreement.py calls it,
so the "overall" line this prints reproduces the already-published kappa and the restricted
lines are directly comparable to it.

Local CPU only. Run from inside ssa/anchor_set/:
    py -3 exp8_misbound_kappa.py --artifacts-dir artifacts_flux_hard \
        --annotators annotator3 annotator2 annotator4
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Set

from anchor_common import NON_SUBJECT_LABELS, cohens_kappa, label_key, load_labels
from config import Config

_SIGNIFICANCE_DEFAULTS = Config().significance


def intended_subject_by_key(manifest: dict) -> Dict[str, str]:
    """{label_key: intended_subject} for every attribute of every DETECTED image.

    Undetected images are skipped because they carry no attribute entries at all (see
    build_manifest_entry's `attributes=[]` on detection failure), so they can never
    contribute a labeled row."""
    intended: Dict[str, str] = {}
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            intended[label_key(img["prompt_id"], attr["attribute"])] = attr["intended_subject"]
    return intended


def disobeyed_keys(intended: Dict[str, str], labels: Dict[str, str]) -> Set[str]:
    """Keys where the annotator named a SUBJECT and it is not the one the prompt asked for.

    none/unclear/shared are excluded (NON_SUBJECT_LABELS): they are missing data, not
    evidence of disobedience -- the same rule build_agreement_rows uses for `scored`."""
    return {
        key for key, label in labels.items()
        if key in intended and label not in NON_SUBJECT_LABELS and label != intended[key]
    }


def restrict(labels: Dict[str, str], keys: Set[str]) -> Dict[str, str]:
    return {k: v for k, v in labels.items() if k in keys}


def kappa_or_none(labels_a: Dict[str, str], labels_b: Dict[str, str]) -> dict:
    """cohens_kappa, but returns a stub instead of raising when the pair shares no keys.

    A restricted subset can legitimately leave a pair with zero overlap (e.g. a very small
    disobeyed set), and that is a reportable "not enough data" outcome, not a crash."""
    if not (set(labels_a) & set(labels_b)):
        return dict(n=0, p_observed=None, p_expected=None, kappa=None, categories=[])
    return cohens_kappa(labels_a, labels_b)


def pairwise_report(labels_by_annotator: Dict[str, Dict[str, str]],
                    keys: Set[str] = None) -> Dict[str, dict]:
    """{"a_vs_b": kappa_dict} for every annotator pair, optionally restricted to `keys`."""
    report: Dict[str, dict] = {}
    for a, b in itertools.combinations(sorted(labels_by_annotator), 2):
        la, lb = labels_by_annotator[a], labels_by_annotator[b]
        if keys is not None:
            la, lb = restrict(la, keys), restrict(lb, keys)
        report[f"{a}_vs_{b}"] = kappa_or_none(la, lb)
    return report


def selection_keys(mode: str, intended: Dict[str, str],
                   labels_by_annotator: Dict[str, Dict[str, str]],
                   consensus: Dict[str, str] = None) -> Set[str]:
    """The disobeyed-row subset, under one of two selection rules.

    "consensus" (default, and the one to report): rows the majority-vote consensus label
        calls disobeyed. Symmetric across every pair -- see module docstring.
    "either": rows ANY annotator calls disobeyed. Asymmetric and biased downward; a
        sensitivity check only."""
    if mode == "consensus":
        if consensus is None:
            raise ValueError(
                "selection mode 'consensus' needs a consensus label file; pass "
                "--consensus-name or run build_consensus_labels.py first")
        return disobeyed_keys(intended, consensus)
    if mode == "either":
        keys: Set[str] = set()
        for labels in labels_by_annotator.values():
            keys |= disobeyed_keys(intended, labels)
        return keys
    raise ValueError(f"unknown selection mode {mode!r}; expected 'consensus' or 'either'")


def format_kappa_line(name: str, k: dict) -> str:
    if k["kappa"] is None:
        detail = "n/a" if k["n"] == 0 else "undefined (one category)"
        return f"    {name:<24} n={k['n']:<5} kappa={detail}"
    return (f"    {name:<24} n={k['n']:<5} kappa={k['kappa']:.3f} "
            f"(observed agreement {k['p_observed']:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cohen's kappa restricted to disobeyed rows (experiment #13)")
    ap.add_argument("--artifacts-dir", default="artifacts_flux_hard")
    ap.add_argument("--annotators", nargs="+", default=["annotator3", "annotator2", "annotator4"])
    ap.add_argument("--consensus-name", default="consensus",
                    help="annotator name of the consensus label file (labels_<name>.json)")
    ap.add_argument("--selection", choices=["consensus", "either"], default="consensus")
    ap.add_argument("--out", default=None,
                    help="optional path to write the report as JSON")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    intended = intended_subject_by_key(manifest)

    labels_by_annotator = {
        a: load_labels(artifacts_dir / f"labels_{a}.json") for a in args.annotators
    }
    consensus_path = artifacts_dir / f"labels_{args.consensus_name}.json"
    consensus = load_labels(consensus_path) if consensus_path.exists() else None

    dis_keys = selection_keys(args.selection, intended, labels_by_annotator, consensus)
    all_labeled = {k for labels in labels_by_annotator.values() for k in labels}
    obeyed_keys = {k for k in all_labeled if k in intended} - dis_keys

    overall = pairwise_report(labels_by_annotator)
    disobeyed = pairwise_report(labels_by_annotator, dis_keys)
    obeyed = pairwise_report(labels_by_annotator, obeyed_keys)

    print(f"artifacts_dir={artifacts_dir}  annotators={args.annotators}")
    print(f"selection={args.selection}  disobeyed rows selected={len(dis_keys)}")
    if args.selection == "consensus" and consensus is None:
        print("  WARNING: no consensus file found")

    print("\n=== Overall kappa (all shared keys -- reproduces the published number) ===")
    for name, k in overall.items():
        print(format_kappa_line(name, k))

    print("\n=== Kappa on OBEYED rows only (the easy majority) ===")
    for name, k in obeyed.items():
        print(format_kappa_line(name, k))

    print("\n=== Kappa on DISOBEYED rows only (C3's foundation) ===")
    for name, k in disobeyed.items():
        print(format_kappa_line(name, k))

    kappas = [k["kappa"] for k in disobeyed.values() if k["kappa"] is not None]
    if kappas:
        lo, hi = min(kappas), max(kappas)
        print(f"\n  disobeyed-subset kappa range: {lo:.3f} - {hi:.3f}")
        verdict = ("C3's ground truth HOLDS" if lo >= _SIGNIFICANCE_DEFAULTS.kappa_hold_min
                   else "report C3 WITH the agreement caveat attached" if lo >= _SIGNIFICANCE_DEFAULTS.kappa_caveat_min
                   else "C3's foundation is annotator disagreement -- cannot claim as written")
        print(f"  pre-registered read: {verdict}")
    else:
        print("\n  no disobeyed-subset kappa computable (too few overlapping rows)")

    if args.out:
        payload = dict(artifacts_dir=str(artifacts_dir), annotators=args.annotators,
                       selection=args.selection, n_disobeyed_keys=len(dis_keys),
                       overall=overall, obeyed=obeyed, disobeyed=disobeyed)
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
