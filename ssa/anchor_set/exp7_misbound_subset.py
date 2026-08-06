#!/usr/bin/env python3
"""
Experiment 7: Mis-bound Subset Mechanism Analysis
Computes the mechanism experiment (§5.3).

Restricts to attribute rows where human_label != intended_subject (rendering disobeying prompt),
and tests whether attention predicts the actual rendered owner above chance on these failure cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scipy import stats
from anchor_common import build_agreement_rows, load_labels, binomial_test_vs_chance

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"


def misbound_analysis_report(manifest: dict, labels: Dict[str, str]) -> dict:
    rows = build_agreement_rows(manifest, labels)

    def _stratum(subset: List[dict]) -> dict:
        # Filter to scored AND mis-bound (human_label != intended_subject)
        misbound = [r for r in subset if r["scored"] and r["human_label"] != r["intended_subject"]]
        n_misbound = len(misbound)
        n_correct = sum(1 for r in misbound if r["correct"])
        
        # Chance baseline per subject count n
        ns = {r["n"] for r in subset}
        chance = (1.0 / next(iter(ns))) if len(ns) == 1 else None
        
        acc = (n_correct / n_misbound) if n_misbound else None
        p_val = binomial_test_vs_chance(n_correct, n_misbound, chance) if chance is not None else None

        return {
            "n_misbound": n_misbound,
            "n_correct": n_correct,
            "accuracy": acc,
            "chance": chance,
            "p_value": p_val
        }

    by_stratum = {n: _stratum([r for r in rows if r["n"] == n])
                  for n in sorted({r["n"] for r in rows})}
    return dict(overall=_stratum(rows), by_stratum=by_stratum)


def _fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%" if pct else f"{x:.4f}"


def format_report(report: dict) -> str:
    header = (f"{'stratum':>8} | {'misbound':>8} | {'correct':>7} | "
              f"{'accuracy':>8} | {'chance':>7} | {'p-value':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for n in sorted(report["by_stratum"]):
        s = report["by_stratum"][n]
        lines.append(
            f"{('n='+str(n)):>8} | {s['n_misbound']:>8} | {s['n_correct']:>7} | "
            f"{_fmt(s['accuracy'], pct=True):>8} | {_fmt(s['chance'], pct=True):>7} | "
            f"{_fmt(s['p_value']):>8}")
    o = report["overall"]
    lines.append(sep)
    lines.append(
        f"{'overall':>8} | {o['n_misbound']:>8} | {o['n_correct']:>7} | "
        f"{_fmt(o['accuracy'], pct=True):>8} | {'  -  ':>7} | {'  -  ':>8}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 7: Mis-bound Subset Mechanism Analysis")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()

    global ARTIFACTS_DIR, MANIFEST_PATH
    ARTIFACTS_DIR = Path(args.artifacts_dir)
    MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"

    manifest = json.loads(MANIFEST_PATH.read_text())
    labels = load_labels(ARTIFACTS_DIR / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {ARTIFACTS_DIR}")

    report = misbound_analysis_report(manifest, labels)
    print(f"Experiment 7 -- Mis-bound Subset Analysis ({ARTIFACTS_DIR}, annotator={args.annotator})\n")
    print(format_report(report))
    print("\nEvaluates metric accuracy specifically when intended_subject != human_label.")


if __name__ == "__main__":
    main()
