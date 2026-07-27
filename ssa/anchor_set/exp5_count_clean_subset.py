#!/usr/bin/env python
"""
Part A, Experiment 5: what happens when we restrict to count-clean items only? A prompt asking
for 4 subjects that only renders 3 is a rendering failure, not a binding failure -- two different
things. This isolates the pure binding question, uncorrected by the model's separate tendency to
under-render (or, for SDXL specifically, to render duplicate/interchangeable subjects).

anchor_common.build_agreement_rows(manifest, labels, counts=counts) already implements the
underlying exclusion (count-broken images forced out of the accuracy denominator) --
analyze_agreement.py already wires counts_<annotator>.json into a single filtered view. What
neither does: show both views side by side, so a reader sees whether restricting to count-clean
*changes* accuracy materially or just shrinks n. That comparison is this script's only job.

Design doc: docs/part-a-five-experiment-battery-design.md (Experiment 5).

NOT YET RUN ON REAL DATA -- see docs/part-a-five-experiment-battery-design.md. Run against the
dummy fixture today:
    py -3 exp5_count_clean_subset.py --artifacts-dir artifacts_dummy --annotator dummy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from anchor_common import build_agreement_rows, load_labels, summarize_agreement


def count_clean_vs_all_report(manifest: dict, labels: Dict[str, str],
                              counts: Dict[str, str]) -> dict:
    """Both summarize_agreement() tables on the SAME manifest/labels: `all` (counts=None, every
    detected image) and `count_clean` (counts=<loaded>, count-broken images excluded), plus how
    many rows that exclusion actually removed."""
    all_rows = build_agreement_rows(manifest, labels)
    clean_rows = build_agreement_rows(manifest, labels, counts=counts)
    n_excluded = sum(1 for r in clean_rows if r["count_broken"])
    return dict(all=summarize_agreement(all_rows),
                count_clean=summarize_agreement(clean_rows),
                n_excluded_as_count_broken=n_excluded)


def _fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%" if pct else f"{x:.3f}"


def _table(summary: dict) -> str:
    header = (f"{'stratum':>8} | {'labeled':>7} | {'scored':>6} | {'correct':>7} | "
              f"{'accuracy':>8} | {'chance':>7}")
    sep = "-" * len(header)
    lines = [header, sep]
    for n in sorted(summary["by_stratum"]):
        s = summary["by_stratum"][n]
        lines.append(
            f"{('n='+str(n)):>8} | {s['n_labeled']:>7} | {s['n_scored']:>6} | "
            f"{s['n_correct']:>7} | {_fmt(s['accuracy'], pct=True)} | {_fmt(s['chance'], pct=True)}")
    o = summary["overall"]
    lines.append(sep)
    lines.append(f"{'overall':>8} | {o['n_labeled']:>7} | {o['n_scored']:>6} | "
                 f"{o['n_correct']:>7} | {_fmt(o['accuracy'], pct=True)} | {'  -  ':>7}")
    return "\n".join(lines)


def format_report(report: dict) -> str:
    return (
        "ALL ROWS (no count filter)\n" + _table(report["all"]) +
        "\n\nCOUNT-CLEAN ONLY (count-broken images excluded)\n" + _table(report["count_clean"]) +
        f"\n\n({report['n_excluded_as_count_broken']} row(s) excluded as count-broken -- "
        "rendering failure, not binding failure.)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 5: accuracy restricted to count-clean images vs. all rows")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--counts-annotator", default=None,
                    help="annotator id whose counts_<id>.json to use (default: same as --annotator)")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")
    counts_annotator = args.counts_annotator or args.annotator
    counts = load_labels(artifacts_dir / f"counts_{counts_annotator}.json")
    if not counts:
        print(f"(no counts_{counts_annotator}.json found -- count-clean view will match "
              f"the all-rows view exactly)")

    report = count_clean_vs_all_report(manifest, labels, counts)
    print(f"Experiment 5 -- count-clean subset ({artifacts_dir}, annotator={args.annotator})\n")
    print(format_report(report))


if __name__ == "__main__":
    main()
