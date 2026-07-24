#!/usr/bin/env python
"""
Stage 3 of the metric-A human-agreement anchor set: does the metric's attention-based
`predicted_owner` agree with the human's `labels_<annotator>.json` label?

Joins manifest + labels into one row per (detected image, attribute), drops human
none/unclear rows from the accuracy denominator (reported separately as coverage), and
reports agreement overall and per stratum (n=2/3/4) against the per-stratum chance baseline
1/n. This per-stratum split is the decision-gate artifact and also speaks directly to the
open sub-chance n=2/n=3 binding question: it shows, per subject count, whether attention
tracks human judgment above chance.

Local CPU only (pandas + stdlib). Run from inside ssa/anchor_set/:
    py -3 analyze_agreement.py --annotator chayan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from anchor_common import build_agreement_rows, load_labels, summarize_agreement

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"


def labels_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"labels_{annotator}.json"


def _fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%" if pct else f"{x:.3f}"


def format_summary(summary: dict) -> str:
    """Human-readable table: one row per stratum plus overall, with n, accuracy, chance, and
    margin. `n_scored` is the accuracy denominator; `n_excluded` counts none/unclear."""
    header = (f"{'stratum':>8} | {'labeled':>7} | {'scored':>6} | {'correct':>7} | "
              f"{'accuracy':>8} | {'chance':>7} | {'margin':>7}")
    sep = "-" * len(header)
    lines = [header, sep]
    for n in sorted(summary["by_stratum"]):
        s = summary["by_stratum"][n]
        lines.append(
            f"{('n='+str(n)):>8} | {s['n_labeled']:>7} | {s['n_scored']:>6} | "
            f"{s['n_correct']:>7} | {_fmt(s['accuracy'], pct=True)} | "
            f"{_fmt(s['chance'], pct=True)} | {_fmt(s['margin_over_chance'], pct=True)}")
    o = summary["overall"]
    lines.append(sep)
    lines.append(
        f"{'overall':>8} | {o['n_labeled']:>7} | {o['n_scored']:>6} | "
        f"{o['n_correct']:>7} | {_fmt(o['accuracy'], pct=True)} | "
        f"{'  -  ':>7} | {'  -  ':>7}")
    return "\n".join(lines)


def analyze(annotator: str) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    labels = load_labels(labels_path(annotator))
    if not labels:
        raise SystemExit(f"No labels found at {labels_path(annotator)}; run label_images.py first.")

    rows = build_agreement_rows(manifest, labels)
    summary = summarize_agreement(rows)

    df = pd.DataFrame(rows)
    out_csv = ARTIFACTS_DIR / f"agreement_{annotator}.csv"
    df.to_csv(out_csv, index=False)

    print(f"Annotator: {annotator}  |  rows: {len(rows)}  |  detail CSV: {out_csv}\n")
    print(format_summary(summary))
    excluded = summary["overall"]["n_excluded"]
    if excluded:
        print(f"\n({excluded} labeled rows excluded as none/unclear -- coverage, not error.)")
    print("\nDecision gate (memo SSA-Metric-Memo.md §7): accuracy clearly above the 1/n chance "
          "line at a stratum = attention tracks binding there.")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Agreement analysis for the metric-A anchor set")
    ap.add_argument("--annotator", required=True, help="annotator id used by label_images.py")
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="directory holding manifest.json / labels files, e.g. "
                         "'artifacts_sdxl' for the SDXL run (default: 'artifacts', the SD1.5 run)")
    args = ap.parse_args()
    global ARTIFACTS_DIR, MANIFEST_PATH
    ARTIFACTS_DIR = Path(args.artifacts_dir)
    MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
    analyze(args.annotator)


if __name__ == "__main__":
    main()
