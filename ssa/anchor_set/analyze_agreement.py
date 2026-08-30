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
    py -3 analyze_agreement.py --annotator annotator1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from anchor_common import (
    build_agreement_rows, cohens_kappa, load_labels, summarize_agreement,
)

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"


def labels_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"labels_{annotator}.json"


def counts_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"counts_{annotator}.json"


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


def format_kappa(result: dict, annotator_a: str, annotator_b: str) -> str:
    """Inter-rater reliability block. Reports the raw agreement fraction alongside kappa,
    because on a small anchor set kappa alone hides how few rows it rests on."""
    agreed = round(result["p_observed"] * result["n"])
    kappa = "undefined (both raters used a single category)" if result["kappa"] is None \
        else f"{result['kappa']:.3f}"
    return (f"Inter-rater reliability: {annotator_a} vs {annotator_b}\n"
            f"  overlapping judgments : {result['n']}\n"
            f"  raw agreement         : {agreed}/{result['n']} = {result['p_observed']:.1%}\n"
            f"  chance agreement      : {result['p_expected']:.1%}\n"
            f"  Cohen's kappa         : {kappa}\n"
            f"  categories used       : {', '.join(result['categories'])}")


def analyze(annotator: str, margin_threshold: float = None,
            compare_annotator: str = None) -> dict:
    """Strict agreement table (the frozen, published measurement). With `margin_threshold`,
    additionally reports the shared-aware table, which readmits human `shared` rows as a
    scoreable outcome and lets the metric abstain via its top-two attention margin -- and
    returns {"strict": ..., "shared": ...} instead of the bare strict summary.
    With `compare_annotator`, also prints Cohen's kappa against that annotator's labels.

    Count-clean/count-broken: if counts_<annotator>.json exists (per-image judgments from
    label_images.py's prompt_count_clean()), every attribute row from a count-broken image is
    forced out of the accuracy denominator via build_agreement_rows(..., counts=...) -- a
    count-broken image conflates rendering failure with binding failure (see
    docs/anchor-set-labeling-protocol.md §4), so it can't answer a pure binding question.
    Missing counts file (e.g. the original SD1.5/SDXL runs, which predate this check) is
    equivalent to an empty dict: no row is excluded for count reasons, reproducing every
    already-published number exactly."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    labels = load_labels(labels_path(annotator))
    if not labels:
        raise SystemExit(f"No labels found at {labels_path(annotator)}; run label_images.py first.")
    counts = load_labels(counts_path(annotator))

    rows = build_agreement_rows(manifest, labels, margin_threshold=margin_threshold, counts=counts)
    summary = summarize_agreement(rows)

    df = pd.DataFrame(rows)
    out_csv = ARTIFACTS_DIR / f"agreement_{annotator}.csv"
    df.to_csv(out_csv, index=False)

    print(f"Annotator: {annotator}  |  rows: {len(rows)}  |  detail CSV: {out_csv}\n")
    if counts:
        n_broken = sum(1 for r in rows if r.get("count_broken"))
        print(f"Count-clean check: {counts_path(annotator)} loaded, "
              f"{n_broken} row(s) excluded as count-broken.\n")
    print("STRICT (one subject only; chance = 1/n)")
    print(format_summary(summary))
    excluded = summary["overall"]["n_excluded"]
    if excluded:
        print(f"\n({excluded} labeled rows excluded as none/unclear/shared/count-broken -- "
              f"coverage, not error.)")
    print("\nDecision gate (memo SSA-Metric-Memo.md §7): accuracy clearly above the 1/n chance "
          "line at a stratum = attention tracks binding there.")

    result = summary
    if margin_threshold is not None:
        shared_summary = summarize_agreement(rows, mode="shared")
        print(f"\nSHARED-AWARE (metric may abstain at margin < {margin_threshold}; "
              f"chance = 1/(n+1))")
        print(format_summary(shared_summary))
        recovered = shared_summary["overall"]["n_scored"] - summary["overall"]["n_scored"]
        print(f"\n({recovered} leakage rows recovered as a scoreable outcome rather than "
              "discarded as missing data.)")
        result = {"strict": summary, "shared": shared_summary}

    if compare_annotator is not None:
        other = load_labels(labels_path(compare_annotator))
        if not other:
            raise SystemExit(
                f"No labels found at {labels_path(compare_annotator)}; "
                f"the second annotator has not labeled anything yet.")
        print()
        print(format_kappa(cohens_kappa(labels, other), annotator, compare_annotator))

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Agreement analysis for the metric-A anchor set")
    ap.add_argument("--annotator", required=True, help="annotator id used by label_images.py")
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="directory holding manifest.json / labels files, e.g. "
                         "'artifacts_sdxl' for the SDXL run (default: 'artifacts', the SD1.5 run)")
    ap.add_argument("--margin-threshold", type=float, default=None,
                    help="also report the shared-aware table: the metric predicts `shared` "
                         "when its top-two attention margin falls below this. Scale it to "
                         "the DATA, not to intuition -- observed margins are tiny (SDXL "
                         "median 0.0095, max 0.0689; SD1.5 median 0.0127), so 0.2 abstains "
                         "on literally every row. Sweep something like 0.005-0.03. "
                         "Omit for the strict, already-published measurement only.")
    ap.add_argument("--compare-annotator", default=None,
                    help="second annotator id to compute Cohen's kappa against, e.g. "
                         "'--annotator annotator1 --compare-annotator ane'")
    args = ap.parse_args()
    global ARTIFACTS_DIR, MANIFEST_PATH
    ARTIFACTS_DIR = Path(args.artifacts_dir)
    MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
    analyze(args.annotator, margin_threshold=args.margin_threshold,
            compare_annotator=args.compare_annotator)


if __name__ == "__main__":
    main()
