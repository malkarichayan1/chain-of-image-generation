#!/usr/bin/env python
"""
Part A, Experiment 1 (the headline number): does cross-attention beat random guessing, per
subject count? For each scored attribute, was `predicted_owner` (attention's argmax) the human's
labeled subject? Reports accuracy at n=2/3/4 separately against chance = 1/n (50%/33%/25%), with
a one-sided binomial significance test per stratum -- accuracy alone doesn't say whether a
14/18 vs a 6/8 is "clearly above chance" or "consistent with a lucky run."

Thin wrapper over anchor_common.build_agreement_rows()/summarize_agreement() (already computes
per-stratum accuracy vs 1/n) -- the only new logic here is the significance test, added via
anchor_common.binomial_test_vs_chance() (shared with exp3_attention_scramble.py).

NOT YET RUN ON REAL DATA. Scaffolded against dummy data ahead of Workstream 2 (the expanded,
second-annotator-validated SDXL anchor set). Run against the dummy fixture today:
    py -3 exp1_accuracy_by_n.py --artifacts-dir artifacts_dummy --annotator dummy
Once Workstream 2 lands, point at the real set:
    py -3 exp1_accuracy_by_n.py --artifacts-dir artifacts_sdxl --annotator <name>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from anchor_common import (
    binomial_test_vs_chance, build_agreement_rows, load_labels, summarize_agreement,
)
from config import Config


def _with_p_value(stratum: dict) -> dict:
    out = dict(stratum)
    out["p_value"] = (
        binomial_test_vs_chance(stratum["n_correct"], stratum["n_scored"], stratum["chance"])
        if stratum["chance"] is not None else None)
    return out


def add_significance(summary: dict) -> dict:
    """Augment summarize_agreement()'s output with a per-stratum one-sided binomial p-value
    against that stratum's own 1/n chance. `overall`'s chance is None whenever more than one n
    is present (summarize_agreement's own rule) -- pooling different chance lines has no single
    null to test against, so overall's p_value is None too, not an invented number."""
    return dict(overall=_with_p_value(summary["overall"]),
                by_stratum={n: _with_p_value(s) for n, s in summary["by_stratum"].items()})


def accuracy_by_n_report(manifest: dict, labels: Dict[str, str]) -> dict:
    rows = build_agreement_rows(manifest, labels)
    return add_significance(summarize_agreement(rows))


def _fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%" if pct else f"{x:.4f}"


def format_report(report: dict) -> str:
    header = (f"{'stratum':>8} | {'labeled':>7} | {'scored':>6} | {'correct':>7} | "
              f"{'accuracy':>8} | {'chance':>7} | {'p-value':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for n in sorted(report["by_stratum"]):
        s = report["by_stratum"][n]
        lines.append(
            f"{('n='+str(n)):>8} | {s['n_labeled']:>7} | {s['n_scored']:>6} | "
            f"{s['n_correct']:>7} | {_fmt(s['accuracy'], pct=True)} | "
            f"{_fmt(s['chance'], pct=True)} | {_fmt(s['p_value'])}")
    o = report["overall"]
    lines.append(sep)
    lines.append(
        f"{'overall':>8} | {o['n_labeled']:>7} | {o['n_scored']:>6} | "
        f"{o['n_correct']:>7} | {_fmt(o['accuracy'], pct=True)} | {'  -  ':>7} | {'  -  ':>8}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 1: accuracy per subject count vs. 1/n chance")
    ap.add_argument("--artifacts-dir", default=str(Config().paths.artifacts_dir))
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    manifest_path = artifacts_dir / "manifest.json"

    manifest = json.loads(manifest_path.read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")

    report = accuracy_by_n_report(manifest, labels)
    print(f"Experiment 1 -- accuracy per subject count ({artifacts_dir}, annotator={args.annotator})\n")
    print(format_report(report))
    print("\nSignificant (p < 0.05, one-sided) means that stratum's accuracy clearly beats "
          "its own 1/n chance line -- not a lucky run at this sample size.")


if __name__ == "__main__":
    main()
