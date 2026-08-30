#!/usr/bin/env python3
"""
Experiment 6: Attention vs. Prompt-Obeyed Baseline
Computes the paper's central table (§5.1).

A trivial baseline that never looks at the image — "assume the model rendered
what the prompt asked for" — significantly outperforms the attention-based metric.
This script compares attention accuracy against the prompt baseline on the exact
same scored rows, and runs McNemar's test (exact binomial on discordant pairs)
to check significance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scipy import stats
from anchor_common import build_agreement_rows, load_labels
from config import Config

def exact_mcnemar_p(b: int, c: int) -> float | None:
    """Exact McNemar's test (binomial test on discordant pairs).
    b = method 1 correct, method 2 wrong
    c = method 1 wrong, method 2 correct
    """
    n = b + c
    if n == 0:
        return None
    # Two-sided test against 0.5
    return float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)

def baseline_accuracy_report(manifest: dict, labels: Dict[str, str]) -> dict:
    rows = build_agreement_rows(manifest, labels)
    
    def _stratum(subset: List[dict]) -> dict:
        scored = [r for r in subset if r["scored"]]
        n_scored = len(scored)
        
        att_correct = 0
        base_correct = 0
        b = 0 # attention correct, baseline wrong
        c = 0 # attention wrong, baseline correct
        
        for r in scored:
            att_hit = r["correct"]
            base_hit = (r["human_label"] == r["intended_subject"])
            
            if att_hit: att_correct += 1
            if base_hit: base_correct += 1
            
            if att_hit and not base_hit: b += 1
            if not att_hit and base_hit: c += 1
            
        return {
            "n_scored": n_scored,
            "att_acc": att_correct / n_scored if n_scored else None,
            "base_acc": base_correct / n_scored if n_scored else None,
            "b": b,
            "c": c,
            "p_value": exact_mcnemar_p(b, c)
        }

    by_stratum = {n: _stratum([r for r in rows if r["n"] == n])
                  for n in sorted({r["n"] for r in rows})}
    return dict(overall=_stratum(rows), by_stratum=by_stratum)

def _fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%" if pct else f"{x:.4f}"

def format_report(report: dict) -> str:
    header = (f"{'stratum':>8} | {'scored':>6} | {'attn_acc':>8} | {'base_acc':>8} | "
              f"{'attn>base':>9} | {'base>attn':>9} | {'p-value':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for n in sorted(report["by_stratum"]):
        s = report["by_stratum"][n]
        lines.append(
            f"{('n='+str(n)):>8} | {s['n_scored']:>6} | "
            f"{_fmt(s['att_acc'], pct=True):>8} | {_fmt(s['base_acc'], pct=True):>8} | "
            f"{s['b']:>9} | {s['c']:>9} | {_fmt(s['p_value'])}")
    o = report["overall"]
    lines.append(sep)
    lines.append(
        f"{'overall':>8} | {o['n_scored']:>6} | "
        f"{_fmt(o['att_acc'], pct=True):>8} | {_fmt(o['base_acc'], pct=True):>8} | "
        f"{o['b']:>9} | {o['c']:>9} | {_fmt(o['p_value'])}")
    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 6: Attention vs. Prompt Baseline (McNemar's test)")
    ap.add_argument("--artifacts-dir", default=str(Config().paths.artifacts_dir))
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    manifest_path = artifacts_dir / "manifest.json"

    manifest = json.loads(manifest_path.read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")

    report = baseline_accuracy_report(manifest, labels)
    print(f"Experiment 6 -- Attention vs Prompt Baseline ({artifacts_dir}, annotator={args.annotator})\n")
    print(format_report(report))
    print("\nattn>base: discordant pairs where attention metric was correct but prompt baseline failed.")
    print("base>attn: discordant pairs where prompt baseline was correct but attention metric failed.")
    print("p-value: exact McNemar's test (two-sided).")

if __name__ == "__main__":
    main()
