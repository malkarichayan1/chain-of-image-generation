#!/usr/bin/env python
"""
Reframes Part A's accuracy question as a detection question. Exp 1/4 asked "does
predicted_owner match the human label" -- a metric loses that fight to a trivial baseline
whenever the model usually binds correctly, because the baseline always guesses "bound
correctly" and that guess is right most of the time. It also can never win, since the
baseline is 0% on exactly the rows a faithfulness auditor exists to catch.

This asks the question the project actually cares about: among rows where the model
actually MIS-BOUND the attribute (human_label != intended_subject), does the attention
margin have any pulse for catching that? The trivial "always assume correct" baseline is
provably 0% on this framing (see `confusion_report`'s tp+fn == all actual failures), so it
is not a competitor here -- this measures whether there is any signal at all worth building
on, before spending a GPU rerun on a better readout.

Ground truth: `actual_failure` = human_label disagrees with intended_subject (the model
mis-bound, per the human). Prediction: `predicted_failure` = predicted_owner disagrees with
intended_subject (bare argmax, same field Exp 1/4 already use). `signed_confidence` is a new
continuous score for ranking (AUROC) -- see its own docstring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from scipy import stats

from anchor_common import build_agreement_rows, label_key, load_labels

_EPS = 1e-12


def signed_confidence(model_scores: Dict[str, float], intended_subject: str) -> float:
    """How strongly attention agrees the INTENDED subject holds the attribute, signed and
    normalized to roughly [-1, 1]: positive means intended_subject's attention mass beats
    every other subject's (attention "votes" the binding succeeded); negative means some
    other subject's mass is higher (attention "votes" a specific mis-binding). Near zero
    means the two are close -- the metric's own uncertainty region.

    Deliberately NOT the same quantity as anchor_common.margin_from_scores: that compares
    top1 vs. runner-up regardless of who top1 is (a "how decisive is the argmax" scalar).
    This compares intended_subject specifically against its best rival, which is exactly the
    quantity a mis-binding detector needs and margin_from_scores cannot answer once
    predicted_owner != intended_subject (at that point margin describes the WRONG subject's
    lead, not intended_subject's shortfall)."""
    intended_score = model_scores[intended_subject]
    other_scores = [v for k, v in model_scores.items() if k != intended_subject]
    best_other = max(other_scores) if other_scores else 0.0
    denom = max(intended_score, best_other, _EPS)
    return (intended_score - best_other) / denom


def _model_scores_lookup(manifest: dict) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            out[label_key(img["prompt_id"], attr["attribute"])] = attr["model_scores"]
    return out


def build_failure_rows(manifest: dict, labels: Dict[str, str],
                       counts: Dict[str, str] = None) -> List[dict]:
    """anchor_agreement rows augmented with the failure-detection columns. Reuses
    build_agreement_rows for `scored`/`human_label`/`intended_subject`/`predicted_owner` (single
    source of truth for what counts as scored, including the count-clean exclusion) and joins
    back to the manifest only for `model_scores`, which build_agreement_rows doesn't carry."""
    rows = build_agreement_rows(manifest, labels, counts=counts)
    scores_lookup = _model_scores_lookup(manifest)
    out = []
    for r in rows:
        model_scores = scores_lookup[label_key(r["prompt_id"], r["attribute"])]
        row = dict(r)
        row.update(
            model_scores=model_scores,
            signed_confidence=signed_confidence(model_scores, r["intended_subject"]),
            predicted_failure=(r["predicted_owner"] != r["intended_subject"]),
            actual_failure=(r["scored"] and r["human_label"] != r["intended_subject"]),
        )
        out.append(row)
    return out


def confusion_report(rows: Sequence[dict]) -> dict:
    """Detection performance of `predicted_failure` against `actual_failure`, scored rows
    only. `fisher_p_value` tests the null that predicted_failure is independent of
    actual_failure -- "does this have any pulse at all," the weakest bar a useful detector
    must clear."""
    scored = [r for r in rows if r["scored"]]
    n = len(scored)
    tp = sum(1 for r in scored if r["predicted_failure"] and r["actual_failure"])
    fp = sum(1 for r in scored if r["predicted_failure"] and not r["actual_failure"])
    fn = sum(1 for r in scored if not r["predicted_failure"] and r["actual_failure"])
    tn = sum(1 for r in scored if not r["predicted_failure"] and not r["actual_failure"])
    sensitivity = (tp / (tp + fn)) if (tp + fn) else None
    specificity = (tn / (tn + fp)) if (tn + fp) else None
    precision = (tp / (tp + fp)) if (tp + fp) else None
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None else None)
    fisher_p = float(stats.fisher_exact([[tp, fp], [fn, tn]])[1]) if n else None
    return dict(
        n=n, n_actual_failures=tp + fn, n_actual_successes=fp + tn,
        tp=tp, fp=fp, fn=fn, tn=tn,
        sensitivity=sensitivity, specificity=specificity, precision=precision,
        balanced_accuracy=balanced_accuracy, fisher_p_value=fisher_p,
    )


def auroc_failure_detection(rows: Sequence[dict]) -> Optional[float]:
    """AUROC of -signed_confidence for predicting actual_failure, via the rank-sum identity
    (equivalent to Mann-Whitney U / (n_pos*n_neg), avoids scipy.stats.mannwhitneyu's
    direction bookkeeping). None when a class is entirely absent -- no ROC curve exists with
    only one class."""
    scored = [r for r in rows if r["scored"]]
    failures = [-r["signed_confidence"] for r in scored if r["actual_failure"]]
    successes = [-r["signed_confidence"] for r in scored if not r["actual_failure"]]
    if not failures or not successes:
        return None
    ranks = stats.rankdata(failures + successes)
    n_pos, n_neg = len(failures), len(successes)
    rank_sum_failures = ranks[:n_pos].sum()
    return float((rank_sum_failures - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _stratum_report(rows: Sequence[dict]) -> dict:
    report = confusion_report(rows)
    report["auroc"] = auroc_failure_detection(rows)
    return report


def mis_binding_report(manifest: dict, labels: Dict[str, str],
                       counts: Dict[str, str] = None) -> dict:
    rows = build_failure_rows(manifest, labels, counts=counts)
    by_stratum = {n: _stratum_report([r for r in rows if r["n"] == n])
                 for n in sorted({r["n"] for r in rows})}
    return dict(overall=_stratum_report(rows), by_stratum=by_stratum)


def _fmt(x, pct=False):
    if x is None:
        return "  -  "
    return f"{x*100:5.1f}%" if pct else f"{x:.4f}"


def format_report(report: dict) -> str:
    header = (f"{'stratum':>8} | {'n':>4} | {'failures':>8} | {'sens':>7} | {'spec':>7} | "
              f"{'prec':>7} | {'bal.acc':>8} | {'auroc':>7} | {'fisher p':>8}")
    sep = "-" * len(header)
    lines = [header, sep]
    for n in sorted(report["by_stratum"]):
        s = report["by_stratum"][n]
        lines.append(
            f"{('n='+str(n)):>8} | {s['n']:>4} | {s['n_actual_failures']:>8} | "
            f"{_fmt(s['sensitivity'], pct=True)} | {_fmt(s['specificity'], pct=True)} | "
            f"{_fmt(s['precision'], pct=True)} | {_fmt(s['balanced_accuracy'], pct=True)} | "
            f"{_fmt(s['auroc'])} | {_fmt(s['fisher_p_value'])}")
    o = report["overall"]
    lines.append(sep)
    lines.append(
        f"{'overall':>8} | {o['n']:>4} | {o['n_actual_failures']:>8} | "
        f"{_fmt(o['sensitivity'], pct=True)} | {_fmt(o['specificity'], pct=True)} | "
        f"{_fmt(o['precision'], pct=True)} | {_fmt(o['balanced_accuracy'], pct=True)} | "
        f"{_fmt(o['auroc'])} | {_fmt(o['fisher_p_value'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mis-binding detection: does the attention margin catch real binding failures?")
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

    report = mis_binding_report(manifest, labels, counts=counts)
    print(f"Mis-binding detection ({artifacts_dir}, annotator={args.annotator})\n")
    print(format_report(report))
    print("\nsens = fraction of ACTUAL mis-bindings the metric flags (predicted_owner != "
          "intended_subject). spec = fraction of actual successes it correctly leaves alone. "
          "auroc = 0.5 is no better than a coin flip at ranking failures above successes; "
          "1.0 is perfect separation. fisher p tests whether predicted_failure and "
          "actual_failure are associated at all.")


if __name__ == "__main__":
    main()
