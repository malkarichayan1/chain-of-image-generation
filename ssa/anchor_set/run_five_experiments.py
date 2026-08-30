#!/usr/bin/env python
"""
Orchestrator: runs all five Part A experiments in a single pass against one artifacts
directory, prints a combined report, and writes it to `five_experiments_<annotator>.json` +
`.md`. Experiment 2 degrades gracefully (reports "unavailable" instead of crashing) when the
manifest lacks the model_scores_full/predicted_owner_full fields -- so this still completes
4/5 sections against the current artifacts_sdxl/manifest.json today.

NOT YET RUN ON REAL DATA -- scaffolded ahead of Workstream 2 (the expanded,
second-annotator-validated SDXL anchor set). Run against the dummy fixture today:
    py -3 make_dummy_artifacts.py --out artifacts_dummy --seed 0
    py -3 run_five_experiments.py --artifacts-dir artifacts_dummy --annotator dummy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import exp1_accuracy_by_n as exp1
import exp2_window_ablation as exp2
import exp3_attention_scramble as exp3
import exp4_positional_baseline as exp4
import exp5_count_clean_subset as exp5
from anchor_common import build_agreement_rows, load_labels
from config import Config

DEFAULT_N_SEEDS = Config().seeds.sweep_n_seeds


def run_all(manifest: dict, labels: Dict[str, str], counts: Dict[str, str],
           n_seeds: int = DEFAULT_N_SEEDS) -> dict:
    """Runs all five experiments against an already-loaded manifest/labels/counts and returns
    one combined, JSON-serializable dict keyed experiment_1..experiment_5."""
    results: Dict[str, dict] = {}

    results["experiment_1"] = exp1.accuracy_by_n_report(manifest, labels)

    if exp2.is_full_trajectory_available(manifest):
        rows2 = build_agreement_rows(manifest, labels)
        joined2 = exp2.join_window_rows(rows2, exp2.full_scores_by_key(manifest))
        results["experiment_2"] = dict(
            available=True,
            per_stratum=exp2.window_accuracy_report(joined2),
            mcnemar=exp2.mcnemar_early_vs_full(joined2),
        )
    else:
        results["experiment_2"] = dict(available=False, message=exp2.UNAVAILABLE_MESSAGE)

    results["experiment_3"] = dict(
        sweep=exp3.scramble_sweep(manifest, labels, seeds=range(n_seeds)),
        mcnemar=exp3.real_vs_scrambled_mcnemar(manifest, labels),
    )

    rows4 = build_agreement_rows(manifest, labels)
    df4 = exp4.join_baseline(rows4, exp4.prompts_by_id_from_manifest(manifest))
    results["experiment_4"] = dict(
        headline=exp4.baseline_vs_metric_report(df4),
        mcnemar=exp4.mcnemar_report(df4),
        by_stratum=exp4.stratified_report(df4),
    )

    results["experiment_5"] = exp5.count_clean_vs_all_report(manifest, labels, counts)

    return results


def format_combined_report(results: dict) -> str:
    lines = ["# Part A Five-Experiment Battery\n"]

    lines.append("## Experiment 1 -- accuracy per subject count\n")
    lines.append("```\n" + exp1.format_report(results["experiment_1"]) + "\n```\n")

    lines.append("## Experiment 2 -- early-window vs. full-trajectory attention\n")
    e2 = results["experiment_2"]
    if e2["available"]:
        for n, s in sorted(e2["per_stratum"].items()):
            lines.append(f"- n={n}: early={s['early_accuracy']:.3f} full={s['full_accuracy']:.3f} "
                         f"chance={s['chance']:.3f} (n_scored={s['n_scored']})")
        lines.append(f"- McNemar (early vs. full): {e2['mcnemar']}\n")
    else:
        lines.append(e2["message"] + "\n")

    lines.append("## Experiment 3 -- attention-randomization falsification\n")
    e3 = results["experiment_3"]
    for n, s in sorted(e3["sweep"].items()):
        lines.append(f"- n={n}: median_scrambled_accuracy={s['median_accuracy']:.3f} "
                     f"chance={s['chance']:.3f} falsification_clean_fraction="
                     f"{s['frac_not_significantly_different_from_chance']:.3f} "
                     f"(n_seeds={s['n_seeds']})")
    lines.append(f"- McNemar (real vs. scrambled, seed={exp3.MCNEMAR_SEED}): {e3['mcnemar']}\n")

    lines.append("## Experiment 4 -- nearest-subject-noun baseline\n")
    e4 = results["experiment_4"]
    lines.append(f"- Headline: {e4['headline']}")
    lines.append(f"- McNemar: {e4['mcnemar']}")
    for n, s in sorted(e4["by_stratum"].items()):
        lines.append(f"- n={n}: {s}")
    lines.append("")

    lines.append("## Experiment 5 -- count-clean subset\n")
    lines.append("```\n" + exp5.format_report(results["experiment_5"]) + "\n```\n")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run all five Part A experiments against one artifacts directory")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--counts-annotator", default=None,
                    help="annotator id whose counts_<id>.json to use (default: same as --annotator)")
    ap.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                    help="Experiment 3's scramble-sweep seed count (default 200)")
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")
    counts_annotator = args.counts_annotator or args.annotator
    counts = load_labels(artifacts_dir / f"counts_{counts_annotator}.json")

    results = run_all(manifest, labels, counts, n_seeds=args.n_seeds)
    report = format_combined_report(results)
    print(report)

    out_json = artifacts_dir / f"five_experiments_{args.annotator}.json"
    out_md = artifacts_dir / f"five_experiments_{args.annotator}.md"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    out_md.write_text(report)
    print(f"\nWrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
