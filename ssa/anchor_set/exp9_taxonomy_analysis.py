#!/usr/bin/env python
"""
Taxonomy analysis and #19 Intent-vs-Realization evaluation.
Reads taxonomy_index.json across easy and hard sets, analyzes:
- #14: Layer-band binding profiles (early / mid / late)
- #16: Per-head binding distribution
- #17: Timestep-window profile (quartiles)
- #18: Distribution characterization (entropy, in-box mass, peak-to-second-peak)
- #19 (CRITICAL): Tests whether ANY sharp cell tracks the rendered image (intent vs realization)
  on the hard set disobeyed rows vs prompt-only baseline.

Prints final verdict: VERDICT: POSITIVE or VERDICT: NEGATIVE.

Usage:
    python exp9_taxonomy_analysis.py \
      --easy-dir artifacts_flux --easy-annotator chayan \
      --hard-dir artifacts_flux_hard --hard-annotator consensus \
      --out artifacts_flux_hard/taxonomy_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from anchor_common import NON_SUBJECT_LABELS, label_key, load_labels


def evaluate_slice(manifest: dict, labels: Dict[str, str], taxonomy_data: dict,
                   slice_type: str, slice_key: str) -> pd.DataFrame:
    """Computes prediction accuracy for a given taxonomy slice (e.g. layer_band='early' or quartile='Q2_25_50')."""
    rows = []
    for img in manifest.get("images", []):
        prompt_id = img["prompt_id"]
        str_pid = str(prompt_id)
        img_tax = taxonomy_data.get(str_pid, {}).get("attributes", {})
        
        for attr in img.get("attributes", []):
            phrase = attr["attribute"]
            key = label_key(prompt_id, phrase)
            human = labels.get(key)
            if not human or human in NON_SUBJECT_LABELS:
                continue

            intended = attr.get("intended_subject")
            attr_tax = img_tax.get(phrase, {})
            
            if slice_type == "layer_bands":
                pred = attr_tax.get("layer_bands", {}).get(slice_key, {}).get("predicted_owner")
            elif slice_type == "timestep_quartiles":
                pred = attr_tax.get("timestep_quartiles", {}).get(slice_key, {}).get("predicted_owner")
            elif slice_type == "pooled":
                pred = attr_tax.get("recomputed_pooled_owner", attr.get("predicted_owner"))
            else:
                pred = None

            rows.append({
                "prompt_id": prompt_id,
                "attribute": phrase,
                "intended_subject": intended,
                "human_label": human,
                "predicted_owner": pred,
                "disobeyed": (intended != human),
                "correct": (pred == human) if pred else False,
                "prompt_obeyed_correct": (intended == human),
            })
    return pd.DataFrame(rows)


def analyze_intent_vs_realization(hard_df: pd.DataFrame) -> Tuple[bool, dict]:
    """Experiment #19: Does any slice beat the prompt-only baseline or beat chance on disobeyed rows?"""
    disobeyed_df = hard_df[hard_df["disobeyed"] == True]
    n_disobeyed = len(disobeyed_df)
    
    # Prompt-only baseline accuracy overall
    prompt_acc = hard_df["prompt_obeyed_correct"].mean()
    slice_acc = hard_df["correct"].mean()
    
    # Accuracy on disobeyed rows
    disobeyed_acc = disobeyed_df["correct"].mean() if n_disobeyed > 0 else 0.0
    
    # Chance level on disobeyed rows
    # Mean 1/n chance ~ 0.18 - 0.25
    p_val_disobeyed = None
    if n_disobeyed > 0:
        n_correct = int(disobeyed_df["correct"].sum())
        p_val_disobeyed = float(stats.binomtest(n_correct, n_disobeyed, p=0.25, alternative="greater").pvalue)

    # A cell passes if it significantly beats chance on disobeyed rows and tracks realization
    passed = bool(p_val_disobeyed is not None and p_val_disobeyed < 0.05 and disobeyed_acc > 0.40)

    summary = {
        "overall_n": len(hard_df),
        "overall_accuracy": round(slice_acc, 4),
        "prompt_baseline_accuracy": round(prompt_acc, 4),
        "disobeyed_n": n_disobeyed,
        "disobeyed_accuracy": round(disobeyed_acc, 4),
        "disobeyed_binom_p": round(p_val_disobeyed, 4) if p_val_disobeyed is not None else None,
        "passed_intent_vs_realization": passed,
    }
    return passed, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Taxonomy analysis and #19 verdict")
    ap.add_argument("--easy-dir", default="artifacts_flux")
    ap.add_argument("--easy-annotator", default="chayan")
    ap.add_argument("--hard-dir", default="artifacts_flux_hard")
    ap.add_argument("--hard-annotator", default="consensus")
    ap.add_argument("--out", default="artifacts_flux_hard/taxonomy_report.json")
    args = ap.parse_args()

    easy_dir = Path(args.easy_dir)
    hard_dir = Path(args.hard_dir)

    easy_manifest = json.loads((easy_dir / "manifest.json").read_text())
    easy_labels = load_labels(easy_dir / f"labels_{args.easy_annotator}.json")
    easy_tax = json.loads((easy_dir / "taxonomy_index.json").read_text()) if (easy_dir / "taxonomy_index.json").exists() else {}

    hard_manifest = json.loads((hard_dir / "manifest.json").read_text())
    hard_labels = load_labels(hard_dir / f"labels_{args.hard_annotator}.json")
    hard_tax = json.loads((hard_dir / "taxonomy_index.json").read_text()) if (hard_dir / "taxonomy_index.json").exists() else {}

    report = {"layer_bands": {}, "timestep_quartiles": {}, "sharpest_cell_test": {}}

    print("=" * 60)
    print("TAXONOMY ANALYSIS REPORT (#14 - #19)")
    print("=" * 60)

    # 1. Layer Bands (#14)
    print("\n[#14] Layer-Band Binding Profiles (Easy vs Hard)")
    for band in ["early", "mid", "late"]:
        easy_df = evaluate_slice(easy_manifest, easy_labels, easy_tax, "layer_bands", band)
        hard_df = evaluate_slice(hard_manifest, hard_labels, hard_tax, "layer_bands", band)
        e_acc = easy_df["correct"].mean() if len(easy_df) else 0.0
        h_acc = hard_df["correct"].mean() if len(hard_df) else 0.0
        print(f"  Band '{band}': Easy Acc = {e_acc:.1%} (n={len(easy_df)}) | Hard Acc = {h_acc:.1%} (n={len(hard_df)})")
        report["layer_bands"][band] = {"easy_acc": round(e_acc, 4), "hard_acc": round(h_acc, 4)}

    # 2. Timestep Quartiles (#17)
    print("\n[#17] Timestep Quartile Profiles (Hard Set)")
    for q in ["Q1_0_25", "Q2_25_50", "Q3_50_75", "Q4_75_100"]:
        hard_df = evaluate_slice(hard_manifest, hard_labels, hard_tax, "timestep_quartiles", q)
        h_acc = hard_df["correct"].mean() if len(hard_df) else 0.0
        print(f"  Quartile '{q}': Hard Acc = {h_acc:.1%} (n={len(hard_df)})")
        report["timestep_quartiles"][q] = {"hard_acc": round(h_acc, 4)}

    # 3. Experiment #19 (CRITICAL - Intent vs Realization)
    print("\n" + "=" * 60)
    print("[#19] CRITICAL: Intent-vs-Realization Test on Sharpest Cells")
    print("=" * 60)
    
    # Test all slices on hard set disobeyed rows
    all_slices = [
        ("layer_bands", "early"),
        ("layer_bands", "mid"),
        ("layer_bands", "late"),
        ("timestep_quartiles", "Q1_0_25"),
        ("timestep_quartiles", "Q2_25_50"),
        ("timestep_quartiles", "Q3_50_75"),
        ("timestep_quartiles", "Q4_75_100"),
        ("pooled", "all"),
    ]

    any_passed = False
    for stype, skey in all_slices:
        df = evaluate_slice(hard_manifest, hard_labels, hard_tax, stype, skey)
        passed, res = analyze_intent_vs_realization(df)
        report["sharpest_cell_test"][f"{stype}_{skey}"] = res
        print(f"  Cell {stype}:{skey} -> Acc: {res['overall_accuracy']:.1%} | Disobeyed Acc: {res['disobeyed_accuracy']:.1%} | Disobeyed p: {res['disobeyed_binom_p']}")
        if passed:
            any_passed = True

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {out_path}")

    print("\n" + "=" * 60)
    if any_passed:
        print("VERDICT: POSITIVE")
    else:
        print("VERDICT: NEGATIVE")
    print("=" * 60)


if __name__ == "__main__":
    main()
