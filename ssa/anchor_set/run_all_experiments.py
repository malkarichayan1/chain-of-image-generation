#!/usr/bin/env python
"""
Automated master runner for all remaining experiments:
1. Recomputes boxes for artifacts_flux_hard (recompute_boxes.py)
2. Experiment #8: CLIPScore discriminant (exp10_clipscore_discriminant.py)
3. Experiment #31: VQAScore on hard set (vqa_score_flux.py + vqa_agreement_check.py)
4. Experiments #14/#16/#17/#18: Taxonomy Capture on Easy set (taxonomy_capture_flux.py)
5. Experiments #14/#16/#17/#18: Taxonomy Capture on Hard set (taxonomy_capture_flux.py)
6. Experiment #19: Taxonomy Analysis & Critical Verdict (exp9_taxonomy_analysis.py)

Usage:
    python run_all_experiments.py
"""
import os
import subprocess
import sys
import time

COMMANDS = [
    ("Recompute Subject Boxes (Hard Set)", "python recompute_boxes.py --artifacts-dir artifacts_flux_hard"),
    ("Experiment #8: CLIPScore Discriminant", "python exp10_clipscore_discriminant.py --artifacts-dir artifacts_flux_hard --annotator consensus"),
    ("Experiment #31: VQAScore GPU Scoring", "python vqa_score_flux.py --artifacts-dir artifacts_flux_hard"),
    ("Experiment #31: VQAScore Agreement Check", "python vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus"),
    ("Taxonomy Capture: Easy Set (artifacts_flux)", "python taxonomy_capture_flux.py --artifacts-dir artifacts_flux"),
    ("Taxonomy Capture: Hard Set (artifacts_flux_hard)", "python taxonomy_capture_flux.py --artifacts-dir artifacts_flux_hard"),
    ("Experiment #19: Final Taxonomy Analysis & Verdict", "python exp9_taxonomy_analysis.py --easy-dir artifacts_flux --easy-annotator annotator1 --hard-dir artifacts_flux_hard --hard-annotator consensus --out artifacts_flux_hard/taxonomy_report.json"),
]

def main():
    print("=" * 70)
    print("STARTING COMPLETE AUTOMATED EXPERIMENT PIPELINE")
    print("=" * 70)
    total_start = time.time()

    for i, (title, cmd) in enumerate(COMMANDS, 1):
        print(f"\n[{i}/{len(COMMANDS)}] >>> {title}")
        print(f"Command: {cmd}")
        print("-" * 70)
        
        step_start = time.time()
        res = subprocess.run(cmd, shell=True)
        step_elapsed = time.time() - step_start

        if res.returncode != 0:
            print(f"\n[ERROR] Step {i} failed with return code {res.returncode}")
            sys.exit(res.returncode)
        
        print(f"Completed in {step_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS SUCCESSFULLY FINISHED in {total_elapsed / 60:.1f} minutes!")
    print("=" * 70)

if __name__ == "__main__":
    main()
