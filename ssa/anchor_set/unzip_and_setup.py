#!/usr/bin/env python
"""
Unzips gpu_experiment_bundle.zip if present, and verifies all required files are in place.

Usage:
    python unzip_and_setup.py
"""
import os
import zipfile
from pathlib import Path

ZIP_NAME = "gpu_experiment_bundle.zip"

def main():
    if os.path.exists(ZIP_NAME):
        print(f"Found {ZIP_NAME} ({os.path.getsize(ZIP_NAME) / (1024*1024):.1f} MB). Extracting...")
        with zipfile.ZipFile(ZIP_NAME, "r") as z:
            z.extractall(".")
        print("Extraction complete!")
    else:
        print(f"No {ZIP_NAME} found in current directory. Checking existing files...")

    required = [
        "anchor_common.py",
        "flux_attention_capture.py",
        "exp10_clipscore_discriminant.py",
        "vqa_score_flux.py",
        "taxonomy_capture_flux.py",
        "exp9_taxonomy_analysis.py",
        "artifacts_flux/manifest.json",
        "artifacts_flux_hard/manifest.json",
    ]

    all_ok = True
    for item in required:
        if os.path.exists(item):
            print(f"  [OK] {item}")
        else:
            print(f"  [MISSING] {item}")
            all_ok = False

    if all_ok:
        print("\nAll experiment files and datasets are ready to run!")
    else:
        print("\nSome files are missing. Please check the archive extraction.")

if __name__ == "__main__":
    main()
