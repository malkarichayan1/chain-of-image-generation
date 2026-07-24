#!/usr/bin/env python
"""
Stage 2 of the metric-A human-agreement anchor set: BLIND human labeling.

Iterates the detected images in the manifest in a fixed randomized order and, for each
(image, attribute), opens the image and asks you which subject owns that attribute -- or
`none` (never rendered) / `unclear`. The model's own prediction is NEVER shown, so your
judgment cannot anchor on it; that blindness is what makes the resulting agreement number
meaningful. Every answer is written to labels_<annotator>.json immediately, so you can stop
and resume; already-answered (prompt_id, attribute) pairs are skipped.

Local CPU only (PIL + stdlib). Run from inside ssa/anchor_set/:
    py -3 label_images.py --annotator chayan
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from anchor_common import (
    LABEL_NONE, LABEL_UNCLEAR, label_key, load_labels, save_labels, pending_label_targets,
)

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
SHUFFLE_SEED = 20260723  # fixed so the labeling order is reproducible across resume sessions


def labels_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"labels_{annotator}.json"


def _open_image(path: str) -> None:
    """Best-effort open in the OS default viewer; fall back to PIL. Never fatal -- if the
    image can't be shown, the annotator can open it manually from the printed path.

    manifest.json's image_path strings were written on Kaggle (Linux), so they use forward
    slashes, e.g. "artifacts/images/p6.png". A plain relative forward-slash path passed to
    os.startfile() invokes ShellExecuteW, which -- unlike ordinary file I/O -- can fail to
    resolve that shape of path on Windows (WinError 2) even though the file exists.
    Resolving to an absolute Path first sidesteps that: Path.resolve() both makes it
    absolute and normalizes the separators to backslashes."""
    abs_path = str(Path(path).resolve())
    try:
        if sys.platform.startswith("win"):
            os.startfile(abs_path)  # type: ignore[attr-defined]
            return
        from PIL import Image
        Image.open(abs_path).show()
    except Exception as e:  # noqa: BLE001 -- showing an image must never crash labeling
        print(f"  (could not auto-open image: {type(e).__name__}: {e})")
        print(f"  open it manually: {abs_path}")


def build_menu(subjects: List[str]) -> Tuple[Dict[str, str], str]:
    """Map single-key inputs to labels: 1..n -> subjects, plus n(one)/u(nclear). Returns
    (choice_map, help_text)."""
    choice_map: Dict[str, str] = {str(i + 1): s for i, s in enumerate(subjects)}
    choice_map[LABEL_NONE[0]] = LABEL_NONE        # 'n'
    choice_map[LABEL_UNCLEAR[0]] = LABEL_UNCLEAR  # 'u'
    lines = [f"    {i + 1}) {s}" for i, s in enumerate(subjects)]
    lines.append(f"    n) {LABEL_NONE} (attribute not visibly rendered on anyone)")
    lines.append(f"    u) {LABEL_UNCLEAR} (rendered but ownership ambiguous)")
    return choice_map, "\n".join(lines)


def prompt_one(img: dict, attr: dict, choice_map: Dict[str, str], menu: str,
               input_fn=input) -> str:
    """Ask for one judgment; re-ask until a valid key is entered. `input_fn` is injected so
    tests drive it without a real console."""
    print(f"\n[p{img['prompt_id']}] {img['prompt']}")
    print(f"  Which subject is wearing/holding the '{attr['attribute']}'?")
    print(menu)
    while True:
        raw = input_fn("  > ").strip().lower()
        if raw in choice_map:
            return choice_map[raw]
        print(f"  invalid choice {raw!r}; pick one of {sorted(choice_map)}")


def run(annotator: str, input_fn=input) -> Dict[str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    lpath = labels_path(annotator)
    labels = load_labels(lpath)

    targets = pending_label_targets(manifest, labels)
    random.Random(SHUFFLE_SEED).shuffle(targets)  # de-correlate order from prompt_id

    total = sum(len(i["attributes"]) for i in manifest["images"] if i.get("detected"))
    if not targets:
        print(f"All {total} judgments already recorded in {lpath}. Nothing to do.")
        return labels

    print(f"Annotator: {annotator} | {len(targets)} of {total} judgments remaining "
          f"| labels file: {lpath}")
    print("Answers save after each entry; Ctrl-C to stop and resume later.\n")

    for img, attr in targets:
        _open_image(img["image_path"])
        subjects = img["subjects"]
        choice_map, menu = build_menu(subjects)
        answer = prompt_one(img, attr, choice_map, menu, input_fn=input_fn)
        labels[label_key(img["prompt_id"], attr["attribute"])] = answer
        save_labels(lpath, labels)

    print(f"\nDone. {len(labels)} judgments recorded in {lpath}.")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind human labeling for the metric-A anchor set")
    ap.add_argument("--annotator", required=True,
                    help="short id for this annotator, e.g. 'chayan' -> labels_chayan.json")
    args = ap.parse_args()
    run(args.annotator)


if __name__ == "__main__":
    main()
