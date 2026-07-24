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
    LABEL_NONE, LABEL_SHARED, LABEL_UNCLEAR, label_key, load_labels, save_labels,
    pending_label_targets, resolve_image_path,
)

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
SHUFFLE_SEED = 20260723  # fixed so the labeling order is reproducible across resume sessions


def labels_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"labels_{annotator}.json"


def _open_image(abs_path: Path) -> None:
    """Best-effort open in the OS default viewer; fall back to PIL. Never fatal -- if the
    image can't be shown, the annotator can open it manually from the printed path.

    Takes an already-resolved absolute Path (see resolve_image_path() in anchor_common.py,
    called by run() below) -- NOT the raw manifest.json image_path string, which is baked in
    relative to the GENERATING script's own artifacts folder, not wherever this run's
    artifacts_dir actually is locally. os.startfile() also specifically needs an absolute
    path on Windows: it invokes ShellExecuteW, which -- unlike ordinary file I/O -- can fail
    to resolve a relative forward-slash path (WinError 2) even when the file exists."""
    abs_path = str(abs_path)
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
    """Map single-key inputs to labels: 1..n -> subjects, plus n(one)/u(nclear)/s(hared).
    Returns (choice_map, help_text).

    `shared` is distinct from `unclear` on purpose. "It's painted on three of them" is a
    binding OUTCOME the metric can be scored against (see anchor_common.margin_from_scores);
    "I can't tell who has it" is missing data. Folding both into `unclear`, as the first
    labeling passes did, throws the former away."""
    choice_map: Dict[str, str] = {str(i + 1): s for i, s in enumerate(subjects)}
    choice_map[LABEL_NONE[0]] = LABEL_NONE        # 'n'
    choice_map[LABEL_UNCLEAR[0]] = LABEL_UNCLEAR  # 'u'
    choice_map[LABEL_SHARED[0]] = LABEL_SHARED    # 's'
    lines = [f"    {i + 1}) {s}" for i, s in enumerate(subjects)]
    lines.append(f"    n) {LABEL_NONE} (attribute not visibly rendered on anyone)")
    lines.append(f"    s) {LABEL_SHARED} (visibly rendered on MORE THAN ONE subject)")
    lines.append(f"    u) {LABEL_UNCLEAR} (rendered, but you genuinely cannot tell whose)")
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


def run(annotator: str, input_fn=input, relabel: frozenset = frozenset()) -> Dict[str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    lpath = labels_path(annotator)
    labels = load_labels(lpath)

    targets = pending_label_targets(manifest, labels, relabel)
    random.Random(SHUFFLE_SEED).shuffle(targets)  # de-correlate order from prompt_id

    total = sum(len(i["attributes"]) for i in manifest["images"] if i.get("detected"))
    if not targets:
        print(f"All {total} judgments already recorded in {lpath}. Nothing to do.")
        return labels

    scope = f" (re-examining {sorted(relabel)} rows)" if relabel else ""
    print(f"Annotator: {annotator} | {len(targets)} of {total} judgments remaining{scope} "
          f"| labels file: {lpath}")
    print("Answers save after each entry; Ctrl-C to stop and resume later.\n")

    for img, attr in targets:
        _open_image(resolve_image_path(ARTIFACTS_DIR, img["image_path"]))
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
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="directory holding manifest.json / labels files, e.g. "
                         "'artifacts_sdxl' for the SDXL run, so it never touches the SD1.5 "
                         "run's data in 'artifacts' (default)")
    ap.add_argument("--relabel", nargs="+", default=[],
                    choices=[LABEL_NONE, LABEL_UNCLEAR, LABEL_SHARED],
                    help="re-ask only the rows currently carrying these labels, leaving "
                         "settled subject judgments untouched. Use '--relabel unclear' to "
                         "split an older pass's ambiguous rows into shared vs genuinely "
                         "unclear without redoing the whole set.")
    args = ap.parse_args()
    global ARTIFACTS_DIR, MANIFEST_PATH
    ARTIFACTS_DIR = Path(args.artifacts_dir)
    MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
    run(args.annotator, relabel=frozenset(args.relabel))


if __name__ == "__main__":
    main()
