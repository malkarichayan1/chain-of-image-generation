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
from typing import Dict, List, Optional, Tuple

from anchor_common import (
    COUNT_BROKEN, COUNT_CLEAN, LABEL_ABSENT, LABEL_NONE, LABEL_SHARED, LABEL_UNCLEAR,
    count_key, label_key, load_labels, parse_prompt_layout, redact_attribute_clause,
    save_labels, pending_label_targets, resolve_image_path,
)

ARTIFACTS_DIR = Path("artifacts")
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
SHUFFLE_SEED = 20260723  # fixed so the labeling order is reproducible across resume sessions

# 2026-07-30 labeling guidelines (FLUX round: chayan/grace/akhil, full double coverage).
# Printed once at the start of every run() call -- including a resume with nothing
# pending -- so it's the first thing an annotator sees whenever they launch the tool, not
# just something they were told once in a message. Wording matches the official guideline
# text verbatim. `Wrong Subject` is not a new stored value -- it's the existing behavior of
# picking a non-intended subject from the menu, now named and explained so annotators don't
# read "which subject has X" as "does the intended subject have X". `Subject Absent / Not
# Evaluable` IS new: see LABEL_ABSENT in anchor_common.py.
GUIDELINES_BANNER = """\
============================================================
 Labeling Guidelines for Workstream 2 (REVISED FOR FLUX)
============================================================
COUNT CHECK (once per image):
  Count-Clean  = the image renders the exact number of distinct subjects
                 requested in the prompt.
  Count-Broken = the image is missing subjects, merges subjects together,
                 adds extra subjects, or duplicates a subject. Small or
                 out-of-focus background bystanders do NOT count as extra
                 subjects unless they duplicate a primary character.

ATTRIBUTE CHECK (once per attribute):
  Present        = the requested attribute is clearly visible and clearly
                    attaches to ONLY the correct subject.
  Wrong Subject  = the attribute is rendered, but attached to the wrong
                    person. Pick whoever ACTUALLY has it from the menu --
                    that records the Observed Owner.
  Missing        = the attribute is completely absent from the image. If
                    you cannot confidently confirm OR rule it out, use
                    Unclear rather than Missing.
  Shared         = the attribute accidentally leaks onto / is shared by
                    multiple subjects, one of which is the correct subject.
  Unclear        = you can't definitively confirm if the attribute is
                    present or who owns it. Includes: the item sits right
                    between two people, subject boundaries overlap, colors
                    are borderline, professional identities can't be told
                    apart, or the item is mostly cut off / hidden.
  Subject Absent / Not Evaluable
                 = the intended subject was never drawn in the image
                    (Count-Broken). Use this INSTEAD of Missing so count
                    failures don't get mixed up with binding failures.

Example -- prompt: "A barista in a red apron and a chef in a blue hat"
  Count check : both a barista AND a chef shown -> Count-Clean.
                only one person shown -> Count-Broken.
  Attribute   : barista has a red apron -> Present.
                chef is wearing the red apron instead -> Wrong Subject
                  (pick "chef" -- that's the Observed Owner).
                neither person has an apron -> Missing.
                both wearing red aprons -> Shared.
                chef's hat completely hidden off-screen -> Unclear.
                the barista was never drawn in the image -> Subject
                  Absent / Not Evaluable.
============================================================
"""


def labels_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"labels_{annotator}.json"


def counts_path(annotator: str) -> Path:
    return ARTIFACTS_DIR / f"counts_{annotator}.json"


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


def build_menu(subjects: List[str],
               layout: Optional[List[Tuple[str, Optional[str]]]] = None,
               attribute: Optional[str] = None,
               ) -> Tuple[Dict[str, str], str]:
    """Map single-key inputs to labels: 1..n -> subjects, plus n(one)/u(nclear)/s(hared)/
    a(bsent). Returns (choice_map, help_text).

    `shared` is distinct from `unclear` on purpose. "It's painted on three of them" is a
    binding OUTCOME the metric can be scored against (see anchor_common.margin_from_scores);
    "I can't tell who has it" is missing data. Folding both into `unclear`, as the first
    labeling passes did, throws the former away. `absent` is distinct from `none` the same
    way: the intended SUBJECT never rendered at all (a count failure), not just this
    attribute never showing up on a subject who does exist.

    `layout` (optional, from anchor_common.parse_prompt_layout): when given, each subject's
    menu line is tagged with its position (e.g. "[far left]") and, only when the subject
    name doesn't literally appear in its own prompt segment, a gloss of how the prompt
    actually phrases it (surfaces cases like FLUX's "cyclist" -> "a man wearing a cycling
    jersey"). None (the default, and what every pre-FLUX caller still passes) renders
    exactly as before -- no tags.

    `attribute` (optional): the attribute THIS question is asking about. When given, any
    gloss is passed through anchor_common.redact_attribute_clause first -- otherwise a
    subject's own gloss can spell out the answer to the very question being asked (e.g.
    cyclist's gloss mentions their "yellow bike helmet", which is exactly what "which
    subject has the yellow helmet?" is trying to find out blind)."""
    choice_map: Dict[str, str] = {str(i + 1): s for i, s in enumerate(subjects)}
    choice_map[LABEL_NONE[0]] = LABEL_NONE        # 'n'
    choice_map[LABEL_UNCLEAR[0]] = LABEL_UNCLEAR  # 'u'
    choice_map[LABEL_SHARED[0]] = LABEL_SHARED    # 's'
    choice_map[LABEL_ABSENT[0]] = LABEL_ABSENT    # 'a'
    lines = ["    (pick whoever ACTUALLY has it -- correct subject = Present, "
              "wrong subject = Wrong Subject:)"]
    for i, subject in enumerate(subjects):
        tag = ""
        if layout is not None:
            position, gloss = layout[i]
            tag = f"   [{position}]"
            if gloss:
                if attribute is not None:
                    gloss = redact_attribute_clause(gloss, attribute)
                tag += f' (prompt calls them: "{gloss}")'
        lines.append(f"    {i + 1}) {subject}{tag}")
    lines.append(f"    n) {LABEL_NONE} / Missing (attribute completely absent from the image)")
    lines.append(f"    s) {LABEL_SHARED} / Shared (leaks onto / is shared by multiple subjects)")
    lines.append(f"    u) {LABEL_UNCLEAR} / Unclear (too blurry/occluded/low-quality to confirm)")
    lines.append(f"    a) {LABEL_ABSENT} / Subject Absent -- Not Evaluable "
                  "(intended subject was never drawn; use instead of Missing)")
    return choice_map, "\n".join(lines)


def prompt_count_clean(img: dict, input_fn=input) -> str:
    """Asked ONCE per image, before its attribute questions: did every intended subject
    render as a visually DISTINCT person? Passing Mask R-CNN's headcount check is not
    enough -- two subjects can render as duplicate/interchangeable people (see
    anchor_common.COUNT_CLEAN's docstring), which makes the binding question for that image
    ill-posed no matter how good the metric is. Re-asks until 'y' or 'n' is entered.

    Shows the full prompt (previously omitted entirely -- there was no text to check the
    rendered subject count/order against) plus, when parse_prompt_layout can read the
    prompt's positional markers, each subject tagged with its position and, where the
    prompt's own wording doesn't contain the subject name, a clarifying note."""
    print(f"\n[p{img['prompt_id']}] COUNT CHECK")
    print(f"  prompt: {img['prompt']}")
    layout = parse_prompt_layout(img["prompt"], img["subjects"])
    if layout is not None:
        tagged = [f"{s} [{pos}]" for s, (pos, _gloss) in zip(img["subjects"], layout)]
        print(f"  intended subjects (left -> right): {', '.join(tagged)}")
        for subject, (_pos, gloss) in zip(img["subjects"], layout):
            if gloss:
                print(f'    note: prompt calls the "{subject}" -> "{gloss}"')
    else:
        print(f"  intended subjects: {', '.join(img['subjects'])}")
    print("  Does the image render the EXACT number of distinct subjects requested --")
    print("  not missing anyone, not merging two subjects into one, not adding extras?")
    print("    y) yes -- Count-Clean")
    print("    n) no  -- Count-Broken (missing subjects, merged subjects, or extra subjects)")
    # The "count" marker in the prompt string lets callers/tests distinguish this y/n
    # question from prompt_one's subject question (both otherwise share the "> " prompt).
    while True:
        raw = input_fn("  [count: y/n] > ").strip().lower()
        if raw == "y":
            return COUNT_CLEAN
        if raw == "n":
            return COUNT_BROKEN
        print(f"  invalid choice {raw!r}; enter 'y' or 'n'")


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
    print(GUIDELINES_BANNER)
    manifest = json.loads(MANIFEST_PATH.read_text())
    lpath = labels_path(annotator)
    labels = load_labels(lpath)
    cpath = counts_path(annotator)
    counts = load_labels(cpath)

    targets = pending_label_targets(manifest, labels, relabel)
    random.Random(SHUFFLE_SEED).shuffle(targets)  # de-correlate order from prompt_id

    total = sum(len(i["attributes"]) for i in manifest["images"] if i.get("detected"))
    if not targets:
        print(f"All {total} judgments already recorded in {lpath}. Nothing to do.")
        return labels

    scope = f" (re-examining {sorted(relabel)} rows)" if relabel else ""
    print(f"Annotator: {annotator} | {len(targets)} of {total} judgments remaining{scope} "
          f"| labels file: {lpath} | counts file: {cpath}")
    print("Answers save after each entry; Ctrl-C to stop and resume later.\n")

    for img, attr in targets:
        _open_image(resolve_image_path(ARTIFACTS_DIR, img["image_path"]))

        # Count-clean check happens ONCE per image, the first time any of its attributes
        # comes up in this (shuffled) pass -- not once per attribute. Skipped entirely in a
        # --relabel pass: relabel narrowly re-asks specific attribute labels (e.g. splitting
        # unclear -> shared) and must not disturb the count judgments settled in the original
        # full pass.
        ckey = count_key(img["prompt_id"])
        if not relabel and ckey not in counts:
            counts[ckey] = prompt_count_clean(img, input_fn=input_fn)
            save_labels(cpath, counts)

        subjects = img["subjects"]
        layout = parse_prompt_layout(img["prompt"], subjects)
        choice_map, menu = build_menu(subjects, layout, attr["attribute"])
        answer = prompt_one(img, attr, choice_map, menu, input_fn=input_fn)
        labels[label_key(img["prompt_id"], attr["attribute"])] = answer
        save_labels(lpath, labels)

    print(f"\nDone. {len(labels)} judgments recorded in {lpath}, "
          f"{len(counts)} images count-checked in {cpath}.")
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
                    choices=[LABEL_NONE, LABEL_UNCLEAR, LABEL_SHARED, LABEL_ABSENT],
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
