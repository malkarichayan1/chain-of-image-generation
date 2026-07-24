#!/usr/bin/env python
"""
Pure, torch-free logic shared by the anchor-set's local stages (label_images.py,
analyze_agreement.py) and exercised by the tests. Imports only numpy + stdlib, so it runs
on this machine where torch/diffusers are NOT installed.

generate_anchor_images.py (the GPU/Kaggle single-file kernel) inline-duplicates
`predicted_owner_from_attention` / `build_manifest_entry` rather than importing this module,
because a Kaggle script kernel is one file -- the same "keep in sync" duplication pattern
generate_chains.py already uses for its attention capture. The copies here are the tested
ones; keep the two in sync.

Ground-truth framing: a prompt's *intended* subject->attribute pairing is NOT ground truth.
SD1.5 mis-binds, and that is the phenomenon under test. The human label on the rendered
pixels is ground truth; `predicted_owner` is the metric's guess; agreement between them is
the result the memo's decision gate reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Sentinels a human may record instead of a subject name.
LABEL_NONE = "none"        # the attribute never visibly rendered on anyone
LABEL_UNCLEAR = "unclear"  # rendered but the annotator cannot assign ownership
NON_SUBJECT_LABELS = frozenset({LABEL_NONE, LABEL_UNCLEAR})


# ---------------------------------------------------------------------------
# Prompt-spec loading / validation
# ---------------------------------------------------------------------------

def load_specs(path: Path) -> List[dict]:
    """Read prompt_specs.json and return its `prompts` list (ignores the `_comment` key)."""
    data = json.loads(Path(path).read_text())
    return list(data["prompts"])


def validate_specs(specs: Sequence[dict]) -> None:
    """Raise ValueError on any malformed spec. Enforced invariants: stratification
    (>=1 prompt at each of n=2/3/4), unique ids, n matches pair count, and distinct
    subjects + distinct attributes within a prompt (a repeated subject would make the
    binding question ill-posed; a repeated attribute would make ownership ambiguous)."""
    ids = [s["id"] for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate prompt ids: {ids}")
    strata = {s["n"] for s in specs}
    for required in (2, 3, 4):
        if required not in strata:
            raise ValueError(f"no prompt at stratum n={required}")
    for s in specs:
        pairs = s["pairs"]
        if s["n"] != len(pairs):
            raise ValueError(f"prompt {s['id']}: n={s['n']} but {len(pairs)} pairs")
        subjects = [p[0] for p in pairs]
        attributes = [p[1] for p in pairs]
        if len(set(subjects)) != len(subjects):
            raise ValueError(f"prompt {s['id']}: repeated subject in {subjects}")
        if len(set(attributes)) != len(attributes):
            raise ValueError(f"prompt {s['id']}: repeated attribute in {attributes}")
        for _subject, attribute in pairs:
            if attribute not in s["prompt"]:
                raise ValueError(
                    f"prompt {s['id']}: attribute {attribute!r} not a substring of prompt "
                    f"(token lookup on Kaggle would fail): {s['prompt']!r}")


# ---------------------------------------------------------------------------
# Prediction from attention (DUPLICATED inline in generate_anchor_images.py)
# ---------------------------------------------------------------------------

def mean_mass_in_box(attn_map: np.ndarray, box: Sequence[float]) -> float:
    """Mean attention value inside an [x0, y0, x1, y1] pixel box. Returns 0.0 for a box
    that is empty or entirely off the map, so it can never spuriously win the argmax."""
    h, w = attn_map.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(attn_map[y0:y1, x0:x1].mean())


def predicted_owner_from_attention(
    attn_map: np.ndarray, subject_boxes: Dict[str, Sequence[float]]
) -> Tuple[str, Dict[str, float]]:
    """The metric's binding call for one attribute: the subject whose detected box holds
    the most of that attribute's (early-window) cross-attention mass. Returns
    (predicted_owner, {subject: mass}). Ties break deterministically by the boxes' dict
    order (insertion order), so a rerun on identical inputs is reproducible."""
    if not subject_boxes:
        raise ValueError("subject_boxes is empty; nothing to attribute the attention to")
    scores = {s: mean_mass_in_box(attn_map, b) for s, b in subject_boxes.items()}
    owner = max(scores, key=lambda s: scores[s])
    return owner, scores


# ---------------------------------------------------------------------------
# Manifest entry assembly (DUPLICATED inline in generate_anchor_images.py)
# ---------------------------------------------------------------------------

def build_attribute_entry(attribute: str, intended_subject: str, predicted_owner: str,
                          model_scores: Dict[str, float]) -> dict:
    return dict(attribute=attribute, intended_subject=intended_subject,
                predicted_owner=predicted_owner, model_scores=model_scores)


def build_manifest_entry(prompt_id: int, n: int, prompt: str, subjects: List[str], seed: int,
                         detected: bool, num_people_detected: int, image_path: str,
                         attributes: List[dict]) -> dict:
    """Pure dict assembly, no model calls -- the tested seam of Stage 1. `attributes` is a
    list of build_attribute_entry() dicts (empty when detection failed)."""
    return dict(prompt_id=prompt_id, n=n, prompt=prompt, subjects=subjects, seed=seed,
                detected=detected, num_people_detected=num_people_detected,
                image_path=image_path, attributes=attributes)


# ---------------------------------------------------------------------------
# Labeling helpers
# ---------------------------------------------------------------------------

def label_key(prompt_id: int, attribute: str) -> str:
    """Stable join key for a single (image, attribute) judgment. A second annotator's
    file uses the same keys, so files align row-for-row without rework."""
    return f"{prompt_id}::{attribute}"


def load_labels(path: Path) -> Dict[str, str]:
    if Path(path).exists():
        return dict(json.loads(Path(path).read_text()))
    return {}


def save_labels(path: Path, labels: Dict[str, str]) -> None:
    Path(path).write_text(json.dumps(labels, indent=2, sort_keys=True))


def pending_label_targets(manifest: dict, labels: Dict[str, str]) -> List[Tuple[dict, dict]]:
    """(image_entry, attribute_entry) pairs still needing a human label -- only detected
    images, only attributes not already keyed in `labels`. Order follows the manifest."""
    out: List[Tuple[dict, dict]] = []
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            if label_key(img["prompt_id"], attr["attribute"]) not in labels:
                out.append((img, attr))
    return out


# ---------------------------------------------------------------------------
# Agreement analysis
# ---------------------------------------------------------------------------

def chance_baseline(n: int) -> float:
    """Random-guess accuracy at subject count n: 1/n."""
    return 1.0 / n


def build_agreement_rows(manifest: dict, labels: Dict[str, str]) -> List[dict]:
    """One row per (detected chain, attribute) that has a human label. `human_label` may be
    a subject, `none`, or `unclear`; `scored` is True only when the human named a subject
    (the accuracy denominator). `correct` compares predicted_owner to the human subject."""
    rows: List[dict] = []
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            key = label_key(img["prompt_id"], attr["attribute"])
            if key not in labels:
                continue
            human = labels[key]
            scored = human not in NON_SUBJECT_LABELS
            rows.append(dict(
                prompt_id=img["prompt_id"], n=img["n"], attribute=attr["attribute"],
                intended_subject=attr["intended_subject"], predicted_owner=attr["predicted_owner"],
                human_label=human, scored=scored,
                correct=(scored and attr["predicted_owner"] == human),
            ))
    return rows


def summarize_agreement(rows: Sequence[dict]) -> dict:
    """Overall + per-stratum agreement vs chance, plus coverage (how many labeled rows were
    excludable none/unclear). Strata with zero scored rows report accuracy=None."""
    def _stratum(subset: Sequence[dict]) -> dict:
        scored = [r for r in subset if r["scored"]]
        n_scored = len(scored)
        n_correct = sum(1 for r in scored if r["correct"])
        ns = {r["n"] for r in subset}
        chance = chance_baseline(next(iter(ns))) if len(ns) == 1 else None
        acc = (n_correct / n_scored) if n_scored else None
        return dict(
            n_labeled=len(subset), n_scored=n_scored, n_correct=n_correct,
            n_excluded=len(subset) - n_scored,
            accuracy=acc, chance=chance,
            margin_over_chance=(acc - chance) if (acc is not None and chance is not None) else None,
        )

    by_stratum = {n: _stratum([r for r in rows if r["n"] == n])
                  for n in sorted({r["n"] for r in rows})}
    return dict(overall=_stratum(rows), by_stratum=by_stratum)
