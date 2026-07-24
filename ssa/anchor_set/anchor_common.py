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
LABEL_SHARED = "shared"    # rendered on MORE THAN ONE subject (attribute leakage)
NON_SUBJECT_LABELS = frozenset({LABEL_NONE, LABEL_UNCLEAR, LABEL_SHARED})

# `shared` is deliberately split out of `unclear`. Both are "not one subject", but they are
# different facts: `unclear` is missing data (the annotator could not tell), while `shared`
# is a binding OUTCOME (the model bound the attribute to several subjects at once) that the
# metric can be asked to predict. Strict scoring excludes all three -- preserving every
# already-published accuracy number -- while shared-aware scoring (see `margin_threshold`
# in build_agreement_rows) readmits only `shared` as an extra predictable class.
MISSING_DATA_LABELS = frozenset({LABEL_NONE, LABEL_UNCLEAR})


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
# Confidence margin -> the metric's own "shared" (leakage) call
# ---------------------------------------------------------------------------

def margin_from_scores(model_scores: Dict[str, float]) -> float:
    """How decisively the top subject beat the runner-up: (top1 - top2) / top1, in [0, 1].

    A bare argmax always names somebody, even when two subjects hold near-identical
    attention mass -- which is exactly the leakage case a human marks `shared`. This is the
    scalar that lets the metric abstain instead. Computed from the `model_scores` already
    stored in every manifest attribute entry, so re-thresholding costs no GPU time.

    Degenerate cases: a lone subject has no competitor, so it scores 1.0 (nothing to be
    ambiguous against); an all-zero map has no evidence for anyone, so it scores 0.0
    (maximally ambiguous, NOT maximally confident -- getting this backwards would turn
    empty attention into a confident prediction)."""
    if not model_scores:
        raise ValueError("model_scores is empty; no prediction to measure confidence for")
    ranked = sorted(model_scores.values(), reverse=True)
    if len(ranked) == 1:
        return 1.0
    top1, top2 = ranked[0], ranked[1]
    if top1 <= 0.0:
        return 0.0
    return float((top1 - top2) / top1)


def predicted_label_with_shared(model_scores: Dict[str, float],
                                margin_threshold: float) -> str:
    """The metric's binding call when it is allowed to say `shared`: the argmax subject when
    its margin clears `margin_threshold`, else LABEL_SHARED. The boundary is inclusive on the
    confident side (margin >= threshold names a subject) so threshold=0.0 reproduces plain
    argmax behavior exactly."""
    if margin_from_scores(model_scores) >= margin_threshold:
        return max(model_scores, key=lambda s: model_scores[s])
    return LABEL_SHARED


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


def resolve_image_path(artifacts_dir: Path, stored_path: str) -> Path:
    """manifest.json's image_path strings are baked in at generation time relative to the
    GPU script's OWN artifacts folder -- which both generate_anchor_images.py and
    generate_anchor_images_sdxl.py name literally "artifacts" (a per-Kaggle-session
    convention, unrelated to wherever the manifest gets downloaded to locally). So every
    manifest, SD1.5 or SDXL, stores paths shaped like "artifacts/images/p6.png" regardless
    of run. Naively resolving that string relative to the current working directory always
    lands in the SD1.5 folder, even when --artifacts-dir points elsewhere. This strips that
    baked-in "artifacts/" prefix and rejoins the remainder onto the CURRENT local
    artifacts_dir instead, so the SDXL run's images resolve correctly too."""
    rel = stored_path
    prefix = "artifacts/"
    if rel.startswith(prefix):
        rel = rel[len(prefix):]
    return (Path(artifacts_dir) / rel).resolve()


def pending_label_targets(manifest: dict, labels: Dict[str, str],
                          relabel: frozenset = frozenset()) -> List[Tuple[dict, dict]]:
    """(image_entry, attribute_entry) pairs still needing a human label -- only detected
    images, only attributes not already keyed in `labels`. Order follows the manifest.

    `relabel` re-queues rows whose EXISTING label is in that set, which is how an already
    finished pass gets revisited without discarding it: passing {LABEL_UNCLEAR} re-asks only
    the ambiguous rows (e.g. to split the ones that were really `shared`) and leaves every
    settled subject judgment alone. Empty by default, so an ordinary resume is unchanged."""
    out: List[Tuple[dict, dict]] = []
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            key = label_key(img["prompt_id"], attr["attribute"])
            if key not in labels or labels[key] in relabel:
                out.append((img, attr))
    return out


# ---------------------------------------------------------------------------
# Agreement analysis
# ---------------------------------------------------------------------------

def chance_baseline(n: int, include_shared: bool = False) -> float:
    """Random-guess accuracy at subject count n. Strict scoring picks among the n subjects
    (1/n); shared-aware scoring picks among n subjects plus `shared` (1/(n+1))."""
    return 1.0 / (n + 1) if include_shared else 1.0 / n


def build_agreement_rows(manifest: dict, labels: Dict[str, str],
                         margin_threshold: float = None) -> List[dict]:
    """One row per (detected chain, attribute) that has a human label.

    Strict columns (always present, semantics frozen so published numbers cannot move):
    `human_label` may be a subject or one of none/unclear/shared; `scored` is True only when
    the human named a single subject (the accuracy denominator); `correct` compares
    `predicted_owner` (bare argmax) to that subject.

    Shared-aware columns (only when `margin_threshold` is given): `margin`,
    `predicted_label` (a subject, or `shared` when the metric's top-two attention margin
    falls short), `scored_shared` (human named a subject OR said `shared`), and
    `correct_shared`. `none`/`unclear` stay excluded in both modes -- they are missing data,
    not outcomes."""
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
            row = dict(
                prompt_id=img["prompt_id"], n=img["n"], attribute=attr["attribute"],
                intended_subject=attr["intended_subject"], predicted_owner=attr["predicted_owner"],
                human_label=human, scored=scored,
                correct=(scored and attr["predicted_owner"] == human),
            )
            if margin_threshold is not None:
                predicted_label = predicted_label_with_shared(
                    attr["model_scores"], margin_threshold)
                scored_shared = human not in MISSING_DATA_LABELS
                row.update(
                    margin=margin_from_scores(attr["model_scores"]),
                    predicted_label=predicted_label,
                    scored_shared=scored_shared,
                    correct_shared=(scored_shared and predicted_label == human),
                )
            rows.append(row)
    return rows


def cohens_kappa(labels_a: Dict[str, str], labels_b: Dict[str, str]) -> dict:
    """Inter-rater reliability between two annotators' label files, over the keys they BOTH
    labeled (so a second annotator who stops partway is not penalised for unlabeled rows).

    Returns {n, p_observed, p_expected, kappa, categories}. `kappa` is None when p_expected
    is 1.0 -- both raters used a single identical category, which makes (po-pe)/(1-pe) a 0/0
    and would otherwise surface as a NaN or a ZeroDivisionError."""
    shared_keys = sorted(set(labels_a) & set(labels_b))
    if not shared_keys:
        raise ValueError("no overlapping label keys; nothing to compare between annotators")

    pairs = [(labels_a[k], labels_b[k]) for k in shared_keys]
    n = len(pairs)
    p_observed = sum(1 for x, y in pairs if x == y) / n

    categories = sorted({c for pair in pairs for c in pair})
    p_expected = sum(
        (sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n)
        for c in categories
    )
    kappa = None if p_expected >= 1.0 else (p_observed - p_expected) / (1.0 - p_expected)
    return dict(n=n, p_observed=p_observed, p_expected=p_expected, kappa=kappa,
                categories=categories)


def summarize_agreement(rows: Sequence[dict], mode: str = "strict") -> dict:
    """Overall + per-stratum agreement vs chance, plus coverage (how many labeled rows were
    excluded). Strata with zero scored rows report accuracy=None.

    `mode="strict"` (default) reads the frozen `scored`/`correct` columns against a 1/n
    baseline -- this is what every published anchor-set number used. `mode="shared"` reads
    `scored_shared`/`correct_shared` against 1/(n+1), which requires rows built with a
    `margin_threshold`."""
    if mode not in ("strict", "shared"):
        raise ValueError(f"unknown mode {mode!r}; expected 'strict' or 'shared'")
    include_shared = mode == "shared"
    if include_shared and rows and "scored_shared" not in rows[0]:
        raise ValueError(
            "mode='shared' needs rows built with build_agreement_rows(..., margin_threshold=...); "
            "these rows have no shared-aware columns")
    scored_col, correct_col = (
        ("scored_shared", "correct_shared") if include_shared else ("scored", "correct"))

    def _stratum(subset: Sequence[dict]) -> dict:
        scored = [r for r in subset if r[scored_col]]
        n_scored = len(scored)
        n_correct = sum(1 for r in scored if r[correct_col])
        ns = {r["n"] for r in subset}
        chance = chance_baseline(next(iter(ns)), include_shared) if len(ns) == 1 else None
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
