#!/usr/bin/env python
"""
Taxonomy analysis (experiments #14, #16, #17, #18, #19): pure CPU re-analysis of
taxonomy_capture_flux.py's output. No model calls anywhere in this file.

Reads taxonomy_index.json + taxonomy_cells_p<id>.npz (per-image (layers, steps, heads,
attributes[, subjects]) arrays), joins them against manifest.json's intended_subject and
labels_<annotator>.json's human_label, and reports:

  #14  layer bands       (blocks 0-6 / 7-12 / 13-18, early window, Holm across 3)
  #16  per-head grid      (19 layers x 24 heads, early window, full distribution, FDR)
  #17  timestep windows   (steps 0-6/7-12/13-18/19-24, ALL layers+heads, Holm across 4)
  #18  distribution grid  (heads x layer-bands x timestep-windows, 3 metrics, descriptive)
  #19  sharpest cells     (top-10 by #18's mass-fraction metric on the EASY set, tested
                           on the HARD set only -- see the "no double-dipping" note below)

Grid boundaries, the #19 selection metric, and k=10 are pre-registered in
docs/remaining-experiments-runbook.md Section 5 and frozen as module constants below --
change them there, not by editing a call site, so the record of what was decided before
looking at output stays in one place.

REPRODUCTION GATE, APPLIED HERE (not in the capture): taxonomy_capture_flux.py's own
docstring says the repro checks flag but do not filter -- "which threshold makes a row
unusable is an analysis decision, not a capture decision." This file is where that decision
happens: `load_all_cells` drops any image whose `repro_mean_abs_pixel_diff` exceeds
`--repro-threshold` (default 0.05) and reports the drop count. Never report a taxonomy
number without also reporting how many images were dropped.

NO DOUBLE-DIPPING (#19): cells are RANKED on the easy set (artifacts_flux/) and TESTED on
the hard set (artifacts_flux_hard/) only -- disjoint images, so selecting the sharpest
cells cannot leak into the significance of testing them. See runbook Section 2(c).

POISSON-BINOMIAL, NOT A SINGLE CHANCE RATE: images in this dataset have different subject
counts (n=2..6), so a scored row's chance of a correct guess is 1/n_for_that_row, not one
shared constant. `poisson_binomial_pvalue` combines per-row chances exactly via polynomial
convolution -- the same approach CLAUDE.md's C3 section describes as already used
("pooled ... Poisson-binomial over each row's own 1/n chance"), reimplemented here as a
tested, reusable primitive since no prior implementation exists anywhere in this repo.

Run from inside ssa/anchor_set/, after downloading BOTH Kaggle taxonomy captures:
    py -3 exp9_taxonomy_analysis.py \
        --easy-dir artifacts_flux --easy-annotator chayan \
        --hard-dir artifacts_flux_hard --hard-annotator consensus \
        --out artifacts_flux_hard/taxonomy_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from anchor_common import NON_SUBJECT_LABELS, label_key, load_labels

# ---------------------------------------------------------------------------
# Pre-registered grid (docs/remaining-experiments-runbook.md Section 5). Frozen here.
# ---------------------------------------------------------------------------

LAYER_BANDS: Dict[str, range] = {"early_0_6": range(0, 7), "mid_7_12": range(7, 13),
                                 "late_13_18": range(13, 19)}
TIMESTEP_WINDOWS: Dict[str, range] = {"w1_0_6": range(0, 7), "w2_7_12": range(7, 13),
                                      "w3_13_18": range(13, 19), "w4_19_24": range(19, 25)}
EXPECTED_N_STEPS = 25   # NUM_INFERENCE_STEPS in taxonomy_capture_flux.py; windows assume this
EXP19_TOP_K = 10
SIGNIFICANCE_ALPHA = 0.05
DEFAULT_REPRO_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageCells:
    prompt_id: int
    in_box_mass: np.ndarray   # (layers, steps, heads, attrs, subjects)
    entropy: np.ndarray       # (layers, steps, heads, attrs)
    attributes: List[str]
    subjects: List[str]
    layer_order: List[str]
    max_steps_early: int


def load_image_cells(artifacts_dir: Path, prompt_id: int, entry: dict) -> ImageCells:
    npz = np.load(artifacts_dir / f"taxonomy_cells_p{prompt_id}.npz")
    return ImageCells(
        prompt_id=prompt_id, in_box_mass=npz["in_box_mass"], entropy=npz["entropy"],
        attributes=list(entry["attributes"]), subjects=list(entry["subjects"]),
        layer_order=list(entry["layer_order"]), max_steps_early=int(entry["max_steps_early"]))


def load_all_cells(artifacts_dir: Path, repro_threshold: float = DEFAULT_REPRO_THRESHOLD
                   ) -> Tuple[List[ImageCells], List[int]]:
    """Loads every image in taxonomy_index.json whose reproduction diff is within
    `repro_threshold` (or has no reference image at all, e.g. repro check unavailable --
    treated as pass-through, not silently dropped, since absence isn't evidence of drift).

    Raises ValueError on the first shape mismatch across images (layer count, head count,
    layer order, or step count) -- captures must share one model config, or every
    downstream reduction silently mixes incompatible axes."""
    index = json.loads((artifacts_dir / "taxonomy_index.json").read_text())
    cells: List[ImageCells] = []
    dropped: List[int] = []
    reference: Optional[Tuple[int, int, int, Tuple[str, ...]]] = None
    for pid_str, entry in sorted(index.items(), key=lambda kv: int(kv[0])):
        diff = entry.get("repro_mean_abs_pixel_diff")
        if diff is not None and diff > repro_threshold:
            dropped.append(int(pid_str))
            continue
        ic = load_image_cells(artifacts_dir, int(pid_str), entry)
        shape_key = (ic.in_box_mass.shape[0], ic.in_box_mass.shape[1], ic.in_box_mass.shape[2],
                    tuple(ic.layer_order))
        if reference is None:
            reference = shape_key
        elif shape_key != reference:
            raise ValueError(
                f"prompt_id {ic.prompt_id} has capture shape (layers,steps,heads,order)="
                f"{shape_key}, expected {reference} -- captures must share one model config")
        cells.append(ic)
    return cells, dropped


def load_ground_truth(artifacts_dir: Path, annotator: str) -> Dict[Tuple[int, str], str]:
    """{(prompt_id, attribute): human_label}. intended_subject is not needed here -- it is
    already carried on each cell_rows() row via the manifest join inside that function, to
    keep this a single source of truth for label loading (anchor_common.load_labels)."""
    labels = load_labels(artifacts_dir / f"labels_{annotator}.json")
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    out: Dict[Tuple[int, str], str] = {}
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            key = label_key(img["prompt_id"], attr["attribute"])
            if key in labels:
                out[(img["prompt_id"], attr["attribute"])] = labels[key]
    return out


def load_intended_subjects(artifacts_dir: Path) -> Dict[Tuple[int, str], str]:
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    out: Dict[Tuple[int, str], str] = {}
    for img in manifest["images"]:
        if not img.get("detected"):
            continue
        for attr in img["attributes"]:
            out[(img["prompt_id"], attr["attribute"])] = attr["intended_subject"]
    return out


# ---------------------------------------------------------------------------
# Statistical primitives. No prior implementation exists in this repo -- see module
# docstring. Both are tested against known closed-form / reference values.
# ---------------------------------------------------------------------------

def poisson_binomial_pvalue(successes: int, chances: Sequence[float]) -> float:
    """Exact one-sided P(X >= successes) for a Poisson-binomial variable with per-trial
    success probabilities `chances` -- i.e. each scored row's OWN 1/n chance, not one
    shared rate. Computed by polynomial convolution (DP): O(len(chances)^2), trivial at the
    row counts here (tens to low hundreds) and exact, unlike a normal approximation."""
    if not chances:
        raise ValueError("chances must be non-empty")
    probs = np.array([1.0])
    for p in chances:
        probs = np.convolve(probs, [1.0 - p, p])
    return float(np.clip(probs[successes:].sum(), 0.0, 1.0))


def holm_correction(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjustment, order-preserving. `None` entries (no scored
    rows for that test) pass through unchanged and are excluded from the family size."""
    indices = [i for i, p in enumerate(p_values) if p is not None]
    if not indices:
        return list(p_values)
    m = len(indices)
    order = sorted(indices, key=lambda i: p_values[i])
    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min((m - rank) * p_values[i], 1.0))
        adjusted[i] = running_max
    return [adjusted.get(i) for i in range(len(p_values))]


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg step-up FDR adjustment, same None-passthrough convention as
    holm_correction."""
    indices = [i for i, p in enumerate(p_values) if p is not None]
    if not indices:
        return list(p_values)
    m = len(indices)
    order = sorted(indices, key=lambda i: p_values[i])
    adjusted: Dict[int, float] = {}
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running_min = min(running_min, min(p_values[i] * m / (rank + 1), 1.0))
        adjusted[i] = running_min
    return [adjusted.get(i) for i in range(len(p_values))]


# ---------------------------------------------------------------------------
# Reduction: collapse (layers, steps, heads) index subsets down to (attrs[, subjects]).
# Every one of #14/#16/#17/#18/#19 is this reduction with a different index subset.
# ---------------------------------------------------------------------------

def reduce_mass(cells: ImageCells, layer_idx: Sequence[int], step_idx: Sequence[int],
                head_idx: Sequence[int]) -> np.ndarray:
    """Mean in_box_mass over the given (layer, step, head) index sets -> (attrs, subjects)
    float32. `np.ix_` on 3 index arrays against a 5D array leaves the trailing (attrs,
    subjects) axes as implicit full slices -- standard NumPy indexing, not a manual loop."""
    sub = cells.in_box_mass[np.ix_(list(layer_idx), list(step_idx), list(head_idx))]
    return sub.astype(np.float32).mean(axis=(0, 1, 2))


def reduce_entropy(cells: ImageCells, layer_idx: Sequence[int], step_idx: Sequence[int],
                   head_idx: Sequence[int]) -> np.ndarray:
    sub = cells.entropy[np.ix_(list(layer_idx), list(step_idx), list(head_idx))]
    return sub.astype(np.float32).mean(axis=(0, 1, 2))


def predicted_owners(reduced_mass: np.ndarray, subjects: Sequence[str]) -> List[str]:
    return [subjects[int(np.argmax(row))] for row in reduced_mass]


def cell_rows(images: Sequence[ImageCells], ground_truth: Dict[Tuple[int, str], str],
             intended: Dict[Tuple[int, str], str], layer_idx: Sequence[int],
             step_idx: Sequence[int], head_idx: Sequence[int]) -> List[dict]:
    """One row per (image, attribute) with ground truth, for the given reduction. Row shape
    mirrors anchor_common.build_agreement_rows so the accuracy/McNemar helpers below read
    identically to the published pipeline's own analysis code. `n` is the subject count for
    THAT row's image -- the per-row chance denominator for poisson_binomial_pvalue."""
    rows: List[dict] = []
    for img in images:
        reduced = reduce_mass(img, layer_idx, step_idx, head_idx)
        owners = predicted_owners(reduced, img.subjects)
        for attr, owner in zip(img.attributes, owners):
            human = ground_truth.get((img.prompt_id, attr))
            intended_subject = intended.get((img.prompt_id, attr))
            if human is None or intended_subject is None:
                continue
            scored = human not in NON_SUBJECT_LABELS
            rows.append(dict(
                prompt_id=img.prompt_id, attribute=attr, n=len(img.subjects),
                predicted_owner=owner, intended_subject=intended_subject, human_label=human,
                scored=scored, correct=(scored and owner == human)))
    return rows


def accuracy_report(rows: Sequence[dict]) -> dict:
    scored = [r for r in rows if r["scored"]]
    n_scored = len(scored)
    n_correct = sum(1 for r in scored if r["correct"])
    chances = [1.0 / r["n"] for r in scored]
    return dict(
        n_scored=n_scored, n_correct=n_correct,
        accuracy=(n_correct / n_scored) if n_scored else None,
        mean_chance=(float(np.mean(chances)) if chances else None),
        p_value=(poisson_binomial_pvalue(n_correct, chances) if chances else None))


# ---------------------------------------------------------------------------
# #14 -- layer bands
# ---------------------------------------------------------------------------

def layer_band_report(images: Sequence[ImageCells], ground_truth, intended) -> dict:
    max_steps_early = images[0].max_steps_early
    n_heads = images[0].in_box_mass.shape[2]
    step_idx, head_idx = range(max_steps_early), range(n_heads)

    reports = {name: accuracy_report(
        cell_rows(images, ground_truth, intended, list(layer_idx), list(step_idx), list(head_idx)))
        for name, layer_idx in LAYER_BANDS.items()}
    adjusted = holm_correction([reports[n]["p_value"] for n in LAYER_BANDS])
    for name, p_holm in zip(LAYER_BANDS, adjusted):
        reports[name]["p_value_holm"] = p_holm
    return reports


# ---------------------------------------------------------------------------
# #17 -- timestep windows. Full layer/head range, ALL steps (not just the early window) --
# the point is to look across the whole trajectory the early-window convention discards.
# ---------------------------------------------------------------------------

def timestep_window_report(images: Sequence[ImageCells], ground_truth, intended) -> dict:
    n_steps = images[0].in_box_mass.shape[1]
    if n_steps != EXPECTED_N_STEPS:
        raise ValueError(
            f"captured {n_steps} steps, but TIMESTEP_WINDOWS assumes {EXPECTED_N_STEPS} "
            "(NUM_INFERENCE_STEPS in taxonomy_capture_flux.py) -- update the grid, don't "
            "silently mis-slice a different step count")
    n_layers = len(images[0].layer_order)
    n_heads = images[0].in_box_mass.shape[2]
    layer_idx, head_idx = list(range(n_layers)), list(range(n_heads))

    reports = {name: accuracy_report(
        cell_rows(images, ground_truth, intended, layer_idx, list(step_idx), head_idx))
        for name, step_idx in TIMESTEP_WINDOWS.items()}
    adjusted = holm_correction([reports[n]["p_value"] for n in TIMESTEP_WINDOWS])
    for name, p_holm in zip(TIMESTEP_WINDOWS, adjusted):
        reports[name]["p_value_holm"] = p_holm
    return reports


# ---------------------------------------------------------------------------
# #16 -- per-head grid (19 layers x 24 heads, early window). Full distribution, not winner.
# ---------------------------------------------------------------------------

def per_head_grid(images: Sequence[ImageCells], ground_truth, intended) -> dict:
    max_steps_early = images[0].max_steps_early
    n_layers = len(images[0].layer_order)
    n_heads = images[0].in_box_mass.shape[2]
    step_idx = list(range(max_steps_early))

    cells: Dict[str, dict] = {}
    keys: List[str] = []
    for li in range(n_layers):
        for hi in range(n_heads):
            key = f"layer{li}_head{hi}"
            cells[key] = accuracy_report(
                cell_rows(images, ground_truth, intended, [li], step_idx, [hi]))
            keys.append(key)
    adjusted = benjamini_hochberg([cells[k]["p_value"] for k in keys])
    for key, p_fdr in zip(keys, adjusted):
        cells[key]["p_value_fdr"] = p_fdr

    accuracies = [cells[k]["accuracy"] for k in keys if cells[k]["accuracy"] is not None]
    n_sig = sum(1 for k in keys if (cells[k]["p_value_fdr"] is not None
                                    and cells[k]["p_value_fdr"] < SIGNIFICANCE_ALPHA))
    distribution = dict(
        n_cells=len(accuracies),
        mean=float(np.mean(accuracies)) if accuracies else None,
        median=float(np.median(accuracies)) if accuracies else None,
        std=float(np.std(accuracies)) if accuracies else None,
        min=float(np.min(accuracies)) if accuracies else None,
        max=float(np.max(accuracies)) if accuracies else None,
        n_significant_fdr05=n_sig)
    return dict(n_layers=n_layers, n_heads=n_heads, cells=cells, distribution=distribution)


# ---------------------------------------------------------------------------
# #18 -- distribution grid (heads x layer-bands x timestep-windows). Descriptive only.
# ---------------------------------------------------------------------------

def collect_cell_metrics(images: Sequence[ImageCells], ground_truth, intended,
                         layer_idx: Sequence[int], step_idx: Sequence[int],
                         head_idx: Sequence[int]) -> Tuple[List[float], List[float], List[float]]:
    """Per scored (image, attribute) row in this cell: mass fraction landing on the
    CORRECT subject, the peak-to-second-peak ratio across all candidate subjects (how
    decisively the top subject beats the runner-up), and spatial entropy. Rows with a
    non-subject human label (none/unclear/shared) are skipped -- there is no "correct"
    subject to fraction against."""
    mass_fracs: List[float] = []
    ratios: List[float] = []
    entropies: List[float] = []
    for img in images:
        reduced_mass = reduce_mass(img, layer_idx, step_idx, head_idx)
        reduced_ent = reduce_entropy(img, layer_idx, step_idx, head_idx)
        for ai, attr in enumerate(img.attributes):
            human = ground_truth.get((img.prompt_id, attr))
            if human is None or human not in img.subjects:
                continue
            scores = reduced_mass[ai]
            total = float(scores.sum())
            correct_idx = img.subjects.index(human)
            mass_fracs.append(float(scores[correct_idx] / total) if total > 0 else 0.0)
            top, second = np.sort(scores)[::-1][:2] if len(scores) > 1 else (scores[0], 0.0)
            ratios.append(float(top / second) if second > 0 else float("inf"))
            entropies.append(float(reduced_ent[ai]))
    return mass_fracs, ratios, entropies


def _stats(xs: Sequence[float]) -> dict:
    if not xs:
        return dict(mean=None, median=None, std=None, n=0)
    arr = np.array(xs, dtype=np.float64)
    return dict(mean=float(arr.mean()), median=float(np.median(arr)),
               std=float(arr.std()), n=len(xs))


def summarize_distribution(mass_fracs: List[float], ratios: List[float],
                           entropies: List[float]) -> dict:
    finite_ratios = [r for r in ratios if np.isfinite(r)]
    return dict(
        n_rows=len(mass_fracs),
        mass_fraction_correct=_stats(mass_fracs),
        peak_to_second_ratio=_stats(finite_ratios),
        n_infinite_ratio=len(ratios) - len(finite_ratios),
        entropy=_stats(entropies))


def distribution_grid(images: Sequence[ImageCells], ground_truth, intended) -> dict:
    n_heads = images[0].in_box_mass.shape[2]
    grid: Dict[str, dict] = {}
    for band_name, layer_range in LAYER_BANDS.items():
        for window_name, step_range in TIMESTEP_WINDOWS.items():
            for hi in range(n_heads):
                mass_fracs, ratios, entropies = collect_cell_metrics(
                    images, ground_truth, intended, list(layer_range), list(step_range), [hi])
                grid[f"{band_name}|{window_name}|head{hi}"] = summarize_distribution(
                    mass_fracs, ratios, entropies)
    return grid


# ---------------------------------------------------------------------------
# #19 -- select on the easy set, test on the hard set. No double-dipping (see module
# docstring). This is the only taxonomy result with a claim attached to it.
# ---------------------------------------------------------------------------

def select_sharpest_cells(easy_images: Sequence[ImageCells], easy_ground_truth, easy_intended,
                          k: int = EXP19_TOP_K) -> List[Tuple[int, int]]:
    """Top-k (layer, head) cells by mean mass-fraction-correct (the #18 metric), early
    window, on the EASY set only. Ties broken by (layer, head) order for determinism."""
    max_steps_early = easy_images[0].max_steps_early
    n_layers = len(easy_images[0].layer_order)
    n_heads = easy_images[0].in_box_mass.shape[2]
    step_idx = list(range(max_steps_early))

    scored: List[Tuple[Tuple[int, int], float]] = []
    for li in range(n_layers):
        for hi in range(n_heads):
            mass_fracs, _, _ = collect_cell_metrics(
                easy_images, easy_ground_truth, easy_intended, [li], step_idx, [hi])
            if mass_fracs:
                scored.append(((li, hi), float(np.mean(mass_fracs))))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [cell for cell, _ in scored[:k]]


def prompt_baseline_test(rows: Sequence[dict]) -> dict:
    """Mirrors exp6_prompt_baseline.py's question for one cell's rows: does the cell beat
    "assume the prompt was obeyed"? McNemar on discordant scored rows."""
    scored = [r for r in rows if r["scored"]]
    n_scored = len(scored)
    cell_correct = sum(1 for r in scored if r["correct"])
    base_correct = sum(1 for r in scored if r["human_label"] == r["intended_subject"])
    b = sum(1 for r in scored if r["correct"] and r["human_label"] != r["intended_subject"])
    c = sum(1 for r in scored if not r["correct"] and r["human_label"] == r["intended_subject"])
    n_discordant = b + c
    p = (stats.binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n_scored=n_scored,
               cell_acc=(cell_correct / n_scored) if n_scored else None,
               base_acc=(base_correct / n_scored) if n_scored else None,
               b=b, c=c, p_value=p)


def misbound_test(rows: Sequence[dict]) -> dict:
    """Mirrors exp7_misbound_subset.py's question for one cell's rows: restricted to rows
    where the model disobeyed the prompt, is the cell's accuracy above chance?"""
    misbound = [r for r in rows if r["scored"] and r["human_label"] != r["intended_subject"]]
    n_misbound = len(misbound)
    n_correct = sum(1 for r in misbound if r["correct"])
    chances = [1.0 / r["n"] for r in misbound]
    return dict(n_misbound=n_misbound, n_correct=n_correct,
               accuracy=(n_correct / n_misbound) if n_misbound else None,
               mean_chance=(float(np.mean(chances)) if chances else None),
               p_value=(poisson_binomial_pvalue(n_correct, chances) if chances else None))


def sharpest_cell_report(easy_images, easy_gt, easy_intended, hard_images, hard_gt,
                         hard_intended, k: int = EXP19_TOP_K) -> dict:
    cell_ids = select_sharpest_cells(easy_images, easy_gt, easy_intended, k=k)
    step_idx = list(range(hard_images[0].max_steps_early))

    keys, baseline_reports, misbound_reports = [], [], []
    for li, hi in cell_ids:
        rows = cell_rows(hard_images, hard_gt, hard_intended, [li], step_idx, [hi])
        keys.append(f"layer{li}_head{hi}")
        baseline_reports.append(prompt_baseline_test(rows))
        misbound_reports.append(misbound_test(rows))

    baseline_holm = holm_correction([r["p_value"] for r in baseline_reports])
    misbound_holm = holm_correction([r["p_value"] for r in misbound_reports])

    per_cell: Dict[str, dict] = {}
    passes: List[bool] = []
    for key, b_rep, m_rep, b_holm, m_holm in zip(
            keys, baseline_reports, misbound_reports, baseline_holm, misbound_holm):
        b_rep["p_value_holm"] = b_holm
        m_rep["p_value_holm"] = m_holm
        # A cell "passes" #19 only if it beats the baseline (not merely differs from it,
        # and specifically in the cell's favor) AND clears chance on the misbound subset --
        # both after Holm correction. Direction matters: a significant LOSS to the baseline
        # is the expected negative result, not a pass.
        cell_beats_baseline = (
            b_rep["cell_acc"] is not None and b_rep["base_acc"] is not None
            and b_rep["cell_acc"] > b_rep["base_acc"]
            and b_holm is not None and b_holm < SIGNIFICANCE_ALPHA)
        above_chance_misbound = (m_holm is not None and m_holm < SIGNIFICANCE_ALPHA)
        passed = cell_beats_baseline and above_chance_misbound
        passes.append(passed)
        per_cell[key] = dict(prompt_baseline=b_rep, misbound=m_rep, passes_exp19=passed)

    verdict = (
        "POSITIVE: at least one cell tracks the rendered image on cases the model "
        "disobeyed -- this changes the paper's conclusion and should be reported as such"
        if any(passes) else
        "NEGATIVE (bulletproof): no cell in the searched hierarchy beats the prompt-only "
        "baseline while also clearing chance on the misbound subset")
    return dict(selected_cells=keys, per_cell=per_cell, n_cells_tested=len(keys),
               n_cells_passing=sum(passes), verdict=verdict)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_battery(datasets: Dict[str, Tuple[Path, str]],
                     repro_threshold: float = DEFAULT_REPRO_THRESHOLD,
                     exp19_k: int = EXP19_TOP_K) -> dict:
    """`datasets`: {"easy": (artifacts_flux, "chayan"), "hard": (artifacts_flux_hard,
    "consensus")}. #14/#16/#17/#18 run once per named dataset; #19 requires both "easy"
    and "hard" to be present with at least one image each."""
    loaded: Dict[str, dict] = {}
    for name, (artifacts_dir, annotator) in datasets.items():
        images, dropped = load_all_cells(artifacts_dir, repro_threshold)
        loaded[name] = dict(
            images=images, n_images=len(images), n_dropped_repro=len(dropped),
            ground_truth=load_ground_truth(artifacts_dir, annotator),
            intended=load_intended_subjects(artifacts_dir))

    per_dataset: Dict[str, dict] = {}
    for name, d in loaded.items():
        if not d["images"]:
            per_dataset[name] = dict(n_images=0, n_dropped_repro=d["n_dropped_repro"],
                                     error="no images passed the reproduction check")
            continue
        images, gt, intended = d["images"], d["ground_truth"], d["intended"]
        per_dataset[name] = dict(
            n_images=d["n_images"], n_dropped_repro=d["n_dropped_repro"],
            exp14_layer_bands=layer_band_report(images, gt, intended),
            exp16_per_head=per_head_grid(images, gt, intended),
            exp17_timestep_windows=timestep_window_report(images, gt, intended),
            exp18_distribution=distribution_grid(images, gt, intended))

    result: Dict[str, object] = dict(datasets=per_dataset)
    if ("easy" in loaded and "hard" in loaded
            and loaded["easy"]["images"] and loaded["hard"]["images"]):
        e, h = loaded["easy"], loaded["hard"]
        result["exp19"] = sharpest_cell_report(
            e["images"], e["ground_truth"], e["intended"],
            h["images"], h["ground_truth"], h["intended"], k=exp19_k)
    else:
        result["exp19"] = dict(
            error="exp19 requires both 'easy' and 'hard' datasets with captured images")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Taxonomy analysis: experiments #14, #16, #17, #18, #19")
    ap.add_argument("--easy-dir", default="artifacts_flux")
    ap.add_argument("--easy-annotator", default="chayan")
    ap.add_argument("--hard-dir", default="artifacts_flux_hard")
    ap.add_argument("--hard-annotator", default="consensus")
    ap.add_argument("--repro-threshold", type=float, default=DEFAULT_REPRO_THRESHOLD)
    ap.add_argument("--exp19-k", type=int, default=EXP19_TOP_K)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = {"easy": (Path(args.easy_dir), args.easy_annotator),
               "hard": (Path(args.hard_dir), args.hard_annotator)}
    report = run_full_battery(datasets, repro_threshold=args.repro_threshold,
                              exp19_k=args.exp19_k)

    for name, d in report["datasets"].items():
        print(f"\n=== {name}: {d['n_images']} image(s), {d['n_dropped_repro']} dropped "
              f"(repro diff > {args.repro_threshold}) ===")
        if "error" in d:
            print(f"  {d['error']}")
            continue
        print("  #14 layer bands:")
        for band, rep in d["exp14_layer_bands"].items():
            print(f"    {band:<12} acc={rep['accuracy']} n={rep['n_scored']} "
                  f"p_holm={rep['p_value_holm']}")
        print("  #17 timestep windows:")
        for win, rep in d["exp17_timestep_windows"].items():
            print(f"    {win:<12} acc={rep['accuracy']} n={rep['n_scored']} "
                  f"p_holm={rep['p_value_holm']}")
        dist = d["exp16_per_head"]["distribution"]
        print(f"  #16 per-head: {dist['n_cells']} cells, mean_acc={dist['mean']}, "
              f"n_significant_fdr05={dist['n_significant_fdr05']}")
        print(f"  #18 distribution grid: {len(d['exp18_distribution'])} cells (descriptive)")

    print(f"\n=== #19: sharpest cells (selected on easy, tested on hard) ===")
    exp19 = report["exp19"]
    if "error" in exp19:
        print(f"  {exp19['error']}")
    else:
        print(f"  {exp19['n_cells_passing']}/{exp19['n_cells_tested']} cells pass")
        print(f"  {exp19['verdict']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
