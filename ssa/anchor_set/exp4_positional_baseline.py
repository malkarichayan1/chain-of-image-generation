#!/usr/bin/env python
"""
Part A, Experiment 4: does the metric beat a naive positional baseline? Simplest possible
heuristic, no attention needed -- guess whichever subject noun sits *nearest* (by character
distance in the prompt text, either direction) to the attribute. In "a barista wearing a red
apron and a cyclist wearing a yellow helmet", "apron" sits right after "barista" -- guess
barista. If the attention-based metric can't beat this, attention hasn't shown it adds anything.

Mirrors discriminant_validity_check.py's bigbox-baseline pattern (accuracy vs. accuracy, paired
McNemar), substituting "nearest subject noun" for "biggest detected box."

Design note (see docs/part-a-five-experiment-battery-design.md, Experiment 4): checked all 306
real scored rows in artifacts_sdxl/manifest.json -- nearest-subject equals intended_subject
306/306 times, because the current prompt template never lets a second subject intervene between
a subject and its own attribute. This baseline is still exactly what was asked for and still a
meaningful comparison; it just currently coincides with "always guess the intended pairing."

NOT YET RUN ON REAL DATA -- see docs/part-a-five-experiment-battery-design.md. Run against the
dummy fixture today:
    py -3 exp4_positional_baseline.py --artifacts-dir artifacts_dummy --annotator dummy
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from scipy import stats

from anchor_common import build_agreement_rows, chance_baseline, load_labels, locate_attribute_phrase


def _locate_subject_in_prompt(prompt: str, subject: str) -> int:
    """Find the position of a subject in the prompt, with fallback for paraphrased subjects.

    First tries exact substring match. Falls back to finding a word in the prompt that shares
    a common prefix with the subject's words, which handles cases like subject="cyclist"
    appearing in the prompt as "cycling jersey" (both share the "cycl" prefix).

    Raises ValueError if the subject or any matching word cannot be located."""
    # Try exact match first
    idx = prompt.find(subject)
    if idx >= 0:
        return idx

    # Fall back to finding a word with a shared prefix
    subject_words = re.findall(r"[a-zA-Z]+", subject.lower())
    if not subject_words:
        raise ValueError(f"subject {subject!r} has no content words to match")

    # Extract all words from the prompt
    prompt_words = re.finditer(r"[a-zA-Z]+", prompt.lower())

    # For each subject word, try to find a prompt word that shares a prefix (stem matching)
    for subj_word in subject_words:
        best_match_pos = None
        for prompt_match in re.finditer(r"[a-zA-Z]+", prompt.lower()):
            prompt_word = prompt_match.group()
            # Check if they share a prefix (at least 4 chars or at least 70% of the shorter word)
            common_prefix_len = len(
                [c for c, p in zip(subj_word, prompt_word) if c == p]
            )
            min_word_len = min(len(subj_word), len(prompt_word))
            if common_prefix_len >= max(4, int(0.7 * min_word_len)):
                best_match_pos = prompt_match.start()
                break

        if best_match_pos is not None:
            return best_match_pos

    raise ValueError(f"subject {subject!r} not found in prompt {prompt!r}")


def nearest_subject_baseline(prompt: str, subjects: Sequence[str], attribute: str) -> str:
    """The nearest subject noun to `attribute`, preceding occurrences preferred over following
    ones (matches "apron comes right after barista -- guess barista": the intended subject
    normally precedes its own attribute, and the real prompt template never lets a following
    subject beat a preceding one). Falls back to the nearest following subject only when no
    subject precedes the attribute at all. Within a direction, nearest (smallest gap) wins;
    since subjects occupy distinct positions, exact same-direction ties can't occur for real
    prompts, but Python's stable sort resolves them by `subjects` list order regardless."""
    attr_idx, _ = locate_attribute_phrase(prompt, attribute)
    preceding: List[Tuple[int, str]] = []
    following: List[Tuple[int, str]] = []
    for subject in subjects:
        try:
            subj_idx = _locate_subject_in_prompt(prompt, subject)
        except ValueError:
            continue
        (preceding if subj_idx <= attr_idx else following).append(
            (abs(subj_idx - attr_idx), subject))
    if not preceding and not following:
        raise ValueError(f"none of {list(subjects)} found in prompt {prompt!r}")
    pool = preceding if preceding else following
    pool.sort(key=lambda pair: pair[0])
    return pool[0][1]


def prompts_by_id_from_manifest(manifest: dict) -> Dict[int, dict]:
    return {img["prompt_id"]: {"prompt": img["prompt"], "subjects": img["subjects"]}
            for img in manifest["images"] if img.get("detected")}


def join_baseline(rows: List[dict], prompts_by_id: Dict[int, dict]) -> pd.DataFrame:
    """One row per agreement row, augmented with `baseline_subject` and `baseline_correct`
    (baseline's guess vs. the human label, only meaningful when `scored`)."""
    out = []
    for r in rows:
        info = prompts_by_id[r["prompt_id"]]
        baseline_subject = nearest_subject_baseline(info["prompt"], info["subjects"], r["attribute"])
        row = dict(r)
        row.update(
            baseline_subject=baseline_subject,
            baseline_correct=(r["scored"] and baseline_subject == r["human_label"]),
        )
        out.append(row)
    return pd.DataFrame(out)


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["scored"] == True]  # noqa: E712


def baseline_vs_metric_report(df: pd.DataFrame) -> dict:
    sub = _scored(df)
    n = len(sub)
    metric_correct = int(sub["correct"].sum()) if n else 0
    baseline_correct = int(sub["baseline_correct"].sum()) if n else 0
    return dict(n=n,
                metric_accuracy=(metric_correct / n) if n else None,
                baseline_accuracy=(baseline_correct / n) if n else None)


def mcnemar_report(df: pd.DataFrame) -> dict:
    sub = _scored(df)
    metric_correct = sub["correct"].astype(bool)
    baseline_correct = sub["baseline_correct"].astype(bool)
    metric_only = int((metric_correct & ~baseline_correct).sum())
    baseline_only = int((~metric_correct & baseline_correct).sum())
    n_discordant = metric_only + baseline_only
    p = (stats.binomtest(metric_only, n_discordant, 0.5, alternative="two-sided").pvalue
         if n_discordant else None)
    return dict(n=len(sub), metric_only_correct=metric_only,
                baseline_only_correct=baseline_only, n_discordant=n_discordant, p_value=p)


def stratified_report(df: pd.DataFrame) -> Dict[int, dict]:
    out = {}
    for n in sorted(_scored(df)["n"].unique()):
        sub = _scored(df[df["n"] == n])
        m = len(sub)
        out[int(n)] = dict(
            n=m,
            metric_accuracy=(sub["correct"].sum() / m) if m else None,
            baseline_accuracy=(sub["baseline_correct"].sum() / m) if m else None,
            chance=chance_baseline(int(n)),
        )
    return out


def _print_report(name: str, report) -> None:
    print(f"\n{name}")
    if isinstance(report, dict) and all(isinstance(v, dict) for v in report.values()):
        for n, r in sorted(report.items()):
            print(f"  n={n}: {r}")
    else:
        for k, v in report.items():
            print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Experiment 4: metric accuracy vs. nearest-subject-noun baseline")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--annotator", required=True)
    args = ap.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    labels = load_labels(artifacts_dir / f"labels_{args.annotator}.json")
    if not labels:
        raise SystemExit(f"No labels found for annotator {args.annotator!r} in {artifacts_dir}")

    rows = build_agreement_rows(manifest, labels)
    df = join_baseline(rows, prompts_by_id_from_manifest(manifest))
    print(f"Experiment 4 -- nearest-subject-noun baseline ({artifacts_dir}, annotator={args.annotator})")
    _print_report("Headline: metric accuracy vs. baseline accuracy", baseline_vs_metric_report(df))
    _print_report("McNemar (paired, same rows)", mcnemar_report(df))
    _print_report("Per-stratum (n=2/3/4)", stratified_report(df))


if __name__ == "__main__":
    main()
