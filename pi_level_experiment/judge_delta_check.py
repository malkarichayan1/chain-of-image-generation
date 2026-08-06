#!/usr/bin/env python
"""
Binary-delta version of E4: does "the judge says yes now AND said no at the previous
step" survive the lock scenario, where CLIPSeg could not even be tested
(coig_delta_mask_check.py -- appears_at_step ceiling of 0.278 there, vs. the judge's 1.000
on these same images per pilot/causal_relevance_results.csv)?

The lock-immunity argument in pilot/spatial_semantic_alignment.py never actually required
spatial pixels -- it requires "present now AND NOT present before." That is exactly as
well-defined for a yes/no judge call as for a segmentation mask:

    delta(k) = judge(image_k, question) AND NOT judge(image_{k-1}, question)

still forces 0 whenever the attribute was already visible at k-1, by the same logic as
score_chains.delta_mask_from_sigmoid. What is given up relative to the spatial version is
localization -- this measures WHETHER new content appeared, not WHERE -- but Part C's own
ablation already showed the spatial IoU wasn't carrying the published result, so that may
be a loss on paper only.

Zero new judge calls needed. Real and shuffled: every (item, question, step) judgment was
already collected by evaluate_sbs_images.py --all and is sitting in
evaluation_sbs_results.csv -- causal_relevance.py's own docstring says as much for the
claimed/final steps it uses; this module is the first thing to also read the step
*before* the claimed one out of that same file. Substituted: its only claimed_step>=2 row
already has appears_at_step=0 in causal_relevance_results.csv (a fresh call
causal_relevance.py made previously), and delta = curr AND NOT prev is forced to 0
whenever curr is 0 regardless of prev -- so no new call is needed there either.

Dataset limitation worth stating up front: after restricting to claimed_step >= 2 (a
previous frame must exist), every scorable shuffled row happens to be a LATE claim
(claimed_step > true_step) -- the lock scenario. There is no scorable EARLY shuffled row
to contrast against, unlike lock_confound_analysis.py's SD1.5 data. So this cannot
replicate that module's EARLY-vs-LATE degradation comparison; it can only ask "real vs.
shuffled-under-the-lock" directly, which is in fact Track 1's original headline
question (persists_to_final: real=0.833, shuffled=0.833 -- indistinguishable) restated at
the claimed step instead of the final step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REAL, SHUFFLED, SUBSTITUTED = "real", "shuffled", "substituted"


def load_step_lookup(sbs_csv: Path) -> Dict[Tuple[int, str, int], int]:
    """(item_index, question_type, step_num) -> binary judge answer, every step -- the
    same table causal_relevance.load_lookup builds, but this module also reads the
    claimed_step - 1 entries that function never looks up."""
    df = pd.read_csv(sbs_csv)
    return {
        (int(r.item_index), str(r.question_type), int(r.step_num)): int(r.answer_binary)
        for r in df.itertuples()
    }


def load_condition_rows(conditions_dir: Path, condition: str) -> pd.DataFrame:
    rows = json.loads((Path(conditions_dir) / f"{condition}.json").read_text())
    frame = pd.DataFrame(rows)
    for column in ("item_index", "claimed_step", "final_step"):
        frame[column] = frame[column].astype(int)
    frame["condition"] = condition
    return frame


def binary_delta(curr: Optional[int], prev: Optional[int]) -> Optional[int]:
    """curr AND NOT prev, propagating missing data as None rather than guessing."""
    if curr is None or prev is None:
        return None
    return int(bool(curr) and not bool(prev))


def score_from_lookup(conditions: pd.DataFrame, lookup: Dict[Tuple[int, str, int], int],
                      ) -> pd.DataFrame:
    """Real/shuffled: curr and prev both come from the pre-collected per-step lookup.
    Rows with claimed_step == 1 are dropped -- no previous frame exists, so the delta is
    undefined, not zero (the same convention as coig_delta_mask_check.score_coig)."""
    rows = []
    for row in conditions[conditions.claimed_step >= 2].itertuples():
        curr = lookup.get((row.item_index, row.question_col, row.claimed_step))
        prev = lookup.get((row.item_index, row.question_col, row.claimed_step - 1))
        rows.append(dict(
            condition=row.condition, item_index=row.item_index,
            question_col=row.question_col, claimed_step=row.claimed_step,
            curr=curr, prev=prev, delta=binary_delta(curr, prev),
        ))
    return pd.DataFrame(rows)


def score_substituted_from_cache(conditions: pd.DataFrame,
                                 causal_relevance_csv: Path) -> pd.DataFrame:
    """Substituted's curr already exists in causal_relevance_results.csv (a fresh judge
    call causal_relevance.py made previously, since these question/image pairs were never
    asked during evaluate_sbs_images.py --all). No new prev call is fetched: whenever
    curr==0, delta is forced to 0 regardless of prev, and every claimed_step>=2 substituted
    row in this dataset already has curr==0 -- verified by an assertion below rather than
    assumed, so a future data refresh that changes this cannot pass silently."""
    cached = pd.read_csv(causal_relevance_csv)
    cached = cached[cached.condition == SUBSTITUTED]
    scorable = conditions[conditions.claimed_step >= 2]
    rows = []
    for row in scorable.itertuples():
        match = cached[(cached.item_index == row.item_index)
                      & (cached.question_col == row.question_col)
                      & (cached.claimed_step == row.claimed_step)]
        if match.empty:
            continue
        curr = int(match.iloc[0].appears_at_step)
        assert curr == 0, (
            f"substituted row {row.item_index}/{row.question_col} has curr={curr} != 0 -- "
            "a fresh judge call at the previous step is required to score its delta and "
            "this module does not make one; see module docstring")
        rows.append(dict(condition=SUBSTITUTED, item_index=row.item_index,
                         question_col=row.question_col, claimed_step=row.claimed_step,
                         curr=curr, prev=None, delta=0))
    return pd.DataFrame(rows)


def pooled_fisher(real: pd.Series, control: pd.Series, metric: str) -> dict:
    """2x2 Fisher exact on binary outcomes -- the pooled analogue of the continuous
    Mann-Whitney used elsewhere in this repo, appropriate here because curr/delta are 0/1,
    not continuous areas."""
    real_yes, real_no = int(real.sum()), int((~real.astype(bool)).sum())
    ctrl_yes, ctrl_no = int(control.sum()), int((~control.astype(bool)).sum())
    _, p = stats.fisher_exact([[real_yes, real_no], [ctrl_yes, ctrl_no]], alternative="greater")
    return dict(metric=metric, real_rate=float(real.mean()), control_rate=float(control.mean()),
               n_real=len(real), n_control=len(control), p_value=float(p))


def clustered_by_item(real: pd.DataFrame, control: pd.DataFrame, metric: str) -> dict:
    """Per-item paired Wilcoxon on mean(metric), matching
    lock_confound_analysis.clustered_p_value's convention -- item_index is this dataset's
    natural clustering unit, the same role prompt_id plays for the SD1.5 chains."""
    real_by_item = real.groupby("item_index")[metric].mean()
    ctrl_by_item = control.groupby("item_index")[metric].mean()
    aligned_real, aligned_ctrl = real_by_item.align(ctrl_by_item, join="inner")
    diffs = (aligned_real - aligned_ctrl).to_numpy()
    if len(diffs) < 2 or np.all(diffs == diffs[0]):
        return dict(metric=metric, p_value=float("nan"), n_groups=len(diffs),
                    note="insufficient variation across items for Wilcoxon")
    _, p_value = stats.wilcoxon(diffs, alternative="greater")
    return dict(metric=metric, p_value=float(p_value), n_groups=len(diffs),
               mean_diff=float(diffs.mean()))


def format_report(fisher_reports: list, clustered_reports: list, substituted: pd.DataFrame,
                  ) -> str:
    lines = ["Binary-delta check (judge-based), real vs shuffled-under-the-lock", ""]
    lines.append(f"{'metric':>8s} | {'real rate':>9s} | {'shuf rate':>9s} | "
                 f"{'n_real':>6s} | {'n_shuf':>6s} | {'pooled p':>9s}")
    for r in fisher_reports:
        lines.append(f"{r['metric']:>8s} | {r['real_rate']:9.3f} | {r['control_rate']:9.3f} | "
                     f"{r['n_real']:6d} | {r['n_control']:6d} | {r['p_value']:9.4f}")
    lines.append("")
    lines.append(f"{'metric':>8s} | {'n_items':>7s} | {'clustered p':>11s}")
    for r in clustered_reports:
        if "note" in r:
            lines.append(f"{r['metric']:>8s} | {r['n_groups']:7d} | {r['note']}")
        else:
            lines.append(f"{r['metric']:>8s} | {r['n_groups']:7d} | {r['p_value']:11.4f}")
    lines.append("")
    lines.append(f"substituted sanity check: {len(substituted)} scorable row(s), "
                f"delta={'all 0 (as expected)' if (substituted.delta == 0).all() else 'MISMATCH -- see above'}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Judge-based binary delta on real CoIG chains")
    ap.add_argument("--conditions-dir", default=str(repo_root / "pilot" / "conditions"))
    ap.add_argument("--sbs-csv", default=str(repo_root / "pilot" / "evaluation_sbs_results.csv"))
    ap.add_argument("--causal-relevance-csv",
                    default=str(repo_root / "pilot" / "causal_relevance_results.csv"))
    ap.add_argument("--out-csv", default="artifacts/judge_delta_results.csv")
    args = ap.parse_args()

    lookup = load_step_lookup(Path(args.sbs_csv))
    real = load_condition_rows(Path(args.conditions_dir), REAL)
    shuffled = load_condition_rows(Path(args.conditions_dir), SHUFFLED)
    substituted_conditions = load_condition_rows(Path(args.conditions_dir), SUBSTITUTED)

    real_scored = score_from_lookup(real, lookup)
    shuffled_scored = score_from_lookup(shuffled, lookup)
    substituted_scored = score_substituted_from_cache(
        substituted_conditions, Path(args.causal_relevance_csv))

    all_scored = pd.concat([real_scored, shuffled_scored, substituted_scored], ignore_index=True)
    all_scored.to_csv(args.out_csv, index=False)
    print(f"Scored {len(all_scored)} rows -> {args.out_csv}\n")

    fisher_reports = [pooled_fisher(real_scored[m], shuffled_scored[m], m)
                      for m in ("curr", "delta")]
    clustered_reports = [clustered_by_item(real_scored, shuffled_scored, m)
                        for m in ("curr", "delta")]
    print(format_report(fisher_reports, clustered_reports, substituted_scored))


if __name__ == "__main__":
    main()
