# Design: Part A Five-Experiment Battery

Date: 2026-07-27. Branch: `sdxl`. Author: Annotator 1 (with Claude).

Scaffolded ahead of Workstream 2 (the labeling-protocol growth run, `docs/anchor-set-labeling-protocol.md`)
finishing, so all five experiments below can run in a single pass the moment Annotator 2/Annotator 3's
validated labels land on the expanded SDXL anchor set. **Nothing in this document is run against
real data yet.** Every script ships against `artifacts_dummy/` (synthetic, `make_dummy_artifacts.py`)
until pointed at `artifacts_sdxl/` with `--artifacts-dir`.

## Context

Five questions, each defending against a different objection to metric A (the one-shot
attribute-binding metric):

1. Does cross-attention beat random guessing, per subject count (n=2/3/4, chance = 1/n)?
2. Does early-window attention (steps 0-15 of 30) beat full-trajectory averaging (all 30 steps)?
3. Does the signal survive attention randomization (falsification)?
4. Does the metric beat a naive positional (nearest-noun) baseline?
5. What happens restricted to count-clean items only (rendering failure vs. binding failure)?

All five read `anchor_common.py`'s existing manifest/labels/counts schema
([anchor_common.py](../ssa/anchor_set/anchor_common.py)) — no new file formats except where noted
in Experiment 2.

## Non-goals (tracked separately, not attempted here)

- Running any of this against `artifacts_sdxl/` — blocked on Workstream 2's labels landing.
- Re-labeling or re-counting anything; this battery only reads existing `labels_*.json` /
  `counts_*.json` conventions.
- Segmenter diversity, VQAScore correlation, causal intervention (Attend-and-Excite steering) —
  separate, lower-priority Part A/B work per `CLAUDE.md`'s priority list.
- Actually rerunning `generate_anchor_images_sdxl.py` on Kaggle for Experiment 2's full-trajectory
  capture (see Experiment 2 below) — the code is patched additively here; the rerun itself is a
  separate, cheap follow-up task, schedulable independently of Workstream 2.

## Architecture

Five sibling scripts in `ssa/anchor_set/` (flat, matching the existing `discriminant_validity_check.py`
/ `analyze_agreement.py` convention — this directory has no subpackages), one orchestrator, one
dummy-data generator, one shared helper addition:

```
ssa/anchor_set/
  anchor_common.py            # +binomial_test_vs_chance (shared by exp1, exp3)
  exp1_accuracy_by_n.py        # Q1: headline accuracy vs 1/n, per stratum
  exp2_window_ablation.py      # Q2: early-window vs full-trajectory accuracy
  exp3_attention_scramble.py   # Q3: cross-item same-stratum scramble, falsification
  exp4_positional_baseline.py  # Q4: nearest-preceding-subject-noun baseline
  exp5_count_clean_subset.py   # Q5: count-clean-only vs all-rows accuracy
  run_five_experiments.py      # orchestrator: one command, combined report
  make_dummy_artifacts.py      # writes artifacts_dummy/ (manifest + labels + counts)
  generate_anchor_images_sdxl.py  # +model_scores_full capture (additive, not run)
  tests/
    test_exp1_accuracy_by_n.py
    test_exp2_window_ablation.py
    test_exp3_attention_scramble.py
    test_exp4_positional_baseline.py
    test_exp5_count_clean_subset.py
    test_run_five_experiments.py
```

## Experiment 1 — headline accuracy per subject count

Thin wrapper over already-tested `build_agreement_rows()` / `summarize_agreement()`
([anchor_common.py:264-378](../ssa/anchor_set/anchor_common.py#L264-L378)), which already computes
per-stratum accuracy against chance = 1/n. The only new logic: a one-sided binomial significance
test per stratum (`binomial_test_vs_chance`, added to `anchor_common.py` since Experiment 3 needs
the same primitive — extracted rather than duplicated, per this repo's existing DRY discipline
around `predicted_owner_from_attention` being "kept in sync" across files).

**Decision rule, pre-registered:** a stratum is reported as "beats chance" only if its one-sided
binomial p < 0.05 against that stratum's own 1/n. Strata with `n_scored == 0` report `p_value=None`,
not a spurious significant/non-significant call.

**Acceptance:** `exp1_accuracy_by_n.py` reproduces `analyze_agreement.py`'s strict-mode accuracy
numbers exactly when run on the same manifest/labels (regression-tested against a shared fixture),
plus adds the per-stratum p-value column analyze_agreement.py doesn't have.

## Experiment 2 — early-window vs. full-trajectory attention

**The real gap in this battery.** The manifest only ever stored one attention aggregation:
`model_scores`, computed with `max_steps = NUM_INFERENCE_STEPS * EARLY_WINDOW_FRACTION = 15`
([generate_anchor_images_sdxl.py:81-82](../ssa/anchor_set/generate_anchor_images_sdxl.py#L81-L82)).
There has never been a full-trajectory (`max_steps=30`) number to compare against.

Tracing the generation loop
([generate_anchor_images_sdxl.py:712-719](../ssa/anchor_set/generate_anchor_images_sdxl.py#L712-L719)):
`capture.unhook_pipeline()` — which discards `AttentionStore.step_store` — isn't called until
*after* every attribute's `model_scores` is computed. The full per-step attention is still resident
in memory at that point. This means a full-trajectory aggregate is capturable **for free, in the
same generation run, at the same seed, with pixel-identical output** — it just requires one more
call to `phase_b_cross_attention_map(max_steps=NUM_INFERENCE_STEPS)` per attribute before the
unhook. See the additive patch to `generate_anchor_images_sdxl.py` (new `model_scores_full` /
`predicted_owner_full` fields per attribute, plus a `PIN_SEEDS_FROM_MANIFEST` top-level constant
that forces each prompt to reuse its already-recorded seed instead of re-running the retry loop,
guaranteeing a bit-identical rerun with zero risk of landing on a different seed). This is a
constant edited before each Kaggle push, not a CLI flag — this script has no argparse anywhere
and Kaggle kernels don't take custom argv; `GROWTH_PROMPT_IDS` already uses exactly this same
edit-the-constant-before-pushing convention for selecting which prompt ids to run.

**This patch is scaffolded now, not executed.** Real full-trajectory data requires one Kaggle
rerun of the patched script against the pinned seeds already in `artifacts_sdxl/manifest.json` —
independent of Workstream 2 (it doesn't change any image pixel, so it can happen before, during,
or after the labeling pass without invalidating any label already collected).

`exp2_window_ablation.py` reads a single manifest and compares `model_scores`-derived accuracy
against `model_scores_full`-derived accuracy on the same rows (paired McNemar, mirroring
`discriminant_validity_check.py`'s `mcnemar_report`
([discriminant_validity_check.py:98-110](../ssa/anchor_set/discriminant_validity_check.py#L98-L110))).
If a manifest lacks `model_scores_full` on any attribute, the script reports
`"unavailable — rerun generate_anchor_images_sdxl.py with the model_scores_full patch"` rather than
crashing, so `run_five_experiments.py` can still complete 4/5 sections against `artifacts_sdxl/`
today.

**Decision rule, pre-registered:** if full-trajectory accuracy is not meaningfully lower than
early-window accuracy (McNemar p >= 0.05, or full-trajectory actually higher), the earlier
sub-chance n=2/n=3 result (`CLAUDE.md`'s open puzzle) is NOT explained by aggregation window, and
that puzzle stays open. If full-trajectory is significantly worse, that supports the
"late/texture-focused steps dilute the binding signal" hypothesis already on file.

**Acceptance:** runs against `artifacts_dummy/`'s manifest (which `make_dummy_artifacts.py`
populates with both fields) today; reports "unavailable" cleanly against the current
`artifacts_sdxl/manifest.json` (no `model_scores_full` yet) without crashing.

## Experiment 3 — attention-randomization falsification

**Design note, pre-registered before implementation:** the obvious design — permute which subject
each attribute's `model_scores` values belong to, *within* the same item — degenerates at n=2.
A 2-element permutation has exactly one non-identity option (a forced swap), so the scrambled
prediction becomes deterministically `argmin` instead of `argmax` of the *same two numbers* —
scrambled accuracy at n=2 would equal `1 − real accuracy` by arithmetic, not by any property of
attention. This is not an independent falsification test at that stratum.

Instead, `exp3_attention_scramble.py` scrambles **across items within the same n-stratum**: for
each scored attribute, draw a different item's `model_scores` dict (same n, so the same subject
slots exist) and use *that* item's scores to make the prediction — i.e., ask whether the metric
would have done just as well pretending it saw a completely unrelated item's attention. This is
non-degenerate at every n and mirrors the `attn_scrambled_crosschain` condition metric B already
uses (`pi_level_experiment/`, per `CLAUDE.md`'s Part D). Run as a sweep over many seeds (200,
matching `pi_level_experiment/rng_sweep.py`'s precedent) rather than a single shuffle — a lone draw
at n=2 or n=3 is exactly the kind of "lucky/unlucky single permutation" this repo has previously
flagged as insufficient evidence on its own.

**Decision rule, pre-registered:** report, per stratum, the fraction of the 200 seeds where a
two-sided exact binomial test (scrambled correct count vs. n_scored, against that stratum's own
1/n) does NOT reject the null (p >= 0.05) — a contrast is "falsification-clean" only if scrambled
accuracy is statistically indistinguishable from chance in at least 95% of seeds. Also report a
paired McNemar between real and scrambled correctness at one fixed, pre-registered seed (42,
matching this repo's existing `DEFAULT_SEED` convention in `pi_level_experiment/`) on the same
rows: real should beat scrambled if attention content is doing the work, not just box geometry or
prompt structure.

**Acceptance:** raises a clear error (not a silent skip) if a stratum has fewer than 2 distinct
items to draw a cross-item partner from — the real anchor set has 18-23+ items per stratum, but
`artifacts_dummy/` must be sized to match or the test would never exercise the real code path.

## Experiment 4 — positional (nearest-noun) baseline

Mirrors `discriminant_validity_check.py`'s bigbox-baseline pattern exactly
([discriminant_validity_check.py:86-110](../ssa/anchor_set/discriminant_validity_check.py#L86-L110)):
join each scored row to "which subject would the trivial nearest-preceding-noun heuristic have
picked," then compare accuracy and run McNemar. For attribute at character offset `a_i` in the
prompt, the baseline's guess is the subject whose own first occurrence is the closest *preceding*
occurrence to `a_i` (ties broken by subject list order, matching `predicted_owner_from_attention`'s
existing insertion-order tie-break for consistency).

**Design note, pre-registered before implementation:** checked all 306 real scored/labeled
attribute rows in `artifacts_sdxl/manifest.json` — nearest-preceding-subject equals
`intended_subject` 306/306 times. The current prompt template ("a photo of a `subject` `verb` a
`attribute` and a `subject` ...") never lets a second subject intervene between a subject and its
own attribute, so this baseline is currently mathematically identical to "always guess the
intended pairing." It is still exactly the baseline you specified (nearest subject noun, no
attention needed) and still a meaningful comparison — a metric that can't beat "guess the intended
pairing" hasn't shown attention adds anything — but it will not discriminate differently from an
"always guess intended_subject" baseline until/unless a future prompt template interleaves subjects
and attributes differently. Documented here rather than silently building something more elaborate
than what was asked for.

**Decision rule, pre-registered:** report metric accuracy vs. baseline accuracy vs. chance, all
three, per stratum, plus McNemar between metric and baseline. If the metric does not beat this
baseline at a given stratum, state that plainly — per the prompt template's degeneracy above, this
is equivalent to saying the metric can't beat "guess the intended pairing" there.

**Acceptance:** on the real 306-row sample described above, the baseline scores 100% against
`intended_subject`; the test suite includes at least one hand-built prompt where a second subject
*does* intervene, so the "nearest preceding occurrence" logic itself is exercised on a non-trivial
case, not just validated against the current degenerate template.

## Experiment 5 — count-clean subset

**Correction after re-reading the current code (it had changed since this document's first
draft):** `analyze_agreement.py` already wires `counts_<annotator>.json` into `analyze()`
automatically ([analyze_agreement.py:90-105](../ssa/anchor_set/analyze_agreement.py#L90-L105)) —
count-broken images are already excluded from the reported accuracy whenever a counts file exists
for that annotator. So this is not first-time wiring. What `analyze_agreement.py` does NOT do is
show the comparison Q5 actually asks for ("what HAPPENS when we restrict" implies a before/after):
it reports exactly one view (filtered if a counts file exists, unfiltered if not), never both side
by side. `exp5_count_clean_subset.py`'s actual contribution is calling
`build_agreement_rows`/`summarize_agreement()` twice on the same manifest+labels — once with
`counts=None` (all detected images) and once with `counts=<loaded>` (count-clean only) — and
reporting both tables together, plus how many rows were excluded as count-broken per stratum, so
a reader sees the shift, not just one already-filtered number.

**Decision rule, pre-registered:** report both tables unconditionally, not just the count-clean
one — a reader needs to see whether restricting to count-clean *changes* the accuracy number
materially or just shrinks n. (Real data note, not a decision rule: as of 2026-07-25's growth run,
count-broken is disproportionately an n=4 problem — 8/23 n=4 rows scored count-clean vs. 23/23
scored unfiltered per the current Annotator 3 counts — restricting to count-clean at n=4 will have real
statistical-power consequences worth stating, not just computing.)

**Acceptance:** on real data (when run against `artifacts_sdxl/`), reproduces the row counts already
verified by hand: 105 detected images / 306 raw judgments, count-clean-only strict-scored subtotals
of 18/14/8 at n=2/3/4 (Annotator 3's counts) vs. 18/16/23 unfiltered.

## Orchestrator

`run_five_experiments.py --artifacts-dir <dir> --annotator <name>` runs all five, prints each
report in the existing `_print_report`-style format, and writes a combined
`five_experiments_<annotator>.json` + `.md` to the artifacts directory. Experiment 2 degrades
gracefully (see above) rather than failing the whole run. `make_dummy_artifacts.py` writes a
synthetic `artifacts_dummy/` sized to match real proportions (~20/15/20 detected items per
n=2/3/4 stratum, both `model_scores` and `model_scores_full` populated, a `labels_dummy.json` and
`counts_dummy.json` with a realistic mix of subject/none/unclear/shared and clean/broken) so the
orchestrator is smoke-testable end-to-end today.

## Testing

Every function ships with a test against small hand-built manifest/label dicts, following the
existing `_row()`-helper pattern in `tests/test_discriminant_validity_check.py` — no fixture
files, matching every existing test in this directory. `generate_anchor_images_sdxl.py`'s patch is
**not** locally testable — this machine has no torch/diffusers (the same boundary
`anchor_common.py`'s own docstring already documents), and the patch reuses existing tested
functions (`phase_b_cross_attention_map`, `predicted_owner_from_attention`) with a different
`max_steps` argument rather than adding new logic, so the risk is reviewed, not unit-tested.

## Risks

- **Experiment 2 still can't run on real data the moment Workstream 2 lands.** It additionally
  needs the Kaggle rerun described above. Stated plainly in this doc and in the orchestrator's
  own output, not silently assumed away.
- **Experiment 3's cross-item scramble needs >=2 items per stratum.** True today (18-23+ per
  stratum); `artifacts_dummy/` is sized to match so this isn't accidentally untested.
- **Experiment 4's baseline is currently indistinguishable from "guess intended_subject."** Real
  finding, not a bug — stated above, will be restated in the eventual results writeup rather than
  presented as more novel than it is.
- **The GPU patch (Experiment 2) could regress the existing early-window `model_scores` if done
  carelessly.** Mitigated by making it strictly additive (new dict keys only) and reusing the
  already-tested aggregation function unchanged.
