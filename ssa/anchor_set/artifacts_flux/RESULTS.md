# Results: FLUX.1-dev cross-attention capture — Part A 5-experiment battery

2026-08-03. First real run of `run_five_experiments.py` against `artifacts_flux/` with
actual captured cross-attention (`model_scores`/`model_scores_full`), not the
`make_dummy_artifacts.py` smoke test. Manifest verified beforehand (all 105 images present,
attribute counts match `n`, 0/301 rows still `"unavailable"`, seeds match the SDXL
manifest's pinned seeds on every overlapping `prompt_id`) — merged from
`worktree-flux-attention-hook`, commit `c54cadc`.

Run against all three annotators independently (annotator1/annotator3/annotator2) rather than picking one,
since inter-rater kappa on this label set turned out to be high (0.95+, see Exp 5/agreement
note below) — the three runs agree closely, which is itself part of the evidence this data
is trustworthy. Raw output: `five_experiments_{annotator1,annotator3,annotator2}.{json,md}`,
`agreement_annotator1.csv`.

## Experiment 1 — accuracy per subject count vs. 1/n chance

Does the FLUX attention-based `predicted_owner` prediction beat chance at each stratum?

| annotator | n=2 acc | n=3 acc | n=4 acc | overall | overall n_scored |
|---|---|---|---|---|---|
| annotator1 | 97.5% (78/80) | 92.9% (65/70) | 71.5% (88/123) | 84.6% | 273 |
| annotator3  | 97.5% (77/79) | 94.3% (66/70) | 72.5% (87/120) | 85.5% | 269 |
| annotator2  | 97.4% (76/78) | 95.6% (65/68) | 73.1% (87/119) | 86.0% | 265 |

Chance is 50% / 33.3% / 25.0% at n=2/3/4. Every stratum, every annotator: binomial
p<0.0001. This is the least interesting of the five results on its own — FLUX's *labeled*
binding rate was already known to be high (93.8% in the design doc's pre-capture check) — but
it confirms the newly-captured attention signal isn't degenerate or randomly wrong before
looking at the more diagnostic experiments below.

## Experiment 2 — early-window vs. full-trajectory attention

Does restricting to the early denoising window (steps 0–12 of 25) change the prediction
versus using the full 25-step trajectory?

| annotator | n=2 (early/full) | n=3 (early/full) | n=4 (early/full) | McNemar p |
|---|---|---|---|---|
| annotator1 | 0.975 / 0.975 | 0.929 / 0.929 | 0.715 / 0.707 | 1.0 |
| annotator3  | 0.975 / 0.975 | 0.943 / 0.943 | 0.725 / 0.717 | 1.0 |
| annotator2  | 0.974 / 0.974 | 0.956 / 0.956 | 0.731 / 0.723 | 1.0 |

Effectively no difference — at most 1 discordant prediction out of 265–273 rows in every
run. FLUX's subject-attribute binding decision looks like it's already settled within the
early half of denoising; the late-trajectory steps add nothing (and in the n=4 stratum,
non-significantly hurt slightly). Consistent with the DAAM/Attend-and-Excite framing this
project has been using throughout — early steps decide layout/binding, late steps refine
texture.

## Experiment 3 — attention-randomization falsification (the primary claim)

Per the design doc, this was the specific claim the capture work was built to support
(Exp 4 has no headroom at n=2, so it can't cleanly demonstrate the signal is real). Cross-
item scramble: shuffle `model_scores` across *different* images within the same stratum,
recompute accuracy — if real attention carries no information, scrambled accuracy should
land at chance and be indistinguishable from real.

| annotator | stratum | median scrambled acc | chance | real acc | McNemar p (real vs. scrambled, seed=42) |
|---|---|---|---|---|---|
| annotator1 | n=2/3/4 | 0.500 / 0.329 / 0.252 | 0.500 / 0.333 / 0.250 | 0.975 / 0.929 / 0.715 | **1.77e-33** |
| annotator3  | n=2/3/4 | 0.494 / 0.329 / 0.250 | 0.500 / 0.333 / 0.250 | 0.975 / 0.943 / 0.725 | **1.77e-33** |
| annotator2  | n=2/3/4 | 0.500 / 0.324 / 0.252 | 0.500 / 0.333 / 0.250 | 0.974 / 0.956 / 0.731 | **2.22e-29** |

(200 scramble seeds per run.) Scrambled accuracy sits essentially exactly on the chance
line in every stratum — the falsification test is clean, not just "significant but still
elevated." Real vs. scrambled McNemar comparisons are overwhelming: real gets ~150 rows
right that scrambled gets wrong, vs. ~10-13 the other way, out of ~265-273 discordant-eligible
rows. This is the strongest single result in the battery and the one the design doc
prioritized building toward.

## Experiment 4 — vs. nearest-subject-noun positional baseline

Does the attention-based metric beat the "just guess the nearest subject noun in the
prompt" baseline (bug-fixed this session for FLUX's sub-phrase attribute strings, e.g.
`"yellow helmet"` inside `"...yellow bike helmet..."`)?

| annotator | metric acc | baseline acc | McNemar p | metric-only wins | baseline-only wins |
|---|---|---|---|---|---|
| annotator1 | 84.6% | 77.7% | 0.0271 | 43 | 24 |
| annotator3  | 85.5% | 77.7% | 0.0139 | 44 | 23 |
| annotator2  | 86.0% | 78.9% | 0.0271 | 43 | 24 |

Significant overall for all three annotators. Per-stratum, as the design doc predicted, n=2
is close to a wash (metric 97.4-97.5% vs. baseline 88.5-88.8% — baseline is already strong
there since a 2-subject prompt gives it little room to be wrong), and the gap is more real
at n=3 (metric ~93-96% vs. baseline ~80-82%) and persists at n=4 (metric ~72-73% vs.
baseline ~69-71%). The metric isn't just recovering the positional baseline's information —
it adds real signal on top, most visibly at n=3.

## Experiment 5 — count-clean subset vs. all rows

| annotator | rows excluded as count-broken | all-rows overall acc | count-clean overall acc |
|---|---|---|---|
| annotator1 | 0 / 297 | 84.6% | 84.6% (identical) |
| annotator3  | 0 / 297 | 85.5% | 85.5% (identical) |
| annotator2  | 0 / 297 | 86.0% | 86.0% (identical) |

No count-broken images in this dataset for any annotator, so this experiment is a null
result by construction — included for completeness/consistency with the SDXL battery, where
it mattered a lot (96/306 rows excluded there). Worth flagging as a genuine difference
between the two anchor sets, not a bug: FLUX's Mask R-CNN person-count detection matched the
prompt's intended subject count far more reliably than SDXL's did on the growth batch.

## Inter-rater reliability (context for trusting the above)

`analyze_agreement.py --artifacts-dir artifacts_flux`, 301 overlapping judgments per pair:

| pair | raw agreement | Cohen's kappa |
|---|---|---|
| annotator1 vs. annotator3 | 289/301 = 96.0% | 0.954 |
| annotator1 vs. annotator2 | 290/301 = 96.3% | 0.958 |

Both comfortably clear the κ≥0.7 target that the SDXL anchor set (κ=0.682) missed. The three
annotator-specific 5-experiment runs above agree closely with each other, which is exactly
what you'd expect given this — it's a mutual check, not three independent findings.

## What this does and doesn't establish

**Established:** FLUX.1-dev's double-block cross-attention (the 19 `FluxTransformerBlock`s)
carries real subject-attribute binding information — not a geometric or positional artifact,
since it survives cross-item randomization comparison (Exp 3) and beats a positional
baseline (Exp 4) — and that information is available early in denoising (Exp 2), on
well-agreed ground truth (kappa 0.95+).

**Not established / explicitly out of scope:**
- The 38 `FluxSingleTransformerBlock`s were never hooked (design doc non-goal — no clean
  text/image token boundary to hook without more invasive changes). This result speaks only
  to the double blocks.
- Part C's robustness battery (Holm-Bonferroni correction across the 5 experiments,
  leave-one-prompt-out, RNG-seed sweep stability) has been run against the SD1.5 chain data
  (`experiments/`) but **not yet against this FLUX data** — the p-values above are
  from single runs at the frozen operating point, not yet stress-tested the way that project's
  numbers were.
- This is one-shot binding (Part A framing), not the chain/lock-confound setting (Part B) —
  it says nothing about whether FLUX-based chain generation would suffer the same
  compositional-lock confound Track 1 found for the original CoIG pipeline.
