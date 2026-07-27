# Metric-A Anchor Set — Labeling Protocol (2026-07-25 growth round)

Who: 2 annotators, **Grace and Akhil**, each independently labeling the **full** growth
batch (full double coverage, not a partial-overlap sample — see §5). Model: **SDXL only**
(`stabilityai/stable-diffusion-xl-base-1.0`) — the SD1.5 anchor set (`artifacts/`) is frozen
historical data and is never relabeled or merged into this round; see CLAUDE.md.

## 1. What's being labeled

Each item is one generated image + one attribute (e.g. "who is wearing the red apron?").
The image was generated from a prompt naming 2, 3, or 4 subjects, each with one visual
attribute (a barista's red apron, a chef's white hat, etc.) from a fixed, locked vocabulary
of 7 subjects x 8 attributes — no new vocabulary is ever introduced by a growth batch, so
growth-batch items are not a confound with the original 24-prompt set.

The 2026-07-25 growth batch adds 85 new images (ids 100-184) on top of the existing 23
detected images (ids 0-23), built by recombining the same locked vocabulary into new
subject combinations at pinned seeds (`build_growth_specs.py`). Combined:

| stratum | images (target) | judgments (target) |
|---|---|---|
| n=2 | ~50 | ~100 |
| n=3 | ~33 | ~99 |
| n=4 | ~25-26 | ~100-104 |
| **total** | **~108-109** | **~300** |

Raw judgment count is not the number that matters — after excluding `unclear`/`none`/
`shared`/count-broken rows, the true "effective n" for binding accuracy is lower (today's
23-image set: 68 raw judgments -> 35 effective, ~51% strict pass rate). ~300 raw is sized to
clear a ~150 effective floor even after that shrinkage.

## 2. The blind design (why it's trustworthy)

`label_images.py` never shows the annotator the metric's own prediction
(`predicted_owner`/`model_scores`), only the image and the question. This is what makes
agreement between the human label and the metric's guess a meaningful number rather than an
anchored one — if the annotator could see the prediction, agreement would be inflated by
suggestion, not signal.

## 3. Per-image and per-attribute questions — Workstream 2 guidelines (2026-07-25)

For each detected image, the annotator answers two kinds of question. `label_images.py`
now prints this exact reminder at the start of every session (including a no-op resume),
so it's shown every time the tool launches, not just communicated once:

**A. Count check (once per image).**

- **Count-Clean** — the image renders the *exact* number of distinct subjects requested in
  the prompt.
- **Count-Broken** — the image is missing subjects, merges subjects together, or adds extra
  subjects.

This check exists because headcount passing Mask R-CNN's automated detection is a weak
proxy for "the prompt rendered correctly" — detection can find the right *number* of people
while two of them are visually merged/interchangeable, or the image otherwise doesn't match
what was asked for in a way headcount alone wouldn't catch.

**B. Attribute check (once per attribute).** Judge only what's visible in the rendered
image — never what the prompt *intended*. The prompt's intended pairing is not ground
truth; the human label on the rendered pixels is.

```
    (pick the subject if the attribute is PRESENT on them, else:)
    1) barista
    2) cyclist
    n) none / Missing   (attribute completely absent from the image)
    s) shared / Shared  (leaks onto / is shared by multiple subjects)
    u) unclear / Unclear (too blurry/occluded/low-quality to confirm)
```

- **Present** — the requested attribute clearly attaches to the correct subject (pick that
  subject from the menu).
- **Missing** (`none`) — the attribute is completely absent from the image.
- **Shared** (`shared`) — the attribute accidentally leaks onto / is shared by multiple
  subjects (e.g. both characters wearing red hats when only one was supposed to). This is
  different from Unclear: it's a real outcome the metric can be scored against (via its own
  top-two attention margin), not missing data.
- **Unclear** (`unclear`) — the render is too blurry, occluded, or low quality to
  definitively confirm if the attribute is present. This is missing data, not an outcome.

**Worked example** — prompt: *"A barista in a red apron and a chef in a blue hat"*
- Count check: both a barista AND a chef shown -> Count-Clean. Only one person shown ->
  Count-Broken.
- Attribute check: barista has a red apron -> Present. Chef has a white hat instead of blue
  -> Missing. Both wearing red aprons -> Shared. Chef's hat completely hidden off-screen ->
  Unclear.

## 4. Edge-case handling — pre-registered before labeling starts

Decided upfront, not adjusted after seeing results:

| label | meaning | how it enters analysis |
|---|---|---|
| single subject | clean binding outcome | **scored** — the primary accuracy denominator, against 1/n chance |
| `shared` | leakage (rendered on 2+ subjects) | **scored separately**, shared-aware mode: the metric may abstain via its top-two attention margin; chance = 1/(n+1). Reported as its own line, never pooled into the strict number. (`analyze_agreement.py --margin-threshold`) |
| `unclear` | can't assign ownership | **excluded** — reported as coverage, not error |
| `none` | never rendered | **excluded** from binding |
| **count-broken** | subjects not visually distinct | **excluded from pure binding evaluation**, logged and reported separately as a rendering-failure rate. `build_agreement_rows(..., counts=...)` forces `scored=False` on every attribute row of a count-broken image, regardless of the individual attribute answer. |

Rationale for keeping `shared` out of the strict number: folding it into `unclear` (an
earlier labeling pass's mistake) throws away a real, scoreable outcome. Rationale for
excluding count-broken entirely rather than scoring it: it conflates two different failure
modes (the model's binding logic vs. its rendering fidelity) that a pure binding metric
cannot and should not try to disentangle.

## 5. Two annotators, full double coverage

Grace and Akhil **each independently label 100% of the growth batch** — not a partial
overlap sample. Both run the identical tool, identical rules, blind (neither sees the
other's answers or the metric's prediction) until both are done.

- **Why full coverage instead of a 30-50% subset:** stronger kappa (computed over the whole
  set, not a sample of it) at the cost of roughly 2x total labeling effort split across the
  two of them — an explicit tradeoff, not a default.
- **No code or file changes needed for this design.** `cohens_kappa()` already computes
  agreement over whichever `(prompt_id, attribute)` keys **both** annotators have actually
  answered at the time it's run (`anchor_common.py`'s `shared_keys = sorted(set(labels_a) &
  set(labels_b))`) — so it works correctly on partial progress mid-labeling and
  automatically converges to full-set kappa once both finish. Neither annotator is blocked
  waiting on the other.
- **Target: Cohen's kappa >= 0.7.** Run `analyze_agreement.py --annotator grace
  --compare-annotator akhil` (or vice versa — symmetric) once both are far enough along to
  be useful; rerun anytime for a running readout as labeling progresses.
- If kappa falls short at completion, the disagreement rows (not the whole set) get a
  documented adjudication pass — decided if and when it's actually needed.
- **Open question, not blocking labeling:** which annotator's file becomes the "reference"
  human label set for `analyze_agreement.py --annotator <X>`'s metric-vs-human accuracy
  table (§7 of the memo's decision gate) — Grace's, Akhil's, or an adjudicated merge of
  both? Punt this until both files exist; it doesn't affect how either annotator labels.

## 6. Deliverables (per the original task brief)

1. **A labels file per annotator**: `labels_grace.json` + `counts_grace.json`, and
   `labels_akhil.json` + `counts_akhil.json` (per-attribute / per-image count-clean).
2. **An inter-rater agreement number**: Cohen's kappa between Grace and Akhil, printed by
   `analyze_agreement.py --annotator grace --compare-annotator akhil`.
3. **This document**: the labeling protocol and edge-case handling.

## 7. Model choice consistency

SDXL (`stabilityai/stable-diffusion-xl-base-1.0`) is canonical for this and all downstream
work — the discriminant-validity check, VQAScore baseline, and this growth round all target
it. `prompt_specs.json` and both generator scripts (`generate_anchor_images.py`,
`generate_anchor_images_sdxl.py`) keep an identical `ANCHOR_PROMPTS` literal for drift-guard
purposes, but `generate_anchor_images.py`'s `main()` is filtered to `FROZEN_PROMPT_IDS`
(0-23) so a rerun can never regenerate the growth batch on the wrong model.
