# Pilot Design: 10 Real vs. 10 Shuffled vs. 10 Substituted

Status: **planned, not yet executed** — waiting on a Gemini 2.5 Flash API key.

## Goal

Before committing to the full ~100-prompt study, run a cheap 10-chain-per-condition
pilot to decide whether CoIG's Causal Relevance metric is worth auditing further.
If Real, Shuffled, and Substituted score similarly, the metric is likely dominated
by the compositional lock's mechanical persistence rather than genuine semantic
faithfulness — that's the finding the full study would chase. If Real clearly beats
the negative controls, the audit angle is weak and effort should shift to Track 2
(a new faithfulness metric).

## Locked decisions

- **Dataset:** Entity Collapse (EC) benchmark — the repo's built-in ~300 profession
  prompts (bartender, pharmacist, airman, ...), NOT T2I-CompBench Color. Chosen
  because the repo's CSP → ARM → judge pipeline is already wired end-to-end for EC,
  giving the fastest path to a go/no-go result.
- **CR method:** no-regen alignment. Images are generated once from the real,
  correctly-ordered sub-prompts with the compositional lock ON. Only the sub-prompt
  *text* is swapped across the three conditions — no image is ever regenerated from
  a shuffled or substituted sub-prompt. This matches the PI's correction: regenerating
  per condition would build a faithful chain of a different image and destroy the
  ground-truth unfaithfulness the pilot depends on.

## Why this reuses (and departs from) the paper's own Causal Relevance protocol

The paper's own Causal Relevance evaluation (Appendix 9) is scoped to **T2I-CompBench**,
not EC. Its protocol: perturb one attribute in a step's sub-prompt (e.g. "red bowl" →
"blue bowl"), **regenerate that step and everything downstream under the lock**, then
check whether the change appears in the intermediate image and persists to the final
image. MLLM = Gemini 2.5 Flash, using a targeted QA template ("Is the {object} present?
Is it {count} in count and {color} in color?"). Reported numbers: unperturbed ≈ 1.35,
perturbed-at-step ≈ 85.32, persists-to-final ≈ 89.52.

That is a *different, narrower* experiment than this pilot: it tests whether a genuine
content change survives the lock, not whether lock-driven persistence gets mistaken for
faithfulness. This pilot is a negative-control extension of that idea onto EC — the
85.32 / 89.52 numbers from the paper are not directly comparable to whatever this pilot
produces, and that should be stated explicitly in any writeup.

## Why EC's own judge protocol, not a new one

The Causal Relevance description in §4 is too vague to safely reconstruct a judge
prompt from scratch. Instead, the pilot reuses EC's own **already-implemented and
paper-validated** evaluation protocol from Appendix 9 — the "visual census":

- MLLM: Gemini 2.5 Pro in the paper (nuance needed for entity-collapse judgments);
  this pilot defaults to **Gemini 2.5 Flash** to match the key being provided, noting
  the accuracy tradeoff as a documented limitation. A Pro spot-check on a few chains
  is worth doing if Flash's entity counts look unreliable.
- Protocol: enumerate every visible entity in an image, assign a unique ID (P1, P2, ...),
  and — under a strict visual-evidence-only constraint (ignore anything mentioned in
  text but not actually rendered) — output structured JSON binding attributes and
  interactions to each ID.
- Already implemented in the forked repo's `evaluate/evaluate_images.py`.

## Why the compositional lock predicts a specific failure pattern here

The CSP system prompt (Appendix 9, rule 3, "Entity Persistence and Immutable Locking")
states outright that once a region is generated, the model is instructed to **"strictly
forbid any alterations to their shape, position, or appearance"** in later steps. This
is a direct textual confirmation — not an inference from behavior — of the mechanical
confound this pilot is built to expose:

- **Real:** step's own sub-prompt → attribute appears at `I_t` *and* persists to `I_n`
  → high on both.
- **Shuffled** (a real sub-prompt from a *different* step of the same chain): the
  attribute is somewhere in the final image because the lock preserved it — so
  **persists-to-final stays high** even though **appears-at-step drops**, since the
  claimed attribute wasn't generated at that particular step. This split — high
  persistence despite a step-level mismatch — is where the confound shows up first.
- **Substituted** (a sub-prompt from an unrelated chain/profession): the named
  attribute was never generated anywhere in this image → low on both.

The pilot logs **appears-at-step** and **persists-to-final** as two separate numbers
per chain/condition, not a single blended score, specifically so this split is visible.

## Steps

1. Read the real repo I/O before writing anything: `create_prompt/create_prompt_sbs.py`,
   `create_images/generate_multi_step_image_genai_simple.py`,
   `evaluate/evaluate_images.py`, `evaluate/evaluate_sbs_images.py`. Confirm the exact
   per-item folder layout, step-file naming, and judge JSON schema.
2. Environment + smoke test: install `coig/requirements.txt`, set `GOOGLE_AI_API_KEY`,
   push one EC prompt through CSP → ARM → judge to confirm the key and quota work
   before spending on 10.
3. Freeze 10 EC prompts into `pilot_prompts.csv` (`item_index` 0–9). Prefer prompts
   whose attributes are visually unambiguous — avoid gray, which the ARM uses as a
   placeholder color per the paper's own evaluation notes.
4. Generate the base chains once: CSP → ARM, lock ON. This is the only generation
   step in the whole pilot. Save per-step PNGs per the repo's existing layout.
5. Build three text manifests over the same fixed images — `real.json`,
   `shuffled.json` (permuted sub-prompt order within each chain), `substituted.json`
   (each step's text swapped for an out-of-chain sub-prompt). No new images.
6. Run the EC visual-census judge (from `evaluate_images.py`) once per step image and
   once per final image, producing entity→attribute JSON for each. For every
   condition, check whether the attribute *named in that condition's sub-prompt* for
   step `t` appears in `I_t`'s JSON (appears-at-step) and in `I_n`'s JSON
   (persists-to-final).
7. Score and compare: mean ± sd of appears-at-step and persists-to-final per
   condition across the 30 chains (10 × 3), plus a paired comparison
   Real–Shuffled / Real–Substituted. n = 10 → read effect sizes, not p-values.

## Go / no-go

- **Persists-to-final stays high across all three conditions** (especially Shuffled)
  → Causal Relevance is riding the lock, not tracking semantics → finding →
  continue to the full study.
- **Real clearly beats Substituted on both measures** → the metric tracks real
  semantic influence → the audit angle is weak → pivot effort to Track 2 (the new
  Anchor / attention-alignment metric).
- **Real > Shuffled > Substituted, graded** → partial sensitivity → still a
  publishable nuance, worth the full ~100-prompt run to firm up.

## Known gaps / open items

- Not yet confirmed whether the EC CSP's attribute vocabulary produces a
  well-formed Substituted pool (need to read `create_prompt_sbs.py` first).
- The CR judge prompt has not been validated against the paper's own numbers yet —
  do that on the Real condition before trusting Shuffled/Substituted results.
- Flash vs. Pro for the EC judge is an open accuracy/cost tradeoff (see above).
