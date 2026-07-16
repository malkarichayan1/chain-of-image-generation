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
prompt from scratch. Instead, the pilot reuses EC's own **already-implemented**
per-step evaluator rather than inventing a new judge prompt.

**Correction after reading the actual repo code (`evaluate/evaluate_images.py` and
`evaluate/evaluate_sbs_images.py`):** the "visual census" JSON protocol (entity IDs
P1/P2, Gemini 2.5 Pro, entity/attribute/interaction JSON) lives in
`evaluate_images.py`, but that script only grades a single `baseline.png` per item
against the full ground-truth list — it's the tool behind the separate
baseline-vs-CoIG entity-collapse comparison (paper Section 6), not a per-step
evaluator, and it cannot be pointed at an arbitrary step image. The script that
actually operates on individual step images is `evaluate_sbs_images.py`: for every
step image it asks a set of yes/no questions already present in
`generated_prompts.csv` (`question_attr_1..4`, `question_interaction_1..2`,
`question_count`), each tied to one specific ground-truth attribute or interaction.
This — not `evaluate_images.py` — is the pilot's judge foundation.

This turns out to simplify the pilot considerably: running
`evaluate_sbs_images.py --all` once (every step × every ground-truth question, for
all 10 chains) gives a complete truth table of "does attribute k appear in step t's
image" with zero extra design work. Real and Shuffled are then just different
*relabelings* of that same already-collected table — no new judge prompt, no new API
calls. Only Substituted needs new questions, since it asks about an attribute that
was never part of a given chain's ground truth and so was never asked against that
chain's images. See [`../pilot/README.md`](../pilot/README.md) for the exact
mechanics and [`../pilot/causal_relevance.py`](../pilot/causal_relevance.py) for the
implementation.

MLLM: `evaluate_sbs_images.py` defaults to Gemini 2.5 Pro in the repo; this pilot's
new Substituted-condition calls (`pilot/judge.py`) default to **Gemini 2.5 Flash** to
match the key being provided, noting the accuracy tradeoff as a documented
limitation. A Pro spot-check on a few chains is worth doing if Flash's answers look
unreliable.

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

Implemented in [`../pilot/`](../pilot/); see [`../pilot/README.md`](../pilot/README.md)
for exact commands.

1. `pilot/select_prompts.py` — freeze 10 EC `item_index` values into
   `pilot_item_indexes.txt`, filtering out prompts whose attributes mention
   gray/grey (the ARM's placeholder color per the paper's evaluation notes).
2. `coig/create_prompt/create_prompt_sbs.py` (vendored, unmodified) — CSP
   decomposition of the 10 prompts into ordered sub-prompt sequences.
3. `coig/create_images/generate_multi_step_image_genai_simple.py` (vendored,
   unmodified) — ARM generates the base image chains once, lock ON. This is the
   only generation step in the whole pilot.
4. `coig/evaluate/evaluate_sbs_images.py --all` (vendored, unmodified) — judges
   every step image against every ground-truth attribute/interaction question,
   for all 10 chains. This single run supplies everything Real and Shuffled need.
5. `pilot/build_conditions.py` — relabels that truth table into three condition
   manifests (`conditions/{real,shuffled,substituted}.json`): Real claims each
   attribute at its true introduction step; Shuffled claims it at a different
   step of the same chain; Substituted claims an attribute that belongs to a
   different chain entirely.
6. `pilot/causal_relevance.py` — for Real/Shuffled, pulls appears-at-step and
   persists-to-final straight from step 4's results (no new API calls). For
   Substituted, asks two new yes/no questions per chain (the only new judge
   calls in the whole pilot).
7. `pilot/score_pilot.py` — mean ± sd of appears-at-step and persists-to-final
   per condition, plus the go/no-go read. n = 10 chains → read effect sizes, not
   p-values.

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

- Code is written (`pilot/`) but not yet run — blocked on `GOOGLE_AI_API_KEY`.
- The CR judge prompt (`pilot/judge.py`) has not been validated against the
  paper's own numbers yet — sanity-check the Real condition's numbers before
  trusting Shuffled/Substituted.
- Flash vs. Pro for the Substituted-condition judge is an open accuracy/cost
  tradeoff (see above); `evaluate_sbs_images.py`'s own Real/Shuffled data was
  collected with whatever `--model` it's run with (defaults to Pro in the
  vendored script).
- `build_conditions.py` resolves each attribute's "true step" as the earliest
  step where the judge already said yes — if an attribute is never detected in
  any step (a judge miss, not a lock failure), that chain/attribute is silently
  excluded from all three conditions. Worth checking how many chains this drops
  once the pilot actually runs.
