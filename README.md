# Chain of Image Generation — Faithfulness Audit

**What cross-attention in text-to-image diffusion actually encodes — and why it cannot be used
to score faithfulness.**

CPGA Summer 2026. Builds on [arXiv:2512.08645](https://arxiv.org/abs/2512.08645), *"Chain-of-Image
Generation: Toward Monitorable and Controllable Image Generation."*

## The finding

We set out to build a faithfulness metric that reads a diffusion model's own cross-attention to
decide which subject an attribute is bound to — is the red apron on the barista or the cyclist?
Across two model families (SDXL, FLUX.1-dev), three annotators, and ~600 human-labeled attribute
judgments, the metric beats chance and survives a falsification battery.

Then we ran the one baseline our own pre-registered plan named but never executed: *assume the
model rendered what the prompt asked for.* No image, no attention, no computation.

**It beats the attention metric on every model and every annotator — six comparisons, six losses,
all p < 0.05.** And on the subset where the model *disobeyed* the prompt — the only rows where a
faithfulness metric can add value over reading the prompt — attention's accuracy at predicting the
rendered outcome is not distinguishable from chance in any of six tests.

Attention is not noise; on FLUX it beats a randomization control at p ≈ 1e-33. But what it carries
appears to be information about **the prompt**, predictive of the image only because these models
usually obey. This is the diffusion analogue of NLP's 2019 "Attention is not Explanation" argument.

The motivating result is separate and self-contained: CoIG's own Causal Relevance score cannot
distinguish a faithful chain from a shuffled one (`persists_to_final` = 0.83 for both), because the
compositional lock mechanically preserves whatever is already in frame.

## Read this first

**[`docs/raw-attention-paper-briefing.md`](docs/raw-attention-paper-briefing.md)** — the ground-up
writeup: claims and what supports each, the measurement apparatus, the reframe, literature
positioning, and what is still missing. Single source of truth for the paper.

**[`CLAUDE.md`](CLAUDE.md)** — working context: repo state, which results are solid vs.
underpowered, and what to do next.

## Contents

| Path | What |
|---|---|
| [`ssa/anchor_set/`](ssa/anchor_set/) | The single-image audit: image generation, attention capture, the seven experiments, human anchor sets for SD1.5 / SDXL / FLUX |
| [`pi_level_experiment/`](pi_level_experiment/) | The chain track: generate → segment → score → analyze, plus [`RESULTS.md`](pi_level_experiment/RESULTS.md) |
| [`pilot/`](pilot/) | The Causal Relevance audit (the paper's motivation) and `spatial_semantic_alignment.py`, the chain metric |
| [`docs/`](docs/) | Paper briefing, labeling protocol, and the pre-registered experiment designs |
| [`proposal/`](proposal/) | Research proposal and the original SSA metric memo |
| [`coig/`](coig/) | Submodule: fork of the original authors' implementation ([youngkyungkim93/coig](https://github.com/youngkyungkim93/coig)) — CSP, ARM, and the MLLM evaluation pipeline, used unmodified |

## Status

Both tracks executed. The Causal Relevance pilot ran 2026-07-18 (10 chains × three conditions) and
confirmed the lock confound. The single-image audit ran across SDXL and FLUX.1-dev with full
double-annotator coverage (κ = 0.954 on FLUX). The chain metric produced a significant result whose
Part C ablation then showed was carried by the segmentation mask, not attention.

Two things gate a submission, both documented in the briefing's §9: the models are too obedient
(~94% correct binding on FLUX leaves only 11–16 discriminative rows per annotator, so the mechanism
claim is underpowered), and on real CoIG images CLIPSeg detects the target attributes at most 27.8%
of the time, which bounds every chain-track number to a known non-random subset.

## Repository conventions

Everything lives on `main`. Retired branches are preserved as `archive/*` tags on origin. GPU work
runs on Kaggle; tests run from inside each package directory:

```bash
cd ssa/anchor_set && py -3 -m pytest tests/ -q
cd pi_level_experiment && py -3 -m pytest tests/ -q
```
