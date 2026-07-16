# Chain of Image Generation — Causal Relevance Audit

Auditing whether CoIG's ("Chain-of-Image Generation") **Causal Relevance** metric
measures genuine step-level semantic faithfulness, or whether it's confounded by
CoIG's own compositional lock, which architecturally forbids later steps from
altering content generated earlier.

Builds on [arXiv:2512.08645](https://arxiv.org/abs/2512.08645), *"Chain-of-Image
Generation: Toward Monitorable and Controllable Image Generation."*

## Contents

- [`proposal/CPGA-Research-Proposal.md`](proposal/CPGA-Research-Proposal.md) —
  the full research proposal: motivation, related work, methods, PI feedback, and
  the resulting two-track restructuring (lock mechanism check + new faithfulness
  metric).
- [`docs/pilot-design.md`](docs/pilot-design.md) — the locked design for a small
  pilot (10 chains × Real/Shuffled/Substituted) meant to decide, cheaply, whether
  the full audit is worth running.
- [`coig/`](coig/) — git submodule pointing at a fork of the original authors'
  implementation ([youngkyungkim93/coig](https://github.com/youngkyungkim93/coig)):
  Compositional Strategy Planner (CSP), Autoregressive Refinement Model (ARM), and
  the MLLM evaluation pipeline. Pilot scripts will be added on top of this fork as
  they're built.

## Core hypothesis

CoIG's Causal Relevance score may just be detecting the compositional lock's
mechanical persistence (later steps are explicitly forbidden from altering earlier
content) rather than genuine causal faithfulness between a step's text and its
visual content. The pilot tests this with a matched negative-control design:
generate one fixed image sequence per prompt, then compare Causal Relevance under
three text conditions laid over the *same* images — Real (correct sub-prompts),
Shuffled (sub-prompts from other steps in the same chain), and Substituted
(sub-prompts from unrelated chains). No image is ever regenerated for the negative
controls, so any remaining "faithfulness" signal there is genuine unfaithfulness by
construction.

See [`docs/pilot-design.md`](docs/pilot-design.md) for the full protocol, the
appendix findings that shaped it, and the go/no-go criteria.

## Status

Planning complete. Execution is pending a paid Gemini 2.5 Flash API key
(`GOOGLE_AI_API_KEY`).
