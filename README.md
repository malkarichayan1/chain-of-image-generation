# Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience

[![Tests](https://img.shields.io/badge/tests-382%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![Venue](https://img.shields.io/badge/NeurIPS%202026-VLM4RWD%20Workshop-orange.svg)]()

> **Core Thesis:** Automated faithfulness metrics for text-to-image (T2I) synthesis—including internal cross-attention spatial mass and VLM-based judges (VQAScore)—predominantly encode *prompt intent* rather than *visual realization*. Consequently, they achieve high aggregate accuracy on standard benchmarks solely because models usually obey their prompts, while providing no reliable diagnostic signal precisely on the prompt-disobeyed failure cases they are deployed to catch.

---

## 1. Key Findings

1. **The Prompt-Only Baseline Wins Everywhere (10/10 Comparisons):**
   A trivial baseline that reads zero image pixels and simply assumes the model followed the prompt significantly outperforms cross-attention on every tested architecture, prompt set, and annotator ($p < 10^{-5}$). Harder compositional prompts widen the baseline's margin rather than allowing attention to close it (e.g., $80.1\%$ baseline vs. $42.8\%$ attention on the FLUX Hard Set, $p < 10^{-20}$).

2. **Toolkit-Wide Failure on Model Disobedience:**
   On the subset of images where the model *disobeys* the prompt (where a faithfulness metric is actually needed):
   - **Cross-Attention:** Accuracy collapses towards chance ($28.6$--$29.0\%$ vs. $18.5\%$ chance floor).
   - **VLM Judges (VQAScore):** Achieves only $41.9\%$ accuracy on hard misbound rows ($26/62$), statistically indistinguishable from attention ($p = 0.263$).
   - **Sequential Causal Relevance (CoIG):** Persistence-to-final score is identical ($0.837$ Real vs. $0.845$ Shuffled, $p = 1.0$) due to mechanical compositional locking.

3. **No Configuration Rescues Attention (456-Cell Taxonomy Search):**
   An exhaustive sweep across FLUX.1-dev's hierarchy ($19 \text{ double blocks} \times 24 \text{ heads} = 456 \text{ cells}$ over 4 denoising quartiles) reveals that $0/10$ of the sharpest attention cells beat the prompt-only baseline on unseen hard data (Holm-corrected $p = 9.4 \times 10^{-20}$).

4. **MMDiT Attention Capture Architecture:**
   A custom, mathematically faithful attention hook for FLUX.1-dev that recomputes Query/Key/Value projections, RMSNorm variants, RoPE embeddings, and manual softmax to extract explicit attention probability matrices despite fused kernel optimizations ($\max |\Delta| < 2 \times 10^{-7}$ vs. stock SDPA).

---

## 2. Repository Structure

```
chain-of-image-generation/
├── ssa/anchor_set/                 # Single-image audit & MMDiT experiments
│   ├── artifacts_sdxl/             # SDXL images, bounding boxes, & 3x human annotations
│   ├── artifacts_flux/             # FLUX Easy Set (n=2..4) images, boxes, & annotations
│   ├── artifacts_flux_hard/        # FLUX Adversarial Hard Set (n=4..6, 100 prompts)
│   ├── flux_attention_capture.py   # Custom MMDiT fused-kernel attention processor
│   ├── build_hard_prompts.py       # Generator for adversarial hard prompt set
│   ├── build_consensus_labels.py   # Majority-vote human ground truth consensus resolver
│   ├── exp1_accuracy_by_n.py       # Stratified accuracy vs. 1/n chance
│   ├── exp2_window_ablation.py     # Early-window vs. full-trajectory comparison
│   ├── exp3_attention_scramble.py  # Cross-item value scrambling control
│   ├── exp3b_within_item_permutation.py # Within-item token derangement control
│   ├── exp4_positional_baseline.py # Nearest-subject-noun syntactic baseline
│   ├── exp5_count_clean_subset.py  # Synthesis failure / count-broken filtering
│   ├── exp6_prompt_baseline.py     # Central diagnostic: Prompt-only baseline comparison
│   ├── exp7_misbound_subset.py     # Evaluation on prompt-violating disobedience subset
│   ├── exp9_taxonomy_analysis.py   # 456-cell layer x head x timestep taxonomy sweep
│   ├── vqa_agreement_check.py      # Head-to-head VQAScore judge comparison
│   └── tests/                      # 274 unit and regression tests (pytest)
├── pi_level_experiment/            # Chain-level step-wise evaluation track
│   ├── run_chain_experiment.py     # Sequential pipeline orchestrator
│   ├── score_chains.py             # Delta-mask and attention IoU scoring
│   ├── rng_sweep.py                # RNG sensitivity and robustness battery
│   ├── RESULTS.md                  # Detailed writeup of chain-track empirical results
│   └── tests/                      # 108 tests for chain evaluation pipeline
├── pilot/                          # Motivating Causal Relevance (CR) audit
│   ├── causal_relevance.py         # Real / Shuffled / Substituted CR scorer
│   └── spatial_semantic_alignment.py # 864-line validated SSA chain metric
├── docs/                           # Documentation, protocols, and LaTeX drafts
│   ├── methods_and_setup_draft.tex # Publication-ready LaTeX source for Methods & Setup
│   ├── references.bib              # Verified, complete BibTeX citations
│   ├── raw-attention-paper-briefing.md # Ground-up technical briefing & claim evidence
│   └── anchor-set-labeling-protocol.md # Pre-registered human annotation protocol
└── proposal/                       # Initial project proposals & SSA metric memo
```

---

## 3. Quickstart & Reproduction

All evaluation scripts run locally on standard CPU in $<30$ seconds using cached scores.

### Running the Test Suite (382 Tests)
```bash
# Run ssa/anchor_set test suite (274 tests)
cd ssa/anchor_set
python -m pytest tests/ -q

# Run pi_level_experiment test suite (108 tests)
cd ../../pi_level_experiment
python -m pytest tests/ -q
```

### Reproducing Core Paper Results

From `ssa/anchor_set/`:

```bash
# 1. Evaluate the Prompt-Only Baseline against Cross-Attention (Table 1 & 2)
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux --annotator chayan
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux_hard --annotator consensus

# 2. Evaluate on the Disobeyed Subset (Section 6.6)
python exp7_misbound_subset.py --artifacts-dir artifacts_flux_hard --annotator consensus

# 3. Run the 456-Cell Taxonomy Search & Intent-vs-Realization Test (Section 6.4)
python exp9_taxonomy_analysis.py \
  --easy-dir artifacts_flux --easy-annotator chayan \
  --hard-dir artifacts_flux_hard --hard-annotator consensus

# 4. Compare Head-to-Head against VQAScore (Section 6.5)
python vqa_agreement_check.py --artifacts-dir artifacts_flux --annotator chayan
python vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus

# 5. Run the Sharper Within-Item Token Derangement Falsification Control (Section 6.2)
python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux --annotator chayan
python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

---

## 4. Citation & Reference

If utilizing this codebase, benchmark protocols, or MMDiT attention extraction pipeline, please cite:

```bibtex
@inproceedings{faithfulness2026assumption,
  title     = {Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience},
  author    = {Anonymous Authors},
  booktitle = {NeurIPS 2026 Workshop on Grounded and Faithful Vision-Language Models for Real-World Deployment (VLM4RWD)},
  year      = {2026}
}
```
