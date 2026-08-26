# Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience

[![Tests](https://img.shields.io/badge/tests-382%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![Venue](https://img.shields.io/badge/NeurIPS%202026-VLM4RWD%20Workshop-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

> **Core Thesis:** Automated faithfulness metrics for text-to-image (T2I) synthesis—including internal cross-attention spatial mass and VLM-based judges (VQAScore)—predominantly encode *prompt intent* rather than *visual realization*. Consequently, they achieve high aggregate accuracy on standard benchmarks solely because models usually obey their prompts, while providing no reliable diagnostic signal precisely on the prompt-disobeyed failure cases they are deployed to catch.

This repository contains the official, verified codebase and data for the paper **"Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience"**, submitted to the NeurIPS 2026 Workshop on Grounded and Faithful Vision-Language Models for Real-World Deployment (VLM4RWD).

---

## 1. Key Empirical Findings

1. **The Prompt-Only Baseline Outperforms Internal Attention (10/10 Comparisons):**
   A trivial baseline that reads zero image pixels and simply predicts the prompt-intended entity binding significantly outperforms cross-attention spatial mass across every tested architecture (SDXL, FLUX.1-dev), prompt set (Standard, Adversarial Hard), and human annotator ($p < 10^{-5}$ up to $p < 10^{-23}$, McNemar test). On hard compositional prompts, the gap widens (80.1% baseline vs. 42.8% attention on the FLUX Hard Set).

2. **Toolkit-Wide Failure on Model Disobedience:**
   On the critical subset of generated images where the synthesis model *disobeys* the prompt:
   - **Cross-Attention:** Accuracy collapses near chance levels (28.6% down to 0.0% vs. an 18.5% random floor).
   - **VLM Judges (VQAScore):** Achieves only 41.9% accuracy on hard misbound rows (26/62), statistically indistinguishable from internal attention ($p = 0.263$).
   - **Sequential Causal Relevance (CoIG):** Persistence-to-final score is identical (0.837 Real vs. 0.845 Shuffled, $p = 1.0$) due to mechanical compositional locking.

3. **No Attention Configuration Rescues Spatial Binding (456-Cell Taxonomy Search):**
   An exhaustive search across FLUX.1-dev's MMDiT architecture (19 double blocks $\times$ 24 heads = 456 individual cells across 4 denoising quartiles) reveals that 0 out of 10 top-performing attention cells beat the prompt-only baseline on unseen hard data (Holm-corrected $p = 9.4 \times 10^{-20}$).

4. **MMDiT Attention Capture Architecture:**
   A custom, mathematically faithful attention hook for FLUX.1-dev that recomputes Query/Key/Value projections, RMSNorm variants, RoPE embeddings, and explicit softmax normalization to extract attention probability matrices despite fused kernel optimizations ($\max |\Delta| < 2 \times 10^{-7}$ relative error vs. stock SDPA).

---

## 2. Repository Structure

The repository is organized into two primary empirical tracks:

```
chain-of-image-generation/
├── ssa/anchor_set/                 # Track A: Single-Image Audit & MMDiT Attention
│   ├── artifacts_flux/             # Standard FLUX generations & annotations (2-4 entities)
│   ├── artifacts_flux_hard/        # Adversarial Hard Set (4-6 entities, prior fights, duplicates)
│   ├── artifacts_sdxl/             # SDXL generations & multi-annotator labels
│   ├── flux_attention_capture.py   # Custom MMDiT attention extraction hook
│   ├── anchor_common.py            # Shared evaluation, metrics, and agreement logic
│   ├── exp3b_within_item_permutation.py  # Within-item token derangement falsification
│   ├── exp6_prompt_baseline.py     # Prompt-only baseline vs. cross-attention
│   ├── exp7_misbound_subset.py     # Disobeyed subset audit (Section 6.6)
│   ├── exp9_taxonomy_analysis.py   # 456-cell attention taxonomy sweep (Section 6.4)
│   ├── vqa_agreement_check.py      # VQAScore VLM judge evaluation (Section 6.5)
│   └── tests/                      # Unit and integration test suite (274 tests)
│
├── pi_level_experiment/            # Track B: Sequential Chain Audit (CoIG Track)
│   ├── run_chain_experiment.py     # Sequential generation orchestrator
│   ├── score_chains.py             # Delta-mask & attention IoU scoring
│   ├── RESULTS.md                  # Comprehensive empirical report on chain dynamics
│   └── tests/                      # Chain pipeline test suite (108 tests)
│
├── pilot/                          # Initial Causal Relevance & Spatial-Semantic experiments
└── docs/                           # Paper drafts, LaTeX sources, and BibTeX references
```

---

## 3. Quickstart & Reproduction

All evaluation scripts execute locally on standard CPU hardware in $<30$ seconds by utilizing pre-extracted attention and score caches.

### Environment Setup

Ensure Python 3.10+ is installed:

```bash
pip install torch diffusers pytest pandas scipy numpy
```

### Running the Test Suite (382 Tests)

To verify numerical correctness, metric computations, and annotation consistency:

```bash
# 1. Single-Image Audit test suite (274 tests)
cd ssa/anchor_set
pytest tests/ -q

# 2. Sequential Chain Audit test suite (108 tests)
cd ../../pi_level_experiment
python -m pytest tests/ -q
```

---

### Reproducing Core Experiments

Navigate to the `ssa/anchor_set/` directory:

```bash
cd ssa/anchor_set/
```

#### 1. Prompt-Only Baseline vs. Cross-Attention
Evaluates whether cross-attention outperforms a zero-compute prompt baseline across datasets and annotators:

```bash
# FLUX Easy Set
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux --annotator chayan

# FLUX Adversarial Hard Set
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

#### 2. Audit on Disobeyed Subsets
Evaluates metric performance specifically on images where the diffusion model failed to follow the prompt:

```bash
python exp7_misbound_subset.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

#### 3. 456-Cell Taxonomy Search & Generalization Sweep
Sweeps FLUX MMDiT blocks and heads, ranking the sharpest cells on Easy data and testing their generalization to Hard data:

```bash
python exp9_taxonomy_analysis.py \
  --easy-dir artifacts_flux --easy-annotator chayan \
  --hard-dir artifacts_flux_hard --hard-annotator consensus
```

#### 4. VLM Judge Audit (VQAScore)
Evaluates VQAScore on the same grounded bounding-box benchmark:

```bash
# FLUX Easy Set
python vqa_agreement_check.py --artifacts-dir artifacts_flux --annotator chayan

# FLUX Adversarial Hard Set
python vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

#### 5. Within-Item Token Derangement Falsification Control
Runs the sharper within-prompt attribute-subject derangement test:

```bash
python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

---

## 4. Citation

```bibtex
@inproceedings{faithfulness2026assumption,
  title     = {Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience},
  author    = {Anonymous Authors},
  booktitle = {NeurIPS 2026 Workshop on Grounded and Faithful Vision-Language Models for Real-World Deployment (VLM4RWD)},
  year      = {2026}
}
```

---
*This repository is anonymized for double-blind peer review.*
