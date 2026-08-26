# Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience

[![Tests](https://img.shields.io/badge/tests-382%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)]()
[![Venue](https://img.shields.io/badge/NeurIPS%202026-VLM4RWD%20Workshop-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

> **Core Thesis:** Automated faithfulness metrics for text-to-image (T2I) synthesis—including internal cross-attention spatial mass and VLM-based judges (VQAScore)—predominantly encode *prompt intent* rather than *visual realization*. Consequently, they achieve high aggregate accuracy on standard benchmarks solely because models usually obey their prompts, while providing no reliable diagnostic signal precisely on the prompt-disobeyed failure cases they are deployed to catch.

This repository contains the official, verified codebase and data for the paper **"Faithful by Assumption: How Text-to-Image Faithfulness Metrics Fail on Model Disobedience"**, submitted to the NeurIPS 2026 Workshop on Grounded and Faithful Vision-Language Models for Real-World Deployment (VLM4RWD).

## 🔬 1. Key Findings

1. **The Prompt-Only Baseline Wins Everywhere (10/10 Comparisons):**
   A trivial baseline that reads zero image pixels and simply assumes the model followed the prompt significantly outperforms cross-attention on every tested architecture, prompt set, and annotator ($p < 10^{-5}$). Harder compositional prompts widen the baseline's margin rather than allowing attention to close it (e.g., 80.1% baseline vs. 42.8% attention on the FLUX Hard Set, $p < 10^{-20}$).

2. **Toolkit-Wide Failure on Model Disobedience:**
   On the subset of images where the model *disobeys* the prompt (where a faithfulness metric is actually needed):
   - **Cross-Attention:** Accuracy collapses towards chance (28.6%–0% vs. 18.5% chance floor).
   - **VLM Judges (VQAScore):** Achieves only 41.9% accuracy on hard misbound rows (26/62), statistically indistinguishable from attention ($p = 0.263$).
   - **Sequential Causal Relevance (CoIG):** Persistence-to-final score is identical (0.837 Real vs. 0.845 Shuffled, $p = 1.0$) due to mechanical compositional locking.

3. **No Configuration Rescues Attention (456-Cell Taxonomy Search):**
   An exhaustive sweep across FLUX.1-dev's hierarchy (19 double blocks $\times$ 24 heads = 456 cells over 4 denoising quartiles) reveals that 0/10 of the sharpest attention cells beat the prompt-only baseline on unseen hard data (Holm-corrected $p = 9.4 \times 10^{-20}$).

4. **MMDiT Attention Capture Architecture:**
   A custom, mathematically faithful attention hook for FLUX.1-dev that recomputes Query/Key/Value projections, RMSNorm variants, RoPE embeddings, and manual softmax to extract explicit attention probability matrices despite fused kernel optimizations ($\max |\Delta| < 2 \times 10^{-7}$ vs. stock SDPA).

---

## 🏗️ 2. Repository Architecture & Contents

The repository is modularized into two primary experimental tracks: **Single-Image Audit** (`ssa/`) and **Sequential Chain Audit** (`pi_level_experiment/`).

### Track A: Single-Image Audit & MMDiT Experiments (`ssa/anchor_set/`)
This track analyzes faithfulness metrics on single image generations, utilizing rigorous human annotations across multiple difficulty tiers.
* **`artifacts_flux/` & `artifacts_flux_hard/`**: Contains generated images, bounding box coordinates, and human annotations for the FLUX model across different complexity levels (2-4 entities vs 4-6 entities).
* **`flux_attention_capture.py`**: The custom MMDiT fused-kernel attention processor.
* **`exp*_*.py`**: The core experiment scripts corresponding to the findings in the paper. E.g., `exp6_prompt_baseline.py` compares attention metrics to the prompt-only baseline, `exp9_taxonomy_analysis.py` performs the 456-cell attention sweep.
* **`tests/`**: Includes 274 unit and regression tests verifying annotation alignment and extraction math.

### Track B: Sequential Chain Audit (`pi_level_experiment/`)
This track evaluates faithfulness in a step-wise chain generation setting (Chain-of-Image-Generation), assessing whether metrics can detect gradual deviations.
* **`run_chain_experiment.py`**: Orchestrates the sequential generation and evaluation pipeline.
* **`score_chains.py`**: Implements delta-mask and attention Intersection-over-Union (IoU) scoring.
* **`RESULTS.md`**: Detailed empirical results and writeup for the chain-track experiments.
* **`tests/`**: Includes 108 tests for the chain evaluation pipeline.

### Supporting Artifacts
* **`docs/`**: Contains LaTeX source code for the paper drafts, complete BibTeX citations, and human annotation protocols.
* **`pilot/`**: Initial causal relevance and spatial-semantic alignment experiments motivating the main audit.

---

## 🚀 3. Quickstart & Reproduction

This codebase is designed for maximum reproducibility. All evaluation scripts execute locally on standard CPU hardware in $<30$ seconds, leveraging cached scores for rapid verification.

### Environment Setup

Ensure you have Python 3.10+ installed. A full environment configuration (e.g. `requirements.txt`) will be provided post-anonymization if required, but standard deep learning libraries (`torch`, `diffusers`, `pytest`, `pandas`, `scipy`) are expected.

### Verified Test Suite (382 Tests)
To guarantee the integrity of our metrics and statistical claims, we provide a comprehensive test suite. We strongly encourage reviewers to run the test suite to verify the environment setup.

```bash
# Run the Single-Image Audit test suite (274 tests)
cd ssa/anchor_set
python -m pytest tests/ -q

# Run the Sequential Chain Audit test suite (108 tests)
cd ../../pi_level_experiment
python -m pytest tests/ -q
```
*Note: Both test suites should pass with 100% success.*

### Reproducing Core Paper Results

Navigate to the `ssa/anchor_set/` directory:
```bash
cd ssa/anchor_set/
```

**1. Evaluate the Prompt-Only Baseline against Cross-Attention (Table 1 & 2)**
```bash
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux --annotator chayan
python exp6_prompt_baseline.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

**2. Evaluate on the Disobeyed Subset (Section 6.6)**
```bash
python exp7_misbound_subset.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

**3. Run the 456-Cell Taxonomy Search & Intent-vs-Realization Test (Section 6.4)**
```bash
python exp9_taxonomy_analysis.py \
  --easy-dir artifacts_flux --easy-annotator chayan \
  --hard-dir artifacts_flux_hard --hard-annotator consensus
```

**4. Compare Head-to-Head against VQAScore (Section 6.5)**
```bash
python vqa_agreement_check.py --artifacts-dir artifacts_flux --annotator chayan
python vqa_agreement_check.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

**5. Run the Sharper Within-Item Token Derangement Falsification Control (Section 6.2)**
```bash
python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux --annotator chayan
python exp3b_within_item_permutation.py --artifacts-dir artifacts_flux_hard --annotator consensus
```

---

## 📜 4. Citation & Reference

If utilizing this codebase, benchmark protocols, or MMDiT attention extraction pipeline, please cite:

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
