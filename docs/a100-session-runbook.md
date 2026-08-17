# A100 Session Runbook — taxonomy capture (#14/#16/#17/#18 → #19)

> **STATUS: EXECUTED 2026-08-17. This plan is kept for the record; two of its premises were
> wrong and are corrected here.**
>
> The capture ran on **Thunder Compute** (`tnr` CLI, 1×A100 80GB, $1.09/GPU-hr,
> per-minute billing, persistent disk), not the metered algoverse JupyterHub this doc was
> written for. Results in CLAUDE.md §3.1.
>
> **Correction 1 — the cost estimate was off by ~25×.** Actual: **~14.1 s/image**, so 103
> easy + 83 hard images = **~45 min wall-clock, ~$2 total**, not "~10–15 GPU-hr per set."
> The old figure was calibrated for Kaggle T4/P100. Do not budget off it.
>
> **Correction 2 — §1.2's persistent-storage workaround is unnecessary on Thunder Compute.**
> `$HOME` survives there by default; only `tnr delete` destroys it (there is no `tnr stop`,
> and there is **no platform-side spending cap** — a timer is the only ceiling).
>
> **What held up:** the §2 smoke-test gate was worth every second. On n=3 it showed 1 image
> over the 0.05 drift threshold, which looked alarming; on the full 186 the real rate was
> 8/186 (4.3%). Run the gate, but read it on n≥20 before panicking.

The A100 is a **metered JupyterHub session**: the clock starts at login, not at first
compute. Everything in §1 is designed so the ~30 GB model download overlaps with setup
instead of being paid for serially, and so a bad capture is caught in minutes rather than
after 10 GPU-hours.

**One job needs this machine.** #8 and #31 already ran on CPU (see §6). #20 and #30 are
sequenced behind #19's verdict and should not consume this window.

Status: written 2026-08-14, before the first session. Branch `hard-prompt-set-retest`.

---

## 0. Before you log in (do this off the clock)

- [ ] **HF token ready.** FLUX.1-dev is gated. Accept the license at
      `huggingface.co/black-forest-labs/FLUX.1-dev` under the account whose token you'll
      paste, and have the token in your clipboard. A gated-repo 401 forty minutes into a
      download is the single most expensive avoidable failure here.
- [ ] **Know whether the instance has persistent storage.** If `$HOME` survives between
      sessions, the ~30 GB download is one-time. If the instance is ephemeral, it recurs
      every session — in that case put `HF_HOME` on whatever persistent volume exists
      (§1.2) rather than re-paying for it each window.
- [ ] Decide the target set. **`artifacts_flux` (easy) first** — #19 selects cells on the
      easy set, so a session that only captures the easy set still advances #19; one that
      only captures hard does not.

## 1. First 10 minutes (the clock is running)

### 1.1 Start the download immediately, in the background

Do this **before** cloning, installing, or reading anything. It is the long pole.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
pip install -q "huggingface_hub[cli]"
nohup hf download black-forest-labs/FLUX.1-dev > ~/hf_dl.log 2>&1 &
```

`nohup ... &` matters: a JupyterHub terminal tab that closes takes its foreground children
with it, and a browser disconnect 25 GB into a download is a real way to lose a window.

### 1.2 If storage is persistent, point the cache at it first

```bash
export HF_HOME=/persistent/path/hf          # BEFORE the download command above
echo 'export HF_HOME=/persistent/path/hf' >> ~/.bashrc
```

### 1.3 While it downloads — set up the repo

```bash
git clone <repo-url> && cd "Chain of Image Generation"
git checkout hard-prompt-set-retest
cd ssa/anchor_set
pip install -q torch diffusers transformers accelerate sentencepiece protobuf
python3 -m pytest tests/ -q     # 391 tests, all green; ~15s, no GPU
```

The test run is not ceremony — it confirms the checkout is the state the capture was
validated against, including the equivalence test proving per-head reduction reproduces
the published pooled path to 1e-6.

### 1.4 Confirm the GPU is what you were promised

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Expect ~40 GB or ~80 GB. If it reports <40 GB, stop and re-read §4 before proceeding —
the memory budget below assumes A100-class headroom.

## 2. Smoke test — 3 images, ~5 minutes, before committing the window

**Do not skip this.** The capture regenerates images that were human-labeled months ago;
if regeneration drifts, the attention describes a picture nobody labeled, silently.

```bash
python3 taxonomy_capture_flux.py --artifacts-dir artifacts_flux --limit 3
```

Then read the two gate fields it prints per image:

```bash
python3 -c "
import json; d=json.load(open('artifacts_flux/taxonomy_index.json'))
for k,v in d.items():
    print(k, v['repro_mean_abs_pixel_diff'], v['pooled_owner_matches_manifest'])
"
```

| Field | Want | If it fails |
|---|---|---|
| `repro_mean_abs_pixel_diff` | ~0 (< 0.05) | **STOP.** Regeneration drifted — different diffusers/torch version, dtype, or scheduler default. Report before burning the window. |
| `pooled_owner_matches_manifest` | `true` on the large majority | A few `false` is tolerable (grid-space vs. pixel-space rounding); broadly false means the capture disagrees with the published pipeline — stop. |

`exp9_taxonomy_analysis.py` drops rows above `--repro-threshold` (default 0.05) and reports
the count, so a partially-drifted run is recoverable — but a wholly-drifted one is 10 wasted
GPU-hours.

## 3. The full run

```bash
nohup python3 taxonomy_capture_flux.py --artifacts-dir artifacts_flux \
    > ~/capture_easy.log 2>&1 &
tail -f ~/capture_easy.log        # prints a per-image ETA
```

Then, if the window allows:

```bash
nohup python3 taxonomy_capture_flux.py --artifacts-dir artifacts_flux_hard \
    > ~/capture_hard.log 2>&1 &
```

**Both `--artifacts-dir` flags are mandatory.** Without one, content search resolves the
first `manifest.json` alphabetically — always `artifacts_flux`, never the hard set — and
writes its index to the cwd rather than beside the artifacts. Two runs sharing one cwd-level
index would make the second resume from the first's entries and merge the datasets.

**Resumption is safe and per-set.** `taxonomy_index.json` is rewritten after every image and
already-captured ids are skipped, so a window that expires mid-run costs at most one image.
Re-running the identical command in the next session continues where it stopped.

## 4. Why this needs the A100 (the memory argument, for reference)

| Component | bf16 |
|---|---|
| FLUX.1-dev transformer | ~24 GB |
| T5-XXL text encoder | ~9–10 GB |
| CLIP encoder + VAE | ~1 GB |
| **Weights subtotal** | **~34 GB** |

On top of that, the capture **deliberately disables the fused attention kernel** — it
replaces `scaled_dot_product_attention` with an explicit `softmax(Q @ Kᵀ)` to obtain a
readable matrix (claim C7). At 1024×1024 that is a ~4,600-token sequence (4,096 image
patches + text), so one layer's explicit matrix across 24 heads is ~2 GB transient, over 19
layers × 25 steps.

A T4 (16 GB) cannot load the weights at all, and lacks native bf16 — casting to fp16 is a
known inf/nan source in FLUX's T5 path. An L4 (24 GB) holds less than the weights alone,
requiring CPU offload, which perturbs execution order and is incompatible with the §2
reproduction gate.

## 5. Getting results off the machine

The deliverables are small — the npz files are float16 and compressed.

```bash
cd ssa/anchor_set
tar czf ~/taxonomy_easy.tgz artifacts_flux/taxonomy_index.json artifacts_flux/taxonomy_cells_p*.npz
tar czf ~/taxonomy_hard.tgz artifacts_flux_hard/taxonomy_index.json artifacts_flux_hard/taxonomy_cells_p*.npz
```

Download both through the Jupyter file browser. **Do not run the analysis on the A100** —
`exp9_taxonomy_analysis.py` is pure CPU re-analysis and belongs on a laptop, not on metered
GPU time:

```bash
python3 exp9_taxonomy_analysis.py \
    --easy-dir artifacts_flux --easy-annotator chayan \
    --hard-dir artifacts_flux_hard --hard-annotator consensus \
    --out artifacts_flux_hard/taxonomy_report.json
```

## 6. What must NOT run on this machine

| Item | Where it actually belongs |
|---|---|
| #8 CLIPScore discriminant | **Already done, CPU, 2026-08-14.** ~10 min locally. |
| #31 VQAScore | **CPU** — blip-vqa-base is ~385M params. `python3 vqa_score_flux.py --artifacts-dir <dir>` |
| #19 analysis | CPU re-analysis of the capture (§5) |
| #20 steering | Needs #19's verdict first, and its current implementation is wrong (perturbs latents, not `attn_probs`) |
| #30 PixArt-Σ | Sequenced after #19; needs a new attention hook + annotation cycle |

## 7. Failure playbook

| Symptom | Cause | Action |
|---|---|---|
| 401 / gated repo | License not accepted for this token's account | Accept on the Hub, re-run download |
| CUDA OOM at load | Something else resident on the GPU | `nvidia-smi`, kill strays; confirm ≥40 GB |
| `repro_mean_abs_pixel_diff` large | Version/dtype drift vs. original generation | **Stop**, record `pip freeze`, report — do not capture the full set |
| Window expires mid-run | Expected and handled | Re-run the identical command next session; it resumes |
| `no manifest.json in <dir>` | Wrong `--artifacts-dir` or incomplete checkout | Confirm cwd is `ssa/anchor_set/` |
