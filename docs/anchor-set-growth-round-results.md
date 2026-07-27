# Metric-A Anchor Set — Growth Round Results (2026-07-27)

Companion to `docs/anchor-set-labeling-protocol.md`. Produced by:
```
py -3 analyze_agreement.py --annotator akhil --artifacts-dir artifacts_sdxl --compare-annotator grace
```
Raw stdout saved verbatim below (summary); row-level detail is in `artifacts_sdxl/agreement_akhil.csv`.
Akhil's labels are used as the reference annotator for the accuracy table because he is the only
one at 100% coverage (306/306 labels, 105/105 counts) as of this run; Grace is at 52.6% (161/306
labels, 92/105 counts) — **still in progress**, not yet the full double coverage the protocol
calls for.

## Inter-rater reliability (the Workstream 2 deliverable)

```
Inter-rater reliability: akhil vs grace
  overlapping judgments : 161
  raw agreement         : 123/161 = 76.4%
  chance agreement      : 26.0%
  Cohen's kappa         : 0.681
  categories used       : absent, barista, chef, cyclist, farmer, none, nurse, pilot, shared, teacher, unclear
```

**κ = 0.681, short of the target κ ≥ 0.7** — but computed on only 161/306 judgments (Grace's
completed share so far), not the full set. Per the protocol (§5), this number is expected to move
as Grace finishes; it is not the final measurement.

87% of disagreements (33/38) involve a boundary/sentinel call — `unclear` vs. `shared` vs. `none`
vs. naming a real subject — not disagreement about *who* owns an attribute once both annotators
agree it's clearly present. The four-way Present/Missing/Shared/Unclear taxonomy's edges are the
soft spot, not core binding judgment.

Count-clean κ (same annotators, per-image judgment, n=92 overlapping): **0.914** — comfortably
above target.

## Metric-vs-human accuracy (Akhil as reference, strict scoring)

```
 stratum | labeled | scored | correct | accuracy |  chance |  margin
--------------------------------------------------------------------
     n=2 |      88 |     18 |       9 |    50.0% |   50.0% |    0.0%
     n=3 |      78 |     14 |       8 |    57.1% |   33.3% |   23.8%
     n=4 |     140 |      8 |       3 |    37.5% |   25.0% |   12.5%
--------------------------------------------------------------------
 overall |     306 |     40 |      20 |    50.0% |     -   |     -
```

96 of 306 rows excluded as **count-broken** (image didn't render the exact requested subject
count — Akhil's own per-image judgment), on top of the usual none/unclear/shared exclusions.
Effective n = 40/306 = **13%** pass rate, well below the ~150-effective floor the protocol sized
the ~300-raw growth batch for (and well below the original 23-image set's ~51% strict pass rate).
This growth batch's generations are shakier than the original set — worth a closer look before
treating the accuracy numbers above as a clean replication, independent of the kappa/coverage gap.

## Status against the original ask

| Deliverable | Status |
|---|---|
| Labels file per annotator | done — copied into `artifacts_sdxl/{labels,counts}_{akhil,grace}.json` |
| Inter-rater agreement number | computed and committed here — 0.681, partial data, short of 0.7 |
| Protocol doc | done — `docs/anchor-set-labeling-protocol.md` |
| Full double coverage (both annotators) | not done — Akhil 100%, Grace 52.6% |
| Effective-n floor (~150) for binding accuracy | not met — 40 achieved, driven mostly by count-broken rate |

**Not done yet.** Grace needs to finish her pass (re-run the kappa command above once she does —
it auto-converges to the full-set number, no code changes needed). The low effective-n from
count-broken images is a separate, real finding worth investigating regardless of Grace's
progress: is the growth batch's prompt/seed pool systematically harder to render than the
original 23 prompts, or is this a one-off from the backfill's retry seeds?
