# Paper 8 — Pilot BVB "Does the brain add anything beyond behaviour?" — rules written 20 Nov 2026 BEFORE any
brain-vs-behaviour model is run on COG-BCI, Pavlov ds003838 or Berlin (only file formats inspected). Tufts was already
seen in room check P8-R5 (performance 0.574 vs fNIRS 0.363), so Tufts counts as the DISCOVERY set; the other datasets
are the CONFIRMATION sets that decide the claims. Never change these rules.

## Datasets and units
- Tufts fNIRS2MW (discovery): fNIRS, 68 people, 16 blocks, n-back 0–3; behaviour = block accuracy; effort = NASA-TLX.
- COG-BCI (Hinss 2023): EEG, 29 people × 3 sessions, n-back levels (and MATB levels if labelled); behaviour = per-block
  accuracy / reaction time as provided; effort = RSME per task/session.
- Pavlov ds003838: EEG (+ pupil, ECG), ~65 people; load = 5 / 9 / 13 digits (memory trials); behaviour = recall
  correctness per trial; unit = trial (or 5-trial group if trials are too few per person).
- Berlin Shin 2018: only if per-trial responses exist in the vendor markers.
Unit of analysis = block (or trial group), because behaviour exists per block.

## Models (leave-one-person-out, logistic regression, per-person z-scoring of every feature, no labels of test person)
- BEH: behaviour features only (accuracy; mean RT and RT spread if available).
- BRAIN: neural features only (EEG: log band power θ/α/β + per-person re-centred covariance tangent features, averaged
  over the unit; fNIRS: P8-A window features averaged over the block).
- FUSED: BEH + BRAIN.
- (Pavlov only, secondary) BODY: pupil size + heart rate features; BODY+BEH.
Metric: balanced accuracy of the load level per unit; paired over people (Wilcoxon signed-rank).

## Claims and pass rules (judged on the CONFIRMATION datasets that have usable behaviour)
- H1 behaviour is a strong baseline: BEH ≥ BRAIN − 0.02 on at least 2 confirmation datasets (or on all, if only 2).
- H2 brain adds little beyond behaviour: FUSED − BEH < 0.03 on at least 2 confirmation datasets.
- H3 (the method part) brain tracks effort that behaviour misses: model the part of the brain score not explained by
  level and performance (residual of BRAIN score regressed on level + BEH within person); correlate with subjective
  effort per unit within person. Pass if mean within-person Spearman ρ ≥ 0.10 with Wilcoxon p < 0.05 on ≥ 1
  confirmation dataset that has unit-level effort ratings (Tufts discovery result reported but not counted).
Outcomes: H1 & H2 hold → critical paper ("behaviour baseline"); + H3 → plus method ("effort beyond performance").
H1 fails (brain clearly better) → the story becomes "when does the brain beat behaviour" (report; decide next step).

## Data-handling note (20 Nov 2026, before any BVB result was seen)
COG-BCI channel sets differ (people 1–9 have 63 channels incl. Cz, others 62 without Cz; channel names for band power
were not exported). So for COG-BCI: band power = mean over channels per band (7 features) and covariance tangent
features use the 18 channels everyone has (Cz dropped). Same for BRAIN and FUSED; behaviour unchanged.
