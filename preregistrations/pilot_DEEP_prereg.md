# Paper 8 — deep-learning decoder check for RB-ME (EEGNet) — rules written 20 Nov 2026 BEFORE running. Never change.
Question: does the RB-ME finding hold when the brain decoder is a standard deep network instead of logistic regression?

## Data and decoder (fixed)
COG-BCI raw EEG (Zenodo 7413650), n-back and MATB recordings, 18 common 10-20 channels (as LEAD), 1–40 Hz, resampled to
128 Hz, per-person per-channel z-score (all that person's data, no labels). 2-s non-overlapping windows from recording
start; window → 1-min unit by start time (same units as BME/RB-ME rows; joined via s|ses|task|unit keys).
EEGNet-8,2 (Lawhern 2018; dropout 0.5), leave-one-person-out, per family; Adam lr 1e-3, batch 256, 15 epochs, seed 0,
Kaggle GPU. Unit score = mean over its windows of the expected level (softmax · [0, 1, 2]).
Then RB-ME exactly as pilot_RBME_prereg.md (ALT halves, caliper 0.15, ≥ 20 matched pairs, Wilcoxon).

## Rules
- **D0 (decoder works):** EEGNet unit-level balanced accuracy > 0.40 (chance 0.333) in both families (reported vs logistic).
- **D1 (main):** MATB RB-ME diff > 0 with p < 0.01.
- D2 (reported, no rule): n-back RB-ME diff.
**PASS = D0 and D1.**
