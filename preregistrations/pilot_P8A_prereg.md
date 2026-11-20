# Paper 8 — Pilot P8-A (go/no-go) — rules written 18 Nov 2026 BEFORE any data is seen. Never change these.

## Question
For a NEW person, can a workload AI (fNIRS, 4 levels 0/1/2/3-back) give an honest RANGE of levels
("level 1–2") that is right ~90% of the time for (almost) every person — and does a simple fix with the
new person's own UNLABELLED signal make that true person-by-person?

## Data
Tufts fNIRS2MW, band-pass filtered `whole_data` (68 eligible subjects, CC-BY-4.0). 30-s windows (150 samples at
5.2 Hz), stride 15 samples (~3 s, fewer near-copies than the official 3-sample stride). Windows must lie inside one
n-back block (single label).

## Splits (official, from the benchmark repo `partition_plan.txt`, "64 generic pool")
17 test buckets × 4 test people. For each bucket: train on its 48 train people, CALIBRATE on its 16 val people
(also people the model never trained on), test on the 4 test people. Every one of the 68 people is a test person once.

## Model (light, laptop)
Per window, per channel (8 channels): mean, std, slope, min, max, last−first → 48 features.
Base classifier: multinomial logistic regression (benchmark baseline type), standardised on the train people.
Secondary check: random forest (300 trees, 3 seeds). Target coverage 90% (alpha = 0.10).

## Methods compared
- M0 plain label (no uncertainty).
- M1 standard conformal label SETS (score = 1 − p(true level)), calibrated on the 16 val people.
- M2 ORDERED conformal INTERVALS: contiguous [a, b] of levels. Score = smallest interval around the model's
  most likely level that holds the true level, grown greedily by probability mass (nested sets → valid conformal).
- M3 = M2 + per-person normalisation: each person's features z-scored with THAT person's own windows (no labels).
  Applied to train, val and test people alike. (Unlabelled, so allowed for a new user.)

## Measures (on test people)
Marginal coverage; average set/interval size; % of M1 sets that are not contiguous (e.g. {0, 2});
PER-PERSON coverage → share of the 68 test people with coverage < 80% ("let-down people").

## Pass rules
- R1 (base model is real): M0 balanced accuracy on new people ≥ 0.30 (chance 0.25).
- R2 (the problem exists): M1 marginal coverage ≥ 0.88 BUT ≥ 25% of people get < 80% coverage.
- R3 (ordered intervals are worth it): M2 coverage ≥ 0.88, M2 mean width ≤ M1 mean set size + 0.10,
  AND ≥ 15% of M1 sets are non-contiguous.
- R4 (the fix works): M3 coverage ≥ 0.88 and its share of let-down people is ≤ HALF of M2's, with mean width no
  more than 0.25 above M2.
GO = R1 and R2 and R3 and R4 (logistic regression; random forest must agree on R2 and R4 direction).
If only R3 fails → the "ordered range" angle is weak; report, decide next step. Any of R1/R2/R4 fails → NO-GO.
