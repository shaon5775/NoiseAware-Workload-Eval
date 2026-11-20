# Paper 8 — (A) comparison with existing confound fixes + (B) session-wise replication of the MATB finding
Rules written 20 Nov 2026 BEFORE running. Never change. Kaggle. Settings S1 n-back, S2 MATB, S4 Pavlov.

## (A) Which methods wrongly claim "brain knows more than performance"? (false-claim rate and power)
Methods, each gives a one-sided Wilcoxon p over people for "brain beyond performance":
M1 Standard AUC (all within-person different-level pairs) > 0.5.
M2 Post-hoc counterbalancing / naive behaviour matching (BME, caliper 0.15 on full-behaviour score) > 0.5.
M3 Cross-validated confound regression (CVCR, Snoek 2019: regress behaviour out of brain features on training people) AUC > 0.5.
M4 SIMEX-corrected BME (NC-BME, 5 repeats per λ) > 0.5.
M5 RB-ME (ours, ALT split halves) brain − bound > 0.
Fake brains, 20 seeds per setting: T = (A+B)/2 real halves; fake halves A, B = T + N(0, 0.5²); full behaviour = (A+B)/2.
- NULL-lin: brain = T·W + noise.  NULL-nl: brain = [T, T², tanh(2T)]·W + noise.  POS: NULL-lin + 0.3·level·v.
False claim = p < 0.01 under NULL. Power = p < 0.01 under POS.
**Rule A1:** RB-ME false-claim rate ≤ 10 % (≤ 2 of 20 seeds) in every setting for both NULL types.
**Rule A2:** RB-ME power ≥ 80 % in ≥ 2 of 3 settings.
**Rule A3:** every other method (M1–M4) has false-claim rate > 10 % in at least one setting × NULL type (shows the need).

## (B) Does the MATB finding replicate in each recording session separately? (S2 only, independent recordings)
RB-ME (ALT, caliper 0.15) run separately on ses-S1, ses-S2, ses-S3 data (brain LOSO model trained and tested within the
session's data only); people with ≥ 10 matched pairs in both directions.
**Rule B1:** diff > 0 with p < 0.01 in ≥ 2 of 3 sessions, and diff > 0 in all 3.
**PASS = A1 and A2 and A3 and B1.**

## ADDENDUM (20 Nov, written after a code test, BEFORE the full Kaggle run) — honest disclosure
A quick code test (2 seeds, laptop) showed part (B) is NOT testable: per session each person has only ~5 MATB minutes per
level, so only 7 / 1 / 1 people reach ≥ 10 matched pairs. B will still be run and reported, but it is marked
"infeasible" and NOT counted. Decision rule for this run becomes PASS = A1 and A2 and A3. (The test also showed the
S1 n-back fake-brain claim rates: M1–M3 100 %, M4 0 %, M5 0 % under NULL; M4/M5 100 % under POS.)
