# Paper 8 — Pilot BME "Behaviour-Matched Evaluation" of neural workload decoders — rules written 20 Nov 2026
(night) BEFORE running. Never change.

## Idea (new evaluation method)
Standard evaluation asks "does the brain decoder rank harder units above easier ones?". But harder units also have
worse performance, so a decoder can score well by reading performance-related brain activity (motor, arousal, error
processing). BME evaluates the decoder ONLY on pairs of units from different levels whose PERFORMANCE IS MATCHED
(behaviour-predicted level nearly equal). The drop from standard to matched evaluation = how much of the claimed neural
workload information is really behaviour. Matched AUC above 0.5 = genuine neural workload information.
Novelty check (20 Nov): no matched-performance evaluation protocol for workload decoders found.

## Settings
S1 COG-BCI n-back, S2 COG-BCI MATB (units = 1 min; features/behaviour as P8-BVB-COG), S3 Tufts fNIRS2MW
(units = 16 blocks per person; behaviour = block accuracy; brain = block mean of P8-A fNIRS features, per-person z).
Pavlov added later as S4 (reported, not needed for the rules).

## Procedure (leave-one-person-out)
Brain score = expected level from a logistic-regression brain classifier; behaviour score = expected level from a
logistic-regression behaviour classifier (both trained on other people). For the held-out person: all pairs (i, j)
with level_i > level_j. Standard AUC = share of pairs with brain_i > brain_j (ties ½). Matched pairs = those with
|behaviour_i − behaviour_j| ≤ 0.15 (levels scaled 0…2; for Tufts 0…3 → caliper 0.2). Matched AUC = same on those.
Inflation = Standard AUC − Matched AUC. People need ≥ 20 matched pairs to count.

## Pass rules
- E1 enough matched data: ≥ 20 people with ≥ 20 matched pairs in at least 2 of S1–S3.
- E2 standard evaluation is inflated: Inflation ≥ 0.05 with Wilcoxon p < 0.01 (over people) in at least 2 of S1–S3.
- E3 matching works: behaviour AUC on matched pairs ≤ 0.58 in those settings.
GO = E1 and E2 and E3. Secondary (reported): is matched brain AUC > 0.5 (Wilcoxon p < 0.05)?
