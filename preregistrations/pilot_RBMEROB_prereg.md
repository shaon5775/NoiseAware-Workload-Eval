# Paper 8 — RB-ME robustness check — rules written 20 Nov 2026 BEFORE running. Never change.
Method exactly as pilot_RBME_prereg.md; only the listed choices vary. Settings S1 COG n-back, S2 COG MATB, S4 Pavlov.

## Variations
- Calipers 0.10, 0.15, 0.20, 0.30.
- Two split types: ALT (as before: odd/even stimuli, alternating 5-s MATB chunks, odd/even digits) and BLOCK
  (contiguous halves: first/second half of the unit's stimuli, first/second 30 s of the MATB minute, first/second half
  of the digit positions). BLOCK halves share less slow drift → harder test for MATB.
- Fake-brain validation at caliper 0.15, ALT split, fake half noise SD 0.3 and 0.8 (reliability ≈ 0.92 and 0.61), 3 seeds.
- Brain decoder variant for the REAL test: band power only (no covariance features), caliper 0.15, both splits.

## Rules
- **K1 method stays valid:** fake NULL mean diff ≤ +0.02 in all settings at both noise levels.
- **K2 method keeps power:** fake POS diff > 0.03 with p < 0.01 in ≥ 2 of 3 settings at both noise levels.
- **K3 finding is stable:** REAL MATB diff > 0 with p < 0.01 at all 4 calipers × 2 splits, and for band-power-only brain
  with both splits.
**ROBUST = K1 and K2 and K3.** Also reported (no rule): n-back and Pavlov real diff under all variations.
Kaggle.
