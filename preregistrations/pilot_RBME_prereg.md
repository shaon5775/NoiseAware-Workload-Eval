# Paper 8 — Idea 2: Reliability-Bounded Matched Evaluation (RB-ME) — rules written 20 Nov 2026 BEFORE running. Never change.
Runs in parallel with LEAD (idea 1). Why: BME and NC-BME failed because matching on a NOISY behaviour score leaves
residual true-performance differences. Fix with TWO INDEPENDENT behaviour measurements (split halves): match on one half,
measure leftover performance difference with the other half, and correct it for its own reliability.

## Data (fixed)
S1 COG n-back, S2 COG MATB, S4 Pavlov (Tufts has no split-half behaviour → excluded). Brain features, levels, LOSO brain
model, caliper 0.15, ≥ 20 matched pairs: exactly as BME. Behaviour halves (make_ab.py): n-back odd/even stimuli (hit,
FA, RT); MATB alternating 5-s chunks (tracking error mean, SD, tank deviation); Pavlov odd/even digit positions
(proportion correct). Each half z-scored within person.

## Method
1. LOSO behaviour models on half A and on half B separately → predicted-level scores sA, sB.
2. Reliability rel = correlation of sA and sB after removing person×level means (clipped to ≥ 0.05).
3. Match different-level within-person pairs on sA (caliper). On matched pairs: brain AUC and half-B AUC (AUC_B).
4. Bound = AUC a PERFECT reader of true performance would reach on those pairs (attenuation correction):
   bound = Φ(Φ⁻¹(AUC_B) / √rel).
5. Swap roles (match on B, test with A); average per person. Statistic per person: brain AUC − bound.
Inference: one-sided Wilcoxon over people (≥ 20 matched pairs in both directions).

## Validation (fake brains, 3 seeds): T = real (A+B)/2; fake halves A = T + N(0, 0.5²), B = T + N(0, 0.5²)
- NULL: brain = T·W + noise (knows only true performance).  POS: NULL + 0.3·level·v.
## Rules
- **V1 (no false claim):** NULL mean(brain − bound) ≤ +0.02 in all 3 settings.
- **V2 (power):** POS mean(brain − bound) > 0.03 with p < 0.01 in ≥ 2 of 3 settings.
- **R1 (finding):** REAL mean(brain − bound) > 0 with p < 0.01 in ≥ 2 of 3 settings, Pavlov included.
**METHOD GO = V1 and V2. FINDING GO = R1.** Kaggle.
