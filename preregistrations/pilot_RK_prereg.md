# Paper 8 — Pilot RK "Rank, don't classify" (EEG) — rules written 19 Nov 2026 BEFORE running. Never change.

## Idea (the new method)
People differ in their EEG "baseline", which kills cross-person workload models. Instead of teaching the AI absolute
levels, teach it only WITHIN-PERSON ORDER: "for the same person, which of these two moments is harder?"
(pairwise ranking loss over pairs from the same person). For a new person, score all their windows, then turn scores
into levels using that person's OWN score distribution (no labels from the new person).
Novelty check 19 Nov: no EEG workload paper using a learning-to-rank / pairwise loss found (cross-subject work uses
domain adaptation, contrastive alignment, Euclidean alignment, prototypes).

## Data / protocol
Shin 2018 n-back (0/2/3-back → 0/1/2), 26 people, EEG only, 84 log band-power features (θ, α, β × 28 channels),
10-s windows, stride 2 s, per-person z-scoring (unlabelled) — identical to P8-MA B2. Leave-one-person-out, 3 seeds,
MLP 84→64→out, same optimiser/epochs for all.

## Models
- C1 CE classifier (= P8-MA B2): (a) argmax, (b) expected-level score + quantile mapping.
- C2 ordinal regression (MSE on level 0/1/2), score + quantile mapping.
- RK (ours): scorer trained with pairwise logistic loss on same-person pairs with different levels; quantile mapping.
Quantile mapping = the new person's scores cut at their own 1/3 and 2/3 quantiles (uses the known balanced design).
Prior-free mapping (secondary) = 1-D k-means with 3 clusters on the new person's scores, ordered by centre.

## Pass rules
- G1: RK (quantile) balanced accuracy ≥ best of {C1a, C1b, C2} + 0.02, AND RK beats that best baseline for ≥ 60%
  of the 26 people.
- G2: RK within-person block-ranking AUC (mean over people, pairs of windows with different levels) ≥ best baseline
  AUC + 0.02.
- G3: RK with prior-free k-means mapping ≥ C1a (argmax) + 0.01.
GO = G1 and G2 and G3.
