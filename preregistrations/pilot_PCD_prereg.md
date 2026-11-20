# Paper 8 — Pilot PCD "Performance-Conditional Demand decoder" — rules written 20 Nov 2026 (night) BEFORE running.
Never change.

## Idea (new method)
P8-BME showed that at EQUAL performance the brain still ranks task demand (matched AUC 0.59 n-back, 0.87 MATB, 0.74
Tufts). Standard decoders are trained to predict the level from all units, so they also learn performance-related
activity. PCD is trained only on what behaviour cannot show: a pairwise logistic model on brain-feature DIFFERENCES
x_i − x_j for within-person pairs of different levels whose behaviour scores are matched (same caliper as BME). It
outputs a per-unit "hidden demand" score w·x. Novelty: no performance-conditional / behaviour-matched training of a
workload decoder found (20 Nov checks).

## Settings and protocol
S1 COG n-back, S2 COG MATB, S3 Tufts (as P8-BME), plus S4 Pavlov if its export is ready before the run.
Leave-one-person-out. Behaviour score for TRAINING people = behaviour classifier trained on the training people
(in-sample, used only to pick training pairs); for the TEST person = trained on the others (as BME). Pairwise logistic
regression on difference vectors (both orders, labels ±1), C = 0.05, StandardScaler on unit features.
Baseline = standard brain level classifier (expected level), exactly as in P8-BME.

## Pass rules (judged on settings with ≥ 20 people having ≥ 20 matched pairs; Tufts uses whatever qualifies)
- P1: PCD matched AUC ≥ baseline matched AUC + 0.03 in at least 2 qualifying settings.
- P2: PCD matched AUC > 0.5 with Wilcoxon p < 0.01 in those settings.
- P3: PCD standard AUC ≥ baseline standard AUC − 0.03 in those settings (does not lose ordinary ranking).
GO = P1 and P2 and P3.
