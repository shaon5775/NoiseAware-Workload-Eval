# Paper 8 — BME-2 CONFIRMATION on untouched Pavlov ds003838 — rules written 20 Nov 2026 (night) BEFORE the Pavlov
features exist on the laptop (export still running on Kaggle). Never change.

## Method being confirmed: Behaviour-Matched Evaluation (BME) protocol
Discovered on COG-BCI and Tufts (P8-BME, P8-BVB-COG): (a) accuracy-based evaluation says the brain adds nothing beyond
behaviour (fused ≈ behaviour), yet (b) on pairs of trials with MATCHED performance but different load, the brain still
ranks load well above chance, while behaviour cannot. BME is proposed as the evaluation protocol that reveals
performance-independent neural workload information which standard accuracy hides.

## Data and protocol (fixed)
Pavlov ds003838, memory trials, load 5 / 9 / 13 digits → 0 / 1 / 2; brain = per-trial EEG features as exported
(7-band log power × channels averaged per band if channel sets differ, + per-person re-centred covariance tangent
features on the common 10-20 channels); behaviour = recall score (partialScore/len, NCorrect/len), z within person.
LOSO logistic regression: BEH (C = 1), BRAIN (C = 0.05), FUSED (C = 0.05). BME exactly as pilot_BME_prereg.md
(expected-level scores, within-person pairs, caliper 0.15, people with ≥ 20 matched pairs).

## Confirmation rules
- C1 genuine neural information at equal performance: matched brain AUC > 0.5, Wilcoxon one-sided p < 0.01.
- C2 matching is valid: behaviour AUC on matched pairs ≤ 0.58.
- C3 accuracy hides it: FUSED − BEH balanced accuracy < 0.03.
- C4 enough data: ≥ 20 people with ≥ 20 matched pairs.
GO (method confirmed on fresh data) = C1 and C2 and C3 and C4.
