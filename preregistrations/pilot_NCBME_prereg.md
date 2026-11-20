# Paper 8 — Noise-Corrected Behaviour-Matched Evaluation (NC-BME) — rules written 20 Nov 2026 BEFORE running. Never change.
Why: P8-BMECHK showed that matching on a NOISY behaviour score leaves residual confounding (fake brain that knows only
true performance → matched AUC 0.62–0.70). NC-BME removes this with simulation-extrapolation (SIMEX, Cook & Stefanski
1994) applied to the behaviour score used for matching.

## Method (fixed)
Settings S1 COG n-back, S2 COG MATB, S3 Tufts, S4 Pavlov; same features, LOSO models, pairs and calipers as BME
(0.15; Tufts 0.2); brain scores are computed once (they do not use behaviour).
1. **Noise size.** Per setting, per behaviour feature: residual r = behz minus its person×level mean. Neighbour reliability
   ρ = correlation of r between consecutive units of the same person and level (time order). Error variance
   σ²_e = var(r)·(1 − max(ρ, 0)). (Conservative: treats all non-shared within-level fluctuation as error.)
2. **Simulation.** For λ ∈ {0, 0.5, 1, 1.5, 2}: add N(0, λσ²_e) to behz, refit the LOSO behaviour model, re-match,
   per-person matched brain AUC; 20 repeats per λ (λ = 0 once), averaged.
3. **Extrapolation.** Quadratic fit of matched AUC vs λ, read at λ = −1 → corrected matched AUC, per person and group.
   People included = ≥ 20 matched pairs at λ = 0.
4. Inference: one-sided Wilcoxon over people of (corrected − 0.5).

## Validation on fake brains (3 seeds, same generators as P8-BMECHK; noise size ESTIMATED by step 1, not given)
- A-err: brain knows only TRUE behaviour, matching uses behaviour + noise (reliability 0.8).
- A-pos: A-err + real load signal (0.3·level·v).

## Rules
- **V1 (method removes the bias):** A-err corrected matched AUC within 0.50 ± 0.03 in ≥ 3 of 4 settings.
- **V2 (method keeps a real signal):** A-pos corrected matched AUC > 0.60 in all 4 settings.
- **R1 (real finding):** real corrected matched AUC > 0.5, Wilcoxon p < 0.01, in ≥ 3 of 4 settings AND in Pavlov.
**METHOD GO = V1 and V2.** **FINDING GO = R1.** Both reported whatever happens.
- Method GO + finding GO → JNE paper: "noise-corrected evaluation; brain carries load info beyond performance".
- Method GO + finding NO → still a methods paper with a warning: "apparent brain advantage is a score-noise artefact".
- Method NO → drop NC-BME.
Runs on Kaggle.
