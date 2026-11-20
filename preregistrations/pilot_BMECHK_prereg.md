# Paper 8 — BME robustness checks (reviewer checks after novelty search) — rules written 20 Nov 2026 BEFORE running. Never change.
Settings: S1 COG n-back, S2 COG MATB, S3 Tufts, S4 Pavlov — same features, behaviour, LOSO models and BME code (bme.py) as
before. Caliper 0.15 (Tufts 0.2) unless stated. "Enough pairs" = ≥ 20 matched pairs per person.

## Check A — null simulations (real levels + real behaviour, SYNTHETIC brain, 20 dims, 3 seeds averaged)
- A-lin: brain = behz·W + noise (brain knows ONLY behaviour, linear).
- A-nl: brain = [behz, behz², tanh(2·behz), pairwise products]·W + noise (only behaviour, nonlinear).
- A-err: brain = behz·W + noise, but BME matches on noisy behaviour (behz + noise, reliability 0.8) — measurement-error bias.
- A-pos: brain = behz·W + 0.3·level·v + noise (positive control: real load signal beyond behaviour).
Rules:
- **A1**: A-lin matched AUC within 0.50 ± 0.02 in all 4 settings.
- **A2**: A-err bias (matched AUC − 0.5) < half of the REAL matched excess (real matched AUC − 0.5) in every setting.
- A-pos: expected matched AUC > 0.55 (sanity; reported, not a rule). A-nl: reported (expected ≈ 0.5).

## Check B — caliper sensitivity + permutation null (real data)
- **B1**: real matched brain AUC > 0.5 with Wilcoxon p < 0.01 at calipers 0.05, 0.10, 0.15, 0.20, 0.30 wherever ≥ 8 people
  have enough pairs (settings S1, S2, S4 must have ≥ 8 at every caliper; Tufts reported where possible).
- **B2**: real group-mean matched AUC above the 99th percentile of 1000 within-person label permutations (LOSO scores
  fixed), in every setting.

## Check C — comparison with cross-validated confound regression (CVCR, Snoek 2019)
CVCR: regress brain features on behaviour in training people only, residualise train + test, then LOSO logistic →
within-person brain AUC on all pairs. Run on real data and on A-lin / A-nl nulls. Descriptive, predictions written now:
CVCR real AUC > 0.5 in all settings (agrees with BME); CVCR ≈ 0.5 on A-lin; CVCR > BME on A-nl (linear regression
misses nonlinear behaviour leakage; matching does not). Not a GO rule.

**PASS (BME robust enough to write the paper) = A1 and A2 and B1 and B2.**
If FAIL: report honestly; fix the protocol (e.g. tighter caliper / reliability correction) only with a new pre-registration.
