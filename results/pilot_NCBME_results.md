# NC-BME (noise-corrected BME, SIMEX) — RESULT: METHOD ❌ NO-GO (V1 fails), finding R1 ✅ (20 Nov 2026, Kaggle paper8-ncbme, 1,177 s)
Rules: pilot_NCBME_prereg.md. Real data and fake brains; matched brain AUC before → after correction.

| Setting | Real: before → corrected (p) | Fake noisy-null A-err: before → corrected (must be 0.47–0.53) | Fake positive A-pos: corrected (must be > 0.60) |
|---|---|---|---|
| S1 COG n-back | 0.589 → 0.536 (p 0.15) | 0.664 → 0.553 ❌ | 0.771 ✅ |
| S2 COG MATB | 0.871 → 0.866 (p < 0.0001) | 0.696 → 0.569 ❌ | 0.799 ✅ |
| S3 Tufts | 0.743 → 0.754 (p 0.003) | 0.682 → 0.633 ❌ | 0.947 ✅ |
| S4 Pavlov | 0.577 → 0.569 (p 0.0001) | 0.620 → 0.505 ✅ | 0.822 ✅ |
V1 ❌ (1 of 4 within ±0.03; need 3) · V2 ✅ · METHOD GO ❌ · R1 ✅ (S2, S3, S4) but cannot be trusted because V1 failed.

## Lessons
- SIMEX (quadratic) removes only part of the noise bias (classic under-correction), except Pavlov where it works.
- Exploratory (not a rule): real curves are almost FLAT as noise is added (MATB, Tufts, Pavlov), while the noise-only fake
  brains rise steeply. That pattern says the real matched effect is not mainly score-noise — but it's a post-hoc observation.
- n-back effect vanishes after correction (0.536, p 0.15) → the n-back "brain beyond behaviour" claim is dead.
