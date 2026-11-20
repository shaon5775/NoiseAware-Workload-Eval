# BME robustness checks — RESULT: ❌ FAIL (20 Nov 2026, laptop, rules pilot_BMECHK_prereg.md)
Code bme_checks.py, log bme_checks_log.txt, numbers bme_checks_results.json.

| Setting | Real matched AUC | Permutation 99th pct | A-lin (only behaviour) | A-nl | **A-err (noisy behaviour)** | A-pos | CVCR real / A-lin / A-nl |
|---|---|---|---|---|---|---|---|
| S1 COG n-back | 0.589 | 0.532 | 0.518 | 0.526 | **0.664** | 0.790 | 0.578 / 0.494 / 0.633 |
| S2 COG MATB | 0.871 | 0.536 | 0.515 | 0.517 | **0.696** | 0.790 | 0.644 / 0.507 / 0.665 |
| S3 Tufts | 0.743 | 0.571 | 0.497 | 0.476 | **0.682** | 0.895 | 0.584 / 0.505 / 0.659 |
| S4 Pavlov | 0.577 | 0.515 | 0.527 | 0.524 | **0.620** | 0.831 | 0.528 / 0.503 / 0.585 |

Calipers 0.05–0.30: real matched AUC stable and p < 0.01 everywhere, except MATB at 0.05 (only 6 people, p 0.016).

Rules: **A1 ❌** (Pavlov linear null 0.527, limit 0.52) · **A2 ❌** (measurement-error null bias is larger than,
or close to, the real effect in every setting) · **B1 ❌** (only MATB at caliper 0.05: 6 people < 8) · **B2 ✅**.

## What it means (honest)
- The killer is **A-err**: if behaviour is measured with ordinary noise (reliability 0.8) and the brain tracks the person's
  TRUE performance, BME gives matched AUC 0.62–0.70 even when the brain has NO load information beyond performance.
  So BME's "brain knows load at equal performance" can be produced by performance that the behaviour score measures
  imperfectly. The GO claim from BME-2 is not safe as stated.
- Caveat in our favour: the simulated brain tracks behaviour much more strongly (standard AUC 0.82–0.91) than the real
  brain (0.65–0.83), so the simulation may exaggerate the bias. Not tested yet — would need a new pre-registration.
- Also found: BME is stable across calipers and far above permutation null; CVCR leaks under nonlinear confounding
  (0.59–0.67 on A-nl) while BME does not (0.48–0.53) — a real, smaller methodological point.
- Lesson: matching on a NOISY confound leaves residual confounding (known "residual confounding" problem in
  epidemiology). Any "beyond behaviour" claim needs a reliability-aware null.
