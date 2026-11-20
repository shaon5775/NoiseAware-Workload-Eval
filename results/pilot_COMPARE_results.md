# Comparison of evaluation methods — RESULT: ❌ FAIL (A1, A3 fail; A2 pass; B infeasible) (20 Nov 2026, Kaggle paper8-compare, 867 s)
Rules: pilot_COMPARE_prereg.md (+ addendum). Rate of claiming "brain beyond performance" (p < 0.01), 20 fake-brain seeds.

| Setting / fake brain | M1 Standard | M2 Matching (BME) | M3 CVCR (Snoek) | M4 SIMEX | M5 RB-ME |
|---|---|---|---|---|---|
| n-back NULL-lin | 100 % | 100 % | 100 % | 0 % | 0 % |
| n-back NULL-nl | 100 % | 100 % | 100 % | 0 % | 0 % |
| n-back POS (power) | 100 % | 100 % | 100 % | 100 % | 100 % |
| MATB NULL-lin | 100 % | 100 % | 100 % | 0 % | 0 % |
| MATB NULL-nl | 100 % | 100 % | 100 % | 0 % | 0 % |
| MATB POS (power) | 100 % | 100 % | 100 % | 95 % | 70 % |
| Pavlov NULL-lin | 100 % | 100 % | 100 % | 0 % | **15 %** |
| Pavlov NULL-nl | 100 % | 100 % | 100 % | 0 % | 0 % |
| Pavlov POS (power) | 100 % | 100 % | 100 % | 100 % | 100 % |
A1 ❌ (RB-ME 15 % > 10 % on Pavlov NULL-lin) · A2 ✅ · A3 ❌ (SIMEX never false-claims) · B infeasible (7/1/1 people).

## What it means (honest)
- Strong, clean finding: the three standard approaches (plain AUC, behaviour matching, confound regression) wrongly
  credit the brain 100 % of the time when behaviour is measured with noise. This is the paper's most solid result.
- Noise-aware approaches fix most of it: SIMEX (with the conservative noise estimate used here) 0 % false claims and
  95–100 % power; RB-ME 0–15 % false claims, 70–100 % power. RB-ME is NOT uniquely best.
- Real-data conflict to resolve: SIMEX (NC-BME run) says Pavlov brain > performance (0.569, p 1e-4); RB-ME says no.
  Both agree: MATB yes, n-back no.
- Note: M4 here used the full residual variance as noise (more conservative than NC-BME's neighbour-reliability rule).
