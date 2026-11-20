# Eye or muscle? — RESULT: E1 ✅, M1 ✅ (21 Nov 2026; Kaggle shaon5775/paper8-artefact2, CPU, 13,718 s)
Rules: pilot_ARTEFACT2_prereg.md. Sanity ✓ (no-ICA +0.305; both-removed +0.077, same as P8-ARTEFACT).

| MATB features (21 people, bound 0.555) | brain AUC | Δ | p |
|---|---|---|---|
| No ICA | 0.860 | +0.305 | 1e-4 |
| Eye components removed only (mean 1.1 of 20) | 0.736 | +0.181 | 0.0028 |
| Muscle components removed only (mean 9.7 of 20) | 0.767 | +0.213 | 0.0014 |
| Both removed (P8-ARTEFACT) | 0.632 | +0.077 | 0.15 |

Regions/bands (Δ): eye-only — delta +0.07, theta +0.07 (slow bands lose the gain), alpha/beta/gamma +0.20–0.21, all regions +0.18–0.24;
muscle-only — every band and region +0.22–0.30 (p ≤ 0.002).
Pre-registered reading (E1 ✓, M1 ✓): neither source alone explains the gain; it vanishes only when both are removed
(≈ half of all components). Cannot tell "eye and muscle each carry it" from "heavy cleaning removes brain signal too".
Paper wording: MATB gain "not robust to full artefact removal"; no neural claim. Story ("two traps") unchanged.
Simulation part (ARTEFACT-NULL): see TEST_LOG P8-ARTNULL (95 % vs 0 %).
