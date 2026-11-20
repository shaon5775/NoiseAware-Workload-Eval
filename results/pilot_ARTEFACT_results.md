# Artefact / task-activity check of the MATB RB-ME result — RESULT: ❌ FAIL (A1, A2 fail; A3 not testable) (20 Nov 2026)
Rules: pilot_ARTEFACT_prereg.md. Kaggle notebook shaon5775/paper8-artefact-fast (CPU, 3,565 s); V3 on laptop (artefact_v3_activity.py).
(A slower duplicate, shaon5775/paper8-artefact, was started first with the same code minus parallelism.)

ICA removed on average 10.3 of 20 components per recording (eye 1.1, muscle 9.7; range 2–20).
Sanity: V0 (no ICA, same recipe, new export) Δ +0.305 — identical to the paper ✓.

| Variant (MATB, 21 people, ρ 0.774, bound 0.555) | brain AUC | Δ | p | bal. acc |
|---|---|---|---|---|
| V0 no ICA, original features | 0.860 | **+0.305** | 1e-4 | 0.647 |
| V1 ICA-cleaned, original features (A1) | 0.632 | +0.077 | 0.15 ❌ | 0.510 |
| V2 ICA, no Fp1/Fp2/F7/F8/T7/T8, 1–20 Hz only (A2) | 0.647 | +0.092 | 0.14 ❌ | 0.533 |
| V3 behaviour + activity counts (laptop) | 0.919 | +0.283 | 0.016 | only 7 people → not testable |
| ICA band power, all 18 channels | 0.686 | +0.131 | 0.056 | 0.568 |

Descriptive (ICA band power, one band or region at a time):
- bands: delta +0.013 (p .54), theta +0.045 (.34), alpha +0.100 (.13), beta +0.111 (.069), gamma 30–40 +0.139 (.044)
- regions: frontal F3/Fz/F4 +0.152 (.013), central +0.109 (.084), parietal +0.113 (.084), occipital +0.042 (.30),
  **artefact-prone Fp1/Fp2/F7/F8/T7/T8 +0.181 (.002)** — the strongest.

Reading: after eye/muscle removal the MATB "beyond performance" effect shrinks from +0.31 to +0.08–0.09 and is no longer
significant; what is left is strongest in the edge channels and the gamma band, where eye and muscle activity live.
By the locked rule the MATB claim must be weakened. Caveat (not a rescue): the muscle detector removed about half of all
components, so some brain signal was probably removed too; the result is "not shown to be neural", not "shown to be artefact".
