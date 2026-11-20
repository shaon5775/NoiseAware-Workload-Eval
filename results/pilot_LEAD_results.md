# LEAD ("brain is faster, not better") — RESULT: ❌ NO-GO (L1 fails in both families) (20 Nov 2026, Kaggle paper8-lead, 9,337 s)
Rules: pilot_LEAD_prereg.md. COG-BCI, 29 people, balanced accuracy (chance 0.333) using only data from the first t seconds.

| t (s) | n-back BRAIN / BEH / FUSED | MATB BRAIN / BEH / FUSED |
|---|---|---|
| 2 | 0.406 / 0.349 / 0.402 | 0.506 / 0.770 / 0.743 |
| 4 | 0.421 / 0.333 / 0.421 | 0.590 / 0.693 / 0.766 |
| 6 | 0.418 / 0.341 / 0.441 | 0.640 / 0.690 / 0.785 |
| 8 | 0.456 / 0.337 / 0.448 | 0.605 / 0.724 / 0.828 |
| 10 | 0.452 / 0.360 / 0.452 | 0.602 / 0.751 / 0.835 |
| 15 | 0.452 / 0.418 / 0.502 | 0.628 / 0.755 / 0.847 |
| 20 | 0.483 / 0.513 / 0.548 | 0.655 / 0.782 / 0.870 |
| 30 | 0.498 / 0.533 / 0.567 | 0.674 / 0.770 / 0.854 |
| 45 | 0.536 / 0.605 / 0.628 | 0.686 / 0.801 / 0.847 |
| 60 | 0.563 / 0.644 / 0.644 | 0.705 / 0.820 / 0.866 |

- n-back: lead time 20 s ✅ (L2), fusion early gain +0.08 ✅ (L3), but brain > behaviour at 4 s only p = 0.028 ❌ (L1 needs < 0.01).
- MATB: behaviour (tracking error) is informative from second 2; brain never ahead (lead 2 s) ❌.
**NO-GO.** Useful side result (could be one paragraph in the RB-ME paper): in memory tasks the brain carries the
load signal for the first ~15–20 s before enough responses accumulate; in continuous multitasking behaviour is
immediate. Notably, in MATB, fusing brain + behaviour DOES help at every time (0.74–0.87 vs 0.69–0.82) — consistent
with RB-ME's "brain beyond performance in multitasking" finding.
