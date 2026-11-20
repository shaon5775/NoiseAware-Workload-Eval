# BME-2 confirmation on untouched Pavlov ds003838 — RESULT: GO (20 Nov 2026, ~06:00 IST, laptop)
Rules: pilot_BME2_prereg.md (written before the Pavlov features existed on the laptop). 65 people, 7,008 memory trials
(load 5 / 9 / 13 digits), 64-ch EEG (63 used), 19 common 10-20 channels for covariance features.

| Measure | Result |
|---|---|
| Balanced accuracy (chance 0.333): behaviour / brain / fused | 0.697 / 0.457 / 0.678 |
| C3 fused − behaviour | −0.018 (accuracy says the brain adds nothing) ✅ |
| Brain AUC standard → behaviour-matched | 0.647 → **0.577** (inflation +0.070, p < 0.0001) |
| C1 matched brain AUC > 0.5 | p < 0.0001 ✅ |
| C2 behaviour AUC on matched pairs | 0.538 (≤ 0.58: matching works) ✅ |
| C4 people with ≥ 20 matched pairs | 65 / 65 ✅ |
**GO — the Behaviour-Matched Evaluation (BME) method is confirmed on a fresh dataset.**

## Full picture across all 4 settings (3 datasets, EEG + fNIRS, ~160 people)
| Setting | Behaviour acc. | Brain acc. | Fused − behaviour | Brain AUC standard | Brain AUC matched | Behaviour AUC matched |
|---|---|---|---|---|---|---|
| COG-BCI n-back (discovery) | 0.666 | 0.467 | −0.050 | 0.663 | 0.589 | 0.585 |
| COG-BCI MATB (discovery) | 0.884 | 0.655 | −0.005 | 0.828 | 0.871 | 0.546 |
| Tufts fNIRS (discovery, 10 people with enough pairs) | 0.578 | 0.414 | +0.013 (late fusion) | 0.713 | 0.743 | 0.541 |
| **Pavlov memory (confirmation, untouched)** | 0.697 | 0.457 | −0.018 | 0.647 | **0.577** | 0.538 |

## What the method shows (paper message)
1. Accuracy-based evaluation (and naive fusion) says neural workload decoders add NOTHING beyond the person's own
   performance — on every dataset.
2. Behaviour-matched evaluation shows the brain DOES carry load information that performance cannot show (matched AUC
   0.58–0.87, all p ≤ 0.001), invisible to standard metrics.
3. In working-memory tasks (n-back, digit span) standard evaluation is also INFLATED by ~7 AUC points (part of the
   "neural" signal is performance-related); in the complex multitask (MATB) it is not.
→ Proposed: BME as the evaluation protocol for neural workload decoders (report matched AUC alongside accuracy).

## Honest caveats
- BME is an EVALUATION method (protocol + metric), not a new decoder. Decoder-side tricks tested tonight all failed.
- Novelty checks were web/OpenAlex searches; a careful journal-database search is still needed before writing.
- The discovery-stage BME rule (inflation in ≥2 settings) failed; GO comes from the separately pre-registered
  confirmation (BME-2) on fresh data. Both are reported.
