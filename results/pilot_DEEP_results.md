# Deep-learning check (EEGNet) — RESULT: ❌ PASS False (D0 fails on n-back), but D1 ✅ strongly (20 Nov 2026, Kaggle paper8-deep, GPU T4, 12,092 s)
Rules: pilot_DEEP_prereg.md. COG-BCI raw EEG, 18 channels, 2-s windows, EEGNet-8,2, leave-one-person-out.

| Family | EEGNet unit balanced acc. (chance 0.333; rule > 0.40) | RB-ME diff (brain − perfect-performance bound) | p | people |
|---|---|---|---|---|
| MATB | 0.476 ✅ | **+0.369** | 9.5e-7 | 21 |
| n-back | 0.336 ❌ (EEGNet did not learn n-back) | −0.251 | 1.0 | 29 |
(Logistic decoder for comparison, RB-ME: MATB +0.305, n-back −0.175.)

- D1 ✅: with a deep decoder the MATB finding is even stronger (+0.369 vs +0.305).
- D0 ❌: EEGNet stayed at chance on n-back with this fixed training recipe (15 epochs, no tuning) → PASS = False
  by the locked rule. Honest reading: the multitasking finding does not depend on the decoder; the n-back result
  ("no brain info beyond performance") is also unchanged, but here partly because EEGNet failed to learn n-back.
