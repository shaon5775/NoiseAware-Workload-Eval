# Idea 2 RB-ME (Reliability-Bounded Matched Evaluation) — RESULT: ✅ METHOD GO (V1 + V2), finding R1 ❌ (20 Nov 2026, Kaggle paper8-rbme, 144 s)
Rules: pilot_RBME_prereg.md. Statistic = brain AUC on behaviour-matched pairs − bound (best a PERFECT performance reader could do).

| Setting | Reliability of half | REAL brain / bound / diff (p) | Fake NULL diff (must ≤ +0.02) | Fake POS diff (must > 0.03, p < 0.01) |
|---|---|---|---|---|
| S1 COG n-back | 0.28 | 0.630 / 0.805 / **−0.175** (p 1.0) | −0.009 ✅ | +0.163 (p 5e-8) ✅ |
| S2 COG MATB | 0.77 | 0.860 / 0.555 / **+0.305** (p 5e-5) | −0.077 ✅ | +0.089 (p 2e-4) ✅ |
| S4 Pavlov | 0.74 | 0.578 / 0.615 / **−0.036** (p 1.0) | −0.007 ✅ | +0.143 (p 2e-12) ✅ |
V1 ✅ · V2 ✅ (3/3) → **METHOD GO** — first Paper 8 method that passes its own fake-data validation (it never credits a
brain that only reads performance, and it does detect real extra signal). R1 ❌ (only MATB; Pavlov not).

## What it shows
- Working-memory tasks (n-back, digit span): the brain's apparent "beyond performance" signal is fully explained by
  performance once noise is properly bounded. Earlier BME/BME-2 claims for these tasks were measurement artefacts.
- Complex multitasking (MATB): the brain beats even a perfect performance reader by +0.31 AUC → genuine neural load
  information that behaviour can't show.
- Message: "Standard and matched evaluations overstate neural workload decoders in memory tasks; a reliability-bounded
  test separates real neural information from performance leakage; real extra information exists only in multitasking."
Caveat: n-back half reliability is low (0.28) → its bound is generous; MATB halves are 5-s chunks (may share slow drift).
Seen before run: S1 quick test on laptop (same numbers).
