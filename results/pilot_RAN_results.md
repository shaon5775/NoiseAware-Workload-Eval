# Pilot RAN "balanced-session trap" + Rest-Anchored Normalisation — results (20 Nov 2026, Kaggle paper8-ran-pilot,
1,305 s CPU). Rules: pilot_RAN_prereg.md. Shin 2018 n-back EEG, 26 people, LR, 150-window test sessions × 10 resamples.
Rest calibration = 18 rest windows (first ~60 s of between-block rest) per person.

| Session mix | Z-own (usual) | Oracle | RAN (rest mean+std) | RAN-m (rest mean, task std) |
|---|---|---|---|---|
| balanced | 0.528 | 0.525 | 0.465 | 0.491 |
| mostly easy | 0.506 | 0.523 | 0.462 | 0.493 |
| mostly hard | 0.529 | 0.524 | 0.467 | 0.490 |
| easy+medium only | 0.527 | 0.563 | 0.507 | 0.525 |
| mean of 3 unbalanced | 0.521 | 0.536 | 0.479 | 0.503 |

## Rule check
- T1 trap is real (easy+medium ≥ 5 pt below oracle AND unbalanced mean ≥ 3 pt below): **FAIL** (−3.5 pt and −1.5 pt).
- T2 RAN fixes it (+3 / +5 pt): **FAIL** (RAN is 4 pt WORSE than usual scaling).
- T3 no cost when balanced (≥ −3 pt): **FAIL** (−6.3 pt).
**Verdict: NO-GO.**

## What we learned
1. The balanced-session trap exists but is small on EEG (1.5–3.5 points), smaller than the 7 points on weak fNIRS.
   Per-person z-scoring is more robust to level mix than feared.
2. One minute of between-block rest is a poor reference: rest EEG differs from task EEG and 18 windows give noisy
   statistics for 84 features → −4 to −6 points.
3. Paper 8 total: 6 pre-registered ideas NO-GO on public data (P8-A, N1, M-A, RK, RAN + order check).
