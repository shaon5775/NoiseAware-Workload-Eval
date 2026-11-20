# Pilot RK "Rank, don't classify" (EEG) — results (19–20 Nov 2026, Kaggle paper8-rk-pilot, 4,169 s CPU)
Rules: pilot_RK_prereg.md. Shin 2018 n-back 0/2/3, 26 people, EEG band power (84), leave-one-person-out, 3 seeds.

| Model | Balanced acc. (chance 0.333) | Within-person ranking AUC |
|---|---|---|
| C1a CE classifier, argmax (best baseline) | **0.531** | 0.728 |
| C1b CE score + quantile mapping | 0.525 | 0.728 |
| C2 ordinal regression + quantile | 0.509 | 0.721 |
| RK pairwise ranking (ours) + quantile | 0.506 | 0.708 |
| RK + prior-free k-means mapping | 0.442 | — |

## Rule check
- G1 (+0.02 over best baseline, ≥ 60% people): **FAIL** (−0.026; better for only 19% of people).
- G2 (ranking AUC +0.02): **FAIL** (−0.020).
- G3 (k-means RK ≥ CE argmax + 0.01): **FAIL** (−0.089).
**Verdict: NO-GO.** Same picture in all 3 seeds.

## What we learned
1. Per-person z-scoring of features (which every model here already uses) removes the person "baseline"; a ranking
   loss then has nothing extra to fix, and throwing away absolute-level information costs accuracy.
2. A plain classifier with per-person normalisation (~53%) is a hard baseline on this dataset.
3. Paper 8 total: 5 pre-registered ideas, all NO-GO (P8-A, N1, M-A, RK + order check). Clear pattern: simple
   per-person normalisation captures most of what clever cross-person tricks try to add on public n-back data.
