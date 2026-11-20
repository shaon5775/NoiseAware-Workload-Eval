# Pilot N1 "felt vs set workload" — results (19 Nov 2026, laptop) — rules: pilot_N1_prereg.md

68 people (2 people miss 4 block ratings each: sub_34, sub_40). Ridge, per-person normalised features, 17 buckets.

| | Felt = TLX mental (primary) | Felt = 4-scale mean | "Hard" = low task accuracy |
|---|---|---|---|
| How closely felt load follows set level (mean within-person ρ) | **0.853** | 0.721 | 0.778 |
| People who felt 3-back NOT harder than 2-back | 5.9% | 7.4% | 7.4% |
| G2 within the same level: brain score vs felt (mean ρ) | **0.067** (p = 0.054, 56% of people positive) | 0.025 (p = 0.26) | −0.006 (p = 0.66) |
| SET model block score vs felt (ρ) | 0.305 | 0.245 | 0.291 |
| FELT model block score vs felt (ρ) | 0.284 | 0.211 | 0.291 |
| Block score vs set level (SET / FELT model) | 0.342 / 0.324 | | |

## Rule check (primary)
- G0 ratings wired correctly (≥ 0.30): **PASS** (0.853).
- G1 felt differs from set (≤ 0.85): **FAIL** (0.853 — people's felt load follows the level almost exactly).
- G2 brain follows felt load within the same level (mean ≥ 0.10, p < 0.05): **FAIL** (0.067, p = 0.054).
- G3 felt-trained model better at felt load (+0.05): **FAIL** (0.284 vs 0.305).
**Verdict: NO-GO.**

## What we learned
1. The hypothesis was wrong: people DID feel 2-back and 3-back as different (only ~6% did not). So the labels are fine.
2. The brain signal is the bottleneck: the 2-probe forehead fNIRS cannot separate 2 vs 3-back even though people
   feel the difference. The ~35% ceiling is a SIGNAL limit of this headband, not a label problem.
3. A weak hint (p = 0.054) that brain score follows felt mental demand inside the same level — too small to build on.
4. Three Paper 8 ideas have now failed on fNIRS2MW (conformal ranges, block-order confound, felt labels). Further
   work needs a richer signal (more channels / EEG+fNIRS, e.g. Shin 2018) — or plan B (Karmakar's group).
