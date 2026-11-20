# Pilot P8-A results (run 19 Nov 2026, laptop) — rules: pilot_P8A_prereg.md

Data: 68 people, 20,672 windows (30 s, stride ~3 s). Official 17 Tufts buckets (48 train / 16 calibrate / 4 test).
Main model: logistic regression (48 simple features). Target coverage 90%.

| | no normalisation | per-person normalisation (M3) |
|---|---|---|
| Balanced accuracy, 4 levels (chance 0.25) | 0.344 | 0.356 |
| M1 label sets: coverage | 0.901 | 0.903 |
| M1 mean set size (of 4 levels) | 3.38 | 3.31 |
| M1 non-contiguous sets (e.g. {0,2}) | 12.4% | 13.5% |
| M1 "let-down" people (< 80% coverage) | 11.8% (8 of 68) | 4.4% (3 of 68) |
| M2 ordered intervals: coverage | 0.999 | 1.000 |
| M2 mean width | 4.00 | 4.00 |
| M2 let-down people | 0% | 0% |

## Rule check (logistic regression)
- R1 base model real (≥ 0.30): **PASS** (0.344, barely).
- R2 problem exists (≥ 25% let-down under M1): **FAIL** (11.8%).
- R3 ordered intervals worth it: **FAIL** (width 4.00 > 3.38 + 0.10; non-contiguous 12.4% < 15%).
- R4 fix works: not meaningful (M2 already 0% let-down because it always outputs all 4 levels).
**Verdict: NO-GO.** Random forest (3 seeds) is a secondary check and cannot change this verdict (GO needs LR).

## What we learned
1. The base model is too weak: with 34% accuracy on 4 levels, an honest 90% answer must include ~3.4 of the 4 levels,
   and the ordered interval becomes "level 0–3" (= no information). Uncertainty methods cannot rescue a model that
   barely beats chance; any paper here first needs a much stronger new-user model.
2. Plain conformal already holds well per person on this dataset (only 12% let-down), so the "it breaks for new users"
   story is weak here.
3. Per-person normalisation (no labels) is a cheap real gain: let-down people 11.8% → 4.4%, accuracy +1.2 points.
4. Greedy ordered intervals from argmax are wasteful when probabilities are flat (they grow to the full range).

## Random-forest side check (3 seeds, finished 19 Nov) — confirms NO-GO
Bal. acc 0.372–0.374 (per-person norm 0.381–0.386); M1 coverage 0.905, set size 3.24 (3.17 normed), let-down 4.4%
(2.9–5.9% normed), non-contiguous 19%; M2 always width 4.0. Same picture: stronger model, still uninformative ranges,
and R2 fails (let-down far below 25%).
