# Pilot M-A "EEG-taught fNIRS" — results (19 Nov 2026, Kaggle notebook paper8-ma-pilot, 4,435 s, CPU)
Rules: pilot_MA_prereg.md. Shin 2018 n-back (0/2/3-back), 26 people, 432 windows each (10 s, stride 2 s),
leave-one-person-out, MLP 64, 3 seeds. fNIRS 216 features (36 ch × HbO/HbR × mean/std/slope), EEG 84 (28 ch × θ/α/β).

| Model (balanced accuracy, chance 0.333) | seed 0 | seed 1 | seed 2 | final |
|---|---|---|---|---|
| B1 fNIRS only (deploy baseline) | 0.425 | 0.422 | 0.419 | **0.419** |
| B2 EEG only (teacher) | 0.533 | 0.536 | 0.534 | **0.534** |
| B3 EEG + fNIRS | 0.538 | 0.536 | 0.537 | **0.537** |
| D0 plain distillation (same time) | 0.409 | 0.412 | 0.412 | **0.412** |
| D-lag (ours, per-person delay) | 0.408 | 0.410 | 0.408 | **0.408** |
Chosen lags (seed 0, all folds): 0 s 208, 2 s 122, 4 s 86, 6 s 91, 8 s 143 — no clear delay.

## Rule check
- R1 teacher has something to give (+0.03): **PASS** (EEG +0.115 over fNIRS).
- R2 distillation helps fNIRS (+0.02, ≥ 60% people): **FAIL** (−0.007, 50% of people).
- R3 lag beats no-lag (+0.01, ≥ 55%): **FAIL** (−0.004, 42%).
**Verdict: NO-GO.**

## What we learned
1. On this data EEG is far better than fNIRS for workload (0.53 vs 0.42), and adding fNIRS to EEG adds almost nothing
   (+0.004). The workload information lives in EEG band power, not in the blood-flow features.
2. Because fNIRS features do not carry that information, an EEG teacher cannot pass it on: soft targets just add
   noise (−0.7 points). Distillation needs the student's signal to contain the knowledge.
3. The learned delay is all over the place (0–8 s), so there is no stable neurovascular lag to exploit at 10-s windows.
4. Practical direction this points to: the cheap deployable device should be a FEW-ELECTRODE EEG, not fNIRS
   (close to Karmakar's "fewest EEG electrodes" line — plan B territory).
