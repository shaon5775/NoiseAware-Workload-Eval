# Noise-aware evaluation of EEG/fNIRS mental-workload decoders

Code, pre-registrations and full results for the paper (in preparation):
*"Does the brain know more than performance shows? Two traps in evaluating EEG and fNIRS mental-workload decoders."*

## What this repository shows
1. Behaviour (task performance) alone predicts workload better than EEG/fNIRS decoders, and adding the brain does not help (COG-BCI, Tufts fNIRS2MW, Pavlov ds003838).
2. Standard ways of asking "does the brain add information beyond performance" (plain AUC, behaviour matching / post-hoc counterbalancing, cross-validated confound regression) give false positives in 100 % of simulated cases when the behaviour measure is noisy (residual confounding; Westfall & Yarkoni 2016).
3. **RB-ME (Reliability-Bounded Matched Evaluation):** split behaviour into two independent halves, match trials on one half, and compare the decoder with the attenuation-corrected bound that a perfect reader of true performance could reach, measured with the other half. Validated on simulated decoders; compared with SIMEX correction.
4. Second trap: a performance-controlled test shows that a decoder knows more than performance, not that this knowledge is neural. RB-ME credits a simulated decoder that reads only task activity in 95 % of runs; on real MATB data a decoder using only key-press counts (no EEG) beats the bound by more than the EEG does.
5. Real data: the EEG gain beyond performance in multitasking (MATB, Δ = +0.31) falls to +0.08 (p = 0.15) after eye and muscle ICA removal; it survives removing eye-only (+0.18) or muscle-only (+0.21) components. No gain in working-memory tasks.

## Layout
- `preregistrations/` — every rule file, written before the corresponding run (never changed; addenda are marked).
- `results/` — result files for every test, including all failures.
- `TEST_HISTORY.md` — chronological list of all tests and lessons (also as `ESM_1_test_history.pdf`, Online Resource 1 of the paper).
- `code/kaggle/` — heavy runs (data export from the public datasets, RB-ME, robustness, comparison, EEGNet). Run on Kaggle notebooks.
- `code/laptop/` — light analysis scripts.

## Data (public, not redistributed here)
- COG-BCI: Hinss et al. (2023), Zenodo https://doi.org/10.5281/zenodo.7413650 (CC-BY-4.0)
- Pavlov digit span: OpenNeuro ds003838 (CC0)
- Tufts fNIRS2MW (CC-BY)

## Licence
Code: MIT. Text and result files: CC-BY-4.0.
