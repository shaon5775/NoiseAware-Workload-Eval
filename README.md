# Noise-aware evaluation of EEG/fNIRS mental-workload decoders

Code, dated rule files, derived data and full results for the paper:
*"Does the brain know more than performance shows? Two traps in evaluating EEG and fNIRS mental-workload decoders"*
(S. Bandyopadhyay and R. K. Mondal; under review).

## What this repository shows
1. Behaviour (task performance) alone classifies workload better than EEG/fNIRS decoders, and adding the brain improves accuracy by at most about one point (COG-BCI, Pavlov digit span, Tufts fNIRS2MW).
2. **First trap — noisy behaviour.** Plain AUC, behaviour matching (post-hoc counterbalancing) and cross-validated confound regression credit decoders that read only true performance in 100 % of simulations when behaviour is measured with ordinary noise (residual confounding).
3. **RB-ME (reliability-bounded matched evaluation).** Split behaviour into two independent halves, match trials on one half, and compare the decoder with the attenuation-corrected bound of a perfect performance reader, measured with the other half. Validated on simulated decoders and compared with SIMEX.
4. **Second trap — non-neural signal.** A test that controls for performance shows that a decoder knows more than performance, not that this is neural: RB-ME credits a simulated decoder that reads only task activity in 95 % of runs, and on real MATB data a decoder using only key-press counts (no EEG) beats the bound by more than the EEG does.
5. Real data: the EEG gain beyond performance in multitasking (MATB, Δ = +0.31) falls to +0.08 (p = 0.15) after eye and muscle ICA removal; it survives removing eye-only (+0.18) or muscle-only (+0.21) components. RB-ME finds no gain in working-memory tasks.

## Layout
- `preregistrations/` — the dated rule file of every formal test, written before the run (never changed; addenda are marked).
- `results/` — result files (`.md`, `.json`) for every test, including all failures.
- `TEST_HISTORY.md` and `ESM_1_test_history.pdf` — chronological list of all tests and lessons (the paper's supplementary file).
- `code/kaggle/` — heavy runs on Kaggle notebooks (data export from the public datasets, RB-ME, robustness, method comparison, EEGNet, ICA artefact checks, dataset inspection).
- `code/laptop/` — light analysis scripts and the figure script (`make_figs.py`).
- `data/` — small **derived** files (features and split-half behaviour) so that RB-ME can be re-run without re-downloading the raw recordings: `data/bme/bme_*.npz` (unit-level features and behaviour, settings S1–S4), `data/ab_halves*.npz` (split-half behaviour, alternating and contiguous), `data/unit_keys.npz`, `data/ses_info.npz`. On Kaggle, upload them as a dataset; the scripts find them under `/kaggle/input/`.

## Where each result in the paper comes from
| Paper section / item | Script(s) | Result file(s) |
|---|---|---|
| Behaviour-baseline audit (Table 2, Fig 2) | `code/laptop/bvb_cog.py`, `bvb_pav.py`, `room_check_behav.py` | `TEST_HISTORY.md` (P8-BVB-COG, P8-BME2, P8-R5); rules in `preregistrations/pilot_BVB_prereg.md` |
| Behaviour matching and its weakness (Fig 3a) | `code/laptop/bme.py`, `bme2_pav.py`, `bme_checks.py` | `results/pilot_BME*_results.md`, `bme_checks_results.json` |
| Five-method comparison (Table 3, Fig 3b–c) | `code/kaggle/compare.py` | `results/pilot_COMPARE_results.md`, `compare_results.json` |
| RB-ME validation and real data (Fig 4–5, Table 5) | `code/kaggle/rbme.py`, `code/laptop/per_person_rbme.py` | `results/pilot_RBME_results.md`, `rbme_results.json`, `per_person_rbme.json` |
| SIMEX correction | `code/kaggle/ncbme.py` | `results/pilot_NCBME_results.md`, `ncbme_results.json` |
| Split-half reliabilities (Table 4), EEGNet accuracy check | `code/laptop/rel_and_rounding.py`, `make_ab.py` | `results/rel_and_rounding_results.json` |
| Robustness (Fig 6) | `code/kaggle/rbme_rob.py`, `deep_cog.py` | `results/pilot_RBMEROB_results.md`, `rbme_rob_results.json`, `pilot_DEEP_results.md` |
| Time course (Fig 7) | `code/kaggle/lead_cog.py` | `results/pilot_LEAD_results.md`, `lead_results.json` |
| Eye/muscle ICA checks (Table 6, Fig 8) | `code/kaggle/artefact_body.py`, `artefact2_body.py` (+ `build_artefact*_nb.py`, generated `artefact*_kaggle.py`) | `results/pilot_ARTEFACT_results.md`, `pilot_ARTEFACT2_results.md`, `artefact2_summary.json` |
| ARTEFACT-NULL simulation, activity matching, key-press decoder | `code/laptop/artefact_null_sim.py`, `artefact_v3_activity.py`, `activity_decoder_real.py` | `results/artefact_null_results.json`, `artefact_v3_results.json`, `activity_decoder_real_results.json` |
| Independent-dataset check (iNCog-EEG; not usable) | `code/kaggle/incog_inspect.py` | `TEST_HISTORY.md` (P8-INCOG) |
| All figures | `code/laptop/make_figs.py` | — |

## Data (public; raw recordings not redistributed here)
- COG-BCI: Hinss et al. (2023), Zenodo https://doi.org/10.5281/zenodo.7413650 (CC-BY-4.0)
- Pavlov digit span: OpenNeuro ds003838 (CC0)
- Tufts fNIRS2MW: https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html (CC-BY-4.0)
The derived files in `data/` are computed from these datasets and are shared under the same terms, with attribution to the original authors.

## Licence
Code: MIT. Text, derived data and result files: CC-BY-4.0.
