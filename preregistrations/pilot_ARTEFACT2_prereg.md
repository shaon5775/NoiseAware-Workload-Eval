# Paper 8 — pre-registered follow-up: eye or muscle? (written 20 Nov 2026 night, BEFORE the run; after P8-ARTEFACT failed)

Why: in P8-ARTEFACT, removing eye AND muscle ICA components together cut the MATB RB-ME Δ from +0.305 to +0.077
(p 0.15). The muscle detector removed ~9.7 of 20 components, so it is unclear whether eye movements, muscle activity,
or over-cleaning of brain signal caused the drop. This test separates them. It does NOT replace the failed test;
P8-ARTEFACT stays reported as a fail whatever happens here.

Same data, units, halves, features, decoder and RB-ME code as P8-ARTEFACT. One ICA fit per recording (same settings,
random_state 0, decim 4); from it three cleaned versions:
- EYE: remove only find_bads_eog components (Fp1, Fp2 proxies).
- MUSCLE: remove only find_bads_muscle components.
- BOTH: remove both (replicates P8-ARTEFACT V1; sanity: Δ within ±0.03 of +0.077).
Also a no-ICA version (sanity: Δ within ±0.02 of +0.305).

Rules (fixed now):
- E1: EYE Δ > 0 and p < 0.01.   M1: MUSCLE Δ > 0 and p < 0.01.
- Interpretation table (decided now):
  E1 ✓, M1 ✗ → effect depends on muscle-band activity (or on signal the muscle detector removes); not shown to be neural.
  E1 ✗, M1 ✓ → effect depends on eye movements.
  E1 ✗, M1 ✗ → both contribute; not shown to be neural.
  E1 ✓, M1 ✓ → neither alone explains it; the BOTH result (P8-ARTEFACT) is then over-cleaning-sensitive — still reported
  as "not robust to full artefact removal".
- No claim that EEG carries workload information beyond performance in MATB will be made unless E1 and M1 both pass
  AND the paper reports the P8-ARTEFACT failure next to it.
- Descriptive: region-wise and band-wise Δ for EYE and MUSCLE versions (same groups as P8-ARTEFACT).

Second part (laptop, simulation, rules fixed now) — "ARTEFACT-NULL": does RB-ME credit a decoder that reads only
task activity (key presses, events, messages per unit; real COG-BCI MATB counts) and true performance, but no brain?
Fake brain = random linear mix of [true performance T, activity counts] + noise, 20 seeds, behaviour noise SD 0.5.
- S1: RB-ME false-claim rate (p < 0.01) ≥ 50 % for the ARTEFACT-NULL decoder → RB-ME alone cannot separate "beyond
  performance" from "beyond performance but non-neural"; artefact/activity control is a necessary second step.
- S2 (control): the plain NULL (T only) false-claim rate stays ≤ 10 %.
