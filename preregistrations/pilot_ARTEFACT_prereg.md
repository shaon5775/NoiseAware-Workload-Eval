# Paper 8 — pre-registered artefact / task-activity check of the MATB RB-ME result (written 20 Nov 2026, before any run)

Why: reviewer self-check M1 — the MATB "brain beyond performance" result (RB-ME Δ +0.305) could come from eye/muscle
activity or from more task events/key presses at harder levels, not from workload.

Data: COG-BCI MATB (29 people, 3 sessions, easy/med/diff), same 1-min units and same split-half behaviour (ab_halves.npz,
alternating 5-s chunks), caliper 0.15, LOSO logistic C=0.05, RB-ME code unchanged (rbme_rob.py).

Export (Kaggle CPU, paper8-artefact): per recording, 1–40 Hz, ICA (fastica, 20 comps, fit on 1-Hz-filtered data),
remove components flagged by find_bads_eog (Fp1, Fp2 as EOG proxies) and find_bads_muscle (default threshold);
10-s windows / 5-s step as before; per window: log band power (7 bands) per named channel (18 shared channels, no Cz),
mean over all EEG channels, covariance (18 ch, 1–40 Hz) and covariance on 1–20 Hz data of the 12 central/parietal/
occipital/mid-frontal channels (F3 Fz F4 C3 C4 P7 P3 Pz P4 P8 O1 O2). Same features also saved WITHOUT ICA (sanity).

Variants (all RB-ME, MATB):
- V0 no ICA, original feature recipe (mean band power + tangent 18 ch): sanity, expect Δ within ±0.05 of +0.305.
- V1 ICA-cleaned, original recipe.
- V2 ICA-cleaned, no Fp1/Fp2/F7/F8/T7/T8, 1–20 Hz only (5 bands × 12 channels band power + tangent of 1–20 Hz covariance).
- V3 task-activity matching (laptop, original features): each behaviour half gets 3 more features = counts of key presses
  (KB1), scheduled events (EVENT) and audio messages (COMM) in that half; then RB-ME as usual.

Rules (fixed now):
- A1: V1 Δ > 0 and p < 0.01.   A2: V2 Δ > 0 and p < 0.01.   A3: V3 Δ > 0 and p < 0.01 (needs ≥ 8 people with pairs;
  if fewer, A3 = "not testable", reported).
- PASS = A1 and A2 and (A3 or not testable). If PASS: paper says the MATB result survives artefact removal and
  task-activity matching. If FAIL: the MATB claim is weakened in abstract, results and conclusion.
- Descriptive only (no pass/fail): band-wise (theta 4–8, alpha 8–13, beta 13–30, delta 1–4, gamma 30–40) and
  region-wise (frontal F3 Fz F4; central C3 C4; parietal P3 Pz P4 P7 P8; occipital O1 O2) RB-ME Δ using ICA-cleaned
  band power of that band/region only (no covariance).
