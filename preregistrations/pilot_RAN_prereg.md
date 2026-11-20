# Paper 8 — Pilot RAN "balanced-session trap" + Rest-Anchored Normalisation — rules written 20 Nov 2026 BEFORE
running on this data. Never change.

## Question
(1) On EEG, does the usual per-person z-scoring (statistics from the new person's own task data) lose accuracy when
the new person's session is NOT balanced across workload levels? (2) Does normalising by a short REST recording
(Rest-Anchored Normalisation, RAN) remove that loss?

## Data
Shin 2018 n-back EEG (0/2/3-back → 0/1/2), 26 people. Task windows: 10 s, stride 2 s inside task blocks (as P8-RK).
Rest windows: 10 s, stride 2 s inside each gap between blocks, from block end + 2 s to next block start − 2 s.
"1-minute rest calibration" = the person's FIRST rest windows in time, up to 60 s of rest data (rest windows whose
start lies in the first 60 s of the rest periods, in order). Features: 84 raw log band powers (θ, α, β × 28 EEG ch).

## Normalisations (applied the same way to training people and the test person)
- Z-own (usual): mean/std of the person's task windows (for the test person: of the windows in the test session).
- Oracle: mean/std of the person's FULL balanced task data (not available in real use; upper reference).
- RAN (ours, primary): mean/std of the person's 1-minute rest calibration windows.
- RAN-m (secondary): rest mean, task-session std.

## Protocol
Leave-one-person-out. Classifier: logistic regression (multinomial, C = 1, StandardScaler on train), trained on 25
people's full balanced task data. Test sessions: 150 windows sampled from the test person with level mix
balanced (1/3 each), mostly easy (0.60/0.25/0.15), mostly hard (0.15/0.25/0.60), easy+medium only (0.5/0.5/0);
10 random resamples per mix (seeded). Metric: balanced accuracy over the levels present.

## Pass rules (logistic regression)
- T1 trap is real on EEG: Z-own(easy+medium only) ≤ Oracle(easy+medium only) − 0.05 AND mean of Z-own over the 3
  unbalanced mixes ≤ mean of Oracle over them − 0.03.
- T2 RAN fixes it: mean RAN over the 3 unbalanced mixes ≥ mean Z-own over them + 0.03 AND RAN(easy+medium only) ≥
  Z-own(easy+medium only) + 0.05.
- T3 no big cost in the lab case: RAN(balanced) ≥ Z-own(balanced) − 0.03.
GO = T1 and T2 and T3. If T1 fails → the trap is too small on this data (NO-GO for the paper story).
