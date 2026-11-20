# Paper 8 — Pilot M-A "EEG-taught fNIRS" (lag-aware privileged distillation) — rules written 19 Nov 2026,
BEFORE any model is trained on this data. Never change these. (Data FORMAT may be inspected first; that does not
touch labels vs signals.)

## Question
If we train with EEG + fNIRS but deploy with fNIRS ONLY, does an EEG "teacher" make the fNIRS-only model better for a
NEW person — and does letting the student learn from the EEG a few seconds EARLIER (neurovascular delay) help more
than plain same-time distillation?

## Data
Shin et al. 2018 (TU Berlin), dataset A n-back (0-, 2-, 3-back), 26 people, EEG + NIRS MATLAB files.
fNIRS: HbO and HbR from all NIRS channels, band-pass 0.01–0.1 Hz (if not already filtered), per-person normalised.
EEG: band power (theta 4–8, alpha 8–13, beta 13–30 Hz, log) per EEG channel, per-person normalised.
Samples: windows inside task blocks only (exact window length/stride fixed after seeing the block timing:
the window = 10 s, stride 2 s if blocks are ≥ 20 s; otherwise the whole block minus 2 s at each end).

## Evaluation
Leave-one-person-out (26 folds). Metric: balanced accuracy, 3 classes (chance 0.333), per test person and mean.
Same small model family for everyone: softmax regression / 1-hidden-layer MLP (64 units) in PyTorch, CPU, 3 seeds.

## Models
- B1 fNIRS-only (student without teacher)  ← the deploy baseline to beat
- B2 EEG-only (teacher)
- B3 EEG + fNIRS fusion (upper bound; needs EEG at deploy, so not our goal)
- D0 plain distillation: fNIRS student trained on hard labels + teacher's soft probabilities from the SAME time window
  (loss = 0.5 CE(label) + 0.5 KL(teacher, T = 2)).
- D-lag (ours): same, but teacher probabilities from the EEG window ending L seconds EARLIER than the fNIRS window,
  L from {0, 2, 4, 6, 8} s chosen per training person by teacher–student agreement (inner check on train people only).
Test people never give EEG or labels to the student.

## Pass rules
- R1 (teacher has something to give): B3 − B1 ≥ 0.03 OR B2 − B1 ≥ 0.03 (mean balanced accuracy).
- R2 (distillation helps the deployable model): best of D0/D-lag − B1 ≥ 0.02 AND improved for ≥ 60% of the 26 people.
- R3 (our lag idea matters): D-lag − D0 ≥ 0.01, with ≥ 55% of people improved.
GO = R1 and R2 and R3. If R1 fails → NO-GO (EEG adds nothing on this data). If only R3 fails → plain cross-modal
distillation for workload remains (smaller novelty) — report, decide next step.
