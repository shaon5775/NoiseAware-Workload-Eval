# Paper 8 — Pilot BOD "Behaviour-Orthogonal effort Decoder" — rules written 20 Nov 2026 (night) BEFORE running.
Never change.

## Idea (new method)
Lab workload decoders are trained on task difficulty, which behaviour already reveals (P8-BVB: behaviour beats brain on
Tufts and COG-BCI). The useful thing a brain sensor could add is EFFORT THAT BEHAVIOUR CANNOT SHOW (two people with the
same score, one straining). BOD learns exactly that:
1. Target = each recording's self-rated effort (RSME) after removing, within the person, what level and behaviour
   explain (residual of RSME ~ level dummies + behaviour).
2. Inputs = per-person-normalised brain features with the BEHAVIOUR SUBSPACE PROJECTED OUT: on training people, fit
   brain → behaviour (ridge) and remove those directions, so the decoder cannot just re-read performance.
3. Ridge regression, leave-one-person-out. Output for a new person = their "hidden effort" score per recording.
Novelty check: Boring et al. 2020 (JNE) showed a load decoder correlates with self-reported effort, but did not remove
behaviour or train for effort beyond performance. No behaviour-orthogonal effort decoder found.

## Data
COG-BCI, 29 people, recording level (each person: 3 sessions × [0/1/2-back + MATB easy/med/diff] = up to 18
recordings with RSME). Brain = mean over the recording's 1-min units of the P8-BVB-COG brain features (7 channel-mean
band powers + 153 tangent features, 18 common channels), z-scored within person. Behaviour = recording mean of the
P8-BVB-COG behaviour features (n-back: hit rate, false-alarm rate, RT; MATB: tracking error, tank deviation, tracking
spread), z-scored within person and task family. Both task families pooled (family indicator included in the level
dummies).

## Compared decoders (all LOSO, same features)
- H3-base: level-trained brain classifier's expected-level score (as in P8-BVB-COG), residualised like the target.
- PLAIN: ridge on brain → effort residual (no behaviour projection).
- BOD (ours): ridge on behaviour-projected brain → effort residual.
Metric: per held-out person, Spearman ρ between predicted and true effort residual over their recordings; mean over
people; Wilcoxon signed-rank (one-sided > 0). Ridge alpha fixed at 10 (no tuning).

## Pass rules
- B1: BOD mean ρ ≥ 0.15 and Wilcoxon p < 0.01.
- B2: BOD ≥ H3-base + 0.05.
- B3: behaviour-orthogonality holds: mean within-person |ρ(BOD prediction, behaviour composite)| ≤ 0.15, and
  BOD ≥ PLAIN − 0.02 (removing behaviour does not cost accuracy).
GO = B1 and B2 and B3.
