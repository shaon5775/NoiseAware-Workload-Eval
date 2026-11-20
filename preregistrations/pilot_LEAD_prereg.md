# Paper 8 — "Brain is faster, not better": Neural Lead Time + Anytime Fusion — rules written 20 Nov 2026 BEFORE any
windowed data exists. Never change.

## Question
With a full minute of behaviour, the brain adds nothing (P8-BVB-COG). Does the brain win in the first seconds of a new
load level, when behaviour has not yet accumulated? For how long ("neural lead time")?

## Data and features (fixed) — COG-BCI (Zenodo 7413650, 29 people × 3 sessions), discovery
- Recordings: zeroBACK/oneBACK/twoBACK (n-back family) and MATBeasy/med/diff (MATB family); one recording = one load level.
- Onset t = 0 = first task event (n-back: first stimulus code 6L2x; MATB: first TRACKING line).
- Times t ∈ {2, 4, 6, 8, 10, 15, 20, 30, 45, 60} s. At time t only data from [0, t] is used.
- Brain(t): mean of 2-s windows (1-s step) fully inside [0, t]: log band power, 7 bands × the 18 common 10-20 channels
  (19 standard minus Cz). Per-person z-score using all that person's windows (no labels, as in earlier pilots).
- Behaviour(t): n-back — hit rate, false-alarm rate, mean RT over stimuli whose 2-s response window closed by t, plus
  number of such stimuli; MATB — mean and SD of tracking error and mean tank deviation |level − 2500| over [0, t].
  Missing (no stimuli yet) → training-set mean. Per-person z-score using that person's full-recording values.
- Models per t and family, leave-one-person-out logistic regression: BRAIN (C = 0.05), BEH (C = 1),
  FUSED = anytime fusion = time-specific logistic on [brain, behaviour] (C = 0.05). Balanced accuracy per person.

## Rules (per family; chance 0.333)
- **L1 brain is faster:** at t = 4 s, BRAIN > BEH, Wilcoxon over people one-sided p < 0.01.
- **L2 meaningful lead:** neural lead time (first t at which group BEH ≥ group BRAIN) ≥ 10 s.
- **L3 fusion helps:** FUSED mean over t ≤ 20 s ≥ BEH mean + 0.03, and FUSED ≥ max(BRAIN, BEH) − 0.02 at every t.
**GO = L1 and L2 and L3 in at least one family.** If GO → separate pre-registered confirmation on Pavlov (encoding EEG
before recall) and Tufts (fNIRS: predicted NO lead, slow signal). Runs on Kaggle.
