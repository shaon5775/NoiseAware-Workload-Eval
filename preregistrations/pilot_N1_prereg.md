# Paper 8 — Pilot N1 "felt vs set workload" (go/no-go) — rules written 19 Nov 2026 BEFORE linking ratings to fNIRS

## Question
Does the fNIRS signal follow how hard a block FELT to the person (their NASA-TLX rating), not only the n-back level
they were given? If yes, labs should label workload by felt load, and that explains the ~35% "level" accuracy ceiling.

## Data
Tufts fNIRS2MW, 68 people, 16 blocks each (19 windows of 30 s per block, as in P8-A).
Per block: NASA-TLX subscales mental, effort, frustration, performance (0–100) from
serial_feedbacktrain_1_1_k (block k = 1..12) and serial_feedbacktest_1_2_k (block 12+k); sub_1 uses
nasa_workload_index_* files with the same numbering. Per block n-back accuracy from task_accuracy (secondary only).
**Felt load (primary) = TLX "mental" score.** Secondary: mean of mental, effort, frustration, 100−performance.
Each person's felt scores are z-scored within that person (people use the scale differently).
Blocks or people with missing ratings are dropped and reported.

## Model and splits
Same 17 official buckets as P8-A (train+val people → train, 4 test people). Per-person normalised window features
(48 features, best setting in P8-A). Ridge regression (alpha chosen by 5-fold CV over train people from
{0.1, 1, 10, 100}). Block brain score = mean of the 19 window predictions.
- SET model: target = n-back level 0–3.
- FELT model: target = the person's z-scored felt load of that block.

## Pass rules
- G0 (ratings are wired correctly): mean within-person Spearman ρ(level, felt) over people ≥ 0.30.
- G1 (felt really differs from set): mean within-person ρ(level, felt) ≤ 0.85.
- G2 (main): WITHIN the same level, do blocks that felt harder get a higher brain score from the SET model?
  Per person: Spearman between brain score and felt load after removing level (residuals of each within its level).
  Pass if mean ρ ≥ 0.10 AND Wilcoxon signed-rank p < 0.05 over people (one-sided, > 0).
- G3 (practical): on new people, FELT model block score vs felt load (mean within-person ρ) ≥ SET model block score
  vs felt load + 0.05.
GO = G0 and G1 and G2 and G3. G0 fail → stop and check the file mapping (not a result).
Report also: same tests with the secondary felt score and with task accuracy (lower accuracy = harder).
