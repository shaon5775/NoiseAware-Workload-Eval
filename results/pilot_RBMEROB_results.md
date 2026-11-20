# RB-ME robustness check — RESULT: ✅ ROBUST (K1, K2, K3 all pass) (20 Nov 2026, Kaggle paper8-rbme-rob, 156 s)
Rules: pilot_RBMEROB_prereg.md. diff = brain AUC on matched pairs − bound (perfect performance reader).

## Real data: diff (p) — calipers 0.10 / 0.15 / 0.20 / 0.30, then band-power-only brain
| Setting | ALT split (reliability) | BLOCK split (reliability) |
|---|---|---|
| COG MATB | +0.297 / +0.305 / +0.298 / +0.249 (all p ≤ 0.002); BP-only +0.333 (rel 0.77) | +0.227 / +0.179 / +0.181 / +0.151 (all p ≤ 0.0024); BP-only +0.207 (rel 0.47) |
| COG n-back | −0.18 to −0.21 everywhere (rel 0.28) | −0.20 to −0.25 (rel 0.22) |
| Pavlov | −0.04 to −0.07 (rel 0.74) | −0.18 to −0.19 (rel 0.36) |

## Fake-brain validation (caliper 0.15, ALT): NULL must be ≤ +0.02; POS must be > 0.03, p < 0.01
| Setting | NULL noise 0.3 / 0.8 | POS noise 0.3 / 0.8 |
|---|---|---|
| n-back | −0.025 / +0.008 ✅ | +0.182 / +0.147 ✅ |
| MATB | −0.058 / −0.036 ✅ | +0.145 / +0.088 ✅ |
| Pavlov | −0.016 / −0.027 ✅ | +0.177 / +0.082 ✅ |

**K1 ✅ K2 ✅ K3 ✅ → ROBUST.** The method stays valid at both noise levels, and the MATB finding holds at every caliper,
with both split types and with a simpler decoder. Memory-task result (no brain information beyond performance) is
also stable. Note: S2 quick test was run on the laptop before this Kaggle run (same numbers).
