"""Paper 8 room check (exploration): NEURAL EARLY WARNING. In MATB (continuous performance log), does the brain in
minute t predict the change in performance from t to t+1 beyond current performance and level? LOSO ridge."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, wilcoxon
import bvb_cog as B
df = B.build("matb").reset_index(drop=True)
raw = np.stack(df.beh.values)[:, 0]                       # tracking error (raw, per unit)
rows = []
for (s, ses, task), g in df.groupby(["s", "ses", "task"]):
    g = g.sort_values("unit")
    for a, b in zip(g.index[:-1], g.index[1:]):
        if df.unit[b] == df.unit[a] + 1 and np.isfinite(raw[a]) and np.isfinite(raw[b]):
            rows.append(dict(s=s, rec=(ses, task), level=df.level[a], cur=raw[a], nxt=raw[b], brain=df.brain[a]))
R = pd.DataFrame(rows)
for s, g in R.groupby("s"):                                # per-person z of performance
    mu, sd = g.cur.mean(), g.cur.std() + 1e-9; R.loc[g.index, "curz"] = (g.cur - mu) / sd; R.loc[g.index, "dz"] = (g.nxt - g.cur) / sd
out = {"perf+level": [], "perf+level+brain": []}
for te in R.s.unique():
    tr, ts = R.s != te, R.s == te
    base = lambda d: np.c_[d.curz.values, np.eye(3)[d.level.values]]
    full = lambda d: np.c_[base(d), np.stack(d.brain.values)]
    for k, f in (("perf+level", base), ("perf+level+brain", full)):
        m = Ridge(alpha=10).fit(f(R[tr]), R[tr].dz.values); out[k].append(spearmanr(m.predict(f(R[ts])), R[ts].dz.values)[0])
M = {k: np.nan_to_num(np.array(v)) for k, v in out.items()}
for k, v in M.items(): print(f"{k:18s} within-person rho {v.mean():+.3f}")
d = M["perf+level+brain"] - M["perf+level"]; print("brain adds", round(d.mean(), 3), "p", round(wilcoxon(d).pvalue, 4), "share", f"{np.mean(d > 0):.0%}", "| pairs", len(R))
