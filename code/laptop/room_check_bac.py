"""Paper 8 room check (exploration): Behaviour-Anchored Calibration (BAC). Brain ranks well WITHIN a person but is weak
at ABSOLUTE level across people; behaviour is strong at absolute level. For a new person: use behaviour-predicted
levels as pseudo-labels to fit a person-specific mapping from brain score to level (no true labels), then fuse."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
import bvb_cog as B, pilot_P8A as P
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])
def run(df, name):
    R = []
    for te in df.s.unique():
        tr, ts = (df.s != te).values, (df.s == te).values; L = df.level.values; K = len(np.unique(L))
        XB, XH = np.stack(df.brain.values), np.stack(df.behz.values)
        sb = StandardScaler().fit(XB[tr]); cb = LogisticRegression(max_iter=3000, C=0.05).fit(sb.transform(XB[tr]), L[tr])
        sh = StandardScaler().fit(XH[tr]); ch = LogisticRegression(max_iter=3000, C=1.0).fit(sh.transform(XH[tr]), L[tr])
        pb, ph = cb.predict_proba(sb.transform(XB[ts])), ch.predict_proba(sh.transform(XH[ts])); y = L[ts]
        late = np.argmax(np.log(ph + 1e-9) + np.log(pb + 1e-9), 1)                     # plain late fusion
        # BAC: person-specific 1-D calibration of the brain score using behaviour pseudo-labels (confident ones)
        s = (pb @ np.arange(K)).reshape(-1, 1); pl = ph.argmax(1); conf = ph.max(1); keep = conf >= np.quantile(conf, 0.3)
        if len(np.unique(pl[keep])) == K:
            cal = LogisticRegression(max_iter=2000).fit(s[keep], pl[keep]); pc = cal.predict_proba(s)
        else: pc = pb
        bac = np.argmax(np.log(ph + 1e-9) + np.log(pc + 1e-9), 1)
        R.append(dict(BEH=bacc(y, ph.argmax(1)), BRAIN=bacc(y, pb.argmax(1)), LATE=bacc(y, late), BRAIN_cal=bacc(y, pc.argmax(1)), BAC=bacc(y, bac)))
    R = pd.DataFrame(R); print(f"{name:14s}", {k: round(v, 3) for k, v in R.mean().items()},
                               "| BAC vs BEH p", round(wilcoxon(R.BAC, R.BEH).pvalue, 4), "share", f"{np.mean(R.BAC > R.BEH):.0%}")
for fam, nm in (("nback", "COG n-back"), ("matb", "COG MATB")): run(B.build(fam).reset_index(drop=True), nm)
rows = []
for s in sorted({x for b in P.buckets() for part in b for x in part}):
    F, Y = P.features(s); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    acc = pd.read_csv(f"data/task_accuracy/sub_{s}.csv").acc.values; acc = (acc - acc.mean()) / (acc.std() + 1e-9)
    for k in range(16): rows.append(dict(s=s, level=int(Y[k * 19]), brain=F[k * 19:(k + 1) * 19].mean(0), behz=np.array([acc[k]])))
run(pd.DataFrame(rows), "Tufts")
