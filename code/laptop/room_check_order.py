"""Paper 8 room check (exploration, no pass rule): is fNIRS2MW workload accuracy partly 'which block / when in session'?
Every subject has the SAME block order (0123 1230 2301 3012) and level 0 always follows level 3."""
import numpy as np, pandas as pd, pilot_P8A as P
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score as bacc

ORDER = [0,1,2,3,1,2,3,0,2,3,0,1,3,0,1,2]
def feats(sub):
    d = pd.read_csv(f"data/sub_{sub}.csv"); x = d[P.CH].values; l = d.label.values
    st = np.r_[0, np.flatnonzero(np.diff(l)) + 1, len(l)]; t = np.arange(P.WIN) - (P.WIN - 1) / 2
    F, Y, B, Pos = [], [], [], []
    for bi, (a, b) in enumerate(zip(st[:-1], st[1:])):
        for k, s in enumerate(range(a, b - P.WIN + 1, P.STRIDE)):
            w = x[s:s + P.WIN]; sl = (t[:, None] * (w - w.mean(0))).sum(0) / (t ** 2).sum()
            F.append(np.r_[w.mean(0), w.std(0), sl, w.min(0), w.max(0), w[-1] - w[0]]); Y.append(l[a]); B.append(bi); Pos.append(k)
    F = np.array(F); F = (F - F.mean(0)) / (F.std(0) + 1e-9)          # per-person norm (best from P8-A)
    return F, np.array(Y), np.array(B), np.array(Pos)

subs = sorted({s for b in P.buckets() for part in b for s in part}); D = {s: feats(s) for s in subs}
assert all((np.array([D[s][1][D[s][2] == i][0] for i in range(16)]) == ORDER).all() for s in subs)
cols = {"all": slice(0, 48), "mean_only": slice(0, 8), "shape_only(slope,last-first)": np.r_[16:24, 40:48]}
out = {}
for name, c in cols.items():
    yy, pp, pos, bb, pb = [], [], [], [], []
    for tr, va, te in P.buckets():
        tr = tr + va
        X = np.vstack([D[s][0][:, c] for s in tr]); sc = StandardScaler().fit(X)
        m = LogisticRegression(max_iter=3000).fit(sc.transform(X), np.concatenate([D[s][1] for s in tr]))
        mb = LogisticRegression(max_iter=3000).fit(sc.transform(X), np.concatenate([D[s][2] for s in tr])) if name == "all" else None
        for s in te:
            F, Y, B, Pos = D[s]; Xs = sc.transform(F[:, c])
            yy.append(Y); pp.append(m.predict(Xs)); pos.append(Pos); bb.append(B)
            if mb is not None: pb.append(mb.predict(Xs))
    yy, pp, pos, bb = map(np.concatenate, (yy, pp, pos, bb))
    r = {"label_bacc": bacc(yy, pp)}
    for lo, hi, tag in [(0, 4, "first 4 windows (0-40 s)"), (4, 10, "mid"), (10, 19, "last 9 windows")]:
        k = (pos >= lo) & (pos < hi); r[f"bacc {tag}"] = bacc(yy[k], pp[k])
    if pb:
        pb = np.concatenate(pb); r["block_index_acc (chance 0.0625)"] = (pb == bb).mean()
        r["label via predicted block"] = bacc(yy, np.array(ORDER)[pb])
        r["block in correct quarter (chance .25)"] = (pb // 4 == bb // 4).mean()
    out[name] = r; print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
