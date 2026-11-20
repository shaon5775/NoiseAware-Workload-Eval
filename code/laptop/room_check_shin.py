"""Paper 8 room checks on Shin 2018 EEG features (exploration, no pass rules — they only decide what to pre-register).
A: stronger features (Riemannian tangent space with per-person re-centering) — how high is the ceiling?
B: temporal context — how much does averaging over the last k windows help (no labels needed)?
C: temporal-consistency test-time adaptation (TC-TTA): smooth, pseudo-label the new person's confident windows, refit.
"""
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import glob, os
Z = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob("data/shin/*.npy")}
P = sorted({int(k.split("_")[1]) for k in Z})
def logm(C):
    w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-10))[..., None, :]) @ np.swapaxes(V, -1, -2)
def isqrtm(C):
    w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-10)))) @ V.T
iu = None
D = {}
for p in P:
    task = Z[f"y_{p}"] >= 0
    bp = Z[f"bp_{p}"][task].reshape(task.sum(), -1); bp = (bp - bp.mean(0)) / (bp.std(0) + 1e-9)
    C = Z[f"cov_{p}"][task].astype(float); C = C + 1e-6 * np.eye(C.shape[1]) * np.trace(C, axis1=1, axis2=2).mean() / C.shape[1]
    M = np.exp(0) * np.mean(C, 0); R = isqrtm(M); T = logm(R @ C @ R)            # per-person re-centred tangent space
    if iu is None: iu = np.triu_indices(C.shape[1])
    tg = T[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1, np.sqrt(2)); tg = (tg - tg.mean(0)) / (tg.std(0) + 1e-9)
    o = np.lexsort((Z[f"t_{p}"][task], Z[f"ses_{p}"][task]))
    D[p] = dict(bp=bp[o], tg=tg[o], y=Z[f"y_{p}"][task][o], ses=Z[f"ses_{p}"][task][o])
print("people", len(P), "task windows", sorted({len(d["y"]) for d in D.values()}), "bp", D[P[0]]["bp"].shape[1], "tg", D[P[0]]["tg"].shape[1])
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
def smooth(pr, ses, k):
    out = pr.copy()
    for s in np.unique(ses):
        i = np.flatnonzero(ses == s); c = np.cumsum(np.vstack([np.zeros((1, 3)), pr[i]]), 0)
        for j in range(len(i)): a = max(0, j - k + 1); out[i[j]] = (c[j + 1] - c[a]) / (j + 1 - a)
    return out
res = {}
def add(k, v): res.setdefault(k, []).append(v)
for te in P:
    tr = [p for p in P if p != te]; Y = np.concatenate([D[p]["y"] for p in tr]); d = D[te]
    for feat in ("bp", "tg", "both"):
        get = (lambda q: np.c_[D[q]["bp"], D[q]["tg"]]) if feat == "both" else (lambda q, f=feat: D[q][f])
        X = np.vstack([get(p) for p in tr]); sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.05 if feat != "bp" else 1.0).fit(sc.transform(X), Y)
        Xt = sc.transform(get(te)); pr = clf.predict_proba(Xt)
        add(f"{feat} plain", bacc(d["y"], pr.argmax(1)))
        for k in (3, 5, 10):
            add(f"{feat} +avg last {k}", bacc(d["y"], smooth(pr, d["ses"], k).argmax(1)))
        if feat == "bp":
            ps = smooth(pr, d["ses"], 5); conf = ps.max(1); keep = conf >= np.quantile(conf, 0.5)
            clf2 = LogisticRegression(max_iter=2000).fit(np.vstack([sc.transform(X), Xt[keep]]), np.r_[Y, ps.argmax(1)[keep]],
                                                         sample_weight=np.r_[np.ones(len(Y)), np.full(keep.sum(), 3.0)])
            p2 = clf2.predict_proba(Xt)
            add("bp TC-TTA", bacc(d["y"], p2.argmax(1))); add("bp TC-TTA +avg last 5", bacc(d["y"], smooth(p2, d["ses"], 5).argmax(1)))
for k, v in res.items(): print(f"{k:28s} {np.mean(v):.3f}   (people better than 'bp plain': {np.mean(np.array(v) > np.array(res['bp plain'])):.0%})")
