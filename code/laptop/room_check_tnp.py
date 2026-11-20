"""Paper 8 room check (exploration): Time-Nuisance Projection (TNP). Learn, from TRAINING people only, the feature
directions that track time-in-recording (drift: gel, fatigue, warm-up) and project them out for everyone. Works live
for a new person (no future data, no labels). Levels are counterbalanced over time, so these directions are not load."""
import glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open("room_check_session.py").read().split("def bacc")[0].split("print(")[0])
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
T = {p: Z[f"t_{p}"][Z[f"y_{p}"] >= 0].astype(float) for p in P}
base = {p: feats(p, "person") for p in P}
def basis(F, t, d=3):
    tt = (t - t.mean()) / t.std(); V = np.vander(tt, d + 1)[:, :-1]          # time polynomials (no constant)
    return np.linalg.lstsq(np.c_[V, np.ones(len(t))], F, rcond=None)[0][:-1]  # d × features: drift loadings
res = {}
for te in P:
    tr = [p for p in P if p != te]
    W = np.vstack([basis(base[p][0], T[p]) for p in tr])                       # stacked drift directions of 25 people
    U, S, Vt = np.linalg.svd(W - 0 * W.mean(0), full_matrices=False)
    for k in (0, 2, 5, 10, 20):
        Q = Vt[:k].T
        proj = (lambda F: F - (F @ Q) @ Q.T) if k else (lambda F: F)
        X = np.vstack([proj(base[p][0]) for p in tr]); Y = np.concatenate([base[p][1] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(X), Y)
        F, y, b, s = base[te]; pr = clf.predict(sc.transform(proj(F)))
        res.setdefault(k, []).append((bacc(y, pr), np.mean([np.bincount(pr[b == q], minlength=3).argmax() == y[b == q][0] for q in np.unique(b)])))
a0 = np.array([r[0] for r in res[0]])
for k, r in res.items():
    r = np.array(r); print(f"TNP remove {k:2d} drift directions: window {r[:,0].mean():.3f} | block {r[:,1].mean():.3f} | people improved {np.mean(r[:,0] > a0):.0%}")
