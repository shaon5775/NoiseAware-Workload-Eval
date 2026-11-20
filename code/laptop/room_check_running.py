"""Paper 8 room check (exploration): LIVE per-person drift removal = normalise each window by the mean/std of the same
person's PAST windows only (last M minutes, unlabelled). Training people get the same treatment."""
import glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open("room_check_session.py").read().split("def bacc")[0].split("print(")[0])
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
T = {p: Z[f"t_{p}"][Z[f"y_{p}"] >= 0].astype(float) for p in P}
raw = {}
for p in P:
    F, y, b, s = feats(p, "person"); o = np.argsort(T[p]); raw[p] = (F[o], y[o], b[o], T[p][o])
def running(F, t, M):
    out = np.zeros_like(F)
    for i in range(len(F)):
        k = (t >= t[i] - M * 60) & (t <= t[i]);
        if k.sum() < 20: k = np.arange(len(F)) < 20                # warm-up: first 20 windows
        out[i] = (F[i] - F[k].mean(0)) / (F[k].std(0) + 1e-9)
    return out
for M in (None, 3, 6, 10, 15):
    D = {p: ((raw[p][0] if M is None else running(raw[p][0], raw[p][3], M)),) + raw[p][1:3] for p in P}
    acc, blk = [], []
    for te in P:
        tr = [p for p in P if p != te]; X = np.vstack([D[p][0] for p in tr]); Y = np.concatenate([D[p][1] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(X), Y)
        F, y, b = D[te]; pr = clf.predict(sc.transform(F)); acc.append(bacc(y, pr))
        blk.append(np.mean([np.bincount(pr[b == q], minlength=3).argmax() == y[b == q][0] for q in np.unique(b)]))
    if M is None: a0 = np.array(acc)
    print(f"{'whole-session z' if M is None else f'running past {M:2d} min'}: window {np.mean(acc):.3f} | block {np.mean(blk):.3f} | people improved {np.mean(np.array(acc) > a0):.0%}")
