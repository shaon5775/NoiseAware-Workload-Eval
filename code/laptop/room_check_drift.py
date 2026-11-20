"""Paper 8 room check (exploration): whole blocks go wrong. Is it slow DRIFT over the ~1-hour recording (gel drying,
fatigue)? Remove a smooth time trend from each feature, per person, without labels (polynomial in time, degree d)."""
import glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open("room_check_session.py").read().split("def bacc")[0].split("print(")[0])
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
T = {p: Z[f"t_{p}"][Z[f"y_{p}"] >= 0].astype(float) for p in P}
base = {p: feats(p, "person") for p in P}
# how strongly do features drift with time? mean |corr(feature, time)|, and does level depend on time?
print("mean |corr(feature,time)|:", round(float(np.mean([np.nanmean(np.abs([np.corrcoef(base[p][0][:, j], T[p])[0, 1] for j in range(base[p][0].shape[1])])) for p in P])), 3),
      "| corr(level,time):", round(float(np.mean([np.corrcoef(base[p][1], T[p])[0, 1] for p in P])), 3))
for d in (0, 1, 2, 3, 5):
    D = {}
    for p in P:
        F, y, b, s = base[p]; t = (T[p] - T[p].mean()) / T[p].std()
        if d: V = np.vander(t, d + 1); F = F - V @ np.linalg.lstsq(V, F, rcond=None)[0]; F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        D[p] = (F, y, b)
    acc, blk = [], []
    for te in P:
        tr = [p for p in P if p != te]; X = np.vstack([D[p][0] for p in tr]); Y = np.concatenate([D[p][1] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(X), Y)
        F, y, b = D[te]; pr = clf.predict(sc.transform(F)); acc.append(bacc(y, pr))
        blk.append(np.mean([np.bincount(pr[b == k], minlength=3).argmax() == y[b == k][0] for k in np.unique(b)]))
    print(f"detrend degree {d}: window bal.acc {np.mean(acc):.3f} | block majority {np.mean(blk):.3f} | people better than d=0 ...")
    if d == 0: a0 = np.array(acc)
    else: print("      share of people improved vs no detrend:", round(float(np.mean(np.array(acc) > a0)), 2))
