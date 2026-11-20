"""Paper 8 room check (exploration), Tufts fNIRS2MW: the app always knows the person's performance (answers right/wrong)
but not their workload. Does performance alone already tell the level? Does adding the brain score help beyond it?"""
import numpy as np, pandas as pd, pilot_P8A as P
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
subs = sorted({s for b in P.buckets() for part in b for s in part}); D = {}
for s in subs:
    F, Y = P.features(s); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    acc = pd.read_csv(f"data/task_accuracy/sub_{s}.csv").acc.values
    D[s] = (F, Y, np.repeat(np.arange(16), 19), (acc - acc.mean()) / (acc.std() + 1e-9))
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])
rows = []
for tr, va, te in P.buckets():
    tr = tr + va; X = np.vstack([D[s][0] for s in tr]); Y = np.concatenate([D[s][1] for s in tr])
    sc = StandardScaler().fit(X); m = Ridge(alpha=10).fit(sc.transform(X), Y)
    # block-level training table for fusion: [brain score (out-of-sample-ish via same model), performance]
    Btr = np.vstack([np.c_[[m.predict(sc.transform(D[s][0]))[D[s][2] == k].mean() for k in range(16)], D[s][3]] for s in tr])
    Ytr = np.concatenate([D[s][1][::19] for s in tr])
    fb = LogisticRegression(max_iter=2000).fit(Btr[:, :1], Ytr); fp = LogisticRegression(max_iter=2000).fit(Btr[:, 1:], Ytr)
    ff = LogisticRegression(max_iter=2000).fit(Btr, Ytr)
    for s in te:
        F, Y, b, perf = D[s]; bs = np.array([m.predict(sc.transform(F))[b == k].mean() for k in range(16)]); yb = Y[::19]
        Bt = np.c_[bs, perf]
        rows.append(dict(brain=bacc(yb, fb.predict(Bt[:, :1])), perf=bacc(yb, fp.predict(Bt[:, 1:])), fused=bacc(yb, ff.predict(Bt)),
                         rho_brain=spearmanr(bs, yb)[0], rho_perf=spearmanr(-perf, yb)[0]))
df = pd.DataFrame(rows); print(df.mean().round(3).to_string()); print("people where fused > perf alone:", round(float((df.fused > df.perf).mean()), 2))
