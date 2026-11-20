"""Paper 8 room check (exploration, no pass rule): per-person z-scoring silently assumes the new person's session has the
SAME mix of workload levels as training (balanced). Real sessions are not balanced (e.g. mostly easy). Does accuracy
collapse when the test session is imbalanced, and does a prior-aware normalisation (SPN-EM) fix it?"""
import numpy as np, pilot_P8A as P
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score as bacc

subs = sorted({s for b in P.buckets() for part in b for s in part}); RAW = {s: P.features(s) for s in subs}
z = lambda F, m, sd: (F - m) / (sd + 1e-9)
MIXES = {"balanced": [.25, .25, .25, .25], "mostly easy": [.55, .25, .12, .08], "mostly hard": [.08, .12, .25, .55],
         "easy+medium only": [.5, .5, 0, 0]}
rng = np.random.RandomState(0)
def sample(Y, mix, n=200):
    idx = []
    for k, w in enumerate(mix):
        c = np.flatnonzero(Y == k); m = int(round(w * n))
        if m: idx += list(rng.choice(c, m, replace=m > len(c)))
    return np.array(idx)
res = {k: {"z_own": [], "z_oracle": [], "spn": []} for k in MIXES}
for tr, va, te in P.buckets():
    tr = tr + va
    X = np.vstack([z(RAW[s][0], RAW[s][0].mean(0), RAW[s][0].std(0)) for s in tr]); Y = np.concatenate([RAW[s][1] for s in tr])
    sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=3000).fit(sc.transform(X), Y)
    mu = np.array([X[Y == k].mean(0) for k in range(4)])                    # class means in person-normalised space
    for s in te:
        F, Ys = RAW[s]
        for name, mix in MIXES.items():
            for rep in range(5):
                i = sample(Ys, mix); Fi, yi = F[i], Ys[i]
                p_own = clf.predict(sc.transform(z(Fi, Fi.mean(0), Fi.std(0))))
                p_or = clf.predict(sc.transform(z(Fi, F.mean(0), F.std(0))))       # oracle: full balanced session stats
                # SPN-EM: jointly estimate the person's offset and the session's level mix (no labels)
                m, sd, pi = Fi.mean(0), Fi.std(0), np.full(4, .25)
                for _ in range(30):
                    Zi = z(Fi, m, sd); pr = clf.predict_proba(sc.transform(Zi))
                    pr = pr * (pi / .25); pr /= pr.sum(1, keepdims=True)              # prior-shift correction (Saerens)
                    pi = 0.5 * pi + 0.5 * pr.mean(0)
                    m = m + sd * (Zi.mean(0) - pi @ mu)                               # remove the mix-induced offset
                pr = clf.predict_proba(sc.transform(z(Fi, m, sd))) * (pi / .25)
                res[name]["z_own"].append(bacc(yi, p_own)); res[name]["z_oracle"].append(bacc(yi, p_or))
                res[name]["spn"].append(bacc(yi, pr.argmax(1)))
for name, r in res.items():
    print(f"{name:18s}", {k: round(float(np.mean(v)), 3) for k, v in r.items()})
