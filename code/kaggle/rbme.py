"""Paper 8 idea 2 RB-ME (rules: pilot_RBME_prereg.md). Inputs: bme_*.npz (paper8-bme-data) + ab_halves.npz."""
import glob, json, sys, time, numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
T0 = time.time(); QUICK = "--quick" in sys.argv
F = sorted(glob.glob("/kaggle/input/**/bme_*.npz", recursive=True)) or sorted(glob.glob("../data/ncbme_upload/bme_*.npz"))
FA = (glob.glob("/kaggle/input/**/ab_halves.npz", recursive=True) or glob.glob("../data/ab_halves.npz"))[0]
Z = {}
for f in F:
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
AB = dict(np.load(FA, allow_pickle=True))
# Pavlov rows in the bme npz are split by person (first 33 sorted people, then rest); reorder halves the same way
s4 = AB["S4_s"].astype(str); ppl = sorted(set(s4))
o = np.r_[np.flatnonzero(np.isin(s4, ppl[:33])), np.flatnonzero(np.isin(s4, ppl[33:]))]
for f in ("s", "y", "A", "B"): AB["S4_" + f] = AB["S4_" + f][o]
SET = [a for a in sys.argv if a in ("S1", "S2", "S4")] if QUICK else ["S1", "S2", "S4"]; CAL = 0.15; SEEDS = 1 if QUICK else 3
for S in SET:
    assert (Z[S + "_s"].astype(str) == AB[S + "_s"].astype(str)).all() and (Z[S + "_y"] == AB[S + "_y"]).all(), S + " misaligned"

def loso(s, y, X, C):
    def fold(te):
        tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X[tr]), y[tr]); return ts, clf.predict_proba(sc.transform(X[ts])) @ clf.classes_
    out = np.zeros(len(y))
    for ts, v in Parallel(n_jobs=-1)(delayed(fold)(te) for te in np.unique(s)): out[ts] = v
    return out

def resid(x, s, y):
    r = x.copy()
    for p in np.unique(s):
        for l in np.unique(y):
            m = (s == p) & (y == l)
            if m.any(): r[m] -= r[m].mean()
    return r

def auc(x, i, j): return np.mean((x[i] > x[j]) + 0.5 * (x[i] == x[j]))

def rbme(s, y, brain, A, B, sb=None):
    if sb is None: sb = loso(s, y, brain, 0.05)
    sA, sB = loso(s, y, A, 1.0), loso(s, y, B, 1.0)
    rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], 0.05)
    stat, braina, bnd = [], [], []
    for p in np.unique(s):
        m = s == p; L = y[m]; b, a_, b_ = sb[m], sA[m], sB[m]
        i, j = np.where(L[:, None] > L[None, :]); vals = []
        for mt, tst in ((a_, b_), (b_, a_)):
            k = np.abs(mt[i] - mt[j]) <= CAL
            if k.sum() < 20: vals = None; break
            ab = np.clip(auc(tst, i[k], j[k]), 1e-3, 1 - 1e-3); bound = norm.cdf(norm.ppf(ab) / np.sqrt(rel))
            vals.append((auc(b, i[k], j[k]), bound))
        if vals: v = np.mean(vals, 0); braina.append(v[0]); bnd.append(v[1]); stat.append(v[0] - v[1])
    stat = np.array(stat); p = wilcoxon(stat, alternative="greater").pvalue if len(stat) > 5 else np.nan
    return dict(n=len(stat), rel=float(rel), brain=float(np.mean(braina)), bound=float(np.mean(bnd)), diff=float(stat.mean()), p=float(p)), sb

res = {}
for S in SET:
    s, y = Z[S + "_s"].astype(str), Z[S + "_y"].astype(int); brain = Z[S + "_brain"].astype(float); A, B = AB[S + "_A"].astype(float), AB[S + "_B"].astype(float)
    r = {}; r["real"], _ = rbme(s, y, brain, A, B); print(S, "REAL", r["real"], f"{time.time()-T0:.0f}s", flush=True)
    T = (A + B) / 2
    for kind in ("null", "pos"):
        vals = []
        for sd in range(SEEDS):
            g = np.random.default_rng(sd); X = T @ g.normal(size=(T.shape[1], 20)) / np.sqrt(T.shape[1]) + g.normal(size=(len(T), 20))
            if kind == "pos": X += 0.3 * (y - y.mean())[:, None] * g.normal(size=20)[None, :]
            vals.append(rbme(s, y, X, T + g.normal(scale=.5, size=T.shape), T + g.normal(scale=.5, size=T.shape))[0])
        r[kind] = dict(diff=float(np.mean([v["diff"] for v in vals])), p=float(np.median([v["p"] for v in vals])),
                       brain=float(np.mean([v["brain"] for v in vals])), bound=float(np.mean([v["bound"] for v in vals])))
        print(S, kind.upper(), r[kind], f"{time.time()-T0:.0f}s", flush=True)
    res[S] = r
V1 = all(res[S]["null"]["diff"] <= .02 for S in SET)
V2 = sum(res[S]["pos"]["diff"] > .03 and res[S]["pos"]["p"] < .01 for S in SET) >= 2
ok = {S: res[S]["real"]["diff"] > 0 and res[S]["real"]["p"] < .01 for S in SET}
R1 = sum(ok.values()) >= 2 and ok.get("S4", False)
print("V1", V1, "| V2", V2, "| METHOD GO", V1 and V2, "| R1", R1, ok, "| FINDING GO", R1)
json.dump(res, open("rbme_results.json", "w"), indent=1)
