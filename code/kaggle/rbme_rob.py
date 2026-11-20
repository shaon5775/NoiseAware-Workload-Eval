"""Paper 8 RB-ME robustness (rules: pilot_RBMEROB_prereg.md). Reuses RB-ME code. Inputs: bme_*.npz (paper8-bme-data) + ab_halves.npz."""
import glob, json, sys, time, numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
T0 = time.time(); QUICK = "--quick" in sys.argv
F = sorted(glob.glob("/kaggle/input/**/bme_*.npz", recursive=True)) or sorted(glob.glob("../data/ncbme_upload/bme_*.npz"))
FA = (glob.glob("/kaggle/input/**/ab_halves.npz", recursive=True) or glob.glob("../data/ab_halves.npz"))[0]
FB = (glob.glob("/kaggle/input/**/ab_halves_block.npz", recursive=True) or glob.glob("../data/ab_halves_block.npz"))[0]
Z = {}
for f in F:
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
ABS = {}
for nm, fp in (("ALT", FA), ("BLOCK", FB)):
    AB = dict(np.load(fp, allow_pickle=True))
    s4 = AB["S4_s"].astype(str); ppl = sorted(set(s4))
    o = np.r_[np.flatnonzero(np.isin(s4, ppl[:33])), np.flatnonzero(np.isin(s4, ppl[33:]))]
    for f in ("s", "y", "A", "B"): AB["S4_" + f] = AB["S4_" + f][o]
    ABS[nm] = AB
SET = [a for a in sys.argv if a in ("S1", "S2", "S4")] if QUICK else ["S1", "S2", "S4"]; CAL = 0.15; SEEDS = 1 if QUICK else 3
for S in SET:
    for AB in ABS.values(): assert (Z[S + "_s"].astype(str) == AB[S + "_s"].astype(str)).all() and (Z[S + "_y"] == AB[S + "_y"]).all(), S + " misaligned"

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

def rbme(s, y, brain, A, B, sb=None, CAL=0.15):
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


BPCOLS = {"S1": 7, "S2": 7, "S4": 441}
res = {}
for S in SET:
    s, y = Z[S + "_s"].astype(str), Z[S + "_y"].astype(int); brain = Z[S + "_brain"].astype(float); r = {}
    sb_full = loso(s, y, brain, 0.05); sb_bp = loso(s, y, brain[:, :BPCOLS[S]], 0.05)
    for sp, AB in ABS.items():
        A, B = AB[S + "_A"].astype(float), AB[S + "_B"].astype(float)
        for cal in (0.10, 0.15, 0.20, 0.30):
            r[f"real_{sp}_{cal}"] = rbme(s, y, brain, A, B, sb_full, cal)[0]; print(S, sp, cal, r[f"real_{sp}_{cal}"], f"{time.time()-T0:.0f}s", flush=True)
        r[f"realBP_{sp}"] = rbme(s, y, brain, A, B, sb_bp, 0.15)[0]; print(S, sp, "band-power only", r[f"realBP_{sp}"], flush=True)
    AB = ABS["ALT"]; T = (AB[S + "_A"] + AB[S + "_B"]).astype(float) / 2
    for nz in (0.3, 0.8):
        for kind in ("null", "pos"):
            vals = []
            for sd in range(SEEDS):
                g = np.random.default_rng(sd); X = T @ g.normal(size=(T.shape[1], 20)) / np.sqrt(T.shape[1]) + g.normal(size=(len(T), 20))
                if kind == "pos": X += 0.3 * (y - y.mean())[:, None] * g.normal(size=20)[None, :]
                vals.append(rbme(s, y, X, T + g.normal(scale=nz, size=T.shape), T + g.normal(scale=nz, size=T.shape))[0])
            r[f"{kind}_{nz}"] = dict(diff=float(np.mean([v["diff"] for v in vals])), p=float(np.median([v["p"] for v in vals])))
            print(S, kind, nz, r[f"{kind}_{nz}"], f"{time.time()-T0:.0f}s", flush=True)
    res[S] = r
K1 = all(res[S][f"null_{nz}"]["diff"] <= .02 for S in SET for nz in (0.3, 0.8))
K2 = all(sum(res[S][f"pos_{nz}"]["diff"] > .03 and res[S][f"pos_{nz}"]["p"] < .01 for S in SET) >= 2 for nz in (0.3, 0.8))
m = res.get("S2", {}); keys = [k for k in m if k.startswith("real")]
K3 = bool(keys) and all(m[k]["diff"] > 0 and m[k]["p"] < .01 for k in keys)
print("K1", K1, "| K2", K2, "| K3", K3, "| ROBUST", K1 and K2 and K3)
json.dump(res, open("rbme_rob_results.json", "w"), indent=1)
