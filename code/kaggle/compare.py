"""Paper 8 (A) comparison of confound fixes + (B) session-wise MATB replication. Rules: pilot_COMPARE_prereg.md."""
import glob, json, sys, time, numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
T0 = time.time(); QUICK = "--quick" in sys.argv
g_ = lambda pat, loc: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or glob.glob(loc))
Z = {}
for f in sorted(g_("bme_*.npz", "../data/ncbme_upload/bme_*.npz")):
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
AB = dict(np.load(g_("ab_halves.npz", "../data/ab_halves.npz")[0], allow_pickle=True))
SES = dict(np.load(g_("ses_info.npz", "../data/ses_info.npz")[0], allow_pickle=True))
s4 = AB["S4_s"].astype(str); ppl = sorted(set(s4))
o = np.r_[np.flatnonzero(np.isin(s4, ppl[:33])), np.flatnonzero(np.isin(s4, ppl[33:]))]
for f in ("s", "y", "A", "B"): AB["S4_" + f] = AB["S4_" + f][o]
SET = [a for a in sys.argv if a in ("S1", "S2", "S4")] if QUICK else ["S1", "S2", "S4"]
NSEED = 2 if QUICK else 20; CAL = 0.15
for S in SET: assert (Z[S + "_s"].astype(str) == AB[S + "_s"].astype(str)).all()
for S in ("S1", "S2"): assert (SES[S + "_s"] == Z[S + "_s"].astype(str)).all()

def loso(s, y, X, C, par=True):
    def fold(te):
        tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X[tr]), y[tr]); return ts, clf.predict_proba(sc.transform(X[ts])) @ clf.classes_
    out = np.zeros(len(y)); R = Parallel(n_jobs=-1)(delayed(fold)(te) for te in np.unique(s)) if par else [fold(te) for te in np.unique(s)]
    for ts, v in R: out[ts] = v
    return out
def auc(x, i, j): return np.mean((x[i] > x[j]) + 0.5 * (x[i] == x[j]))
def pairs(L): return np.where(L[:, None] > L[None, :])
def wil(d): return wilcoxon(d, alternative="greater").pvalue if len(d) > 5 and np.any(d != 0) else 1.0
def resid(x, s, y):
    r = x.copy()
    for p in np.unique(s):
        for l in np.unique(y):
            m = (s == p) & (y == l); r[m] -= r[m].mean() if m.any() else 0
    return r

def m1(s, y, sb):
    v = [auc(sb[s == p], *pairs(y[s == p])) - .5 for p in np.unique(s)]; return wil(np.array(v))
def matched(s, y, sb, sh, minp=20):
    v = []
    for p in np.unique(s):
        m = s == p; i, j = pairs(y[m]); k = np.abs(sh[m][i] - sh[m][j]) <= CAL
        if k.sum() >= minp: v.append(auc(sb[m], i[k], j[k]) - .5)
    return np.array(v)
def m2(s, y, sb, sh): return wil(matched(s, y, sb, sh))
def m3(s, y, brain, Bfull):
    sb = np.zeros(len(y))
    for te in np.unique(s):
        tr, ts = s != te, s == te; lr = LinearRegression().fit(Bfull[tr], brain[tr])
        Rtr, Rts = brain[tr] - lr.predict(Bfull[tr]), brain[ts] - lr.predict(Bfull[ts]); sc = StandardScaler().fit(Rtr)
        clf = LogisticRegression(max_iter=3000, C=.05).fit(sc.transform(Rtr), y[tr]); sb[ts] = clf.predict_proba(sc.transform(Rts)) @ clf.classes_
    return m1(s, y, sb)
def m4(s, y, sb, Bfull):
    ev =resid(Bfull, s, y).var(0); lam = [0, .5, 1, 1.5, 2]; reps = 5; curves = []
    for l in lam:
        acc = []
        for r in (range(reps) if l > 0 else [0]):
            g = np.random.default_rng(1000 + r); sh = loso(s, y, Bfull + g.normal(size=Bfull.shape) * np.sqrt(l * ev), 1.0, par=False)
            v = {}
            for p in np.unique(s):
                m = s == p; i, j = pairs(y[m]); k = np.abs(sh[m][i] - sh[m][j]) <= CAL
                v[p] = auc(sb[m], i[k], j[k]) if k.sum() else np.nan
            acc.append(v)
        curves.append({p: np.nanmean([a[p] for a in acc]) for p in np.unique(s)})
    corr = np.array([np.polyval(np.polyfit(lam, [c[p] for c in curves], 2), -1) for p in np.unique(s)])
    return wil(corr[np.isfinite(corr)] - .5)
def m5(s, y, sb, A, B, minp=20):
    sA, sB = loso(s, y, A, 1.0, par=False), loso(s, y, B, 1.0, par=False)
    rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], .05); st = []
    for p in np.unique(s):
        m = s == p; i, j = pairs(y[m]); vals = []
        for mt, ts in ((sA[m], sB[m]), (sB[m], sA[m])):
            k = np.abs(mt[i] - mt[j]) <= CAL
            if k.sum() < minp: vals = None; break
            ab = np.clip(auc(ts, i[k], j[k]), 1e-3, 1 - 1e-3); vals.append(auc(sb[m], i[k], j[k]) - norm.cdf(norm.ppf(ab) / np.sqrt(rel)))
        if vals: st.append(np.mean(vals))
    st = np.array(st); return wil(st), (float(st.mean()) if len(st) else np.nan), len(st)

def all_methods(s, y, brain, A, B):
    Bf = (A + B) / 2; sb = loso(s, y, brain, .05, par=False); sh = loso(s, y, Bf, 1.0, par=False)
    return dict(M1=m1(s, y, sb), M2=m2(s, y, sb, sh), M3=m3(s, y, brain, Bf), M4=m4(s, y, sb, Bf), M5=m5(s, y, sb, A, B)[0])

def one_seed(S, kind, sd):
    s, y = Z[S + "_s"].astype(str), Z[S + "_y"].astype(int); T = (AB[S + "_A"] + AB[S + "_B"]).astype(float) / 2
    g = np.random.default_rng(sd); F = T if kind != "nl" else np.c_[T, T ** 2, np.tanh(2 * T)]
    F = (F - F.mean(0)) / (F.std(0) + 1e-9); X = F @ g.normal(size=(F.shape[1], 20)) / np.sqrt(F.shape[1]) + g.normal(size=(len(T), 20))
    if kind == "pos": X += .3 * (y - y.mean())[:, None] * g.normal(size=20)[None, :]
    return all_methods(s, y, X, T + g.normal(scale=.5, size=T.shape), T + g.normal(scale=.5, size=T.shape))

res = {"A": {}, "B": {}}
for S in SET:
    for kind in ("lin", "nl", "pos"):
        P = Parallel(n_jobs=-1)(delayed(one_seed)(S, kind, sd) for sd in range(NSEED))
        rate = {m: float(np.mean([p[m] < .01 for p in P])) for m in P[0]}
        res["A"][f"{S}_{kind}"] = rate; print(S, kind, "claim rate (p<0.01)", rate, f"{time.time()-T0:.0f}s", flush=True)
# (B) MATB per session
s, y = Z["S2_s"].astype(str), Z["S2_y"].astype(int); brain = Z["S2_brain"].astype(float); A, B = AB["S2_A"].astype(float), AB["S2_B"].astype(float)
for ses in sorted(set(SES["S2_ses"])):
    m = SES["S2_ses"] == ses; sb = loso(s[m], y[m], brain[m], .05)
    p, d, n = m5(s[m], y[m], sb, A[m], B[m], minp=10); res["B"][ses] = dict(p=float(p), diff=d, n=n)
    print("MATB", ses, res["B"][ses], flush=True)
SETs = SET
A1 = all(res["A"][f"{S}_{k}"]["M5"] <= .10 for S in SETs for k in ("lin", "nl"))
A2 = sum(res["A"][f"{S}_pos"]["M5"] >= .80 for S in SETs) >= 2
A3 = all(any(res["A"][f"{S}_{k}"][m] > .10 for S in SETs for k in ("lin", "nl")) for m in ("M1", "M2", "M3", "M4"))
B = list(res["B"].values()); B1 = sum(b["diff"] > 0 and b["p"] < .01 for b in B) >= 2 and all(b["diff"] > 0 for b in B)
print("A1", A1, "| A2", A2, "| A3", A3, "| B1", B1, "| PASS", A1 and A2 and A3 and B1)
json.dump(res, open("compare_results.json", "w"), indent=1)
