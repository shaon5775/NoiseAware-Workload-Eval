"""Paper 8 NC-BME (rules: pilot_NCBME_prereg.md). Self-contained; runs on Kaggle. Input: bme_data.npz (4 settings)."""
import glob, json, sys, time, numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

QUICK = "--quick" in sys.argv
files = sorted(glob.glob("/kaggle/input/**/bme_*.npz", recursive=True)) or sorted(glob.glob("../data/ncbme_upload/bme_*.npz"))
print("input files", files); Z = {}
for f in files:
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
SET = [a for a in sys.argv if a in ("S1","S2","S3","S4")] if QUICK else ["S1", "S2", "S3", "S4"]; CAL = {"S1": .15, "S2": .15, "S3": .2, "S4": .15}
LAMS = [0.0, 0.5, 1.0, 1.5, 2.0]; REPS = 5 if QUICK else 20; SEEDS = 1 if QUICK else 3

def loso_score(s, y, X, C):
    out = np.zeros(len(y))
    for te in np.unique(s):
        tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X[tr]), y[tr])
        out[ts] = clf.predict_proba(sc.transform(X[ts])) @ clf.classes_
    return out

def person_auc(s, y, sb, sh, cal):
    res = {}
    for p in np.unique(s):
        m = s == p; L, b, h = y[m], sb[m], sh[m]
        i, j = np.where(L[:, None] > L[None, :]); mk = np.abs(h[i] - h[j]) <= cal
        res[p] = (int(mk.sum()), np.mean((b[i[mk]] > b[j[mk]]) + 0.5 * (b[i[mk]] == b[j[mk]])) if mk.sum() else np.nan)
    return res

def noise_var(s, y, B):
    """Neighbour reliability per feature -> error variance (conservative)."""
    R = B.copy()
    for p in np.unique(s):
        for l in np.unique(y):
            m = (s == p) & (y == l)
            if m.any(): R[m] -= R[m].mean(0)
    a, b = [], []
    for p in np.unique(s):
        for l in np.unique(y):
            idx = np.flatnonzero((s == p) & (y == l))
            if len(idx) > 1: a.append(R[idx[:-1]]); b.append(R[idx[1:]])
    a, b = np.vstack(a), np.vstack(b)
    rho = np.array([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(B.shape[1])])
    return R.var(0) * (1 - np.clip(np.nan_to_num(rho), 0, 1)), rho

def ncbme(s, y, brain, B, cal, seed=0):
    sb = loso_score(s, y, brain, 0.05)
    base = person_auc(s, y, sb, loso_score(s, y, B, 1.0), cal)
    keep = [p for p, (n, a) in base.items() if n >= 20]
    ev, rho = noise_var(s, y, B)
    def one(lam, r):
        rng = np.random.default_rng(seed * 100000 + int(lam * 10) * 1000 + r)
        Bn = B + rng.normal(size=B.shape) * np.sqrt(lam * ev)
        pa = person_auc(s, y, sb, loso_score(s, y, Bn, 1.0), cal); return lam, np.array([pa[p][1] for p in keep])
    jobs = [(l, r) for l in LAMS[1:] for r in range(REPS)]
    out = Parallel(n_jobs=-1)(delayed(one)(l, r) for l, r in jobs)
    curve = {0.0: np.array([base[p][1] for p in keep])}
    for l in LAMS[1:]: curve[l] = np.nanmean(np.stack([v for lam, v in out if lam == l]), 0)
    M = np.stack([curve[l] for l in LAMS])                     # λ × people
    lam = np.array(LAMS); corr = np.array([np.polyval(np.polyfit(lam, M[:, k], 2), -1.0) for k in range(M.shape[1])])
    grp = np.polyval(np.polyfit(lam, np.nanmean(M, 1), 2), -1.0)
    p = wilcoxon(corr - 0.5, alternative="greater").pvalue if len(keep) > 5 else np.nan
    return dict(n_people=len(keep), rho=rho.tolist(), curve=np.nanmean(M, 1).tolist(), corrected_group=float(grp),
                corrected_mean=float(np.nanmean(corr)), p=float(p))

def synth(y, B, kind, seed):
    rng = np.random.default_rng(seed)
    X = B @ rng.normal(size=(B.shape[1], 20)) / np.sqrt(B.shape[1]) + rng.normal(size=(len(B), 20))
    if kind == "pos": X += 0.3 * (y - y.mean())[:, None] * rng.normal(size=20)[None, :]
    return X, B + rng.normal(scale=0.5, size=B.shape)            # brain from TRUE behaviour; matching on noisy behaviour

res = {}; t0 = time.time()
for S in SET:
    s, y = Z[S + "_s"].astype(str), Z[S + "_y"].astype(int); brain, B = Z[S + "_brain"].astype(float), Z[S + "_beh"].astype(float)
    r = {"real": ncbme(s, y, brain, B, CAL[S])}
    print(S, "REAL", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["real"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    for kind in ("err", "pos"):
        vals = [ncbme(s, y, *synth(y, B, kind, sd), CAL[S], seed=sd + 1) for sd in range(SEEDS)]
        r["A-" + kind] = dict(uncorrected=float(np.mean([v["curve"][0] for v in vals])),
                              corrected=float(np.mean([v["corrected_group"] for v in vals])))
        print(S, "A-" + kind, r["A-" + kind], f"{time.time()-t0:.0f}s", flush=True)
    res[S] = r

V1 = sum(abs(res[S]["A-err"]["corrected"] - .5) <= .03 for S in SET) >= 3
V2 = all(res[S]["A-pos"]["corrected"] > .60 for S in SET)
ok = {S: res[S]["real"]["corrected_group"] > .5 and res[S]["real"]["p"] < .01 for S in SET}
R1 = sum(ok.values()) >= 3 and ok.get("S4", False)
print("V1", V1, "| V2", V2, "| METHOD GO", V1 and V2, "| R1", R1, ok, "| FINDING GO", R1)
json.dump(res, open("ncbme_results.json", "w"), indent=1)
