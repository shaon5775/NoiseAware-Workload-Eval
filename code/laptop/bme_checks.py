"""Paper 8 BME robustness checks (rules: pilot_BMECHK_prereg.md)."""
import pickle, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from bme import scores

DF = pickle.load(open("data/bme_dfs.pkl", "rb"))
NLEV = {"S1": 3, "S2": 3, "S3": 4, "S4": 3}; CAL = {"S1": .15, "S2": .15, "S3": .2, "S4": .15}

def pairs(df, sb, sh, cal, lev=None):
    lev = df.level.values if lev is None else lev; out = []
    for s in df.s.unique():
        m = (df.s == s).values; L, b, h = lev[m], sb[m], sh[m]
        i, j = np.where(L[:, None] > L[None, :]); mk = np.abs(h[i] - h[j]) <= cal
        if mk.sum() == 0: out.append((s, 0, np.nan, np.nan)); continue
        a = lambda x, k: np.mean((x[i[k]] > x[j[k]]) + 0.5 * (x[i[k]] == x[j[k]]))
        out.append((s, int(mk.sum()), a(b, mk), a(b, np.ones(len(i), bool))))
    return pd.DataFrame(out, columns=["s", "n", "match", "std"])

def bme(df, cal, sb=None, sh=None):
    if sb is None: sb, sh = scores(df, 0)
    R = pairs(df, sb, sh, cal); ok = R[R.n >= 20]
    p = wilcoxon(ok.match - 0.5, alternative="greater").pvalue if len(ok) > 5 else np.nan
    return dict(n_ok=len(ok), match=ok.match.mean(), std=ok["std"].mean(), p=p), sb, sh

def synth(df, kind, seed):
    rng = np.random.default_rng(seed); B = np.stack(df.behz.values); d = df.copy()
    if kind == "nl":
        k = B.shape[1]; prods = [B[:, a] * B[:, b] for a in range(k) for b in range(a + 1, k)]
        Z = np.column_stack([B, B ** 2, np.tanh(2 * B)] + [p[:, None] for p in prods])
    else: Z = B
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    X = Z @ rng.normal(size=(Z.shape[1], 20)) / np.sqrt(Z.shape[1]) + rng.normal(size=(len(B), 20))
    if kind == "pos": X += 0.3 * (df.level.values - df.level.mean())[:, None] * rng.normal(size=20)[None, :]
    d["brain"] = list(X)
    if kind == "err": d["behz"] = list(B + rng.normal(scale=0.5, size=B.shape))     # reliability 1/1.25 = 0.8
    return d

def cvcr(df):
    """Cross-validated confound regression, then within-person AUC on all pairs."""
    sb = np.zeros(len(df)); Xb, Bh = np.stack(df.brain.values), np.stack(df.behz.values)
    for te in df.s.unique():
        tr, ts = (df.s != te).values, (df.s == te).values
        lr = LinearRegression().fit(Bh[tr], Xb[tr]); Rtr, Rts = Xb[tr] - lr.predict(Bh[tr]), Xb[ts] - lr.predict(Bh[ts])
        sc = StandardScaler().fit(Rtr); clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(Rtr), df.level.values[tr])
        sb[ts] = clf.predict_proba(sc.transform(Rts)) @ clf.classes_
    R = pairs(df, sb, sb, 1e9); return R["std"].mean()

res = {}
for S, df in DF.items():
    r = {}; real, sb, sh = bme(df, CAL[S]); r["real"] = real
    print(S, "real", {k: round(v, 4) for k, v in real.items()}, flush=True)
    # B1 calipers
    r["cal"] = {c: bme(df, c, sb, sh)[0] for c in (0.05, 0.10, 0.15, 0.20, 0.30)}
    for c, v in r["cal"].items(): print(f"  caliper {c}: ok {v['n_ok']} matched {v['match']:.3f} p {v['p']:.2g}", flush=True)
    # B2 permutation (levels shuffled within person, scores fixed)
    rng = np.random.default_rng(0); lev = df.level.values; null = []
    for _ in range(1000):
        pl = lev.copy()
        for s in df.s.unique():
            m = np.flatnonzero((df.s == s).values); pl[m] = rng.permutation(lev[m])
        R = pairs(df, sb, sh, CAL[S], pl); null.append(R[R.n >= 20].match.mean())
    r["perm99"] = float(np.nanpercentile(null, 99)); print(f"  permutation 99th pct {r['perm99']:.3f}", flush=True)
    # A nulls
    for kind in ("lin", "nl", "err", "pos"):
        vals = [bme(synth(df, kind, sd), CAL[S])[0] for sd in range(3)]
        r["A-" + kind] = dict(match=float(np.mean([v["match"] for v in vals])), std=float(np.mean([v["std"] for v in vals])))
        print(f"  A-{kind}: standard {r['A-'+kind]['std']:.3f} matched {r['A-'+kind]['match']:.3f}", flush=True)
    # C CVCR
    r["cvcr_real"] = cvcr(df); r["cvcr_lin"] = np.mean([cvcr(synth(df, "lin", sd)) for sd in range(3)])
    r["cvcr_nl"] = np.mean([cvcr(synth(df, "nl", sd)) for sd in range(3)])
    print(f"  CVCR real {r['cvcr_real']:.3f} | A-lin {r['cvcr_lin']:.3f} | A-nl {r['cvcr_nl']:.3f}", flush=True)
    res[S] = r

A1 = all(abs(r["A-lin"]["match"] - .5) <= .02 for r in res.values())
A2 = all(r["A-err"]["match"] - .5 < 0.5 * (r["real"]["match"] - .5) for r in res.values())
B1 = all(v["match"] > .5 and v["p"] < .01 for S, r in res.items() for v in r["cal"].values() if v["n_ok"] >= 8) and \
     all(v["n_ok"] >= 8 for S in ("S1", "S2", "S4") for v in res[S]["cal"].values())
B2 = all(r["real"]["match"] > r["perm99"] for r in res.values())
print("A1", A1, "| A2", A2, "| B1", B1, "| B2", B2, "| PASS", A1 and A2 and B1 and B2)
json.dump(res, open("bme_checks_results.json", "w"), indent=1, default=float)
