"""Paper 8 pilot PCD (rules: pilot_PCD_prereg.md)."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fitpred(Xtr, ytr, Xte, C):
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(Xtr), ytr)
    return clf, sc, clf.predict_proba(sc.transform(Xte)) @ clf.classes_

def run(df, cal, name, rng=np.random.RandomState(0)):
    rows = []
    for te in df.s.unique():
        tr, ts = (df.s != te).values, (df.s == te).values
        XB, XH, L = np.stack(df.brain.values), np.stack(df.behz.values), df.level.values
        _, _, base = fitpred(XB[tr], L[tr], XB[ts], 0.05)                       # baseline brain score (test)
        clf_h, sch, h_te = fitpred(XH[tr], L[tr], XH[ts], 1.0)                  # behaviour score (test)
        h_tr = clf_h.predict_proba(sch.transform(XH[tr])) @ clf_h.classes_     # behaviour score (train, in-sample)
        sc = StandardScaler().fit(XB[tr]); Z = sc.transform(XB)
        D, Yd = [], []
        Str = df.s.values[tr]; Ztr, Ltr = Z[tr], L[tr]
        for s in np.unique(Str):
            m = np.flatnonzero(Str == s); i, j = np.where(Ltr[m][:, None] > Ltr[m][None, :])
            k = np.abs(h_tr[m][i] - h_tr[m][j]) <= cal; i, j = m[i[k]], m[j[k]]
            if len(i) > 400: sel = rng.choice(len(i), 400, replace=False); i, j = i[sel], j[sel]
            D.append(Ztr[i] - Ztr[j]); D.append(Ztr[j] - Ztr[i]); Yd += [1] * len(i) + [0] * len(i)
        D = np.vstack(D); pcd = LogisticRegression(max_iter=3000, C=0.05, fit_intercept=False).fit(D, np.array(Yd))
        p = Z[ts] @ pcd.coef_[0]
        Lt = L[ts]; i, j = np.where(Lt[:, None] > Lt[None, :]); mk = np.abs(h_te[i] - h_te[j]) <= cal
        auc = lambda x, k: np.mean((x[i[k]] > x[j[k]]) + 0.5 * (x[i[k]] == x[j[k]])) if k.sum() else np.nan
        al = np.ones(len(i), bool)
        rows.append(dict(s=te, n=int(mk.sum()), base_std=auc(base, al), base_m=auc(base, mk), pcd_std=auc(p, al), pcd_m=auc(p, mk)))
    R = pd.DataFrame(rows); ok = R[R.n >= 20]
    g = ok.pcd_m - ok.base_m
    p_gain = wilcoxon(g).pvalue if len(ok) > 5 else np.nan; p_gt = wilcoxon(ok.pcd_m - 0.5, alternative="greater").pvalue if len(ok) > 5 else np.nan
    print(f"{name:14s} people {len(ok)} | matched AUC: base {ok.base_m.mean():.3f} PCD {ok.pcd_m.mean():.3f} (gain {g.mean():+.3f}, p={p_gain:.4f}, share {np.mean(g>0):.0%}) "
          f"| standard AUC: base {ok.base_std.mean():.3f} PCD {ok.pcd_std.mean():.3f} | PCD matched>0.5 p={p_gt:.4f}")
    q = len(ok) >= 20 or name.startswith("S3")
    return dict(q=q, P1=g.mean() >= 0.03, P2=p_gt < 0.01, P3=ok.pcd_std.mean() >= ok.base_std.mean() - 0.03)

if __name__ == "__main__":
    import bvb_cog as B, pilot_P8A as P
    res = {}
    for fam, nm in (("nback", "S1 COG n-back"), ("matb", "S2 COG MATB")):
        res[nm] = run(B.build(fam).reset_index(drop=True), 0.15, nm)
    rows = []
    for s in sorted({x for b in P.buckets() for part in b for x in part}):
        F, Y = P.features(s); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        acc = pd.read_csv(f"data/task_accuracy/sub_{s}.csv").acc.values; acc = (acc - acc.mean()) / (acc.std() + 1e-9)
        for k in range(16): rows.append(dict(s=s, level=int(Y[k * 19]), brain=F[k * 19:(k + 1) * 19].mean(0), behz=np.array([acc[k]])))
    res["S3 Tufts"] = run(pd.DataFrame(rows), 0.2, "S3 Tufts")
    Q = [k for k, v in res.items() if v["q"]]
    P1 = sum(res[k]["P1"] for k in Q) >= 2; win = [k for k in Q if res[k]["P1"]]
    P2 = all(res[k]["P2"] for k in win) if win else False; P3 = all(res[k]["P3"] for k in win) if win else False
    print(res); print("qualifying", Q, "| P1", P1, "| P2", P2, "| P3", P3, "| GO", P1 and P2 and P3)
