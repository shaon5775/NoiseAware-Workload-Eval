"""Paper 8 pilot BME (rules: pilot_BME_prereg.md): behaviour-matched evaluation on COG-BCI (n-back, MATB) and Tufts."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def scores(df, nlev):
    """LOSO expected-level scores from brain and from behaviour."""
    sb, sh = np.zeros(len(df)), np.zeros(len(df))
    for te in df.s.unique():
        tr, ts = (df.s != te).values, (df.s == te).values
        for col, out, C in (("brain", sb, 0.05), ("behz", sh, 1.0)):
            X = np.stack(df[col].values[tr]); sc = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X), df.level.values[tr])
            out[ts] = clf.predict_proba(sc.transform(np.stack(df[col].values[ts]))) @ clf.classes_
    return sb, sh

def evaluate(df, nlev, cal, name):
    sb, sh = scores(df, nlev); r = []
    for s in df.s.unique():
        m = (df.s == s).values; L, b, h = df.level.values[m], sb[m], sh[m]
        i, j = np.where(L[:, None] > L[None, :])
        auc = lambda x, k: np.mean((x[i[k]] > x[j[k]]) + 0.5 * (x[i[k]] == x[j[k]])) if k.sum() else np.nan
        allk = np.ones(len(i), bool); mk = np.abs(h[i] - h[j]) <= cal
        r.append(dict(s=s, n_match=int(mk.sum()), std=auc(b, allk), match=auc(b, mk), beh_match=auc(h, mk), beh_std=auc(h, allk)))
    R = pd.DataFrame(r); ok = R[R.n_match >= 20]
    infl = ok["std"] - ok["match"]
    p_inf = wilcoxon(infl, alternative="greater").pvalue if len(ok) > 5 else np.nan
    p_gt = wilcoxon(ok["match"] - 0.5, alternative="greater").pvalue if len(ok) > 5 else np.nan
    print(f"{name:14s} people ok {len(ok):2d}/{len(R)} | brain AUC standard {ok['std'].mean():.3f} matched {ok['match'].mean():.3f} "
          f"| inflation {infl.mean():+.3f} p={p_inf:.4f} | behaviour AUC matched {ok['beh_match'].mean():.3f} (standard {ok['beh_std'].mean():.3f}) "
          f"| matched brain > 0.5 p={p_gt:.4f}")
    return dict(E1=len(ok) >= 20, E2=bool(infl.mean() >= 0.05 and p_inf < 0.01), E3=bool(ok["beh_match"].mean() <= 0.58))

if __name__ == "__main__":
    import bvb_cog as B
    res = {}
    for fam, nm in (("nback", "S1 COG n-back"), ("matb", "S2 COG MATB")):
        res[nm] = evaluate(B.build(fam).reset_index(drop=True), 3, 0.15, nm)
    # Tufts (fNIRS blocks, behaviour = block accuracy)
    import pilot_P8A as P
    rows = []
    for s in sorted({x for b in P.buckets() for part in b for x in part}):
        F, Y = P.features(s); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        acc = pd.read_csv(f"data/task_accuracy/sub_{s}.csv").acc.values; acc = (acc - acc.mean()) / (acc.std() + 1e-9)
        for k in range(16): rows.append(dict(s=s, level=int(Y[k * 19]), brain=F[k * 19:(k + 1) * 19].mean(0), behz=np.array([acc[k]])))
    res["S3 Tufts"] = evaluate(pd.DataFrame(rows), 4, 0.2, "S3 Tufts")
    E1 = sum(v["E1"] for v in res.values()) >= 2; E2 = sum(v["E2"] for v in res.values()) >= 2
    E3 = all(v["E3"] for v in res.values() if v["E2"])
    print(res); print("E1", E1, "| E2", E2, "| E3", E3, "| GO", E1 and E2 and E3)
