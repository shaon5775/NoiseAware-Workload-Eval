"""Paper 8 pilot BOD (rules: pilot_BOD_prereg.md) on COG-BCI. Reuses feature building from bvb_cog.py."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import bvb_cog as B

rows = []
for fam in ("nback", "matb"):
    df = B.build(fam); df["fam"] = fam
    # H3-base score: LOSO level classifier expected level (same as P8-BVB-COG)
    score = np.zeros(len(df)); df = df.reset_index(drop=True)
    for te in df.s.unique():
        tr, ts = df.s != te, df.s == te
        X = np.stack(df[tr].brain.values); sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(X), df[tr].level.values)
        score[ts.values] = clf.predict_proba(sc.transform(np.stack(df[ts].brain.values))) @ np.arange(3)
    df["score"] = score
    g = df.groupby(["s", "ses", "task"])
    rec = g.agg(level=("level", "first"), score=("score", "mean"), fam=("fam", "first")).reset_index()
    rec["brain"] = list(g.brain.apply(lambda v: np.stack(v.values).mean(0)).values)
    rec["beh"] = list(g.behz.apply(lambda v: np.stack(v.values).mean(0)).values)
    rec["sess"] = rec.ses.str.extract(r"(\d)").astype(int)[0]
    rec["rsme"] = [B.rsme[(B.rsme.sbj == s) & (B.rsme.Session == n) & (B.rsme.condition == B.RSMEKEY[t])].Score.mean()
                   for s, n, t in zip(rec.s, rec.sess, rec.task)]
    rows.append(rec)
R = pd.concat(rows).dropna(subset=["rsme"]).reset_index(drop=True)
# within-person preparation: z-score brain; behaviour z within person × family; residualise targets on level+family+behaviour
P = {}
for s, g in R.groupby("s"):
    if len(g) < 10: continue
    Xb = np.stack(g.brain.values); Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-9)
    Bh = np.zeros((len(g), 3))
    for fam in ("nback", "matb"):
        m = (g.fam == fam).values; b = np.stack(g[m].beh.values); Bh[m] = (b - b.mean(0)) / (b.std(0) + 1e-9)
    Bh = np.nan_to_num(Bh)
    Z = np.c_[np.eye(6)[(g.level + 3 * (g.fam == "matb")).values], Bh]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    P[s] = dict(X=Xb, B=Bh, y=res(g.rsme.values.astype(float)), h3=res(g.score.values), bcomp=Bh.mean(1))
people = sorted(P); print("people", len(people), "recordings per person", sorted({len(P[p]["y"]) for p in people}))
out = {k: [] for k in ("H3-base", "PLAIN", "BOD")}; orth = []
for te in people:
    tr = [p for p in people if p != te]
    X = np.vstack([P[p]["X"] for p in tr]); Y = np.concatenate([P[p]["y"] for p in tr]); Bt = np.vstack([P[p]["B"] for p in tr])
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    W = Ridge(alpha=10).fit(Xs, Bt).coef_                                  # behaviour directions in brain space (3 × d)
    Q, _ = np.linalg.qr(W.T); proj = lambda A: A - (A @ Q) @ Q.T
    plain = Ridge(alpha=10).fit(Xs, Y); bod = Ridge(alpha=10).fit(proj(Xs), Y)
    Xt = sc.transform(P[te]["X"]); yt = P[te]["y"]
    pb = bod.predict(proj(Xt))
    out["PLAIN"].append(spearmanr(plain.predict(Xt), yt)[0]); out["BOD"].append(spearmanr(pb, yt)[0])
    out["H3-base"].append(spearmanr(P[te]["h3"], yt)[0]); orth.append(abs(spearmanr(pb, P[te]["bcomp"])[0]))
M = {k: np.nan_to_num(np.array(v)) for k, v in out.items()}
for k, v in M.items(): print(f"{k:8s} mean rho {v.mean():+.3f}  p(>0)={wilcoxon(v, alternative='greater').pvalue:.4f}  share>0 {np.mean(v>0):.0%}")
o = float(np.nanmean(orth)); print(f"BOD |rho with behaviour| {o:.3f}")
B1 = M["BOD"].mean() >= 0.15 and wilcoxon(M["BOD"], alternative="greater").pvalue < 0.01
B2 = M["BOD"].mean() >= M["H3-base"].mean() + 0.05
B3 = o <= 0.15 and M["BOD"].mean() >= M["PLAIN"].mean() - 0.02
print("B1", B1, "| B2", B2, "| B3", B3, "| GO", B1 and B2 and B3)
