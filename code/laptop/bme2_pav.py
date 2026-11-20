"""Paper 8 BME-2 confirmation on Pavlov ds003838 (rules: pilot_BME2_prereg.md, fixed before data arrived)."""
import pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from bme import evaluate                                  # same BME procedure as the discovery pilot
D = pickle.load(open("data/pav/pav_features.pkl", "rb"))
def logm(C): w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
def isqrtm(C): w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T
c19 = [set(d["ch19"]) for d in D.values()]; common = sorted(set.intersection(*c19)) if c19 else []
nch = {d["bp"].shape[2] for d in D.values()}; print("people", len(D), "band-power channel counts", nch, "common 10-20 channels", len(common))
LV = {5: 0, 9: 1, 13: 2}; rows = []; skipped = 0
for sid, d in D.items():
    m = d["meta"]; mem = (m.cond == "memory").values
    if mem.sum() < 60: skipped += 1; continue
    order = np.argsort(m[mem].start.values); idx = np.flatnonzero(mem)[order]; meta = m[mem].iloc[order].reset_index(drop=True)
    beh = d["beh"]; beh = beh[beh.condition.astype(str).str.fullmatch(r"\d+")].copy()
    beh = beh[beh.task.astype(str).str.contains("remember|memor", case=False)].sort_values("trial").reset_index(drop=True)
    if len(beh) != len(meta) or not (beh.condition.astype(int).values == meta.len.values).all():
        print(sid, "behaviour/EEG mismatch", len(beh), len(meta)); skipped += 1; continue
    bp = d["bp"][idx].astype(float); bp = bp.mean(2) if len(nch) > 1 else bp.reshape(len(idx), -1)
    bp = (bp - bp.mean(0)) / (bp.std(0) + 1e-9)
    ci = [d["ch19"].index(c) for c in common]; C = d["cov"][idx][:, ci][:, :, ci].astype(float)
    C += 1e-6 * np.eye(len(ci)) * np.trace(C, axis1=1, axis2=2).mean() / len(ci)
    R = isqrtm(C.mean(0)); iu = np.triu_indices(len(ci)); T = logm(R @ C @ R)[:, iu[0], iu[1]]; T = (T - T.mean(0)) / (T.std(0) + 1e-9)
    L = meta.len.values.astype(float); bh = np.c_[beh.partialScore.values / L, beh.NCorrect.values / L]
    bh = (bh - bh.mean(0)) / (bh.std(0) + 1e-9)
    for k in range(len(idx)): rows.append(dict(s=sid, level=LV[int(L[k])], brain=np.r_[bp[k], T[k]], behz=bh[k]))
df = pd.DataFrame(rows); print("people used", df.s.nunique(), "skipped", skipped, "trials", len(df))
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])
acc = {k: [] for k in ("BEH", "BRAIN", "FUSED")}
for te in df.s.unique():
    tr, ts = (df.s != te).values, (df.s == te).values
    for k, get, Cc in (("BEH", lambda x: np.stack(x.behz.values), 1.0), ("BRAIN", lambda x: np.stack(x.brain.values), 0.05),
                       ("FUSED", lambda x: np.c_[np.stack(x.behz.values), np.stack(x.brain.values)], 0.05)):
        X = get(df[tr]); sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=3000, C=Cc).fit(sc.transform(X), df.level.values[tr])
        acc[k].append(bacc(df.level.values[ts], clf.predict(sc.transform(get(df[ts])))))
A = {k: np.array(v) for k, v in acc.items()}
for k, v in A.items(): print(f"{k:6s} balanced accuracy {v.mean():.3f}")
r = evaluate(df, 3, 0.15, "S4 Pavlov")
gain = (A["FUSED"] - A["BEH"]).mean(); print("FUSED - BEH", round(gain, 3))
