"""Paper 8 pilot BVB on Pavlov ds003838 (rules: pilot_BVB_prereg.md). Unit = one memory trial (5 / 9 / 13 digits).
Behaviour = recall score of that trial (partialScore / length, NCorrect / length). Brain = log band power (7 bands × ch)
+ per-person re-centred covariance tangent features of the last 10 s of encoding. LOSO logistic regression."""
import pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
D = pickle.load(open("data/pav/pav_features.pkl", "rb"))
def logm(C): w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
def isqrtm(C): w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T
LV = {5: 0, 9: 1, 13: 2}; P = {}
for sid, d in D.items():
    m = d["meta"]; mem = (m.cond == "memory").values
    if mem.sum() < 60: continue
    meta = m[mem].sort_values("start").reset_index(drop=True); idx = np.flatnonzero(mem)[np.argsort(m[mem].start.values)]
    beh = d["beh"]; beh = beh[beh.task.astype(str).str.contains("remember|memory", case=False)].sort_values("trial").reset_index(drop=True)
    if len(beh) != len(meta) or not (beh.condition.astype(int).values == meta.len.values).all():
        print(sid, "behaviour/EEG trial mismatch", len(beh), len(meta)); continue
    B = d["bp"][idx].reshape(len(idx), -1).astype(float); B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    C = d["cov"][idx].astype(float); C += 1e-6 * np.eye(C.shape[1]) * np.trace(C, axis1=1, axis2=2).mean() / C.shape[1]
    R = isqrtm(C.mean(0)); iu = np.triu_indices(C.shape[1]); T = logm(R @ C @ R)[:, iu[0], iu[1]]; T = (T - T.mean(0)) / (T.std(0) + 1e-9)
    L = meta.len.values.astype(float)
    bh = np.c_[beh.partialScore.values / L, beh.NCorrect.values / L]; bh = (bh - bh.mean(0)) / (bh.std(0) + 1e-9)
    P[sid] = dict(brain=np.c_[B, T], beh=bh, y=np.array([LV[int(x)] for x in L]))
print("people used:", len(P))
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
res = {k: [] for k in ("BEH", "BRAIN", "FUSED")}
people = sorted(P)
for te in people:
    tr = [p for p in people if p != te]
    for k, get, C_ in (("BEH", lambda d: d["beh"], 1.0), ("BRAIN", lambda d: d["brain"], 0.05), ("FUSED", lambda d: np.c_[d["beh"], d["brain"]], 0.05)):
        X = np.vstack([get(P[p]) for p in tr]); Y = np.concatenate([P[p]["y"] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=3000, C=C_).fit(sc.transform(X), Y)
        res[k].append(bacc(P[te]["y"], clf.predict(sc.transform(get(P[te])))))
M = {k: np.array(v) for k, v in res.items()}
for k, v in M.items(): print(f"{k:6s} {v.mean():.3f}")
print("H1 BEH >= BRAIN-0.02:", M["BEH"].mean() >= M["BRAIN"].mean() - 0.02,
      "| H2 FUSED-BEH < 0.03:", (M["FUSED"] - M["BEH"]).mean() < 0.03,
      "| FUSED vs BEH Wilcoxon p:", round(wilcoxon(M["FUSED"], M["BEH"]).pvalue, 4), "| share FUSED>BEH", round(float(np.mean(M["FUSED"] > M["BEH"])), 2))
