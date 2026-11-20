"""Reviewer fixes M4/M5/M7 (laptop, light).
M4: split-half reliability of the real behaviour halves per dataset (alternating and contiguous splits).
M5: split-half reliability of simulated behaviour halves at noise SD 0.3/0.5/0.8 (T = mean of the two real halves).
M7: MATB logistic balanced accuracy with classes from argmax vs from the rounded expected score (the EEGNet rule)."""
import glob, json, numpy as np
from serial_loso import loso, resid, bacc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def order_s4(AB):
    s4 = AB["S4_s"].astype(str); ppl = sorted(set(s4)); o = np.r_[np.flatnonzero(np.isin(s4, ppl[:33])), np.flatnonzero(np.isin(s4, ppl[33:]))]
    for f in ("s", "y", "A", "B"): AB["S4_" + f] = AB["S4_" + f][o]
    return AB
ALT = order_s4(dict(np.load("data/ab_halves.npz", allow_pickle=True))); BLK = order_s4(dict(np.load("data/ab_halves_block.npz", allow_pickle=True)))
def rel(s, y, A, B): return round(float(np.corrcoef(resid(loso(s, y, A, 1.0), s, y), resid(loso(s, y, B, 1.0), s, y))[0, 1]), 3)
out = {"real": {}, "sim": {}}
for S in ("S1", "S2", "S4"):
    s, y = ALT[S + "_s"].astype(str), ALT[S + "_y"].astype(int)
    out["real"][S] = dict(alternating=rel(s, y, ALT[S + "_A"].astype(float), ALT[S + "_B"].astype(float)),
                          contiguous=rel(s, y, BLK[S + "_A"].astype(float), BLK[S + "_B"].astype(float)))
    T = (ALT[S + "_A"] + ALT[S + "_B"]).astype(float) / 2; r = {}
    for nz in (0.3, 0.5, 0.8):
        rng = np.random.default_rng(0); r[nz] = rel(s, y, T + rng.normal(scale=nz, size=T.shape), T + rng.normal(scale=nz, size=T.shape))
    out["sim"][S] = r; print(S, out["real"][S], "sim", r, flush=True)
Z = {}
for f in sorted(glob.glob("data/ncbme_upload/bme_*.npz")):
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
s, y, X = Z["S2_s"].astype(str), Z["S2_y"].astype(int), Z["S2_brain"].astype(float); am, rd = [], []
for te in np.unique(s):
    tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=3000, C=0.05).fit(sc.transform(X[tr]), y[tr])
    P = c.predict_proba(sc.transform(X[ts])); am.append(bacc(y[ts], P.argmax(1))); rd.append(bacc(y[ts], np.clip(np.rint(P @ np.arange(3)), 0, 2)))
out["matb_logistic_bacc_argmax"] = float(np.mean(am)); out["matb_logistic_bacc_rounded"] = float(np.mean(rd))
print("MATB logistic bacc argmax", round(out["matb_logistic_bacc_argmax"], 3), "rounded expected score", round(out["matb_logistic_bacc_rounded"], 3))
json.dump(out, open("rel_and_rounding_results.json", "w"), indent=1)
