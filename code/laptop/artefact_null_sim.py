"""ARTEFACT-NULL simulation (rules: pilot_ARTEFACT2_prereg.md, part 2). Laptop, light.
Fake brain reads true performance T and real MATB task-activity counts (key presses, events, messages), no load signal of
its own. Does RB-ME credit it? Control: plain NULL (T only)."""
import json, sys, numpy as np
from scipy.stats import wilcoxon, norm
from serial_loso import loso, resid, auc
sys.argv = [sys.argv[0]]
import bvb_cog as B

def rbme(s, y, sb, sA, sB, CAL=0.15):
    rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], 0.05); st = []
    for p in np.unique(s):
        m = s == p; L = y[m]; i, j = np.where(L[:, None] > L[None, :]); vals = []
        for mt, ts in ((sA[m], sB[m]), (sB[m], sA[m])):
            k = np.abs(mt[i] - mt[j]) <= CAL
            if k.sum() < 20: vals = None; break
            ab = np.clip(auc(ts, i[k], j[k]), 1e-3, 1 - 1e-3); vals.append(auc(sb[m], i[k], j[k]) - norm.cdf(norm.ppf(ab) / np.sqrt(rel)))
        if vals: st.append(np.mean(vals))
    st = np.array(st); return float(st.mean()), float(wilcoxon(st, alternative="greater").pvalue) if len(st) > 5 else np.nan, len(st)

def counts_unit(s, ses, task, k, unit_s=60.0):
    c = np.zeros(3)
    for o, d in B.EVT[(s, ses, task)]:
        p = d.split()
        if k * unit_s <= o < (k + 1) * unit_s and len(p) >= 2 and p[1] in ("KB1", "EVENT", "COMM"): c[("KB1", "EVENT", "COMM").index(p[1])] += 1
    return c

df = B.build("matb").reset_index(drop=True)
AB = dict(np.load("data/ab_halves.npz", allow_pickle=True)); s, y = AB["S2_s"].astype(str), AB["S2_y"].astype(int)
assert (df.s.astype(str).values == s).all()
act = np.stack([counts_unit(r.s, r.ses, r.task, r.unit) for r in df.itertuples()])
for p in np.unique(s):
    m = s == p; act[m] = (act[m] - act[m].mean(0)) / (act[m].std(0) + 1e-9)
T = (AB["S2_A"] + AB["S2_B"]).astype(float) / 2; res = {}
for kind in ("null", "artefact_null"):
    rows = []
    for sd in range(20):
        g = np.random.default_rng(sd); Z = T if kind == "null" else np.c_[T, act]
        X = Z @ g.normal(size=(Z.shape[1], 20)) / np.sqrt(Z.shape[1]) + g.normal(size=(len(T), 20))
        sb = loso(s, y, X, 0.05)
        sA, sB = loso(s, y, T + g.normal(scale=0.5, size=T.shape), 1.0), loso(s, y, T + g.normal(scale=0.5, size=T.shape), 1.0)
        rows.append(rbme(s, y, sb, sA, sB))
    d = np.array([r[0] for r in rows]); pv = np.array([r[1] for r in rows])
    res[kind] = dict(mean_diff=float(d.mean()), false_claim_rate=float(np.mean(pv < 0.01)), median_p=float(np.median(pv)), n_people=int(np.median([r[2] for r in rows])))
    print(kind, res[kind], flush=True)
res["S1"] = res["artefact_null"]["false_claim_rate"] >= 0.5; res["S2"] = res["null"]["false_claim_rate"] <= 0.10
print("S1", res["S1"], "| S2", res["S2"]); json.dump(res, open("artefact_null_results.json", "w"), indent=1)
