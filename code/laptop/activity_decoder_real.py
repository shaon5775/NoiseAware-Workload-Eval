"""Real-data demonstration of trap 2 (laptop, light; descriptive, not a pass/fail test).
Decoder input = real MATB task-activity counts per 1-min unit (key presses KB1, scheduled events EVENT, audio messages COMM),
z-scored within person; NO brain data. Same LOSO logistic (C=0.05), same behaviour halves and RB-ME as the paper."""
import json, sys, numpy as np
from scipy.stats import wilcoxon, norm
from serial_loso import loso, resid, auc, bacc
sys.argv = [sys.argv[0]]
import bvb_cog as B

def rbme(s, y, sb, sA, sB, CAL=0.15):
    rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], 0.05); st, br, bd = [], [], []
    for p in np.unique(s):
        m = s == p; L = y[m]; i, j = np.where(L[:, None] > L[None, :]); vals = []
        for mt, ts in ((sA[m], sB[m]), (sB[m], sA[m])):
            k = np.abs(mt[i] - mt[j]) <= CAL
            if k.sum() < 20: vals = None; break
            ab = np.clip(auc(ts, i[k], j[k]), 1e-3, 1 - 1e-3); vals.append((auc(sb[m], i[k], j[k]), norm.cdf(norm.ppf(ab) / np.sqrt(rel))))
        if vals: v = np.mean(vals, 0); br.append(v[0]); bd.append(v[1]); st.append(v[0] - v[1])
    st = np.array(st); return dict(n=len(st), brain=float(np.mean(br)), bound=float(np.mean(bd)), diff=float(st.mean()), p=float(wilcoxon(st, alternative="greater").pvalue))

df = B.build("matb").reset_index(drop=True)
AB = dict(np.load("data/ab_halves.npz", allow_pickle=True)); s, y = AB["S2_s"].astype(str), AB["S2_y"].astype(int)
assert (df.s.astype(str).values == s).all()
act = np.zeros((len(df), 3))
for n, r in enumerate(df.itertuples()):
    for o, d in B.EVT[(r.s, r.ses, r.task)]:
        p = d.split()
        if r.unit * 60 <= o < (r.unit + 1) * 60 and len(p) >= 2 and p[1] in ("KB1", "EVENT", "COMM"): act[n, ("KB1", "EVENT", "COMM").index(p[1])] += 1
for p in np.unique(s):
    m = s == p; act[m] = (act[m] - act[m].mean(0)) / (act[m].std(0) + 1e-9)
sA, sB = loso(s, y, AB["S2_A"].astype(float), 1.0), loso(s, y, AB["S2_B"].astype(float), 1.0)
res = {}
for nm, cols in (("all_activity", [0, 1, 2]), ("key_presses_only", [0]), ("events_only", [1]), ("messages_only", [2])):
    sb = loso(s, y, act[:, cols], 0.05); r = rbme(s, y, sb, sA, sB)
    r["bacc"] = float(np.mean([bacc(y[s == q], np.clip(np.rint(sb[s == q]), 0, 2)) for q in np.unique(s)])); res[nm] = r
    print(nm, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open("activity_decoder_real_results.json", "w"), indent=1)

# Does the EEG decoder score track key presses WITHIN person and level? (descriptive)
import glob
from scipy.stats import spearmanr
Z = {}
for f in sorted(glob.glob("data/ncbme_upload/bme_*.npz")):
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
assert (Z["S2_s"].astype(str) == s).all()
sb_eeg = loso(s, y, Z["S2_brain"].astype(float), 0.05)
rho = [spearmanr(resid(sb_eeg, s, y)[s == p], resid(act[:, 0], s, y)[s == p])[0] for p in np.unique(s)]
rho = np.nan_to_num(np.array(rho)); w = wilcoxon(rho, alternative="greater").pvalue
res["eeg_vs_keypress_within_level"] = dict(mean_rho=float(rho.mean()), share_pos=float(np.mean(rho > 0)), p=float(w), n=len(rho))
print("EEG score vs key presses within person x level:", res["eeg_vs_keypress_within_level"])
json.dump(res, open("activity_decoder_real_results.json", "w"), indent=1)
