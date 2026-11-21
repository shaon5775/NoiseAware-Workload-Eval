"""Per-person RB-ME values for Fig. 5 (same code as kaggle/rbme.py, real data only). Light: ~2 min."""
import sys, numpy as np, json
sys.argv = ["x"]; src = open("kaggle/rbme.py", encoding="utf8").read()
src = src.replace('F = sorted(glob.glob("/kaggle/input/**/bme_*.npz", recursive=True)) or sorted(glob.glob("../data/ncbme_upload/bme_*.npz"))',
                  'F = sorted(glob.glob("data/ncbme_upload/bme_*.npz"))').replace('or glob.glob("../data/ab_halves.npz"))[0]', 'or glob.glob("data/ab_halves.npz"))[0]')
exec(src[:src.index("res = {}\nfor S in SET:")])
out = {}
for S in ["S1", "S2", "S4"]:
    s, y = Z[S + "_s"].astype(str), Z[S + "_y"].astype(int); brain = Z[S + "_brain"].astype(float); A, B = AB[S + "_A"].astype(float), AB[S + "_B"].astype(float)
    sb = loso(s, y, brain, 0.05); sA, sB = loso(s, y, A, 1.0), loso(s, y, B, 1.0)
    rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], 0.05); rows = []
    for p in np.unique(s):
        m = s == p; L = y[m]; i, j = np.where(L[:, None] > L[None, :]); vals = []
        for mt, tst in ((sA[m], sB[m]), (sB[m], sA[m])):
            k = np.abs(mt[i] - mt[j]) <= CAL
            if k.sum() < 20: vals = None; break
            ab = np.clip(auc(tst, i[k], j[k]), 1e-3, 1 - 1e-3); vals.append((auc(sb[m], i[k], j[k]), norm.cdf(norm.ppf(ab) / np.sqrt(rel))))
        if vals: v = np.mean(vals, 0); rows.append([float(v[0]), float(v[1])])
    out[S] = rows; R = np.array(rows); print(S, len(rows), "brain", R[:, 0].mean().round(3), "bound", R[:, 1].mean().round(3), "diff", (R[:, 0] - R[:, 1]).mean().round(3))
json.dump(out, open("figs/per_person_rbme.json", "w"), indent=1)
