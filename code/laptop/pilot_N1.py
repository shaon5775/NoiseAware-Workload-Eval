"""Paper 8 pilot N1 'felt vs set workload'. Rules: pilot_N1_prereg.md (fixed before running)."""
import os, json, numpy as np, pandas as pd, pilot_P8A as P
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

def ratings(sub):
    d = f"data/nasa_tlx/sub_{sub}"; out = {}
    for f in os.listdir(d):
        for pre, off in (("train_1_1_", 0), ("test_1_2_", 12)):
            if pre in f:
                k = int(f.split(pre)[1].split(".")[0]) + off
                r = pd.read_csv(f"{d}/{f}").iloc[0]
                out[k - 1] = dict(mental=float(r["mental"]), effort=float(r["effort"]),
                                  frustration=float(r["frustration"]), performance=float(r["performance"]))
    acc = pd.read_csv(f"data/task_accuracy/sub_{sub}.csv")
    acc = {int(t.split("_")[1]) - 1: a for t, a in zip(acc.task_name, acc.acc)}
    return out, acc

subs = sorted({s for b in P.buckets() for part in b for s in part})
W, R = {}, {}
for s in subs:
    F, Y = P.features(s); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    blk = np.repeat(np.arange(16), 19); assert len(Y) == 304
    rt, acc = ratings(s)
    lv = Y[::19]
    felt = np.array([rt[b]["mental"] if b in rt else np.nan for b in range(16)])
    felt2 = np.array([np.mean([rt[b]["mental"], rt[b]["effort"], rt[b]["frustration"], 100 - rt[b]["performance"]])
                      if b in rt else np.nan for b in range(16)])
    hard = np.array([-acc.get(b, np.nan) for b in range(16)])
    z = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    W[s] = (F, Y, blk); R[s] = dict(level=lv, felt=z(felt), felt2=z(felt2), hard=z(hard))
miss = {s: int(np.isnan(R[s]["felt"]).sum()) for s in subs if np.isnan(R[s]["felt"]).any()}
print("people with missing block ratings:", miss)

def fit(Xtr, ytr, groups):
    best = None
    for a in (0.1, 1, 10, 100):
        err = []
        for i, j in GroupKFold(5).split(Xtr, ytr, groups):
            sc = StandardScaler().fit(Xtr[i]); m = Ridge(alpha=a).fit(sc.transform(Xtr[i]), ytr[i])
            err.append(((m.predict(sc.transform(Xtr[j])) - ytr[j]) ** 2).mean())
        if best is None or np.mean(err) < best[0]: best = (np.mean(err), a)
    sc = StandardScaler().fit(Xtr); return sc, Ridge(alpha=best[1]).fit(sc.transform(Xtr), ytr)

def resid_within_level(v, lv):
    v = v.copy()
    for l in range(4): k = lv == l; v[k] = v[k] - np.nanmean(v[k])
    return v

def evaluate(target):
    """target in felt, felt2, hard: returns per-person stats for SET and FELT-target models."""
    rows = []
    for tr, va, te in P.buckets():
        tr = tr + va
        ok = [s for s in tr if not np.isnan(R[s][target]).all()]
        X = np.vstack([W[s][0] for s in ok]); g = np.concatenate([[s] * 304 for s in ok])
        yset = np.concatenate([W[s][1] for s in ok])
        yfelt = np.concatenate([np.repeat(R[s][target], 19) for s in ok]); keep = ~np.isnan(yfelt)
        ms = fit(X, yset, g); mf = fit(X[keep], yfelt[keep], g[keep])
        for s in te:
            F, Y, blk = W[s]; lv, tg = R[s]["level"], R[s][target]
            bset = np.array([ms[1].predict(ms[0].transform(F))[blk == b].mean() for b in range(16)])
            bfelt = np.array([mf[1].predict(mf[0].transform(F))[blk == b].mean() for b in range(16)])
            k = ~np.isnan(tg)
            if k.sum() < 8: continue
            rows.append(dict(sub=s,
                rho_level_target=spearmanr(lv[k], tg[k])[0],
                within_level=spearmanr(resid_within_level(bset, lv)[k], resid_within_level(tg, lv)[k])[0],
                set_vs_target=spearmanr(bset[k], tg[k])[0], felt_vs_target=spearmanr(bfelt[k], tg[k])[0],
                set_vs_level=spearmanr(bset, lv)[0], felt_vs_level=spearmanr(bfelt, lv)[0]))
    return pd.DataFrame(rows)

res = {}
for target in ("felt", "felt2", "hard"):
    df = evaluate(target).fillna(0); df.to_csv(f"pilot_N1_perperson_{target}.csv", index=False)
    wl = df.within_level
    r = dict(n_people=len(df), G0_G1_mean_rho_level_target=df.rho_level_target.mean(),
             G2_within_level_mean=wl.mean(), G2_wilcoxon_p=wilcoxon(wl, alternative="greater").pvalue,
             G2_share_positive=(wl > 0).mean(),
             G3_set_vs_target=df.set_vs_target.mean(), G3_felt_vs_target=df.felt_vs_target.mean(),
             set_vs_level=df.set_vs_level.mean(), felt_vs_level=df.felt_vs_level.mean(),
             share_3back_felt_not_harder_than_2back=float(np.mean([
                 np.nanmean(R[s][target][R[s]["level"] == 3]) <= np.nanmean(R[s][target][R[s]["level"] == 2]) for s in df["sub"]])))
    res[target] = r; print(target, {k: round(float(v), 4) for k, v in r.items()}, flush=True)
json.dump(res, open("pilot_N1_results.json", "w"), indent=1)
