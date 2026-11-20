"""Paper 8 pilot P8-A. Rules: pilot_P8A_prereg.md (fixed before running)."""
import re, sys, json, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

DATA = "data"; WIN, STRIDE, ALPHA, K = 150, 15, 0.10, 4
CH = ["AB_I_O","AB_PHI_O","AB_I_DO","AB_PHI_DO","CD_I_O","CD_PHI_O","CD_I_DO","CD_PHI_DO"]

def buckets():
    txt = open("partition_plan.txt").read()
    sec = txt.split("64 generic pool size")[1].split("16 generic pool size")[0]
    get = lambda name: [list(map(int, re.findall(r"\d+", m))) for m in re.findall(name + r"\s*=\s*\[([^\]]*)\]", sec)]
    return list(zip(get("train_subjects"), get("val_subjects"), get("test_subjects")))

def features(sub):
    d = pd.read_csv(f"{DATA}/sub_{sub}.csv"); x = d[CH].values; l = d.label.values
    starts = np.r_[0, np.flatnonzero(np.diff(l)) + 1, len(l)]
    t = np.arange(WIN) - (WIN - 1) / 2
    F, Y = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        for s in range(a, b - WIN + 1, STRIDE):
            w = x[s:s + WIN]
            slope = (t[:, None] * (w - w.mean(0))).sum(0) / (t ** 2).sum()
            F.append(np.r_[w.mean(0), w.std(0), slope, w.min(0), w.max(0), w[-1] - w[0]]); Y.append(l[a])
    F = np.array(F); return F, np.array(Y)

def nested_intervals(p):
    """Greedy contiguous intervals grown from argmax; returns list of (lo, hi, mass)."""
    lo = hi = int(np.argmax(p)); out = [(lo, hi, p[lo])]
    while lo > 0 or hi < K - 1:
        left = p[lo - 1] if lo > 0 else -1; right = p[hi + 1] if hi < K - 1 else -1
        if left >= right: lo -= 1
        else: hi += 1
        out.append((lo, hi, p[lo:hi + 1].sum()))
    return out

def ord_score(p, y):
    for lo, hi, m in nested_intervals(p):
        if lo <= y <= hi: return m

def ord_interval(p, q):
    for lo, hi, m in nested_intervals(p):
        if m >= q - 1e-12: return lo, hi
    return 0, K - 1

def qhat(scores):
    n = len(scores); return np.quantile(scores, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")

def run(model_fn, feats, norm, tag):
    rows = []
    for tr, va, te in buckets():
        X = lambda S: np.vstack([feats[norm][s][0] for s in S]); Yf = lambda S: np.concatenate([feats[norm][s][1] for s in S])
        sc = StandardScaler().fit(X(tr)); m = model_fn().fit(sc.transform(X(tr)), Yf(tr))
        Pv = m.predict_proba(sc.transform(X(va))); yv = Yf(va)
        q1 = qhat(1 - Pv[np.arange(len(yv)), yv]); q2 = qhat(np.array([ord_score(p, y) for p, y in zip(Pv, yv)]))
        for s in te:
            Xs, ys = feats[norm][s]; P = m.predict_proba(sc.transform(Xs))
            sets = (1 - P) <= q1 + 1e-12
            cov1 = sets[np.arange(len(ys)), ys]; size1 = sets.sum(1)
            noncontig = np.array([st.sum() > 0 and (np.ptp(np.flatnonzero(st)) + 1) != st.sum() for st in sets])
            iv = np.array([ord_interval(p, q2) for p in P]); cov2 = (iv[:, 0] <= ys) & (ys <= iv[:, 1]); w2 = iv[:, 1] - iv[:, 0] + 1
            rows.append(dict(model=tag, norm=norm, sub=s, n=len(ys), pred=P.argmax(1).tolist(), y=ys.tolist(),
                             cov1=cov1.mean(), size1=size1.mean(), nonc=noncontig.mean(), cov2=cov2.mean(), w2=w2.mean()))
    return rows

def summarize(rows):
    df = pd.DataFrame(rows); out = {}
    for (mdl, norm), g in df.groupby(["model", "norm"]):
        w = g.n / g.n.sum(); yy = np.concatenate(g.y.values); pp = np.concatenate(g.pred.values)
        out[f"{mdl}|{norm}"] = dict(
            bal_acc=balanced_accuracy_score(yy, pp),
            M1_cov=(g.cov1 * w).sum(), M1_size=(g.size1 * w).sum(), M1_noncontig=(g.nonc * w).sum(),
            M1_letdown=(g.cov1 < 0.80).mean(),
            M2_cov=(g.cov2 * w).sum(), M2_width=(g.w2 * w).sum(), M2_letdown=(g.cov2 < 0.80).mean(),
            per_person_cov2_min=g.cov2.min(), per_person_cov2_p10=g.cov2.quantile(0.10))
    return out

if __name__ == "__main__":
    t0 = time.time()
    subs = sorted({s for b in buckets() for part in b for s in part}); assert len(subs) == 68, len(subs)
    raw = {s: features(s) for s in subs}
    pn = {s: ((F - F.mean(0)) / (F.std(0) + 1e-9), Y) for s, (F, Y) in raw.items()}   # per-person, unlabelled
    feats = {"none": raw, "person": pn}
    print("windows:", sum(len(v[1]) for v in raw.values()), f"({time.time()-t0:.0f}s)", flush=True)
    rows = []
    lr = lambda: LogisticRegression(max_iter=3000, C=1.0)
    for norm in ["none", "person"]:
        rows += run(lr, feats, norm, "LR"); print("LR", norm, f"done {time.time()-t0:.0f}s", flush=True)
    for k, v in summarize(rows).items(): print("EARLY", k, {a: round(b, 3) for a, b in v.items()}, flush=True)
    json.dump(summarize(rows), open("pilot_P8A_results_LR.json", "w"), indent=1)
    for seed in range(3):
        rf = lambda: RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
        for norm in ["none", "person"]:
            rows += run(rf, feats, norm, f"RF{seed}"); print("RF", seed, norm, f"done {time.time()-t0:.0f}s", flush=True)
    res = summarize(rows)
    json.dump(res, open("pilot_P8A_results.json", "w"), indent=1)
    pd.DataFrame(rows).drop(columns=["pred", "y"]).to_csv("pilot_P8A_perperson.csv", index=False)
    for k, v in res.items(): print(k, {a: round(b, 3) for a, b in v.items()})
