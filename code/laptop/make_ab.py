"""Split-half behaviour (A = odd items, B = even items) per unit, aligned with bme_dfs rows. Light (parsing only)."""
import pickle, numpy as np, pandas as pd
import sys
import bvb_cog as B
BLOCK = "--block" in sys.argv
def halves_unit(s, ses, task, k, unit_s=60.0):
    ev = B.EVT[(s, ses, task)]; out = np.full((2, 3), np.nan)
    if "BACK" in task:
        stim = [(o, d[-1]) for o, d in ev if len(d) == 4 and d[2] == "2" and k * unit_s <= o < (k + 1) * unit_s]
        ro = np.array([o for o, d in ev if len(d) == 4 and d[2] == "3"])
        def hit(o): r = ro[(ro > o) & (ro < o + 2.0)] if len(ro) else ro; return r[0] - o if len(r) else None
        for h in (0, 1):
            S = (stim[:len(stim)//2] if h == 0 else stim[len(stim)//2:]) if BLOCK else stim[h::2]; tg = [hit(o) for o, x in S if x == "2"]; nt = [hit(o) for o, x in S if x == "1"]
            rt = [x for x in tg if x is not None]
            if tg: out[h] = [np.mean([x is not None for x in tg]), np.mean([x is not None for x in nt]) if nt else 0, np.mean(rt) if rt else 2.0]
    else:
        tr, rm = [], []
        for o, d in ev:
            if not (k * unit_s <= o < (k + 1) * unit_s): continue
            p = d.split(); c = int((o - k * unit_s) >= 30) if BLOCK else int((o - k * unit_s) // 5) % 2          # alternating 5-s chunks
            if len(p) >= 5 and p[1] == "TRACKING": tr.append((c, np.hypot(float(p[3]), float(p[4]))))
            if len(p) >= 7 and p[1] == "RESMAN": rm.append((c, abs(float(p[3]) - 2500) + abs(float(p[5]) - 2500)))
        for h in (0, 1):
            a = [v for c, v in tr if c == h]; b = [v for c, v in rm if c == h]
            if a: out[h] = [np.mean(a), np.mean(b) if b else np.nan, np.std(a)]
    return out
def zp(df, col):
    out = np.zeros((len(df), 3))
    for s in df.s.unique():
        m = (df.s == s).values; X = np.stack(df[col].values[m]); X = np.where(np.isnan(X), np.nanmean(X, 0), X)
        X = np.where(np.isnan(X), 0, X); out[m] = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return out
AB = {}
for S, fam in (("S1", "nback"), ("S2", "matb")):
    df = B.build(fam).reset_index(drop=True); H = [halves_unit(r.s, r.ses, r.task, r.unit) for r in df.itertuples()]
    df["A"] = [h[0] for h in H]; df["Bh"] = [h[1] for h in H]
    AB[S] = dict(s=df.s.astype(str).values, y=df.level.values, A=zp(df, "A"), B=zp(df, "Bh"))
import pav_df_ab as P
df = P.df; A, Bb = [], []
for r in df.itertuples():
    c = str(int(r.tc)).zfill(r.L) if str(r.tc).strip().isdigit() else "0" * r.L
    v = np.array([ch == "1" for ch in c[:r.L]], float); A.append([v[:r.L//2].mean() if BLOCK else v[0::2].mean()]); Bb.append([v[r.L//2:].mean() if BLOCK else v[1::2].mean()])
df["A"] = A; df["Bh"] = Bb
def z1(df, col):
    out = np.zeros((len(df), 1))
    for s in df.s.unique():
        m = (df.s == s).values; X = np.stack(df[col].values[m]); out[m] = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return out
AB["S4"] = dict(s=df.s.astype(str).values, y=df.level.values, A=z1(df, "A"), B=z1(df, "Bh"))
for k, v in AB.items():
    r = [np.corrcoef(v["A"][:, j], v["B"][:, j])[0, 1] for j in range(v["A"].shape[1])]
    print(k, len(v["s"]), "A-B correlation per feature", np.round(r, 3))
np.savez_compressed("data/ab_halves_block.npz" if BLOCK else "data/ab_halves.npz", **{f"{k}_{f}": v[f] for k, v in AB.items() for f in v})
