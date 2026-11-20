"""Paper 8 artefact check V3 (rules: pilot_ARTEFACT_prereg.md): MATB RB-ME with behaviour halves + task-activity counts
(key presses KB1, scheduled events EVENT, audio messages COMM) per half. Also reports reliabilities (M4/M5). Laptop, light."""
import json, sys, glob, numpy as np
import bvb_cog as B
src = open("kaggle/artefact_body.py", encoding="utf-8").read().split("MUSCLE_FAIL = [0]")[0].split("def feats")[0]
g = {}; exec(src.replace("try:\n    import mne\nexcept ImportError:\n    subprocess.run([\"pip\", \"install\", \"-q\", \"mne\"], check=True); import mne\nmne.set_log_level(\"ERROR\")", ""), g)
body = open("kaggle/artefact_body.py", encoding="utf-8").read()
exec(body[body.index("def loso"):body.index("def bacc")], g)
rbme, loso = g["rbme"], g["loso"]
def zp(df, col):   # same as make_ab.zp (per-person z-score)
    out = np.zeros((len(df), len(df[col].iloc[0])))
    for p in df.s.unique():
        m = (df.s == p).values; X = np.stack(df[col].values[m]); out[m] = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return out

def counts_unit(s, ses, task, k, unit_s=60.0):
    ev = B.EVT[(s, ses, task)]; out = np.zeros((2, 3))
    for o, d in ev:
        if not (k * unit_s <= o < (k + 1) * unit_s): continue
        p = d.split(); h = int((o - k * unit_s) // 5) % 2
        if len(p) >= 2 and p[1] in ("KB1", "EVENT", "COMM"): out[h, ("KB1", "EVENT", "COMM").index(p[1])] += 1
    return out

df = B.build("matb").reset_index(drop=True)
Z = {}
for f in sorted(glob.glob("data/ncbme_upload/bme_*.npz")):
    for k, v in np.load(f, allow_pickle=True).items(): Z[k] = np.concatenate([Z[k], v]) if k in Z else v
AB = dict(np.load("data/ab_halves.npz", allow_pickle=True))
s, y = AB["S2_s"].astype(str), AB["S2_y"].astype(int); assert (df.s.astype(str).values == s).all() and (df.level.values == y).all()
C = [counts_unit(r.s, r.ses, r.task, r.unit) for r in df.itertuples()]
df["cA"] = [c[0] for c in C]; df["cB"] = [c[1] for c in C]
cA, cB = zp(df, "cA"), zp(df, "cB")
print("mean counts per half-unit (KB1, EVENT, COMM) by level:", [np.stack(df.cA.values)[y == l].mean(0).round(2).tolist() for l in range(3)])
brain = Z["S2_brain"].astype(float); sb = loso(s, y, brain, 0.05)
A, Bh = AB["S2_A"].astype(float), AB["S2_B"].astype(float)
res = {"orig": rbme(s, y, sb, loso(s, y, A, 1.0), loso(s, y, Bh, 1.0)),
       "activity": rbme(s, y, sb, loso(s, y, np.c_[A, cA], 1.0), loso(s, y, np.c_[Bh, cB], 1.0)),
       "activity_only": rbme(s, y, sb, loso(s, y, cA, 1.0), loso(s, y, cB, 1.0))}
for k, v in res.items(): print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
v = res["activity"]; A3 = ("not testable" if v["n"] < 8 else bool(v["diff"] > 0 and v["p"] < .01)); print("A3", A3)
res["A3"] = A3; json.dump(res, open("artefact_v3_results.json", "w"), indent=1)
