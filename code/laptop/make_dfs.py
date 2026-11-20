"""Cache the 4 BME dataframes (S1-S4) built exactly as in bme.py / bme2_pav.py."""
import pickle, numpy as np, pandas as pd
import bvb_cog as B, pilot_P8A as P
out = {"S1": B.build("nback").reset_index(drop=True), "S2": B.build("matb").reset_index(drop=True)}
rows = []
for s in sorted({x for b in P.buckets() for part in b for x in part}):
    F, Y = P.features(s); F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    acc = pd.read_csv(f"data/task_accuracy/sub_{s}.csv").acc.values; acc = (acc - acc.mean()) / (acc.std() + 1e-9)
    for k in range(16): rows.append(dict(s=s, level=int(Y[k * 19]), brain=F[k * 19:(k + 1) * 19].mean(0), behz=np.array([acc[k]])))
out["S3"] = pd.DataFrame(rows)
import pav_df; out["S4"] = pav_df.df
for k, d in out.items(): out[k] = d[["s", "level", "brain", "behz"]].reset_index(drop=True); print(k, len(d), d.s.nunique())
pickle.dump(out, open("data/bme_dfs.pkl", "wb"))
