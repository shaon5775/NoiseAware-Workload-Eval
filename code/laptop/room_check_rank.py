"""Paper 8 room check (exploration): absolute 4-level is ~35%. Is RELATIVE workload (which of two moments is harder,
same person) much easier? And are errors mostly between neighbouring levels?"""
import numpy as np, pandas as pd, pilot_P8A as P
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

subs = sorted({s for b in P.buckets() for part in b for s in part})
D = {}
for s in subs:
    F, Y = P.features(s); D[s] = ((F - F.mean(0)) / (F.std(0) + 1e-9), Y)
yy, pp, per = [], [], []
blockpair = []
for tr, va, te in P.buckets():
    tr = tr + va; X = np.vstack([D[s][0] for s in tr]); sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=3000).fit(sc.transform(X), np.concatenate([D[s][1] for s in tr]))
    for s in te:
        F, Y = D[s]; Pr = m.predict_proba(sc.transform(F)); score = Pr @ np.arange(4)   # expected level
        yy.append(Y); pp.append(Pr.argmax(1))
        r = {"sub": s}
        for g in (1, 2, 3):
            a = []
            for lo in range(4 - g):
                k = (Y == lo) | (Y == lo + g); a.append(roc_auc_score(Y[k] == lo + g, score[k]))
            r[f"gap{g}"] = np.mean(a)
        # block-level: average score over each 19-window block, rank 16 blocks
        bs = np.array([score[i*19:(i+1)*19].mean() for i in range(16)]); bl = Y[::19][:16]
        pairs = [(bs[i] > bs[j]) == (bl[i] > bl[j]) for i in range(16) for j in range(16) if bl[i] != bl[j] and i < j]
        r["block_rank_acc"] = np.mean(pairs)
        per.append(r)
yy, pp = np.concatenate(yy), np.concatenate(pp)
cm = confusion_matrix(yy, pp, normalize="true"); print("confusion (rows=true 0..3):\n", np.round(cm, 2))
print("errors to a neighbour level:", round(np.mean(np.abs(yy - pp)[yy != pp] == 1), 3))
df = pd.DataFrame(per); print(df.drop(columns="sub").describe().loc[["mean", "25%", "50%"]].round(3))
print("people with block ranking > 0.6:", round((df.block_rank_acc > 0.6).mean(), 3))
