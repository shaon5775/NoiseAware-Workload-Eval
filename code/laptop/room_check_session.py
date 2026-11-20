"""Paper 8 room check (exploration): errors are whole-block, not window noise. Is it SESSION drift?
Compare per-person vs per-session normalisation (unlabelled), and count blocks whose majority vote is right."""
import glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
Z = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob("data/shin/*.npy")}
P = sorted({int(k.split("_")[1]) for k in Z})
def logm(C): w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-10))[..., None, :]) @ np.swapaxes(V, -1, -2)
def isqrtm(C): w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-10)))) @ V.T
def feats(p, unit):
    task = Z[f"y_{p}"] >= 0; bp = Z[f"bp_{p}"][task].reshape(task.sum(), -1).astype(float)
    C = Z[f"cov_{p}"][task].astype(float); C += 1e-6 * np.eye(C.shape[1]) * np.trace(C, axis1=1, axis2=2).mean() / C.shape[1]
    ses = Z[f"ses_{p}"][task]; g = ses if unit == "session" else np.zeros_like(ses)
    iu = np.triu_indices(C.shape[1]); tg = np.zeros((len(C), len(iu[0])))
    for s in np.unique(g):
        k = g == s; bp[k] = (bp[k] - bp[k].mean(0)) / (bp[k].std(0) + 1e-9)
        R = isqrtm(C[k].mean(0)); T = logm(R @ C[k] @ R); tg[k] = T[:, iu[0], iu[1]] * np.where(iu[0] == iu[1], 1, np.sqrt(2))
        tg[k] = (tg[k] - tg[k].mean(0)) / (tg[k].std(0) + 1e-9)
    return np.c_[bp, tg], Z[f"y_{p}"][task], Z[f"blk_{p}"][task], ses
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
print("sessions per person:", sorted({len(np.unique(Z[f'ses_{p}'])) for p in P}), "blocks per session:",
      sorted({len(np.unique(Z[f'blk_{p}'][(Z[f'ses_{p}'] == 0) & (Z[f'y_{p}'] >= 0)])) for p in P}))
for unit in ("person", "session"):
    D = {p: feats(p, unit) for p in P}; acc, blk = [], []
    for te in P:
        tr = [p for p in P if p != te]; X = np.vstack([D[p][0] for p in tr]); Y = np.concatenate([D[p][1] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(X), Y)
        F, y, b, s = D[te]; pr = clf.predict(sc.transform(F)); acc.append(bacc(y, pr))
        blk.append(np.mean([np.bincount(pr[b == k], minlength=3).argmax() == y[b == k][0] for k in np.unique(b)]))
    print(f"normalise per {unit:8s}: window bal.acc {np.mean(acc):.3f} | blocks right by majority vote {np.mean(blk):.3f}")
