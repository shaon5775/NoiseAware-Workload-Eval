"""Paper 8 room check (exploration): LOCAL REST REFERENCING. Lessons: offsets are solved by per-person scaling; what is
left is slow person-specific drift that makes whole blocks wrong, and only better features ever helped. So: describe
each task window RELATIVE to the rest just before its block (~20 s earlier). Removes offset AND drift, works live."""
import glob, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
Z = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob("data/shin/*.npy")}
P = sorted({int(k.split("_")[1]) for k in Z})
def logm(C): w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-10))[..., None, :]) @ np.swapaxes(V, -1, -2)
def isqrtm(C): w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-10)))) @ V.T
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
iu = None; D = {}
for p in P:
    y, blk, rest = Z[f"y_{p}"], Z[f"blk_{p}"], Z[f"rest_{p}"].astype(bool)
    bp = Z[f"bp_{p}"].reshape(len(y), -1).astype(float)
    C = Z[f"cov_{p}"].astype(float); C += 1e-6 * np.eye(C.shape[1]) * np.trace(C, axis1=1, axis2=2).mean() / C.shape[1]
    if iu is None: iu = np.triu_indices(C.shape[1])
    task = ~rest; blocks = np.unique(blk[task])
    # global per-person reference (usual)
    Rg = isqrtm(C[task].mean(0)); TG = logm(Rg @ C @ Rg)[:, iu[0], iu[1]]
    G = np.c_[bp - bp[task].mean(0), TG]
    # local reference: rest windows after the PREVIOUS block (rest with blk == k-1); first block uses global ref
    Lc = np.zeros_like(G); has = np.zeros(len(y), bool)
    for k in blocks:
        tk = task & (blk == k); rk = rest & (blk == k - 1)
        if rk.sum() >= 3:
            Rl = isqrtm(C[rk].mean(0)); TL = logm(Rl @ C[tk] @ Rl)[:, iu[0], iu[1]]
            Lc[tk] = np.c_[bp[tk] - bp[rk].mean(0), TL]; has[tk] = True
        else: Lc[tk] = G[tk]
    zs = lambda A: (A - A[task].mean(0)) / (A[task].std(0) + 1e-9)
    D[p] = dict(G=zs(G)[task], L=zs(Lc)[task], y=y[task], b=blk[task], has=has[task])
print("share of task windows with a local rest reference:", round(float(np.mean(np.concatenate([D[p]['has'] for p in P]))), 3))
for name, get in [("global per-person ref (usual)", lambda d: d["G"]), ("LOCAL pre-block rest ref", lambda d: d["L"]),
                  ("both (global + local)", lambda d: np.c_[d["G"], d["L"]])]:
    acc, blkacc = [], []
    for te in P:
        tr = [p for p in P if p != te]; X = np.vstack([get(D[p]) for p in tr]); Y = np.concatenate([D[p]["y"] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(X), Y)
        d = D[te]; pr = clf.predict_proba(sc.transform(get(d))); acc.append(bacc(d["y"], pr.argmax(1)))
        bl = np.unique(d["b"]); blkacc.append(bacc(np.array([d["y"][d["b"] == k][0] for k in bl]), np.array([pr[d["b"] == k].mean(0).argmax() for k in bl])))
    if name.startswith("global"): a0, b0 = np.array(acc), np.array(blkacc)
    print(f"{name:32s} window {np.mean(acc):.3f} | block {np.mean(blkacc):.3f} | people better (window) {np.mean(np.array(acc) > a0):.0%} (block) {np.mean(np.array(blkacc) > b0):.0%}")
