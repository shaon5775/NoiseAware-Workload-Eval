# Paper 8 pilot RK "Rank, don't classify" (EEG). Rules: pilot_RK_prereg.md (fixed).
import os, subprocess, glob, zipfile, time, json, numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, welch
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
D = "/kaggle/tmp/shin"; os.makedirs(D, exist_ok=True)
if not glob.glob(f"{D}/EEG/*"):
    p = f"{D}/eeg.zip"
    subprocess.run(["wget", "-q", "-O", p, "http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/EEG/EEG_01-26_MATLAB.zip"], check=True)
    zipfile.ZipFile(p).extractall(f"{D}/EEG"); os.remove(p)
log("data ready")
def L(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False); return [v for k, v in m.items() if not k.startswith("__")][0]
def as_list(o): return list(o.flat) if isinstance(o, np.ndarray) and o.dtype == object else [o]
def blocks(mrk):
    out = []
    for m in as_list(mrk):
        t = np.atleast_1d(m.time).astype(float) / 1000.0
        y = np.atleast_2d(m.y); y = y if y.shape[1] == len(t) else y.T
        lev = np.full(len(t), -1)
        for ci, c in enumerate([str(c) for c in np.atleast_1d(m.className)]):
            for l in (0, 2, 3):
                if c.strip().startswith(f"{l}-back"): lev[y[ci] > 0] = l
        k = lev >= 0; t, lev = t[k], lev[k]; o = np.argsort(t); t, lev = t[o], lev[o]; i = 0
        while i < len(t):
            j = i
            while j + 1 < len(t) and lev[j + 1] == lev[i] and t[j + 1] - t[j] < 5: j += 1
            out.append((t[i], t[j] + 2 if j > i else t[i] + 40, int(lev[i]))); i = j + 1
    return out
def person(vp):
    cs = as_list(L(f"{D}/EEG/VP{vp:03d}-EEG/cnt_nback.mat")); bl = blocks(L(f"{D}/EEG/VP{vp:03d}-EEG/mrk_nback.mat"))
    segs = [[bl[0]]]
    for b in bl[1:]:
        if b[0] < segs[-1][-1][0]: segs.append([b])
        else: segs[-1].append(b)
    if len(cs) == 1: segs = [bl]
    assert len(segs) == len(cs), (len(segs), len(cs))
    F, Y, B = [], [], []; bi = -1
    for c, sb in zip(cs, segs):
      cl = [str(x) for x in np.atleast_1d(c.clab)]; keep = [i for i, n in enumerate(cl) if "EOG" not in n.upper()]
      x = np.asarray(c.x, float)[:, keep]; fs = float(c.fs)
      x = sosfiltfilt(butter(4, [1, 40], btype="band", fs=fs, output="sos"), x, axis=0)
      for (s, e, lv) in sb:
        bi += 1
        for off in np.arange(0, (e - s) - 10 + 1e-9, 2.0):
            a = int((s + off) * fs); w = x[a:a + int(10 * fs)]
            f, P = welch(w, fs=fs, nperseg=int(2 * fs), axis=0)
            F.append(np.concatenate([np.log(P[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in ((4, 8), (8, 13), (13, 30))]))
            Y.append({0: 0, 2: 1, 3: 2}[lv]); B.append(bi)
    F = np.array(F); return (F - F.mean(0)) / (F.std(0) + 1e-9), np.array(Y), np.array(B)
DATA = {vp: person(vp) for vp in range(1, 27)}
log("loaded", len(DATA), "people; windows", sorted({len(v[1]) for v in DATA.values()}), "feats", DATA[1][0].shape[1])

import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(4)
def net(o): return nn.Sequential(nn.Linear(84, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, o))
def fit(mode, ps, seed, epochs=60):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    X = torch.tensor(np.vstack([DATA[p][0] for p in ps]), dtype=torch.float32)
    Y = np.concatenate([DATA[p][1] for p in ps]); P = np.concatenate([[i] * len(DATA[p][1]) for i, p in enumerate(ps)])
    m = net(3 if mode == "ce" else 1); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(X) / 256)); Yt = torch.tensor(Y)
    idx = {(i, l): np.flatnonzero((P == i) & (Y == l)) for i in range(len(ps)) for l in range(3)}
    for st in range(steps):
        if mode == "rank":
            who = rng.randint(len(ps), size=256); lo = rng.randint(0, 2, size=256); hi = np.array([rng.randint(l + 1, 3) for l in lo])
            a = np.array([rng.choice(idx[(w, l)]) for w, l in zip(who, hi)]); b = np.array([rng.choice(idx[(w, l)]) for w, l in zip(who, lo)])
            loss = Fn.softplus(-(m(X[a]) - m(X[b]))).mean()
        else:
            k = torch.tensor(rng.randint(len(X), size=256)); out = m(X[k])
            loss = Fn.cross_entropy(out, Yt[k]) if mode == "ce" else Fn.mse_loss(out[:, 0], Yt[k].float())
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    def score(Z):
        with torch.no_grad():
            o = m(torch.tensor(Z, dtype=torch.float32))
            return (torch.softmax(o, 1).numpy(), torch.softmax(o, 1).numpy() @ np.arange(3)) if mode == "ce" else (None, o[:, 0].numpy())
    return score
def qmap(s): q = np.quantile(s, [1 / 3, 2 / 3]); return np.digitize(s, q)
def kmap(s, it=50):
    c = np.quantile(s, [1 / 6, 1 / 2, 5 / 6])
    for _ in range(it):
        a = np.argmin(np.abs(s[:, None] - c[None]), 1); c = np.array([s[a == j].mean() if (a == j).any() else c[j] for j in range(3)])
    return np.argsort(np.argsort(c))[np.argmin(np.abs(s[:, None] - c[None]), 1)]
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
def auc(y, s):
    r = [];
    for lo, hi in ((0, 1), (0, 2), (1, 2)):
        a, b = s[y == hi], s[y == lo]; r.append(np.mean(a[:, None] > b[None]) + 0.5 * np.mean(a[:, None] == b[None]))
    return np.mean(r)

R = {k: {} for k in ("C1a", "C1b", "C2", "RK", "RKk", "C1a_auc", "C1b_auc", "C2_auc", "RK_auc")}
for seed in range(3):
    for te in DATA:
        tr = [p for p in DATA if p != te]; X, y, _ = DATA[te]
        pr, s1 = fit("ce", tr, seed)(X); s2 = fit("reg", tr, seed)(X)[1]; s3 = fit("rank", tr, seed)(X)[1]
        vals = dict(C1a=bacc(y, pr.argmax(1)), C1b=bacc(y, qmap(s1)), C2=bacc(y, qmap(s2)), RK=bacc(y, qmap(s3)), RKk=bacc(y, kmap(s3)),
                    C1a_auc=auc(y, s1), C1b_auc=auc(y, s1), C2_auc=auc(y, s2), RK_auc=auc(y, s3))
        for k, v in vals.items(): R[k].setdefault(te, []).append(v)
    log("seed", seed, {k: round(float(np.mean([np.mean(v) for v in R[k].values()])), 4) for k in R})
M = {k: np.array([np.mean(R[k][p]) for p in DATA]) for k in R}; o = {k: float(M[k].mean()) for k in M}
bb = max(("C1a", "C1b", "C2"), key=lambda k: o[k]); ba = max(("C1b_auc", "C2_auc"), key=lambda k: o[k])
o.update(best_baseline=bb, G1_gain=o["RK"] - o[bb], G1_share=float((M["RK"] > M[bb]).mean()),
         G2_gain=o["RK_auc"] - o[ba], G3_gain=o["RKk"] - o["C1a"])
o["G1"] = bool(o["G1_gain"] >= 0.02 and o["G1_share"] >= 0.60); o["G2"] = bool(o["G2_gain"] >= 0.02); o["G3"] = bool(o["G3_gain"] >= 0.01)
o["GO"] = bool(o["G1"] and o["G2"] and o["G3"]); o["per_person"] = {k: M[k].round(4).tolist() for k in M}
print("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != "per_person"}))
json.dump(o, open("/kaggle/working/rk_pilot_results.json", "w"), indent=1)
