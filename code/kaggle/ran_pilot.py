# Paper 8 pilot RAN (rest-anchored normalisation). Rules: pilot_RAN_prereg.md (fixed).
import os, subprocess, glob, zipfile, time, json, numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
def bp(w, fs):
    f, P = welch(w, fs=fs, nperseg=int(2 * fs), axis=0)
    return np.concatenate([np.log(P[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in ((4, 8), (8, 13), (13, 30))])
def person(vp):
    cs = as_list(L(f"{D}/EEG/VP{vp:03d}-EEG/cnt_nback.mat")); bl = blocks(L(f"{D}/EEG/VP{vp:03d}-EEG/mrk_nback.mat"))
    segs = [[bl[0]]]
    for b in bl[1:]:
        if b[0] < segs[-1][-1][0]: segs.append([b])
        else: segs[-1].append(b)
    if len(cs) == 1: segs = [bl]
    assert len(segs) == len(cs)
    TF, TY, RF = [], [], []; rest_used = 0.0
    for c, sb in zip(cs, segs):
        cl = [str(x) for x in np.atleast_1d(c.clab)]; keep = [i for i, n in enumerate(cl) if "EOG" not in n.upper()]
        x = sosfiltfilt(butter(4, [1, 40], btype="band", fs=float(c.fs), output="sos"), np.asarray(c.x, float)[:, keep], axis=0)
        fs = float(c.fs)
        for bi, (s, e, lv) in enumerate(sb):
            for off in np.arange(0, (e - s) - 10 + 1e-9, 2.0):
                a = int((s + off) * fs); TF.append(bp(x[a:a + int(10 * fs)], fs)); TY.append({0: 0, 2: 1, 3: 2}[lv])
            if bi + 1 < len(sb):
                r0, r1 = e + 2, sb[bi + 1][0] - 2
                for off in np.arange(0, (r1 - r0) - 10 + 1e-9, 2.0):
                    if rest_used + off < 60:
                        a = int((r0 + off) * fs); RF.append(bp(x[a:a + int(10 * fs)], fs))
                rest_used += max(0.0, r1 - r0)
    return np.array(TF), np.array(TY), np.array(RF)
DATA = {vp: person(vp) for vp in range(1, 27)}
log("loaded; task windows", sorted({len(v[1]) for v in DATA.values()}), "rest windows", sorted({len(v[2]) for v in DATA.values()}))

MIX = {"balanced": [1/3, 1/3, 1/3], "mostly easy": [.60, .25, .15], "mostly hard": [.15, .25, .60], "easy+medium only": [.5, .5, 0]}
def stats(F): return F.mean(0), F.std(0) + 1e-9
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])
R = {m: {k: [] for k in ("Zown", "Oracle", "RAN", "RANm")} for m in MIX}
for te in DATA:
    tr = [p for p in DATA if p != te]
    for norm in ("Zown", "Oracle", "RAN", "RANm"):
        def tnorm(p):
            F, Y, Rr = DATA[p]
            if norm in ("Zown", "Oracle"): m, s = stats(F)
            elif norm == "RAN": m, s = stats(Rr)
            else: m, s = Rr.mean(0), F.std(0) + 1e-9
            return (F - m) / s
        X = np.vstack([tnorm(p) for p in tr]); Y = np.concatenate([DATA[p][1] for p in tr])
        sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), Y)
        F, Yt, Rr = DATA[te]
        for mname, mix in MIX.items():
            rng = np.random.RandomState(1000 * te + list(MIX).index(mname))
            for rep in range(10):
                idx = np.concatenate([rng.choice(np.flatnonzero(Yt == k), int(round(w * 150)), replace=False) for k, w in enumerate(mix) if w > 0])
                Fi, yi = F[idx], Yt[idx]
                if norm == "Zown": m, s = stats(Fi)
                elif norm == "Oracle": m, s = stats(F)
                elif norm == "RAN": m, s = stats(Rr)
                else: m, s = Rr.mean(0), Fi.std(0) + 1e-9
                R[mname][norm].append(bacc(yi, clf.predict(sc.transform((Fi - m) / s))))
    log("person", te, "done")
o = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in R.items()}
U = ["mostly easy", "mostly hard", "easy+medium only"]; mean = lambda k: float(np.mean([o[m][k] for m in U]))
o["T1"] = bool(o["easy+medium only"]["Zown"] <= o["easy+medium only"]["Oracle"] - 0.05 and mean("Zown") <= mean("Oracle") - 0.03)
o["T2"] = bool(mean("RAN") >= mean("Zown") + 0.03 and o["easy+medium only"]["RAN"] >= o["easy+medium only"]["Zown"] + 0.05)
o["T3"] = bool(o["balanced"]["RAN"] >= o["balanced"]["Zown"] - 0.03)
o["GO"] = bool(o["T1"] and o["T2"] and o["T3"]); o["unbalanced_means"] = {k: mean(k) for k in ("Zown", "Oracle", "RAN", "RANm")}
print("RESULT", json.dumps(o, default=lambda v: round(v, 4)))
json.dump(o, open("/kaggle/working/ran_pilot_results.json", "w"), indent=1)
