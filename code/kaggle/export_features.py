# Paper 8: export Shin 2018 n-back EEG window features once, so idea checks can run fast on the laptop.
import os, subprocess, glob, zipfile, time, numpy as np, scipy.io as sio
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
BANDS = ((1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40))
OUT = {}
for vp in range(1, 27):
    cs = as_list(L(f"{D}/EEG/VP{vp:03d}-EEG/cnt_nback.mat")); bl = blocks(L(f"{D}/EEG/VP{vp:03d}-EEG/mrk_nback.mat"))
    segs = [[bl[0]]]
    for b in bl[1:]:
        if b[0] < segs[-1][-1][0]: segs.append([b])
        else: segs[-1].append(b)
    if len(cs) == 1: segs = [bl]
    BP, COV, Y, BLK, TT, SES, ISREST = [], [], [], [], [], [], []; bi = -1
    for si, (c, sb) in enumerate(zip(cs, segs)):
        cl = [str(x) for x in np.atleast_1d(c.clab)]; keep = [i for i, n in enumerate(cl) if "EOG" not in n.upper()]
        fs = float(c.fs); x = sosfiltfilt(butter(4, [1, 40], btype="band", fs=fs, output="sos"), np.asarray(c.x, float)[:, keep], axis=0)
        spans = []
        for k, (s, e, lv) in enumerate(sb):
            bi += 1; spans.append((s, e, lv, bi, 0))
            if k + 1 < len(sb): spans.append((e + 2, sb[k + 1][0] - 2, -1, bi, 1))
        for s, e, lv, b, rest in spans:
            for off in np.arange(0, (e - s) - 10 + 1e-9, 2.0):
                a = int((s + off) * fs); w = x[a:a + int(10 * fs)]
                f, P = welch(w, fs=fs, nperseg=int(2 * fs), axis=0)
                BP.append(np.stack([np.log(P[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in BANDS]))  # bands × ch
                C = np.cov(w[::4].T); COV.append(C.astype(np.float32))
                Y.append({-1: -1, 0: 0, 2: 1, 3: 2}[lv]); BLK.append(b); TT.append(s + off); SES.append(si); ISREST.append(rest)
    OUT[f"bp_{vp}"] = np.array(BP, np.float32); OUT[f"cov_{vp}"] = np.array(COV, np.float32)
    OUT[f"y_{vp}"] = np.array(Y); OUT[f"blk_{vp}"] = np.array(BLK); OUT[f"t_{vp}"] = np.array(TT, np.float32)
    OUT[f"ses_{vp}"] = np.array(SES); OUT[f"rest_{vp}"] = np.array(ISREST)
    log("person", vp, "windows", len(Y), "rest", int(np.sum(ISREST)), "channels", x.shape[1])
np.savez_compressed("/kaggle/working/shin_nback_eeg_features.npz", **OUT)
log("saved", round(os.path.getsize("/kaggle/working/shin_nback_eeg_features.npz") / 1e6, 1), "MB")
