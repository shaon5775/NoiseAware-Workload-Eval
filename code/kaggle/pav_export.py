# Paper 8 BVB: export Pavlov ds003838 (OpenNeuro, CC0) per-trial EEG features + behaviour, one person at a time.
import os, re, subprocess, time, pickle, urllib.request, io, numpy as np, pandas as pd
from scipy.signal import welch
try:
    import mne
except ImportError:
    subprocess.run(["pip", "install", "-q", "mne"], check=True); import mne
mne.set_log_level("ERROR")
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
S3 = "https://s3.amazonaws.com/openneuro.org"; W = "/kaggle/tmp/pav"; os.makedirs(W, exist_ok=True)
BANDS = ((1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40))
STD19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
def ls(prefix):
    keys, tok = [], None
    while True:
        u = f"{S3}?list-type=2&prefix={prefix}&max-keys=1000" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        x = urllib.request.urlopen(u).read().decode(); keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m: return keys
        tok = m.group(1)
import urllib.parse, h5py
def read_set73(path):
    """EEGLAB .set saved as MATLAB v7.3 (HDF5) with the data embedded."""
    f = h5py.File(path, "r"); E = f["EEG"] if "EEG" in f else f
    srate = float(np.array(E["srate"]).squeeze()); nb = int(np.array(E["nbchan"]).squeeze())
    d = E["data"]
    if isinstance(d, h5py.Dataset) and d.dtype.kind in "fiu": X = np.array(d, dtype=np.float32)
    else: raise RuntimeError("data not embedded")
    if X.shape[1] != nb and X.shape[0] == nb: X = X.T                       # want samples × channels
    labs = []
    try:
        L = E["chanlocs"]["labels"]
        for r in L[:].ravel(): labs.append("".join(chr(int(c)) for c in np.array(f[r]).ravel()))
    except Exception:
        labs = [f"ch{i}" for i in range(nb)]
    f.close(); return X, labs, srate
allk =ls("ds003838/sub-"); subs = sorted({k.split("/")[1] for k in allk if k.endswith("_task-memory_eeg.set")})
log("subjects with memory EEG:", len(subs))
OUT = {}
for sid in subs:
    try:
        base = f"ds003838/{sid}"
        ev = pd.read_csv(io.StringIO(urllib.request.urlopen(f"{S3}/{base}/eeg/{sid}_task-memory_events.tsv").read().decode()), sep="\t")
        beh = pd.read_csv(io.StringIO(urllib.request.urlopen(f"{S3}/{base}/beh/{sid}_task-memory_beh.tsv").read().decode()), sep="\t")
        extra = [k for k in allk if k.startswith(f"{base}/eeg/{sid}_task-memory_eeg.")]
        for k in extra:
            p = f"{W}/{os.path.basename(k)}"; subprocess.run(["wget", "-q", "-O", p, f"{S3}/{k}"], check=True)
        X, ch, fs0 = read_set73(f"{W}/{sid}_task-memory_eeg.set")
        keep = [i for i, c in enumerate(ch) if not re.search(r"EOG|ECG|EMG|PPG|Status|TRIG|EXG|GSR|Resp", c, re.I)]
        X, ch = X[:, keep], [ch[i] for i in keep]
        X = mne.filter.filter_data(X.T.astype(np.float64), fs0, 1, 40, verbose=False)
        X = mne.filter.resample(X, down=fs0 / 250, verbose=False).T; fs = 250.0
        i19 = [ch.index(c) for c in STD19 if c in ch]
        if len(OUT) == 0: log("channels", len(ch), ch[:10], "fs0", fs0, "samples", X.shape)
        # parse digit events: "<cond> <trial>/<len>: ... digit <pos> ..."
        rows = []; trial = 0
        for o, tt in zip(ev.onset, ev.trial_type.astype(str)):
            m = re.match(r"(memory|control)\s+(\d+)/(\d+)", tt)        # "<cond> <position>/<length> ..."
            if m:
                pos = int(m.group(2))
                if pos == 1: trial += 1
                rows.append((m.group(1), trial, int(m.group(3)), pos, float(o)))
        R = pd.DataFrame(rows, columns=["cond", "trial", "len", "pos", "onset"])
        F, meta = [], []
        for (cond, trial, L), g in R.groupby(["cond", "trial", "len"]):
            end = g.onset.max() + 2.0; a = int((end - 10) * fs); b = int(end * fs)
            if a < 0 or b > len(X): continue
            w = X[a:b]; fr, P = welch(w, fs=fs, nperseg=int(2 * fs), axis=0)
            bp = np.stack([np.log(P[(fr >= lo) & (fr < hi)].mean(0) + 1e-20) for lo, hi in BANDS]).astype(np.float32)
            F.append((bp, np.cov(w[::2, i19].T).astype(np.float32))); meta.append((cond, trial, L, g.onset.min(), end))
        OUT[sid] = dict(bp=np.array([f[0] for f in F]), cov=np.array([f[1] for f in F]),
                        meta=pd.DataFrame(meta, columns=["cond", "trial", "len", "start", "end"]), beh=beh, ch=ch, ch19=[ch[i] for i in i19])
        log(sid, "trials", len(F), "memory", int((OUT[sid]["meta"].cond == "memory").sum()), "beh rows", len(beh))
        if len(OUT) == 1: print(R.head(8).to_string()); print(beh.head(5).to_string()); print(OUT[sid]["meta"].head(6).to_string())
    except Exception as e:
        log(sid, "FAILED", repr(e)[:300])
        if not OUT and sid == subs[0]:
            import traceback; traceback.print_exc()
            try:
                f = h5py.File(f"{W}/{sid}_task-memory_eeg.set", "r"); print("TOP KEYS", list(f.keys()))
                E = f["EEG"] if "EEG" in f else f; print("EEG KEYS", list(E.keys())[:60])
                for k in ("data", "srate", "nbchan", "pnts"): print(k, E[k] if k in E else None)
            except Exception as e2: print("h5 inspect failed", e2)
            raise
    for f in os.listdir(W): os.remove(f"{W}/{f}")
    if len(OUT) % 10 == 0: pickle.dump(OUT, open("/kaggle/working/pav_features.pkl", "wb"), protocol=4)
pickle.dump(OUT, open("/kaggle/working/pav_features.pkl", "wb"), protocol=4)
log("saved people", len(OUT), round(os.path.getsize("/kaggle/working/pav_features.pkl") / 1e6, 1), "MB")
