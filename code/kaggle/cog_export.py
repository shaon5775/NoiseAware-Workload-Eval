# Paper 8 BVB: export COG-BCI (Hinss 2023, Zenodo 7413650, CC-BY-4.0) EEG window features + raw behaviour + events,
# one person at a time (download → features → delete). No modelling here.
import os, subprocess, glob, zipfile, time, pickle, shutil, numpy as np, scipy.io as sio
from scipy.signal import welch
try:
    import mne
except ImportError:
    subprocess.run(["pip", "install", "-q", "mne"], check=True); import mne
mne.set_log_level("ERROR")
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
W = "/kaggle/tmp/cog"; os.makedirs(W, exist_ok=True); OUT = "/kaggle/working"
TASKS = ["zeroBACK", "oneBACK", "twoBACK", "MATBeasy", "MATBmed", "MATBdiff", "RS_Beg_EO", "RS_End_EO"]
BANDS = ((1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40))
STD19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
def to_py(o):
    if isinstance(o, dict): return {k: to_py(v) for k, v in o.items() if not k.startswith("__")}
    if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
    if isinstance(o, np.ndarray) and o.dtype == object: return [to_py(v) for v in o.ravel()]
    return o
FEAT, BEH, EVT, INFO = {}, {}, {}, {}
for s in range(1, 30):
    sid = f"sub-{s:02d}"; z = f"{W}/{sid}.zip"
    try:
        subprocess.run(["wget", "-q", "-O", z, f"https://zenodo.org/records/7413650/files/{sid}.zip?download=1"], check=True)
        zipfile.ZipFile(z).extractall(f"{W}/{sid}"); os.remove(z)
    except Exception as e:
        log(sid, "download failed", repr(e)[:200]); continue
    for ses in sorted(glob.glob(f"{W}/{sid}/**/ses-S*", recursive=True)):
        sn = os.path.basename(ses)
        for b in glob.glob(f"{ses}/behavioral/*.mat"):
            try: BEH[(s, sn, os.path.basename(b)[:-4])] = to_py(sio.loadmat(b, simplify_cells=True))
            except Exception as e: BEH[(s, sn, os.path.basename(b)[:-4])] = f"ERR {e!r}"
        for task in TASKS:
            f = glob.glob(f"{ses}/eeg/{task}.set") or glob.glob(f"{ses}/eeg/{task.replace('EO','Eo')}.set")
            if not f: continue
            try:
                raw = mne.io.read_raw_eeglab(f[0], preload=True)
            except Exception as e:
                log(sid, sn, task, "read failed", repr(e)[:150]); continue
            raw.pick("eeg"); raw.filter(1, 40, verbose=False); fs = raw.info["sfreq"]; X = raw.get_data().T
            ch = raw.ch_names; i19 = [ch.index(c) for c in STD19 if c in ch]
            win, st = int(10 * fs), int(5 * fs); bp, cov, t0 = [], [], []
            for a in range(0, len(X) - win + 1, st):
                w = X[a:a + win]; fr, P = welch(w, fs=fs, nperseg=int(2 * fs), axis=0)
                bp.append(np.stack([np.log(P[(fr >= lo) & (fr < hi)].mean(0) + 1e-20) for lo, hi in BANDS]).astype(np.float32))
                cov.append(np.cov(w[::5, i19].T).astype(np.float32)); t0.append(a / fs)
            FEAT[(s, sn, task)] = dict(bp=np.array(bp), cov=np.array(cov), t=np.array(t0, np.float32))
            EVT[(s, sn, task)] = [(float(o), str(d)) for o, d in zip(raw.annotations.onset, raw.annotations.description)]
            INFO[(s, sn, task)] = dict(fs=fs, n_ch=len(ch), ch19=[ch[i] for i in i19], dur_s=len(X) / fs)
    shutil.rmtree(f"{W}/{sid}", ignore_errors=True)
    n = [k for k in FEAT if k[0] == s]
    log(sid, "recordings", len(n), "windows", sum(len(FEAT[k]["t"]) for k in n), "beh files", len([k for k in BEH if k[0] == s]))
    if s == 1:   # print structure once, so the behaviour parser can be written correctly
        for k in sorted(k for k in BEH if k[0] == 1)[:4]:
            v = BEH[k]; print("BEH", k, type(v).__name__, (list(v.keys()) if isinstance(v, dict) else str(v)[:200]))
            if isinstance(v, dict):
                for kk, vv in v.items(): print("   ", kk, type(vv).__name__, getattr(vv, "shape", None), str(vv)[:300])
        for k in sorted(k for k in EVT if k[0] == 1)[:3]:
            from collections import Counter; print("EVT", k, INFO[k], Counter(d for _, d in EVT[k]).most_common(12))
pickle.dump(dict(FEAT=FEAT, BEH=BEH, EVT=EVT, INFO=INFO), open(f"{OUT}/cog_features.pkl", "wb"), protocol=4)
log("saved", round(os.path.getsize(f"{OUT}/cog_features.pkl") / 1e6, 1), "MB", "people", len({k[0] for k in FEAT}))
