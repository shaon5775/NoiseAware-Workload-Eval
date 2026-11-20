# Paper 8 artefact check (rules: pilot_ARTEFACT_prereg.md). COG-BCI MATB: ICA eye/muscle removal, then RB-ME variants.
# The split-half behaviour and unit keys (from ab_halves.npz / unit_keys.npz, MATB only) are embedded as EMB below.
import os, io, glob, zipfile, subprocess, shutil, time, json, base64, numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
try:
    import mne
except ImportError:
    subprocess.run(["pip", "install", "-q", "mne"], check=True); import mne
mne.set_log_level("ERROR")
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
QUICK = os.environ.get("QUICK") == "1"
TASKS = {"MATBeasy": 0, "MATBmed": 1, "MATBdiff": 2}
BANDS = ((1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40))
CH18 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
CH12 = ["F3", "Fz", "F4", "C3", "C4", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
I12 = [CH18.index(c) for c in CH12]

def feats(X, fs):
    """X: samples x channels (all EEG); returns per 10-s window (5-s step) features."""
    win, st = int(10 * fs), int(5 * fs); sos = butter(4, [1, 20], btype="band", fs=fs, output="sos")
    out = dict(bpall=[], bp18=[], cov18=[], cov12lo=[], t=[])
    return win, st, sos, out

def window_feats(X, X18, fs):
    win, st, sos, out = feats(X, fs); X12lo = sosfiltfilt(sos, X18[:, I12], axis=0)
    for a in range(0, len(X) - win + 1, st):
        w = X[a:a + win]; fr, P = welch(w, fs=fs, nperseg=int(2 * fs), axis=0)
        lp = np.stack([np.log(P[(fr >= lo) & (fr < hi)].mean(0) + 1e-20) for lo, hi in BANDS])     # 7 x ch
        out["bpall"].append(lp.mean(1)); w18 = X18[a:a + win]
        fr, P = welch(w18, fs=fs, nperseg=int(2 * fs), axis=0)
        out["bp18"].append(np.stack([np.log(P[(fr >= lo) & (fr < hi)].mean(0) + 1e-20) for lo, hi in BANDS]))
        out["cov18"].append(np.cov(w18[::5].T)); out["cov12lo"].append(np.cov(X12lo[a:a + win][::5].T)); out["t"].append(a / fs)
    return {k: np.array(v, np.float32) for k, v in out.items()}

MUSCLE_FAIL = [0]
def process(raw):
    try: raw.set_montage("standard_1005", on_missing="ignore")
    except Exception: pass
    raw.filter(1, 40, verbose=False); fs = raw.info["sfreq"]
    base = window_feats(raw.get_data().T, raw.copy().pick(CH18).get_data().T, fs)
    ica = mne.preprocessing.ICA(n_components=20, method="fastica", random_state=0, max_iter=500)
    ica.fit(raw, decim=4)
    eog, _ = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"])
    try: mus, _ = ica.find_bads_muscle(raw, h_freq=40)
    except Exception as e:
        mus = []; MUSCLE_FAIL[0] += 1
        if MUSCLE_FAIL[0] <= 3: log("muscle detection failed", repr(e)[:160])
    ica.exclude = sorted(set(eog) | set(mus)); clean = ica.apply(raw.copy())
    cl = window_feats(clean.get_data().T, clean.copy().pick(CH18).get_data().T, fs)
    return base, cl, (len(eog), len(mus), len(ica.exclude))

def process_file(sn, task, fp):
    try:
        import mne; mne.set_log_level("ERROR")
        raw = mne.io.read_raw_eeglab(fp, preload=True); raw.pick("eeg")
        if any(c not in raw.ch_names for c in CH18): return sn, task, "missing channels"
        return sn, task, process(raw)
    except Exception as e:
        return sn, task, "failed " + repr(e)[:200]

def export():
    W = "/kaggle/tmp/cog"; os.makedirs(W, exist_ok=True); R = {}; ninfo = {}
    for s in (range(1, 3) if QUICK else range(1, 30)):
        sid = f"sub-{s:02d}"; z = f"{W}/{sid}.zip"
        try:
            subprocess.run(["wget", "-q", "-O", z, f"https://zenodo.org/records/7413650/files/{sid}.zip?download=1"], check=True)
            zipfile.ZipFile(z).extractall(f"{W}/{sid}"); os.remove(z)
        except Exception as e:
            log(sid, "download failed", repr(e)[:200]); continue
        jobs = []
        for ses in sorted(glob.glob(f"{W}/{sid}/**/ses-S*", recursive=True)):
            for task in TASKS:
                f = glob.glob(f"{ses}/eeg/{task}.set")
                if f: jobs.append((os.path.basename(ses), task, f[0]))
        for sn, task, out in Parallel(n_jobs=-1)(delayed(process_file)(sn, task, fp) for sn, task, fp in jobs):
            if isinstance(out, str): log(sid, sn, task, out); continue
            base, cl, cnt = out; R[(s, sn, task)] = dict(raw=base, ica=cl); ninfo[f"{s}|{sn}|{task}"] = cnt
        shutil.rmtree(f"{W}/{sid}", ignore_errors=True); log(sid, "recordings", sum(1 for k in R if k[0] == s))
    return R, ninfo

def logm(C): w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)
def isqrtm(C): w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T
def zs(X): return (X - X.mean(0)) / (X.std(0) + 1e-9)
def tangent(Cs):
    Cs = Cs + 1e-6 * np.eye(Cs.shape[1]) * np.trace(Cs, axis1=1, axis2=2).mean() / Cs.shape[1]
    Rm = isqrtm(Cs.mean(0)); iu = np.triu_indices(Cs.shape[1]); return zs(logm(Rm @ Cs @ Rm)[:, iu[0], iu[1]])

def unit_table(R, keys):
    """Average window features per 1-min unit (units with >= 6 windows), in the order of keys."""
    U = {}
    for (s, sn, task), d in R.items():
        for ver in ("raw", "ica"):
            f = d[ver]; u = (f["t"] // 60).astype(int)
            for k in range(u.max() + 1):
                m = u == k
                if m.sum() < 6: continue
                U.setdefault(f"{s}|{sn}|{task}|{k}", {})[ver] = {n: f[n][m].mean(0) for n in ("bpall", "bp18", "cov18", "cov12lo")}
    ok = np.array([k in U and len(U[k]) == 2 for k in keys]); return U, ok

def brain_sets(U, keys, s):
    """Per-person normalised feature sets (as in bvb_cog.build)."""
    out = {}
    def per_person(fn):
        X = None
        for p in np.unique(s):
            m = s == p; Xp = fn(m)
            if X is None: X = np.zeros((len(s), Xp.shape[1]))
            X[m] = Xp
        return X
    get = lambda ver, n, m: np.stack([U[k][ver][n] for k in keys[m]])
    for ver in ("raw", "ica"):
        out[f"V_{ver}_orig"] = per_person(lambda m: np.c_[zs(get(ver, "bpall", m)), tangent(get(ver, "cov18", m))])
    out["V2_ica_central_lo"] = per_person(lambda m: np.c_[zs(get("ica", "bp18", m)[:, :5][:, :, I12].reshape(m.sum(), -1)), tangent(get("ica", "cov12lo", m))])
    REG = {"frontal": ["F3", "Fz", "F4"], "central": ["C3", "C4"], "parietal": ["P3", "Pz", "P4", "P7", "P8"], "occipital": ["O1", "O2"],
           "frontopolar+lateral(artefact-prone)": ["Fp1", "Fp2", "F7", "F8", "T7", "T8"]}
    BND = {"delta 1-4": [0], "theta 4-8": [1], "alpha 8-13": [2, 3], "beta 13-30": [4, 5], "gamma 30-40": [6]}
    for nm, bi in BND.items(): out["band_" + nm] = per_person(lambda m, bi=bi: zs(get("ica", "bp18", m)[:, bi, :].reshape(m.sum(), -1)))
    for nm, ch in REG.items():
        ci = [CH18.index(c) for c in ch]; out["region_" + nm] = per_person(lambda m, ci=ci: zs(get("ica", "bp18", m)[:, :, ci].reshape(m.sum(), -1)))
    out["bp18_all_ica"] = per_person(lambda m: zs(get("ica", "bp18", m).reshape(m.sum(), -1)))
    return out

def loso(s, y, X, C):
    def fold(te):
        tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X[tr]), y[tr]); return ts, clf.predict_proba(sc.transform(X[ts])) @ clf.classes_
    out = np.zeros(len(y))
    for ts, v in Parallel(n_jobs=-1)(delayed(fold)(te) for te in np.unique(s)): out[ts] = v
    return out
def resid(x, s, y):
    r = x.copy()
    for p in np.unique(s):
        for l in np.unique(y):
            m = (s == p) & (y == l)
            if m.any(): r[m] -= r[m].mean()
    return r
def auc(x, i, j): return np.mean((x[i] > x[j]) + 0.5 * (x[i] == x[j]))
def rbme(s, y, sb, sA, sB, CAL=0.15):
    rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], 0.05); stat, ba, bd = [], [], []
    for p in np.unique(s):
        m = s == p; L = y[m]; b, a_, b_ = sb[m], sA[m], sB[m]; i, j = np.where(L[:, None] > L[None, :]); vals = []
        for mt, tst in ((a_, b_), (b_, a_)):
            k = np.abs(mt[i] - mt[j]) <= CAL
            if k.sum() < 20: vals = None; break
            ab = np.clip(auc(tst, i[k], j[k]), 1e-3, 1 - 1e-3); vals.append((auc(b, i[k], j[k]), norm.cdf(norm.ppf(ab) / np.sqrt(rel))))
        if vals: v = np.mean(vals, 0); ba.append(v[0]); bd.append(v[1]); stat.append(v[0] - v[1])
    stat = np.array(stat); p = wilcoxon(stat, alternative="greater").pvalue if len(stat) > 5 else np.nan
    return dict(n=len(stat), rel=float(rel), brain=float(np.mean(ba)), bound=float(np.mean(bd)), diff=float(stat.mean()), p=float(p))
def bacc(y, p): return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))

if __name__ == "__main__" or True:
    E = dict(np.load(io.BytesIO(base64.b64decode(EMB)), allow_pickle=True))
    R, ninfo = export(); keys = E["key"].astype(str); U, ok = unit_table(R, keys)
    log("units with both versions", ok.sum(), "of", len(keys))
    keys, s, y, A, B = keys[ok], E["s"].astype(str)[ok], E["y"].astype(int)[ok], E["A"][ok].astype(float), E["B"][ok].astype(float)
    sA, sB = loso(s, y, A, 1.0), loso(s, y, B, 1.0); res = {"n_units": int(ok.sum()), "ica_removed": ninfo}
    ex = np.array([v[2] for v in ninfo.values()]); res["ica_removed_summary"] = dict(mean=float(ex.mean()), min=int(ex.min()), max=int(ex.max()),
        eog_mean=float(np.mean([v[0] for v in ninfo.values()])), muscle_mean=float(np.mean([v[1] for v in ninfo.values()])))
    log("ICA removed", res["ica_removed_summary"])
    for nm, X in brain_sets(U, keys, s).items():
        sb = loso(s, y, X, 0.05); r = rbme(s, y, sb, sA, sB)
        r["bacc"] = float(np.mean([bacc(y[s == p], np.clip(np.rint(sb[s == p]), 0, 2)) for p in np.unique(s)])); res[nm] = r
        log(nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    A1 = res["V_ica_orig"]["diff"] > 0 and res["V_ica_orig"]["p"] < .01; A2 = res["V2_ica_central_lo"]["diff"] > 0 and res["V2_ica_central_lo"]["p"] < .01
    S0 = abs(res["V_raw_orig"]["diff"] - 0.305) <= 0.05
    print("SANITY V0 within 0.05 of +0.305:", S0, "| A1", A1, "| A2", A2, flush=True)
    json.dump(res, open("/kaggle/working/artefact_results.json", "w"), indent=1)
    print("RESULTS_JSON", json.dumps({k: v for k, v in res.items() if k != "ica_removed"}))
