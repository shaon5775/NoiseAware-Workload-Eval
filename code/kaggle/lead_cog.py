# Paper 8 LEAD (rules: pilot_LEAD_prereg.md): neural lead time + anytime fusion on COG-BCI (Zenodo 7413650, CC-BY-4.0).
# Part 1 export (download one person -> onset-locked 2-s window features + behaviour(t) -> delete). Part 2 analysis.
import os, sys, subprocess, glob, zipfile, time, pickle, shutil, numpy as np
from scipy.signal import welch
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
TASKS = {"zeroBACK": ("nback", 0), "oneBACK": ("nback", 1), "twoBACK": ("nback", 2),
         "MATBeasy": ("matb", 0), "MATBmed": ("matb", 1), "MATBdiff": ("matb", 2)}
BANDS = ((1, 4), (4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40))
CH18 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
TS = [2, 4, 6, 8, 10, 15, 20, 30, 45, 60]

def onset_of(ev, task):
    if "BACK" in task:
        o = [o for o, d in ev if len(d) == 4 and d[2] == "2"]
    else:
        o = [o for o, d in ev if len(d.split()) >= 5 and d.split()[1] == "TRACKING"]
    return min(o) if o else None

def beh_at(ev, task, on, t):
    """Behaviour using only [on, on + t]. n-back: hit, FA, RT, n stimuli; MATB: track err mean, sd, tank dev."""
    if "BACK" in task:
        stim = [(o, d[-1]) for o, d in ev if len(d) == 4 and d[2] == "2" and on <= o <= on + t - 2.0]
        ro = np.array([o for o, d in ev if len(d) == 4 and d[2] == "3"])
        def hit(o): r = ro[(ro > o) & (ro < o + 2.0)] if len(ro) else ro; return r[0] - o if len(r) else None
        h = [hit(o) for o, x in stim if x == "2"]; fa = [hit(o) for o, x in stim if x == "1"]
        rt = [x for x in h if x is not None]
        return np.array([np.mean([x is not None for x in h]) if h else np.nan,
                         np.mean([x is not None for x in fa]) if fa else np.nan,
                         np.mean(rt) if rt else np.nan, float(len(stim))])
    tr, rm = [], []
    for o, d in ev:
        if not (on <= o <= on + t): continue
        p = d.split()
        if len(p) >= 5 and p[1] == "TRACKING": tr.append(np.hypot(float(p[3]), float(p[4])))
        if len(p) >= 7 and p[1] == "RESMAN": rm.append(abs(float(p[3]) - 2500) + abs(float(p[5]) - 2500))
    return np.array([np.mean(tr) if tr else np.nan, np.std(tr) if tr else np.nan, np.mean(rm) if rm else np.nan])

def export():
    import mne; mne.set_log_level("ERROR")
    W = "/kaggle/tmp/cog"; os.makedirs(W, exist_ok=True); REC = []
    for s in range(1, 30):
        sid = f"sub-{s:02d}"; z = f"{W}/{sid}.zip"
        try:
            subprocess.run(["wget", "-q", "-O", z, f"https://zenodo.org/records/7413650/files/{sid}.zip?download=1"], check=True)
            zipfile.ZipFile(z).extractall(f"{W}/{sid}"); os.remove(z)
        except Exception as e:
            log(sid, "download failed", repr(e)[:200]); continue
        for ses in sorted(glob.glob(f"{W}/{sid}/**/ses-S*", recursive=True)):
            for task, (fam, lev) in TASKS.items():
                f = glob.glob(f"{ses}/eeg/{task}.set")
                if not f: continue
                try:
                    raw = mne.io.read_raw_eeglab(f[0], preload=True)
                    ev = [(float(o), str(d)) for o, d in zip(raw.annotations.onset, raw.annotations.description)]
                    on = onset_of(ev, task)
                    if on is None or any(c not in raw.ch_names for c in CH18): log(sid, task, "no onset/channels"); continue
                    raw.pick(CH18); raw.filter(1, 40, verbose=False); fs = raw.info["sfreq"]; X = raw.get_data().T
                    bp, ends = [], []
                    for k in range(0, 59):
                        a = int((on + k) * fs); b = a + int(2 * fs)
                        if b > len(X): break
                        fr, P = welch(X[a:b], fs=fs, nperseg=int(fs), axis=0)
                        bp.append(np.concatenate([np.log(P[(fr >= lo) & (fr < hi)].mean(0) + 1e-20) for lo, hi in BANDS])); ends.append(k + 2)
                    if len(ends) < 59: log(sid, task, "short recording", len(ends)); continue
                    REC.append(dict(s=s, ses=os.path.basename(ses), task=task, fam=fam, level=lev, bp=np.array(bp, np.float32),
                                    ends=np.array(ends), beh=np.stack([beh_at(ev, task, on, t) for t in TS]), behfull=beh_at(ev, task, on, 1e9)))
                except Exception as e:
                    log(sid, task, "failed", repr(e)[:150])
        shutil.rmtree(f"{W}/{sid}", ignore_errors=True)
        log(sid, "recordings so far", len(REC))
        if s == 1:
            for r in REC[:6]: print(r["task"], r["beh"][[1, 4, 9]].round(3).tolist(), r["behfull"].round(3).tolist())
    pickle.dump(REC, open("/kaggle/working/lead_cog.pkl", "wb"), protocol=4); return REC

def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])

def analyse(REC):
    # per-person z of brain windows (all that person's windows), per-person z of behaviour rates (full-recording values)
    for s in {r["s"] for r in REC}:
        R = [r for r in REC if r["s"] == s]; A = np.vstack([r["bp"] for r in R]); mu, sd = A.mean(0), A.std(0) + 1e-9
        for r in R: r["bpz"] = (r["bp"] - mu) / sd
        for fam in ("nback", "matb"):
            Rf = [r for r in R if r["fam"] == fam]
            if not Rf: continue
            F = np.vstack([r["behfull"] for r in Rf]); k = 3; m, d = np.nanmean(F[:, :k], 0), np.nanstd(F[:, :k], 0) + 1e-9
            for r in Rf: r["behz"] = np.c_[(r["beh"][:, :k] - m) / d, r["beh"][:, k:]]
    out = {}
    for fam in ("nback", "matb"):
        R = [r for r in REC if r["fam"] == fam]; s = np.array([r["s"] for r in R]); y = np.array([r["level"] for r in R])
        res = {k: np.full((len(TS), len(np.unique(s))), np.nan) for k in ("BRAIN", "BEH", "FUSED")}; people = np.unique(s)
        for ti, t in enumerate(TS):
            Xb = np.stack([r["bpz"][r["ends"] <= t].mean(0) for r in R]); Xh = np.stack([r["behz"][ti] for r in R])
            for pi, te in enumerate(people):
                tr, ts = s != te, s == te
                Xh_ = Xh.copy(); mu = np.nanmean(Xh_[tr], 0); mu = np.where(np.isnan(mu), 0, mu)
                Xh_ = np.where(np.isnan(Xh_), mu, Xh_)
                for k, X, C in (("BRAIN", Xb, .05), ("BEH", Xh_, 1.0), ("FUSED", np.c_[Xb, Xh_], .05)):
                    sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X[tr]), y[tr])
                    res[k][ti, pi] = bacc(y[ts], clf.predict(sc.transform(X[ts])))
            log(fam, f"t={t:2d}s", " ".join(f"{k} {np.nanmean(v[ti]):.3f}" for k, v in res.items()))
        G = {k: np.nanmean(v, 1) for k, v in res.items()}
        cross = [t for ti, t in enumerate(TS) if G["BEH"][ti] >= G["BRAIN"][ti]]; lead = cross[0] if cross else 999
        d = res["BRAIN"][1] - res["BEH"][1]
        try: p = wilcoxon(d, alternative="greater").pvalue
        except ValueError: p = 1.0
        early = [ti for ti, t in enumerate(TS) if t <= 20]
        L1 = bool(p < 0.01); L2 = lead >= 10
        L3 = bool(G["FUSED"][early].mean() >= G["BEH"][early].mean() + 0.03 and all(G["FUSED"] >= np.maximum(G["BRAIN"], G["BEH"]) - 0.02))
        out[fam] = dict(curves={k: v.round(4).tolist() for k, v in G.items()}, lead_time=lead, p_t4=float(p), L1=L1, L2=L2, L3=L3, GO=L1 and L2 and L3)
        log(fam, "lead time", lead, "| t=4 brain-beh", round(float(np.nanmean(d)), 3), "p", round(float(p), 5), "| L1", L1, "L2", L2, "L3", L3)
    print("GO (any family)", any(v["GO"] for v in out.values()))
    import json; json.dump(dict(TS=TS, res=out), open("lead_results.json", "w"), indent=1)

if __name__ == "__main__":
    if "--test" in sys.argv:          # tiny synthetic run of the analysis only
        rng = np.random.default_rng(0); REC = []
        for s in range(6):
            for ses in range(3):
                for task, (fam, lev) in TASKS.items():
                    nb = 4 if fam == "nback" else 3
                    REC.append(dict(s=s, ses=ses, task=task, fam=fam, level=lev, bp=(rng.normal(size=(59, 126)) + .3 * lev).astype(np.float32),
                                    ends=np.arange(2, 61), beh=rng.normal(size=(len(TS), nb)) + lev * np.linspace(0, 1, len(TS))[:, None],
                                    behfull=rng.normal(size=nb) + lev))
        analyse(REC)
    else:
        analyse(export())
