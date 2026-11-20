# Paper 8 deep-learning check (rules: pilot_DEEP_prereg.md): EEGNet on COG-BCI raw EEG -> 1-min unit scores -> RB-ME.
import os, sys, glob, zipfile, subprocess, shutil, time, json, numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
TASKS = {"zeroBACK": ("nback", 0), "oneBACK": ("nback", 1), "twoBACK": ("nback", 2),
         "MATBeasy": ("matb", 0), "MATBmed": ("matb", 1), "MATBdiff": ("matb", 2)}
CH18 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
FS = 128; WIN = 2 * FS

def export():
    import mne; mne.set_log_level("ERROR")
    W = "/kaggle/tmp/cog"; os.makedirs(W, exist_ok=True); D = {"nback": [], "matb": []}
    for s in range(1, 30):
        sid = f"sub-{s:02d}"; z = f"{W}/{sid}.zip"
        try:
            subprocess.run(["wget", "-q", "-O", z, f"https://zenodo.org/records/7413650/files/{sid}.zip?download=1"], check=True)
            zipfile.ZipFile(z).extractall(f"{W}/{sid}"); os.remove(z)
        except Exception as e:
            log(sid, "download failed", repr(e)[:200]); continue
        recs = []
        for ses in sorted(glob.glob(f"{W}/{sid}/**/ses-S*", recursive=True)):
            for task, (fam, lev) in TASKS.items():
                f = glob.glob(f"{ses}/eeg/{task}.set")
                if not f: continue
                try:
                    raw = mne.io.read_raw_eeglab(f[0], preload=True)
                    if any(c not in raw.ch_names for c in CH18): continue
                    raw.pick(CH18); raw.filter(1, 40, verbose=False); raw.resample(FS, verbose=False); X = raw.get_data().astype(np.float32)
                    n = X.shape[1] // WIN; Wd = X[:, :n * WIN].reshape(18, n, WIN).transpose(1, 0, 2)
                    unit = (np.arange(n) * 2) // 60
                    recs.append((fam, lev, os.path.basename(ses), task, Wd, unit))
                except Exception as e:
                    log(sid, task, "failed", repr(e)[:150])
        if recs:
            A = np.concatenate([r[4] for r in recs]); mu, sd = A.mean((0, 2), keepdims=True), A.std((0, 2), keepdims=True) + 1e-9
            for fam, lev, ses, task, Wd, unit in recs:
                D[fam].append(dict(s=s, ses=ses, task=task, level=lev, X=((Wd - mu) / sd).astype(np.float16), unit=unit))
        shutil.rmtree(f"{W}/{sid}", ignore_errors=True); log(sid, "recordings", len(recs))
    return D

def eegnet(C=18, T=WIN, F1=8, Dm=2, F2=16, ncls=3):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(F1),
        nn.Conv2d(F1, F1 * Dm, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * Dm), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(.5),
        nn.Conv2d(F1 * Dm, F1 * Dm, (1, 16), padding=(0, 8), groups=F1 * Dm, bias=False), nn.Conv2d(F1 * Dm, F2, 1, bias=False),
        nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(.5), nn.Flatten(), nn.LazyLinear(ncls))

def train_predict(Xtr, ytr, Xte, dev):
    import torch
    torch.manual_seed(0); np.random.seed(0); net = eegnet().to(dev)
    net(torch.zeros(2, 1, 18, WIN, device=dev)); opt = torch.optim.Adam(net.parameters(), 1e-3); lossf = torch.nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float16); yt = torch.tensor(np.asarray(ytr, dtype=np.int64))
    for ep in range(15):
        net.train(); perm = torch.randperm(len(yt))
        for b in range(0, len(yt), 256):
            i = perm[b:b + 256]; xb = Xt[i].float().unsqueeze(1).to(dev); yb = yt[i].to(dev)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(Xte), 1024):
            p = torch.softmax(net(torch.tensor(Xte[b:b + 1024]).float().unsqueeze(1).to(dev)), 1).cpu().numpy(); out.append(p @ np.arange(3))
    return np.concatenate(out)

def loso_beh(s, y, X):
    out = np.zeros(len(y))
    for te in np.unique(s):
        tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr])
        out[ts] = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[ts])) @ np.arange(3)
    return out
def auc(x, i, j): return np.mean((x[i] > x[j]) + 0.5 * (x[i] == x[j]))
def resid(x, s, y):
    r = x.copy()
    for p in np.unique(s):
        for l in np.unique(y):
            m = (s == p) & (y == l); r[m] -= r[m].mean() if m.any() else 0
    return r
def rbme(s, y, sb, A, B, cal=0.15):
    sA, sB = loso_beh(s, y, A), loso_beh(s, y, B); rel = max(np.corrcoef(resid(sA, s, y), resid(sB, s, y))[0, 1], .05); st = []
    for p in np.unique(s):
        m = s == p; L = y[m]; i, j = np.where(L[:, None] > L[None, :]); vals = []
        for mt, ts in ((sA[m], sB[m]), (sB[m], sA[m])):
            k = np.abs(mt[i] - mt[j]) <= cal
            if k.sum() < 20: vals = None; break
            ab = np.clip(auc(ts, i[k], j[k]), 1e-3, 1 - 1e-3); vals.append(auc(sb[m], i[k], j[k]) - norm.cdf(norm.ppf(ab) / np.sqrt(rel)))
        if vals: st.append(np.mean(vals))
    st = np.array(st); return dict(n=len(st), rel=float(rel), diff=float(st.mean()), p=float(wilcoxon(st, alternative="greater").pvalue))
def bacc(y, p): return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))

if __name__ == "__main__" or True:
    import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"; log("device", dev)
    D = export()
    g = lambda pat: glob.glob(f"/kaggle/input/**/{pat}", recursive=True)[0]
    AB = dict(np.load(g("ab_halves.npz"), allow_pickle=True)); KEYS = dict(np.load(g("unit_keys.npz"), allow_pickle=True)); res = {}
    for fam, S in (("matb", "S2"), ("nback", "S1")):
        R = D[fam]; people = sorted({r["s"] for r in R}); unit_score = {}
        for te in people:
            Xtr = np.concatenate([r["X"] for r in R if r["s"] != te]); ytr = np.concatenate([np.full(len(r["X"]), r["level"]) for r in R if r["s"] != te])
            te_recs = [r for r in R if r["s"] == te]; Xte = np.concatenate([r["X"] for r in te_recs]); pred = train_predict(Xtr, ytr, Xte, dev); o = 0
            for r in te_recs:
                pr = pred[o:o + len(r["X"])]; o += len(r["X"])
                for u in np.unique(r["unit"]): unit_score[f"{r['s']}|{r['ses']}|{r['task']}|{u}"] = float(pr[r["unit"] == u].mean())
            log(fam, "person", te, "done")
        keys = KEYS[S + "_key"]; ok = np.array([k in unit_score for k in keys]); log(fam, "units matched", ok.sum(), "of", len(keys))
        s = AB[S + "_s"].astype(str)[ok]; y = AB[S + "_y"].astype(int)[ok]; sb = np.array([unit_score[k] for k in keys[ok]])
        A, B = AB[S + "_A"][ok].astype(float), AB[S + "_B"][ok].astype(float)
        pred_lv = np.clip(np.rint(sb), 0, 2).astype(int)
        r = rbme(s, y, sb, A, B); r["bacc"] = np.mean([bacc(y[s == p], pred_lv[s == p]) for p in np.unique(s)]); res[fam] = r; log(fam, "EEGNet", r)
    D0 = all(res[f]["bacc"] > .40 for f in res); D1 = res["matb"]["diff"] > 0 and res["matb"]["p"] < .01
    print("D0", D0, "| D1", D1, "| PASS", D0 and D1, "| n-back (reported)", res["nback"]); json.dump(res, open("deep_results.json", "w"), indent=1)
