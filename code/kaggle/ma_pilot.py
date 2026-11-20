# Paper 8 pilot M-A "EEG-taught fNIRS" (lag-aware privileged distillation). Rules: pilot_MA_prereg.md (fixed).
import os, subprocess, glob, zipfile, time, json, numpy as np, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, welch
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
D = "/kaggle/tmp/shin"; os.makedirs(D, exist_ok=True)
base = "http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS"
for kind, f in [("NIRS", "NIRS_01-26_MATLAB.zip"), ("EEG", "EEG_01-26_MATLAB.zip")]:
    if not glob.glob(f"{D}/{kind}/*"):
        p = f"{D}/{f}"; subprocess.run(["wget", "-q", "-O", p, f"{base}/{kind}/{f}"], check=True)
        zipfile.ZipFile(p).extractall(f"{D}/{kind}"); os.remove(p)
log("data ready")

def L(path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    return [v for k, v in m.items() if not k.startswith("__")][0]
def tree(o, name="", d=0):
    if d > 3: return
    if hasattr(o, "_fieldnames"):
        for fn in o._fieldnames: tree(getattr(o, fn), f"{name}.{fn}", d + 1)
    elif isinstance(o, np.ndarray) and o.dtype == object and o.size <= 6:
        for i, x in enumerate(o.flat): tree(x, f"{name}[{i}]", d + 1)
    else:
        s = getattr(o, "shape", None); small = np.size(o) <= 12
        print(f"   {name}: {type(o).__name__} {s} {str(o)[:160] if small else ''}")
for kind in ("NIRS", "EEG"):
    for what in ("cnt", "mrk"):
        p = glob.glob(f"{D}/{kind}/VP001-{kind}/{what}_nback.mat")[0]; print(f"\n## {kind} {what}"); tree(L(p), what)
mrk0 = L(glob.glob(f"{D}/NIRS/VP001-NIRS/mrk_nback.mat")[0])
for fn in ("className",):
    if hasattr(mrk0, fn): print("className:", getattr(mrk0, fn))
if hasattr(mrk0, "event"): tree(mrk0.event, "event")

def as_list(o): return list(o.flat) if isinstance(o, np.ndarray) and o.dtype == object else [o]

def blocks(mrk):
    """Return list of (start_s, end_s, level) for 0/2/3-back series."""
    out = []
    for m in as_list(mrk):
        t = np.atleast_1d(m.time).astype(float) / 1000.0
        y = np.atleast_2d(m.y); y = y if y.shape[1] == len(t) else y.T
        cn = [str(c) for c in np.atleast_1d(m.className)]
        lev = np.full(len(t), -1)
        for ci, c in enumerate(cn):
            for l in (0, 2, 3):
                if c.strip().startswith(f"{l}-back"): lev[y[ci] > 0] = l
        k = lev >= 0; t, lev = t[k], lev[k]; o = np.argsort(t); t, lev = t[o], lev[o]
        i = 0
        while i < len(t):
            j = i
            while j + 1 < len(t) and lev[j + 1] == lev[i] and t[j + 1] - t[j] < 5: j += 1
            end = t[j] + 2 if j > i else t[i] + 40
            out.append((t[i], end, int(lev[i]))); i = j + 1
    return out

def get_cnt(c, kind):
    cs = as_list(c); res = []
    for c in cs:
        if kind == "NIRS":
            res.append((np.c_[np.asarray(c.oxy.x, float), np.asarray(c.deoxy.x, float)], float(c.oxy.fs)))
        else:
            cl = [str(x) for x in np.atleast_1d(c.clab)]; keep = [i for i, n in enumerate(cl) if "EOG" not in n.upper()]
            res.append((np.asarray(c.x, float)[:, keep], float(c.fs)))
    return res

WIN, STR, LAGS = 10.0, 2.0, (0, 2, 4, 6, 8)
def person(vp):
    nc = get_cnt(L(f"{D}/NIRS/VP{vp:03d}-NIRS/cnt_nback.mat"), "NIRS"); nb = blocks(L(f"{D}/NIRS/VP{vp:03d}-NIRS/mrk_nback.mat"))
    ec = get_cnt(L(f"{D}/EEG/VP{vp:03d}-EEG/cnt_nback.mat"), "EEG"); eb = blocks(L(f"{D}/EEG/VP{vp:03d}-EEG/mrk_nback.mat"))
    assert len(nc) == len(ec) and len(nb) == len(eb), (vp, len(nc), len(ec), len(nb), len(eb))
    # if several sessions are separate arrays, markers restart per session: split blocks by session in order
    def split(bl, n):
        if n == 1: return [bl]
        segs, cur = [], [bl[0]]
        for b in bl[1:]:
            if b[0] < cur[-1][0]:
                segs.append(cur); cur = [b]
            else:
                cur.append(b)
        segs.append(cur); assert len(segs) == n, (len(segs), n); return segs
    NB, EB = split(nb, len(nc)), split(eb, len(ec))
    FN, FE, Y = [], {l: [] for l in LAGS}, []
    for (xn, fsn), (xe, fse), bn, be in zip(nc, ec, NB, EB):
        sos = butter(3, [0.01, 0.1], btype="band", fs=fsn, output="sos"); xn = sosfiltfilt(sos, xn, axis=0)
        sose = butter(4, [1, 40], btype="band", fs=fse, output="sos"); xe = sosfiltfilt(sose, xe, axis=0)
        for (sn, en, lv), (se, ee, lv2) in zip(bn, be):
            assert lv == lv2
            for off in np.arange(0, (en - sn) - WIN + 1e-9, STR):
                a = int((sn + off) * fsn); w = xn[a:a + int(WIN * fsn)]
                if len(w) < int(WIN * fsn): continue
                tt = np.arange(len(w)) - (len(w) - 1) / 2
                FN.append(np.r_[w.mean(0), w.std(0), (tt[:, None] * (w - w.mean(0))).sum(0) / (tt ** 2).sum()])
                for lg in LAGS:
                    b0 = max(0, int((se + off - lg) * fse)); we = xe[b0:b0 + int(WIN * fse)]
                    f, P = welch(we, fs=fse, nperseg=int(2 * fse), axis=0)
                    bp = [np.log(P[(f >= lo) & (f < hi)].mean(0) + 1e-12) for lo, hi in ((4, 8), (8, 13), (13, 30))]
                    FE[lg].append(np.concatenate(bp))
                Y.append(lv)
    z = lambda A: (np.array(A) - np.mean(A, 0)) / (np.std(A, 0) + 1e-9)     # per-person, unlabelled
    return z(FN), {l: z(FE[l]) for l in LAGS}, np.array([{0: 0, 2: 1, 3: 2}[v] for v in Y])

DATA = {}
for vp in range(1, 27):
    try: DATA[vp] = person(vp)
    except Exception as e: log("person", vp, "FAILED:", repr(e)[:300])
log("people loaded:", len(DATA), "windows/person:", sorted({len(v[2]) for v in DATA.values()}),
    "fNIRS feats:", next(iter(DATA.values()))[0].shape[1], "EEG feats:", next(iter(DATA.values()))[1][0].shape[1],
    "class counts VP1:", np.bincount(next(iter(DATA.values()))[2]).tolist())

import torch, torch.nn as nn, torch.nn.functional as Fn
torch.set_num_threads(4)
def train(X, Y=None, soft=None, seed=0, epochs=60):
    torch.manual_seed(seed); X = torch.tensor(X, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    Yt = torch.tensor(Y); St = None if soft is None else torch.tensor(soft, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            k = perm[i:i + 256]; out = net(X[k]); loss = Fn.cross_entropy(out, Yt[k])
            if St is not None:
                loss = 0.5 * loss + 0.5 * 4 * Fn.kl_div(Fn.log_softmax(out / 2, 1), St[k], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return lambda Z: torch.softmax(net(torch.tensor(Z, dtype=torch.float32)), 1).detach().numpy()
def soften(p, T=2.0): q = np.exp(np.log(p + 1e-9) / T); return q / q.sum(1, keepdims=True)
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])

people = sorted(DATA); res = {m: {} for m in ("B1", "B2", "B3", "D0", "Dlag")}; chosen = {}
for seed in range(3):
    for te in people:
        tr = [p for p in people if p != te]
        XN = {p: DATA[p][0] for p in tr}; Y = {p: DATA[p][2] for p in tr}
        cat = lambda d, ps: np.vstack([d[p] for p in ps]); caty = lambda ps: np.concatenate([Y[p] for p in ps])
        yte = DATA[te][2]
        f = train(cat(XN, tr), caty(tr), seed=seed); res["B1"].setdefault(te, []).append(bacc(yte, f(DATA[te][0]).argmax(1)))
        f = train(np.vstack([DATA[p][1][0] for p in tr]), caty(tr), seed=seed); res["B2"].setdefault(te, []).append(bacc(yte, f(DATA[te][1][0]).argmax(1)))
        f = train(np.vstack([np.c_[DATA[p][0], DATA[p][1][0]] for p in tr]), caty(tr), seed=seed)
        res["B3"].setdefault(te, []).append(bacc(yte, f(np.c_[DATA[te][0], DATA[te][1][0]]).argmax(1)))
        # out-of-fold teacher (EEG) probs at every lag, and out-of-fold student (fNIRS) probs, inner 5 folds over train people
        folds = np.array_split(np.random.RandomState(seed).permutation(tr), 5)
        TP = {p: {} for p in tr}; SP = {}
        for fo in folds:
            rest = [p for p in tr if p not in fo]
            ft = train(np.vstack([DATA[p][1][0] for p in rest]), caty(rest), seed=seed)   # teacher trained at lag 0 features
            fs_ = train(cat(XN, rest), caty(rest), seed=seed)
            for p in fo:
                for lg in LAGS: TP[p][lg] = ft(DATA[p][1][lg])
                SP[p] = fs_(XN[p])
        # D0: same-time teacher
        f = train(cat(XN, tr), caty(tr), soft=np.vstack([soften(TP[p][0]) for p in tr]), seed=seed)
        res["D0"].setdefault(te, []).append(bacc(yte, f(DATA[te][0]).argmax(1)))
        # D-lag: per training person, lag with best teacher-student agreement
        lag = {p: max(LAGS, key=lambda lg: np.mean(TP[p][lg].argmax(1) == SP[p].argmax(1))) for p in tr}
        chosen.setdefault(seed, {})[te] = np.bincount([LAGS.index(lag[p]) for p in tr], minlength=5).tolist()
        f = train(cat(XN, tr), caty(tr), soft=np.vstack([soften(TP[p][lag[p]]) for p in tr]), seed=seed)
        res["Dlag"].setdefault(te, []).append(bacc(yte, f(DATA[te][0]).argmax(1)))
    log("seed", seed, "done", {m: round(float(np.mean([np.mean(v) for v in res[m].values()])), 4) for m in res})

M = {m: np.array([np.mean(res[m][p]) for p in people]) for m in res}
out = {m: float(M[m].mean()) for m in M}
best = "D0" if out["D0"] >= out["Dlag"] else "Dlag"
out.update(R1=bool(max(out["B3"], out["B2"]) - out["B1"] >= 0.03),
           R2_gain=out[best] - out["B1"], R2_share=float((M[best] > M["B1"]).mean()),
           R3_gain=out["Dlag"] - out["D0"], R3_share=float((M["Dlag"] > M["D0"]).mean()))
out["R2"] = bool(out["R2_gain"] >= 0.02 and out["R2_share"] >= 0.60); out["R3"] = bool(out["R3_gain"] >= 0.01 and out["R3_share"] >= 0.55)
out["GO"] = bool(out["R1"] and out["R2"] and out["R3"])
out["lag_histogram_seed0_sum"] = np.sum(list(chosen[0].values()), 0).tolist()
out["per_person"] = {m: M[m].round(4).tolist() for m in M}; out["people"] = people
print("RESULT", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k != "per_person"}))
json.dump(out, open("/kaggle/working/ma_pilot_results.json", "w"), indent=1)
