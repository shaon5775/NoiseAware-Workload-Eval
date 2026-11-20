"""Paper 8 pilot BVB on COG-BCI (rules: pilot_BVB_prereg.md). Unit = one minute of a task recording (n-back levels

0/1/2, and MATB easy/med/diff as a second task family). Behaviour per unit from the behavioural .mat (parser below is

filled in after seeing the file structure printed by the Kaggle export). Effort = RSME per task per session."""

import pickle, sys, numpy as np, pandas as pd

from scipy.stats import spearmanr, wilcoxon

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.preprocessing import StandardScaler

D = pickle.load(open("data/cog/cog_features.pkl", "rb")); FEAT, BEH, EVT, INFO = D["FEAT"], D["BEH"], D["EVT"], D["INFO"]

FAMILY = {"nback": {"zeroBACK": 0, "oneBACK": 1, "twoBACK": 2}, "matb": {"MATBeasy": 0, "MATBmed": 1, "MATBdiff": 2}}

BEHFILE = {"zeroBACK": "0-Back", "oneBACK": "1-Back", "twoBACK": "2-Back", "MATBeasy": "MATB_Easy", "MATBmed": "MATB_Med", "MATBdiff": "MATB_Diff"}

RSMEKEY = {"zeroBACK": "ZeroBack", "oneBACK": "OneBack", "twoBACK": "TwoBack", "MATBeasy": "MATB_easy", "MATBmed": "MATB_med", "MATBdiff": "MATB_diff"}

rsme = pd.read_csv("data/cog/RSME.txt")

def logm(C): w, V = np.linalg.eigh(C); return (V * np.log(np.maximum(w, 1e-12))[..., None, :]) @ np.swapaxes(V, -1, -2)

def isqrtm(C): w, V = np.linalg.eigh(C); return (V * (1 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T

def behaviour_per_unit(s, ses, task, n_units, unit_s=60.0):

    """n-back codes 6L2x = stimulus (x=1 non-target, 2 target), 6L3x = response (x=2 hit, 1/3 other).

    MATB: TRACKING lines give cursor x,y (error = distance from centre); RESMAN lines give tank levels (target 2500)."""

    ev = EVT[(s, ses, task)]; out = np.full((n_units, 3), np.nan)

    if "BACK" in task:

        stim = [(o, d[-1]) for o, d in ev if len(d) == 4 and d[2] == "2"]; resp = [(o, d[-1]) for o, d in ev if len(d) == 4 and d[2] == "3"]

        ro = np.array([o for o, _ in resp]) if resp else np.zeros(0)

        for k in range(n_units):

            T = [(o, x) for o, x in stim if k * unit_s <= o < (k + 1) * unit_s]

            tg = [o for o, x in T if x == "2"]; nt = [o for o, x in T if x == "1"]

            def hit(o): r = ro[(ro > o) & (ro < o + 2.0)]; return r[0] - o if len(r) else None

            h = [hit(o) for o in tg]; fa = [hit(o) for o in nt]

            if tg: out[k] = [np.mean([x is not None for x in h]), np.mean([x is not None for x in fa]) if nt else 0,

                             np.nanmean([x for x in h if x is not None]) if any(x is not None for x in h) else 2.0]

    else:

        tr, rm = [], []

        for o, d in ev:

            p = d.split()

            if len(p) >= 5 and p[1] == "TRACKING": tr.append((o, float(p[3]), float(p[4])))

            if len(p) >= 7 and p[1] == "RESMAN": rm.append((o, abs(float(p[3]) - 2500) + abs(float(p[5]) - 2500)))

        tr, rm = np.array(tr), np.array(rm)

        for k in range(n_units):

            a = tr[(tr[:, 0] >= k * unit_s) & (tr[:, 0] < (k + 1) * unit_s)] if len(tr) else tr

            b = rm[(rm[:, 0] >= k * unit_s) & (rm[:, 0] < (k + 1) * unit_s)] if len(rm) else rm

            if len(a): out[k] = [np.mean(np.hypot(a[:, 1], a[:, 2])), np.mean(b[:, 1]) if len(b) else np.nan, np.std(np.hypot(a[:, 1], a[:, 2]))]

    return out

def build(family):

    rows = []

    for (s, ses, task), f in FEAT.items():

        if task not in FAMILY[family]: continue

        t = f["t"]; u = (t // 60).astype(int); n = u.max() + 1; BU = behaviour_per_unit(s, ses, task, n)

        bp = f["bp"].astype(float).mean(2); C = f["cov"].astype(float)          # band power averaged over channels (channel sets differ)
        if C.shape[1] == 19: C = np.delete(np.delete(C, 9, 1), 9, 2)              # drop Cz so all people share 18 channels

        for k in range(n):

            m = u == k

            if m.sum() < 6: continue

            rows.append(dict(s=s, ses=ses, task=task, unit=k, level=FAMILY[family][task], bp=bp[m].mean(0), C=C[m].mean(0),

                             beh=BU[k] if k < len(BU) else np.full(3, np.nan)))

    df = pd.DataFrame(rows)

    # per-person brain features: z-score band power, re-centre covariances on the person's mean (no labels)

    out = []

    for s, g in df.groupby("s"):

        B = np.stack(g.bp.values); B = (B - B.mean(0)) / (B.std(0) + 1e-9)

        Cs = np.stack(g.C.values); Cs += 1e-6 * np.eye(Cs.shape[1]) * np.trace(Cs, axis1=1, axis2=2).mean() / Cs.shape[1]

        R = isqrtm(Cs.mean(0)); iu = np.triu_indices(Cs.shape[1]); T = logm(R @ Cs @ R)[:, iu[0], iu[1]]

        T = (T - T.mean(0)) / (T.std(0) + 1e-9)

        Bh = np.stack(g.beh.values); Bh = np.where(np.isnan(Bh), np.nanmean(Bh, 0), Bh); Bh = (Bh - Bh.mean(0)) / (Bh.std(0) + 1e-9)

        g = g.copy(); g["brain"] = list(np.c_[B, T]); g["behz"] = list(Bh); out.append(g)

    return pd.concat(out)

def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in np.unique(y)])

if __name__ == "__main__":

    for family in ("nback", "matb"):

        df = build(family); df = df[np.isfinite(np.stack(df.beh.values)).all(1) | True].reset_index(drop=True)

        people = sorted(df.s.unique()); res = {k: [] for k in ("BEH", "BRAIN", "FUSED")}; score = np.zeros(len(df))

        for te in people:

            tr = df.s != te; ts = df.s == te

            for k, get, C_ in (("BEH", lambda d: np.stack(d.behz.values), 1.0), ("BRAIN", lambda d: np.stack(d.brain.values), 0.05),

                               ("FUSED", lambda d: np.c_[np.stack(d.behz.values), np.stack(d.brain.values)], 0.05)):

                X, Y = get(df[tr]), df[tr].level.values; sc = StandardScaler().fit(X)

                clf = LogisticRegression(max_iter=3000, C=C_).fit(sc.transform(X), Y); Xt = sc.transform(get(df[ts]))

                res[k].append(bacc(df[ts].level.values, clf.predict(Xt)))

                if k == "BRAIN": score[ts.values] = clf.predict_proba(Xt) @ np.arange(3)

        M = {k: np.array(v) for k, v in res.items()}

        print(f"\n=== COG-BCI {family}: units {len(df)}, people {len(people)}")

        for k, v in M.items(): print(f"  {k:6s} {v.mean():.3f}")

        print(f"  H1 BEH>=BRAIN-0.02: {M['BEH'].mean() >= M['BRAIN'].mean() - 0.02} | H2 FUSED-BEH<0.03: {(M['FUSED']-M['BEH']).mean() < 0.03} "

              f"(gain {(M['FUSED']-M['BEH']).mean():+.3f}, p={wilcoxon(M['FUSED'], M['BEH']).pvalue:.4f}, share {np.mean(M['FUSED']>M['BEH']):.0%})")

        # H3: per recording (person x session x task): brain score, behaviour, RSME effort; residualise on level + behaviour within person

        df["score"] = score; rec = df.groupby(["s", "ses", "task"]).agg(level=("level", "first"), score=("score", "mean")).reset_index()

        rec["beh"] = df.groupby(["s", "ses", "task"]).behz.apply(lambda v: np.stack(v.values).mean(0)).values

        rec["sess"] = rec.ses.str.extract(r"(\d)").astype(int)

        rec["rsme"] = [rsme[(rsme.sbj == s) & (rsme.Session == n) & (rsme.condition == RSMEKEY[t])].Score.mean() for s, n, t in zip(rec.s, rec.sess, rec.task)]

        rho = []

        for s, g in rec.groupby("s"):

            g = g.dropna(subset=["rsme"])

            if len(g) < 6: continue

            Z = np.c_[np.eye(3)[g.level.values], np.stack(g.beh.values)]

            r1 = g.score.values - Z @ np.linalg.lstsq(Z, g.score.values, rcond=None)[0]; r2 = g.rsme.values - Z @ np.linalg.lstsq(Z, g.rsme.values, rcond=None)[0]

            rho.append(spearmanr(r1, r2)[0])

        rho = np.nan_to_num(np.array(rho))

        print(f"  H3 brain residual vs effort residual: mean rho {rho.mean():+.3f}, p(one-sided)={wilcoxon(rho, alternative='greater').pvalue:.4f}, people {len(rho)}, share>0 {np.mean(rho>0):.0%}")

