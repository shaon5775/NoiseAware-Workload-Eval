"""Paper 8 room check (exploration): errors are whole BLOCKS. In real use the app knows WHEN an activity starts/ends
(segment boundaries) but not how hard it is for this person. Idea SEG-OT: average the decoder's probabilities per
segment, then jointly assign levels to ALL segments of the new person (optimal transport / balanced assignment, i.e.
segments of one person must use the levels in roughly the proportions of the curriculum), then self-train on them."""
import glob, os, numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
exec(open("room_check_session.py").read().split("def bacc")[0].split("print(")[0])
def bacc(y, p): return np.mean([np.mean(p[y == c] == c) for c in range(3)])
D = {p: feats(p, "person") for p in P}
def assign(S, prop=None):
    n = len(S); prop = np.full(3, 1/3) if prop is None else prop; cnt = np.round(prop * n).astype(int); cnt[-1] = n - cnt[:-1].sum()
    cols = np.concatenate([[k] * c for k, c in enumerate(cnt)]); cost = -np.log(S[:, cols] + 1e-9)
    r, c = linear_sum_assignment(cost); out = np.zeros(n, int); out[r] = cols[c]; return out
R = {}
def add(k, v): R.setdefault(k, []).append(v)
for te in P:
    tr = [p for p in P if p != te]; X = np.vstack([D[p][0] for p in tr]); Y = np.concatenate([D[p][1] for p in tr])
    sc = StandardScaler().fit(X); clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(X), Y)
    F, y, b, s = D[te]; Xt = sc.transform(F); pr = clf.predict_proba(Xt); blocks = np.unique(b)
    yb = np.array([y[b == k][0] for k in blocks]); Sb = np.array([pr[b == k].mean(0) for k in blocks])
    vote = np.array([np.bincount(pr[b == k].argmax(1), minlength=3).argmax() for k in blocks])
    add("window argmax", bacc(y, pr.argmax(1))); add("block: majority vote", bacc(yb, vote)); add("block: mean prob", bacc(yb, Sb.argmax(1)))
    a = assign(Sb); add("block: SEG-OT assignment", bacc(yb, a))
    # self-train: pseudo-label every window with its segment's assigned level, refit with the person's data, repeat
    lab = a
    for it in range(2):
        yw = lab[np.searchsorted(blocks, b)]
        c2 = LogisticRegression(max_iter=2000, C=0.05).fit(np.vstack([sc.transform(X), Xt]), np.r_[Y, yw],
                                                          sample_weight=np.r_[np.ones(len(Y)), np.full(len(yw), 2.0)])
        p2 = c2.predict_proba(Xt); S2 = np.array([p2[b == k].mean(0) for k in blocks]); lab = assign(S2)
    add("block: SEG-OT + self-train", bacc(yb, lab)); add("window after self-train", bacc(y, p2.argmax(1)))
    # robustness: assignment with a WRONG proportion prior (mostly easy curriculum assumed)
    add("block: SEG-OT, wrong prior (.5/.3/.2)", bacc(yb, assign(Sb, np.array([.5, .3, .2]))))
base = np.array(R["block: mean prob"])
for k, v in R.items(): print(f"{k:40s} {np.mean(v):.3f}   people better than block mean-prob: {np.mean(np.array(v) > base):.0%}")
