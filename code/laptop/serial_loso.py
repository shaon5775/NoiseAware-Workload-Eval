"""Serial leave-one-person-out helpers (joblib is very slow on this Windows laptop)."""
import numpy as np
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def loso(s, y, X, C):
    out = np.zeros(len(y))
    for te in np.unique(s):
        tr, ts = s != te, s == te; sc = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X[tr]), y[tr])
        out[ts] = c.predict_proba(sc.transform(X[ts])) @ c.classes_
    return out
def resid(x, s, y):
    r = x.copy()
    for p in np.unique(s):
        for l in np.unique(y):
            m = (s == p) & (y == l)
            if m.any(): r[m] -= r[m].mean()
    return r
def auc(x, i, j): return np.mean((x[i] > x[j]) + 0.5 * (x[i] == x[j]))
def bacc(y, p): return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))
