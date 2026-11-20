"""Paper 8 figures (Cognitive Neurodynamics rules: 84/174 mm wide, Arial 8 pt, vector PDF, colour + hatching).
All numbers are copied from paper8/pilot_*_results.md and figs/per_person_rbme.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
MM = 1 / 25.4; W1, W2 = 84 * MM, 174 * MM
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]       # Okabe–Ito
H = ["", "///", "...", "xxx", "\\\\\\", "ooo"]
def save(f, n): f.savefig(f"figs/Fig{n}.pdf", bbox_inches="tight"); f.savefig(f"figs/Fig{n}.png", dpi=600, bbox_inches="tight"); plt.close(f)
def panel(ax, s): ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9)

# ---------- Fig 1: schematic ----------
f, ax = plt.subplots(1, 2, figsize=(W2, 70 * MM), gridspec_kw={"width_ratios": [1, 1.25]}); [a.axis("off") for a in ax]
def box(a, x, y, w, h, t, c="#FFFFFF", fs=6.3):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=c, ec="black", lw=0.6)); a.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(a, x1, y1, x2, y2): a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, lw=0.7))
a = ax[0]; a.set_xlim(0, 1); a.set_ylim(0, 1)
box(a, 0.33, 0.82, 0.34, 0.12, "Workload level", "#E8F1FA")
box(a, 0.02, 0.50, 0.38, 0.14, "True performance\n(latent)", "#FDF0DC")
box(a, 0.60, 0.50, 0.38, 0.14, "EEG decoder output")
box(a, 0.02, 0.12, 0.38, 0.14, "Measured behaviour\n(true + noise)", "#FDF0DC")
arr(a, 0.44, 0.82, 0.24, 0.64); arr(a, 0.56, 0.82, 0.76, 0.64); arr(a, 0.40, 0.57, 0.60, 0.57); arr(a, 0.21, 0.50, 0.21, 0.26)
a.text(0.50, 0.60, "leakage", ha="center", fontsize=6, style="italic")
a.text(0.44, 0.19, "matching on this score\nleaves a hidden\nperformance gap", fontsize=6, va="center")
panel(a, "a")
a = ax[1]; a.set_xlim(-0.02, 1.03); a.set_ylim(0, 1)
box(a, 0.00, 0.74, 0.30, 0.18, "Behaviour per unit\nsplit into half A\nand half B", "#FDF0DC")
box(a, 0.35, 0.74, 0.30, 0.18, "Different-load pairs\nmatched on the\nhalf-A score", "#E8F1FA")
box(a, 0.70, 0.74, 0.30, 0.18, "Brain AUC on\nmatched pairs")
box(a, 0.35, 0.42, 0.30, 0.18, "Half-B AUC on the\nsame pairs\n(leftover gap)", "#FDF0DC")
box(a, 0.70, 0.42, 0.30, 0.18, "Bound for a perfect\nperformance reader\n" + r"$\Phi(\Phi^{-1}(\mathrm{AUC}_B)/\sqrt{\rho})$", "#FDF0DC")
box(a, 0.20, 0.06, 0.80, 0.18, "Δ = brain AUC − bound (A and B swapped, averaged)\nWilcoxon over people; Δ > 0 = beyond performance", "#E6F4EE")
arr(a, 0.30, 0.83, 0.35, 0.83); arr(a, 0.65, 0.83, 0.70, 0.83); arr(a, 0.50, 0.74, 0.50, 0.60); arr(a, 0.65, 0.51, 0.70, 0.51)
arr(a, 0.93, 0.74, 0.93, 0.24); arr(a, 0.80, 0.42, 0.80, 0.24)
a.text(0.15, 0.51, "ρ = split-half\nreliability of\nthe two scores", fontsize=6, ha="center", va="center")
panel(a, "b"); save(f, 1)

# ---------- Fig 2: audit ----------
S = ["COG-BCI n-back", "COG-BCI MATB", "Tufts fNIRS", "Pavlov digit span"]
BEH = [0.666, 0.884, 0.578, 0.697]; BR = [0.467, 0.655, 0.414, 0.457]; FU = [0.616, 0.879, 0.591, 0.678]
f, a = plt.subplots(figsize=(W2, 55 * MM)); x = np.arange(4); w = 0.26
for k, (v, n) in enumerate(((BEH, "Behaviour only"), (BR, "Brain only"), (FU, "Behaviour + brain"))):
    a.bar(x + (k - 1) * w, v, w, color=C[k], hatch=H[k], edgecolor="black", lw=0.5, label=n)
    for xi, vi in zip(x + (k - 1) * w, v): a.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7)
a.axhline(1 / 3, ls="--", lw=0.7, color="grey")
a.axhline(0.25, ls=":", lw=0.7, color="grey")
a.set_xticks(x); a.set_xticklabels(S); a.set_ylabel("Balanced accuracy (LOSO)"); a.set_ylim(0, 1); a.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(f, 2)

# ---------- Fig 3: naive matching fooled + method comparison ----------
f, ax = plt.subplots(1, 3, figsize=(W2, 58 * MM), gridspec_kw={"width_ratios": [1.15, 1, 1]})
a = ax[0]; real = [0.589, 0.871, 0.743, 0.577]; lin = [0.518, 0.515, 0.497, 0.527]; err = [0.664, 0.696, 0.682, 0.620]; x = np.arange(4); w = 0.26
for k, (v, n) in enumerate(((real, "Real EEG/fNIRS"), (lin, "Fake: perfect behaviour"), (err, "Fake: noisy behaviour"))):
    a.bar(x + (k - 1) * w, v, w, color=C[[0, 4, 5][k]], hatch=H[k], edgecolor="black", lw=0.5, label=n)
a.axhline(0.5, ls="--", lw=0.7, color="grey"); a.set_ylim(0.4, 0.95); a.set_xticks(x); a.set_xticklabels(["n-back", "MATB", "Tufts", "Pavlov"])
a.set_ylabel("Brain AUC, matched pairs"); a.set_ylim(0.4, 1.12); a.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); a.legend(frameon=False, fontsize=7, loc="upper left", ncol=1, bbox_to_anchor=(0.0, 1.03), handlelength=1.2); panel(a, "a")
M = ["Plain AUC", "Matching", "CVCR", "SIMEX", "RB-ME"]
fc = {"n-back": [100, 100, 100, 0, 0], "MATB": [100, 100, 100, 0, 0], "Pavlov": [100, 100, 100, 0, 15]}     # max over NULL-lin / NULL-nl
pw = {"n-back": [100, 100, 100, 100, 100], "MATB": [100, 100, 100, 95, 70], "Pavlov": [100, 100, 100, 100, 100]}
for a, D, t, p in ((ax[1], fc, "False claims under NULL (%)", "b"), (ax[2], pw, "Detection of real signal (%)", "c")):
    x = np.arange(5)
    for k, s in enumerate(D):
        a.bar(x + (k - 1) * 0.26, D[s], 0.26, color=C[k], hatch=H[k], edgecolor="black", lw=0.5, label=s)
    a.set_xticks(x); a.set_xticklabels(M, fontsize=7, rotation=35, ha="right"); a.set_ylabel(t); a.set_ylim(0, 110); panel(a, p)
ax[1].axhline(10, ls="--", lw=0.7, color="grey"); ax[1].text(3.4, 13, "10 % limit", fontsize=7, ha="center", color="grey"); ax[1].legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0))
f.tight_layout(); save(f, 3)

# ---------- Fig 4: RB-ME validation ----------
f, a = plt.subplots(figsize=(W1, 60 * MM)); nz = [0.3, 0.5, 0.8]
NULL = {"n-back": [-0.025, -0.009, 0.008], "MATB": [-0.058, -0.077, -0.036], "Pavlov": [-0.016, -0.007, -0.027]}
POS = {"n-back": [0.182, 0.163, 0.147], "MATB": [0.145, 0.089, 0.088], "Pavlov": [0.177, 0.143, 0.082]}
mk = ["o", "s", "^"]
for k, s in enumerate(NULL):
    a.plot(nz, NULL[s], marker=mk[k], ls="--", color=C[k], mfc="white", lw=0.8, label=f"{s} NULL")
    a.plot(nz, POS[s], marker=mk[k], ls="-", color=C[k], lw=0.8, label=f"{s} POS")
a.axhline(0, color="black", lw=0.6); a.axhline(0.02, color="grey", ls=":", lw=0.7); a.text(0.82, 0.026, "+0.02 limit", fontsize=7, color="grey", ha="right")
a.set_xlabel("Noise SD added to each behaviour half"); a.set_ylabel("Δ = brain AUC − bound"); a.set_xticks(nz); a.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(f, 4)

# ---------- Fig 5: per-person real data ----------
P = json.load(open("figs/per_person_rbme.json")); nm = {"S1": "COG-BCI n-back", "S2": "COG-BCI MATB", "S4": "Pavlov digit span"}
simex = {"S1": 0.536, "S2": 0.866, "S4": 0.569}; p_txt = {"S1": "Δ = −0.175, p ≈ 1", "S2": "Δ = +0.305, p = 5×10$^{-5}$", "S4": "Δ = −0.036, p ≈ 1"}
f, ax = plt.subplots(1, 3, figsize=(W2, 60 * MM), sharey=True); rng = np.random.default_rng(0)
for a, S in zip(ax, ["S1", "S2", "S4"]):
    R = np.array(P[S]); n = len(R)
    for i in range(n): a.plot([0, 1], R[i], color="grey", lw=0.3, alpha=0.6)
    a.scatter(np.zeros(n) + rng.uniform(-0.04, 0.04, n), R[:, 0], s=8, color=C[0], marker="o", zorder=3, label="Brain AUC")
    a.scatter(np.ones(n) + rng.uniform(-0.04, 0.04, n), R[:, 1], s=8, color=C[1], marker="s", zorder=3, label="Bound (perfect performance)")
    for j, c in ((0, C[0]), (1, C[1])):
        m = R[:, j].mean(); ci = 1.96 * R[:, j].std(ddof=1) / np.sqrt(n); a.errorbar(j + 0.18, m, yerr=ci, fmt="D", color="black", ms=4, capsize=2, lw=0.8)
    a.scatter([0.5], [simex[S]], marker="*", s=40, color=C[2], zorder=4, label="SIMEX-corrected brain AUC")
    a.axhline(0.5, ls="--", lw=0.6, color="grey"); a.set_xticks([0, 1]); a.set_xticklabels(["Brain", "Bound"]); a.set_xlim(-0.35, 1.4)
    a.set_title(f"{nm[S]} (n = {n})\n{p_txt[S]}", fontsize=7)
ax[0].set_ylabel("AUC on behaviour-matched pairs"); ax[0].legend(frameon=False, fontsize=7, loc="lower left"); f.tight_layout(); save(f, 5)

# ---------- Fig 6: robustness forest ----------
rows = [("Logistic, alternating split, caliper 0.10", 0.297), ("Logistic, alternating, 0.15 (main)", 0.305), ("Logistic, alternating, 0.20", 0.298),
        ("Logistic, alternating, 0.30", 0.249), ("Band power only, alternating, 0.15", 0.333), ("Logistic, block split, 0.10", 0.227),
        ("Logistic, block, 0.15", 0.179), ("Logistic, block, 0.20", 0.181), ("Logistic, block, 0.30", 0.151), ("Band power only, block, 0.15", 0.207),
        ("EEGNet, alternating, 0.15", 0.369)]
f, a = plt.subplots(figsize=(W1, 62 * MM)); yy = np.arange(len(rows))[::-1]
for (t, v), yv in zip(rows, yy):
    mkr = "D" if "EEGNet" in t else ("s" if "block" in t else "o"); a.plot(v, yv, marker=mkr, color="black" if "main" in t else C[0], ms=4)
a.axvline(0, color="black", lw=0.6); a.set_yticks(yy); a.set_yticklabels([r[0] for r in rows], fontsize=7); a.set_xlim(-0.05, 0.42)
a.set_xlabel("MATB Δ = brain AUC − bound (all p ≤ 0.0024)"); save(f, 6)

# ---------- Fig 7: time course (LEAD) ----------
t = [2, 4, 6, 8, 10, 15, 20, 30, 45, 60]
nb = {"Brain": [.406, .421, .418, .456, .452, .452, .483, .498, .536, .563], "Behaviour": [.349, .333, .341, .337, .360, .418, .513, .533, .605, .644], "Behaviour + brain": [.402, .421, .441, .448, .452, .502, .548, .567, .628, .644]}
mb = {"Brain": [.506, .590, .640, .605, .602, .628, .655, .674, .686, .705], "Behaviour": [.770, .693, .690, .724, .751, .755, .782, .770, .801, .820], "Behaviour + brain": [.743, .766, .785, .828, .835, .847, .870, .854, .847, .866]}
f, ax = plt.subplots(1, 2, figsize=(W2, 55 * MM), sharey=True)
for a, D, ti, p in ((ax[0], nb, "COG-BCI n-back", "a"), (ax[1], mb, "COG-BCI MATB", "b")):
    for k, (n, v) in enumerate(D.items()): a.plot(t, v, marker=["o", "s", "^"][k], color=C[k], ls=["-", "--", "-."][k], lw=0.9, ms=3, label=n)
    a.axhline(1 / 3, ls=":", lw=0.6, color="grey"); a.set_xscale("log"); a.set_xticks(t); a.set_xticklabels(t); a.set_xlabel("Seconds after load onset"); a.set_title(ti); panel(a, p)
ax[0].set_ylabel("Balanced accuracy"); ax[0].legend(frameon=False, loc="upper left"); f.tight_layout(); save(f, 7)
print("done")

# ---------- Fig 8: artefact check (P8-ARTEFACT; eye-only / muscle-only rows from P8-ARTEFACT2 when available) ----------
import os
A2 = json.load(open("artefact2_summary.json")) if os.path.exists("artefact2_summary.json") else {}
va = [("No\nICA", 0.305, "1e-4")] + [(n, A2[k][0], A2[k][1]) for n, k in (("Eye\nonly", "eye"), ("Muscle\nonly", "muscle")) if k in A2] + \
     [("Eye +\nmuscle", 0.077, "0.15"), ("+ no edge,\n1-20 Hz", 0.092, "0.14")]
vb = [("Edge ch.", 0.181, "0.002"), ("Frontal", 0.152, "0.013"), ("Parietal", 0.113, "0.084"), ("Central", 0.109, "0.084"), ("Occipital", 0.042, "0.30"),
      ("30-40 Hz", 0.139, "0.044"), ("13-30 Hz", 0.111, "0.069"), ("8-13 Hz", 0.100, "0.13"), ("4-8 Hz", 0.045, "0.34"), ("1-4 Hz", 0.013, "0.54")]
f, ax = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={"width_ratios": [len(va), 7.5]})
for a, V, p, ti in ((ax[0], va, "a", "MATB: components removed"), (ax[1], vb, "b", "After eye + muscle removal: one region or band")):
    x = np.arange(len(V)); cols = [C[0] if (i == 0 and p == "a") else (C[5] if (p == "b" and i < 5) else C[2]) for i in range(len(V))]
    hat = ["" if (i == 0 and p == "a") else ("///" if (p == "b" and i < 5) else "...") for i in range(len(V))]
    a.bar(x, [v for _, v, _ in V], 0.65, color=cols, hatch=hat, edgecolor="black", lw=0.5)
    for xi, (_, v, pv) in zip(x, V): a.text(xi, v + 0.008, f"p={pv}", ha="center", fontsize=6.5, rotation=90, va="bottom")
    a.axhline(0, color="black", lw=0.6); a.set_xticks(x); a.set_xticklabels([n for n, _, _ in V], fontsize=7, rotation=0 if p == "a" else 45, ha="center" if p == "a" else "right")
    a.set_ylim(-0.02, 0.38); a.set_title(ti); panel(a, p)
ax[0].set_ylabel("Δ = brain AUC − bound"); f.tight_layout(); save(f, 8)
print("fig8 done")
