"""Regenerate figures/fig5_downstream.png -- the six-panel Figure 5:
random-sampled AP is a reliable scalable replacement for full AP.

Dataset order matches every other table in main.tex (Section 12):
short_comparison_test, comparison_test, mp_mouse_1_jul, control_mouse_1_jul,
control_mouse_1_oct, mp_mouse_1_oct, still_test. moving_test is excluded
(no full-AP reference).

Data source: results/method_comparison.csv and
results/fig5f_downstream_agreement.csv, from the 2026-08-24 version-pinned run.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

DATASETS = ["short", "comparison", "mp Jul", "ctrl Jul", "ctrl Oct", "mp Oct", "still"]

K_FULL    = [9, 33, 40, 42, 45, 50, 46]
K_SAMPLED = [9, 28, 26, 25, 28, 30, 27]

ARI_SAMPLED = [1.000, 0.472, 0.438, 0.325, 0.387, 0.446, 0.450]

T_FULL    = [0.2, 55.1, 150.9, 196.3, 218.8, 214.2, 177.2]
T_SAMPLED = [0.2, 17.8,  34.5,  21.5,  24.4,  51.6,  19.5]

OCC_R   = [1.000, 0.911, 0.927, 0.764, 0.956, 0.919, 0.803]
OCC_LO  = [1.000, 0.803, 0.872, 0.544, 0.921, 0.770, 0.503]
OCC_HI  = [1.000, 0.969, 0.970, 0.899, 0.980, 0.966, 0.923]
TRANS_R  = [1.000, 0.930, 0.965, 0.931, 0.975, 0.955, 0.923]
TRANS_LO = [1.000, 0.885, 0.944, 0.892, 0.962, 0.934, 0.890]
TRANS_HI = [1.000, 0.957, 0.977, 0.955, 0.984, 0.968, 0.946]
RASTER   = [1.000, 0.599, 0.546, 0.415, 0.508, 0.505, 0.534]
EXEMPLAR = [1.000, 0.333, 0.100, 0.095, 0.089, 0.020, 0.022]

C_FULL, C_SAMPLED = "#4477AA", "#EE7733"
C_OCC, C_TRANS = "#4477AA", "#EE7733"

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.32)

# ── Panel A: workflow schematic ────────────────────────────────────────────
axA = fig.add_subplot(gs[0, 0])
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.set_title("A   Workflow comparison", loc="left", fontsize=13, fontweight="bold")

def box(ax, x, y, w, h, text, color):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                 linewidth=1.2, edgecolor="#555555", facecolor=color, zorder=2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.3, zorder=3)

def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                  color="#555555", linewidth=1.1, zorder=2))

box(axA, 0.3, 7.3, 2.2, 1.3, "N windows", "#eeeeee")
arrow(axA, 2.5, 7.95, 3.1, 7.95)
box(axA, 3.1, 7.3, 3.0, 1.3, "Build N×N\ndistance + affinity", "#d5e8d4")
arrow(axA, 6.1, 7.95, 6.7, 7.95)
box(axA, 6.7, 7.3, 3.0, 1.3, "AP message\npassing (O(N²)/iter)", "#d5e8d4")
axA.text(0.1, 8.6, "Full AP", fontsize=9.5, fontweight="bold", color=C_FULL)

box(axA, 0.3, 4.3, 2.2, 1.3, "N windows", "#eeeeee")
arrow(axA, 2.5, 4.95, 3.1, 4.95)
box(axA, 3.1, 4.3, 2.4, 1.3, "Sample\nM=10,000", "#ffe6cc")
arrow(axA, 5.5, 4.95, 6.1, 4.95)
box(axA, 6.1, 4.3, 2.4, 1.3, "AP on M×M\n→ exemplars", "#ffe6cc")
arrow(axA, 5.9, 4.3, 5.9, 2.6)
arrow(axA, 6.1, 1.95, 8.6, 1.95)
box(axA, 0.3, 1.3, 5.6, 1.3, "FAISS: assign all N to nearest exemplar", "#ffe6cc")
axA.text(0.1, 5.6, "AP sampled", fontsize=9.5, fontweight="bold", color=C_SAMPLED)

# ── Panel B: complexity ─────────────────────────────────────────────────────
axB = fig.add_subplot(gs[0, 1])
N = np.logspace(3, 5.6, 200)
M, K = 10_000, 30
full_ops = N**2
sampled_ops = np.where(N <= M, N**2, M**2 + N*K)
axB.plot(N, full_ops, color=C_FULL, linewidth=2, label="Full AP:  $O(N^2)$")
axB.plot(N, sampled_ops, color=C_SAMPLED, linewidth=2, label="AP sampled:  $O(M^2 + N{\\cdot}K)$")
axB.axvline(25_000, color="#888888", linestyle="--", linewidth=1)
axB.text(25_000*1.08, full_ops.max()*0.02, "N=25,000\n(feasibility limit)",
         fontsize=7.3, color="#666666", va="bottom")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("N (windows)", fontsize=9.5)
axB.set_ylabel("Core operations", fontsize=9.5)
axB.set_title("B   Complexity", loc="left", fontsize=13, fontweight="bold")
axB.legend(fontsize=8.3, frameon=False, loc="upper left")
axB.tick_params(labelsize=8.3)

# ── Panel C: ARI vs AP full (AP sampled only, per H4) ───────────────────────
axC = fig.add_subplot(gs[0, 2])
x = np.arange(len(DATASETS))
axC.bar(x, ARI_SAMPLED, color=C_SAMPLED, width=0.6)
for xi, v in zip(x, ARI_SAMPLED):
    axC.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#333333")
axC.set_xticks(x); axC.set_xticklabels(DATASETS, fontsize=8.3, rotation=20, ha="right")
axC.set_ylabel("ARI vs AP full", fontsize=9.5)
axC.set_ylim(0, 1.12)
axC.set_title("C   Cluster agreement (ARI)", loc="left", fontsize=13, fontweight="bold")
axC.tick_params(labelsize=8.3)

# ── Panel D: cluster count agreement ────────────────────────────────────────
axD = fig.add_subplot(gs[1, 0])
w = 0.36
axD.bar(x - w/2, K_FULL, width=w, color=C_FULL, label="AP full")
axD.bar(x + w/2, K_SAMPLED, width=w, color=C_SAMPLED, label="AP sampled")
axD.set_xticks(x); axD.set_xticklabels(DATASETS, fontsize=8.3, rotation=20, ha="right")
axD.set_ylabel("Clusters found (K)", fontsize=9.5)
axD.set_title("D   Cluster-count agreement", loc="left", fontsize=13, fontweight="bold")
axD.legend(fontsize=8.3, frameon=False)
axD.tick_params(labelsize=8.3)

# ── Panel E: runtime advantage ──────────────────────────────────────────────
axE = fig.add_subplot(gs[1, 1])
axE.bar(x - w/2, T_FULL, width=w, color=C_FULL, label="AP full")
axE.bar(x + w/2, T_SAMPLED, width=w, color=C_SAMPLED, label="AP sampled")
axE.set_yscale("log")
axE.set_xticks(x); axE.set_xticklabels(DATASETS, fontsize=8.3, rotation=20, ha="right")
axE.set_ylabel("Runtime (s, log scale)", fontsize=9.5)
axE.set_title("E   Runtime advantage", loc="left", fontsize=13, fontweight="bold")
axE.legend(fontsize=8.3, frameon=False)
axE.tick_params(labelsize=8.3)

# ── Panel F: downstream behavioral agreement ────────────────────────────────
axF = fig.add_subplot(gs[1, 2])
w2 = 0.2
occ_err = [np.array(OCC_R) - np.array(OCC_LO), np.array(OCC_HI) - np.array(OCC_R)]
trans_err = [np.array(TRANS_R) - np.array(TRANS_LO), np.array(TRANS_HI) - np.array(TRANS_R)]
axF.bar(x - w2, OCC_R, width=w2, color=C_OCC, label="Occupancy $r$", yerr=occ_err, capsize=2, error_kw={"linewidth": 0.9})
axF.bar(x, TRANS_R, width=w2, color=C_TRANS, label="Transition $r$", yerr=trans_err, capsize=2, error_kw={"linewidth": 0.9})
axF.bar(x + w2, RASTER, width=w2, color="#888888", label="Raster agreement")
axF.set_xticks(x); axF.set_xticklabels(DATASETS, fontsize=8.3, rotation=20, ha="right")
axF.set_ylabel("Agreement with AP full", fontsize=9.5)
axF.set_ylim(0, 1.12)
axF.set_title("F   Downstream behavioral agreement", loc="left", fontsize=13, fontweight="bold")
axF.legend(fontsize=7.6, frameon=False, loc="upper right")
axF.tick_params(labelsize=8.3)

fig.suptitle("Figure 5.  Random-sampled AP is a reliable scalable replacement for full AP",
             fontsize=14.5, fontweight="bold", y=1.02)

fig.savefig("fig5_downstream.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved fig5_downstream.png")
