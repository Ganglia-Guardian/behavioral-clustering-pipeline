"""Regenerate cluster_count_heatmap.png, silhouette_heatmap.png, and
ari_vs_ap_full.png from the current results/method_comparison.csv.

Run after any batch_compare.py re-run to keep the paper's figures and
tables in sync with the cached results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(REPO_ROOT / "results" / "method_comparison.csv")

DATASETS = ["short_comparison_test", "comparison_test", "mp_mouse_1_jul",
            "control_mouse_1_jul", "control_mouse_1_oct", "mp_mouse_1_oct",
            "still_test", "moving_test"]
DS_SHORT = ["short", "comparison", "mp Jul", "ctrl Jul", "ctrl Oct", "mp Oct", "still", "moving"]
METHODS = ["hdbscan", "ap_full", "ap_sampled", "ap_coreset", "ap_sparse",
           "ap_twolevel", "ap_kmeans", "ap_stratified"]
METHOD_LABELS = ["HDBSCAN", "AP full", "AP sampled", "AP coreset", "AP sparse",
                  "AP two-level", "AP kmeans", "AP stratified"]

def get(dataset, method, col):
    row = df[(df["dataset"] == dataset) & (df["method"] == method)]
    if len(row) == 0 or row.iloc[0]["status"] != "ok":
        return None
    v = row.iloc[0][col]
    return None if pd.isna(v) else v

# ── cluster count heatmap ───────────────────────────────────────────────────
counts = np.full((len(DATASETS), len(METHODS)), np.nan)
for i, d in enumerate(DATASETS):
    for j, m in enumerate(METHODS):
        v = get(d, m, "n_clusters")
        if v is not None:
            counts[i, j] = v

fig, ax = plt.subplots(figsize=(11, 5))
disp = np.log1p(counts)
im = ax.imshow(disp, cmap="PuBuGn", aspect="auto")
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHOD_LABELS, rotation=30, ha="right")
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DS_SHORT)
for i in range(len(DATASETS)):
    for j in range(len(METHODS)):
        if np.isnan(counts[i, j]):
            ax.text(j, i, "skip", ha="center", va="center", fontsize=8, color="#888888")
        else:
            color = "white" if disp[i, j] > 0.75 * np.nanmax(disp) else "black"
            ax.text(j, i, f"{int(counts[i,j])}", ha="center", va="center", fontsize=9, color=color)
cbar = fig.colorbar(im, ax=ax); cbar.set_label("log(1 + clusters)")
ax.set_title("Number of clusters by dataset and method", fontsize=14)
fig.tight_layout()
fig.savefig("cluster_count_heatmap.png", dpi=200, facecolor="white")
plt.close(fig)

# ── silhouette heatmap ──────────────────────────────────────────────────────
sil = np.full((len(DATASETS), len(METHODS)), np.nan)
for i, d in enumerate(DATASETS):
    for j, m in enumerate(METHODS):
        v = get(d, m, "silhouette")
        if v is not None:
            sil[i, j] = v

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(sil, cmap="YlGnBu", aspect="auto", vmin=0, vmax=0.6)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHOD_LABELS, rotation=30, ha="right")
ax.set_yticks(range(len(DATASETS))); ax.set_yticklabels(DS_SHORT)
for i in range(len(DATASETS)):
    for j in range(len(METHODS)):
        if np.isnan(sil[i, j]):
            ax.text(j, i, "skip", ha="center", va="center", fontsize=8, color="#888888")
        else:
            color = "white" if sil[i, j] > 0.75 * 0.6 else "black"
            ax.text(j, i, f"{sil[i,j]:.2f}", ha="center", va="center", fontsize=9, color=color)
cbar = fig.colorbar(im, ax=ax); cbar.set_label("Silhouette")
ax.set_title("Silhouette score by dataset and method", fontsize=14)
fig.tight_layout()
fig.savefig("silhouette_heatmap.png", dpi=200, facecolor="white")
plt.close(fig)

# ── ARI vs AP full (grouped bars, only datasets where AP full ran) ─────────
ari_methods = ["hdbscan", "ap_sampled", "ap_coreset", "ap_sparse", "ap_twolevel", "ap_kmeans", "ap_stratified"]
ari_labels = ["hdbscan", "AP sampled", "AP coreset", "AP sparse", "AP twolevel", "AP kmeans", "AP stratified"]
ari_datasets = [d for d in DATASETS if get(d, "ap_full", "n_clusters") is not None]
ari_short = [DS_SHORT[DATASETS.index(d)] for d in ari_datasets]

colors = ["#7f7f7f", "#4477AA", "#66CCEE", "#CC79A7", "#228833", "#EE7733", "#CC3311"]
fig, ax = plt.subplots(figsize=(12, 4.6))
n = len(ari_methods)
w = 0.8 / n
x = np.arange(len(ari_datasets))
for k, (m, lab, c) in enumerate(zip(ari_methods, ari_labels, colors)):
    vals = [get(d, m, "ari_vs_ap_full") or 0 for d in ari_datasets]
    ax.bar(x + (k - n/2 + 0.5) * w, vals, width=w, label=lab, color=c)
ax.set_xticks(x); ax.set_xticklabels(ari_short)
ax.set_ylabel("ARI vs AP full")
ax.set_ylim(0, 1.05)
ax.set_title("Agreement with full AP reference", fontsize=14)
ax.legend(ncol=4, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig("ari_vs_ap_full.png", dpi=200, facecolor="white")
plt.close(fig)

print("Saved cluster_count_heatmap.png, silhouette_heatmap.png, ari_vs_ap_full.png")
print(f"ARI figure covers {len(ari_datasets)} datasets: {ari_datasets}")
