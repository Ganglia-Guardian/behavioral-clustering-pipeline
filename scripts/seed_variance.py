"""
seed_variance.py
-----------------
Measure seed-to-seed stability of the four sampling-based AP variants
(ap_sampled, ap_coreset, ap_kmeans, ap_stratified). Each method's cached
result in results/ was produced with a single hardcoded seed (42) -- this
script re-runs each method across N_SEEDS different seeds per dataset and
records cluster count (K) and ARI against the cached AP-full reference
(where available), so the paper can report a mean +/- std instead of a
single, possibly lucky, run.

AP full and AP sparse/two-level are excluded: AP full has no sampling
randomness (fixed random_state=0, deterministic given the data), and sparse/
two-level are not the paper's recommended methods, so their seed variance is
lower priority.

Output: results/seed_variance.csv (one row per dataset x method x seed) and
results/seed_variance_summary.csv (mean/std/min/max per dataset x method).

Usage
-----
    python3 scripts/seed_variance.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_pipeline import (
    load_cleaned_motion, process_motion, extract_histogram_features,
    ARENA_BIN_EDGES, _build_cdf_features,
    cluster_ap_sampled, cluster_ap_kmeans_sampled, cluster_ap_stratified_sampled,
)
from sklearn.metrics import adjusted_rand_score

DATASETS = ["short_comparison_test", "comparison_test", "mp_mouse_1_jul",
            "control_mouse_1_jul", "control_mouse_1_oct", "mp_mouse_1_oct",
            "still_test", "moving_test"]
AP_SAMPLE_SIZE = 10_000
SEEDS = list(range(1, 11))  # 10 seeds, distinct from the canonical seed=42


def _labels_from_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    cluster_idx = df["ClusterIdx"].values.astype(int)
    return np.where(cluster_idx == 0, -1, cluster_idx - 1)


def run_method(method, hist, channel_sizes, seed):
    t0 = time.time()
    if method == "ap_sampled":
        labels = cluster_ap_sampled(hist, channel_sizes, sample_size=AP_SAMPLE_SIZE, seed=seed)
    elif method == "ap_coreset":
        labels = cluster_ap_sampled(hist, channel_sizes, sample_size=AP_SAMPLE_SIZE,
                                     use_coreset=True, seed=seed)
    elif method == "ap_kmeans":
        labels = cluster_ap_kmeans_sampled(hist, channel_sizes, sample_size=AP_SAMPLE_SIZE, seed=seed)
    elif method == "ap_stratified":
        labels = cluster_ap_stratified_sampled(hist, channel_sizes, sample_size=AP_SAMPLE_SIZE, seed=seed)
    else:
        raise ValueError(method)
    elapsed = time.time() - t0
    return labels, elapsed


def main():
    rows = []
    methods = ["ap_sampled", "ap_coreset", "ap_kmeans", "ap_stratified"]

    for name in DATASETS:
        out_dir = RESULTS_DIR / name
        clean_csv = out_dir / "combined_harp_data_cleaned.csv"
        if not clean_csv.exists():
            print(f"[{name}] SKIP: {clean_csv} not found")
            continue

        print(f"\n[{name}] loading + featurizing ...")
        raw, ts, folders = load_cleaned_motion(str(clean_csv))
        sensor = process_motion(raw)
        edges = ARENA_BIN_EDGES["3d_wired"]
        channel_sizes = [len(e) - 1 for e in edges]
        hist = extract_histogram_features(sensor, edges)
        N = hist.shape[0]

        if N <= AP_SAMPLE_SIZE:
            print(f"[{name}] SKIP: N={N:,} <= sample_size, sampling methods degenerate to AP full "
                  f"(no seed variance to measure)")
            continue

        ap_full_csv = out_dir / "Cluster_detail_results_ap_full.csv"
        ref_labels = _labels_from_csv(ap_full_csv) if ap_full_csv.exists() else None
        print(f"[{name}] N={N:,}  AP-full reference: {'yes' if ref_labels is not None else 'no'}")

        for method in methods:
            for seed in SEEDS:
                labels, elapsed = run_method(method, hist, channel_sizes, seed)
                K = len(np.unique(labels[labels >= 0]))
                ari = adjusted_rand_score(ref_labels, labels) if ref_labels is not None else None
                rows.append({
                    "dataset": name, "method": method, "seed": seed,
                    "K": K, "ari_vs_ap_full": ari, "t_s": round(elapsed, 2),
                })
                ari_str = f"ARI={ari:.3f}" if ari is not None else "ARI=n/a"
                print(f"  {method:15s} seed={seed:2d}  K={K:3d}  {ari_str}  ({elapsed:.1f}s)")

        # save incrementally so a crash/interrupt doesn't lose completed datasets
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "seed_variance.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "seed_variance.csv", index=False)

    summary = df.groupby(["dataset", "method"]).agg(
        K_mean=("K", "mean"), K_std=("K", "std"), K_min=("K", "min"), K_max=("K", "max"),
        ari_mean=("ari_vs_ap_full", "mean"), ari_std=("ari_vs_ap_full", "std"),
        ari_min=("ari_vs_ap_full", "min"), ari_max=("ari_vs_ap_full", "max"),
        n_seeds=("seed", "count"),
    ).round(4).reset_index()
    summary.to_csv(RESULTS_DIR / "seed_variance_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Seed variance summary")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'seed_variance.csv'}")
    print(f"Saved: {RESULTS_DIR / 'seed_variance_summary.csv'}")


if __name__ == "__main__":
    main()
