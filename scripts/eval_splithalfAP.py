"""
eval_splithalfAP.py
-------------------
Split-half stability test using the real AP-full algorithm.

For each random 50/50 split:
  1. Run AP-full on half A  → exemplar_indices_A, labels_A
  2. Assign half B windows to nearest A-exemplar (L1 dist) → assigned_B
  3. Run AP-full on half B independently → labels_B
  4. AMI(assigned_B, labels_B)   [AMI handles different K across halves]

Repeat N_SPLITS times, report mean ± std AMI.

Usage
-----
    python scripts/eval_splithalfAP.py \
        --harp-csv results/short_comparison_test/combined_harp_data_cleaned.csv
"""

import argparse
import sys
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from pathlib import Path
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, extract_empirical_features,
    _build_cdf_features, ARENA_BIN_EDGES,
)

N_SPLITS = 5
DAMPING  = 0.9
MAX_ITER = 1000
CONV_ITER = 15


def run_ap(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run AP-full on `features` (L1 affinity). Returns (labels, exemplar_indices)."""
    N = features.shape[0]
    dist     = ssd.cdist(features, features, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    pref     = float(affinity.min())
    np.fill_diagonal(affinity, pref)

    ap = AffinityPropagation(
        affinity="precomputed", preference=pref,
        damping=DAMPING, max_iter=MAX_ITER,
        convergence_iter=CONV_ITER, random_state=0,
    )
    labels = ap.fit_predict(affinity)
    return labels, ap.cluster_centers_indices_


def assign_to_exemplars(query: np.ndarray, exemplars: np.ndarray) -> np.ndarray:
    """Assign each query window to its nearest exemplar (L1 distance)."""
    dist = ssd.cdist(query, exemplars, metric="cityblock")
    return np.argmin(dist, axis=1)


def splithalfAP(features: np.ndarray, label: str, rng: np.random.Generator):
    N = features.shape[0]
    ami_scores = []

    for s in range(N_SPLITS):
        perm   = rng.permutation(N)
        idx_A  = perm[: N // 2]
        idx_B  = perm[N // 2 :]

        feat_A = features[idx_A]
        feat_B = features[idx_B]

        print(f"  [{label}] split {s+1}/{N_SPLITS}  N_A={len(idx_A)}  N_B={len(idx_B)}")

        # Cluster half A
        labels_A, exemplar_local = run_ap(feat_A)
        if len(exemplar_local) == 0:
            print(f"    WARNING: split {s+1} half A has no exemplars, skipping")
            continue
        exemplar_feats = feat_A[exemplar_local]   # shape (K_A, dim)
        K_A = len(exemplar_local)

        # Assign half B to A's exemplars
        assigned_B = assign_to_exemplars(feat_B, exemplar_feats)

        # Cluster half B independently
        labels_B, _ = run_ap(feat_B)
        K_B = len(np.unique(labels_B))

        score = adjusted_mutual_info_score(assigned_B, labels_B)
        ami_scores.append(score)
        print(f"    K_A={K_A}  K_B={K_B}  AMI={score:.4f}")

    mean_ami = float(np.mean(ami_scores))
    std_ami  = float(np.std(ami_scores))
    print(f"\n  [{label}]  mean AMI = {mean_ami:.4f} ± {std_ami:.4f}  "
          f"({N_SPLITS} splits)\n")
    return mean_ami, std_ami


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harp-csv", type=Path, required=True)
    p.add_argument("--arena", default="3d_wired")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"\nLoading data: {args.harp_csv}")
    raw, ts, folders = load_cleaned_motion(str(args.harp_csv))
    sensor     = process_motion(raw)
    bin_edges  = ARENA_BIN_EDGES[args.arena]
    chan_sizes  = [len(e) - 1 for e in bin_edges]

    hist_matrix = extract_histogram_features(sensor, bin_edges)
    emp_matrix  = extract_empirical_features(sensor)
    hist_feats  = _build_cdf_features(hist_matrix, chan_sizes).astype(np.float32)
    emp_feats   = emp_matrix.astype(np.float32)

    N = hist_feats.shape[0]
    print(f"Windows: {N}  |  hist dim: {hist_feats.shape[1]}  |  emp dim: {emp_feats.shape[1]}")

    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*60}")
    print("Running split-half AP stability test")
    print(f"{'='*60}")

    mean_h, std_h = splithalfAP(hist_feats, "histogram ", rng)
    mean_e, std_e = splithalfAP(emp_feats,  "empirical ", rng)

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"  Histogram  (30-dim CDF):      AMI = {mean_h:.4f} ± {std_h:.4f}")
    print(f"  Empirical (240-dim sorted):   AMI = {mean_e:.4f} ± {std_e:.4f}")
    delta = mean_e - mean_h
    winner = "empirical" if delta > 0 else "histogram"
    print(f"  ΔAMI (emp - hist) = {delta:+.4f}   Winner: {winner}")


if __name__ == "__main__":
    main()
