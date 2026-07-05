"""
cluster_4d.py
-------------
Run AP-full clustering on the full dataset using three 4D joint W1 methods,
then compare their results.

Methods:
  sliced_w1  : Sliced W1, L=100 random projections   [~30s]
  exact_emd  : Exact W1 via LP (POT emd2)            [~3 min]
  sinkhorn   : Sinkhorn regularized OT               [~5 min with reg=0.5]

Output per method:
  - Number of clusters K
  - Cluster sizes
  - Saves labels to results/<dataset>/Cluster_detail_results_4d_<method>.csv

Cross-method comparison:
  - ARI between every pair of methods

Usage
-----
    python scripts/cluster_4d.py \\
        --harp-csv results/short_comparison_test/combined_harp_data_cleaned.csv \\
        --results-dir results/short_comparison_test
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from pathlib import Path
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import wasserstein_4d as w4d
from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    ARENA_BIN_EDGES,
)

DAMPING   = 0.9
MAX_ITER  = 1000
CONV_ITER = 15


def ap_from_dist(dist: np.ndarray):
    affinity = -(dist ** 2)
    pref     = float(affinity.min())
    np.fill_diagonal(affinity, pref)
    ap = AffinityPropagation(
        affinity='precomputed', preference=pref,
        damping=DAMPING, max_iter=MAX_ITER,
        convergence_iter=CONV_ITER, random_state=0,
    )
    labels = ap.fit_predict(affinity)
    return labels, ap.cluster_centers_indices_


def run_method(joint: np.ndarray, method: str, **kwargs) -> tuple:
    print(f"\n  Computing {method} distance matrix ({len(joint)}×{len(joint)})...")
    t0   = time.time()
    dist = w4d.pairwise(joint, method, **kwargs)
    t_dist = time.time() - t0
    print(f"  Distance matrix done in {t_dist:.1f}s")

    print(f"  Running AP full...")
    t1 = time.time()
    labels, exemplars = ap_from_dist(dist)
    t_ap = time.time() - t1

    K    = len(np.unique(labels))
    sizes = sorted([int((labels == c).sum()) for c in np.unique(labels)], reverse=True)
    print(f"  K={K}  AP time={t_ap:.1f}s")
    print(f"  Cluster sizes: {sizes}")
    return labels, exemplars, dist


def save_labels(labels: np.ndarray, timestamps: np.ndarray,
                out_path: Path, method: str):
    df = pd.DataFrame({
        'Timestamp':  timestamps[:len(labels)],
        'ClusterIdx': labels,
        'Method':     method,
    })
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--harp-csv',    type=Path, required=True)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--method',      choices=['sliced_w1', 'exact_emd', 'sinkhorn'],
                   required=True)
    p.add_argument('--arena',       default='3d_wired')
    p.add_argument('--L',           type=int,   default=100)
    p.add_argument('--reg',         type=float, default=0.5,
                   help='Sinkhorn regularization (default 0.5 for speed)')
    p.add_argument('--seed',        type=int,   default=42)
    args = p.parse_args()

    print(f"\nLoading: {args.harp_csv}")
    raw, ts, folders = load_cleaned_motion(str(args.harp_csv))
    sensor            = process_motion(raw)
    joint             = w4d.extract_joint(sensor)
    N                 = joint.shape[0]
    timestamps        = ts[: N * 60 : 60]   # one timestamp per window
    print(f"Windows: {N}  |  joint shape: {joint.shape}")

    kwargs = {}
    if args.method == 'sliced_w1':
        kwargs = {'L': args.L, 'seed': args.seed}
    elif args.method == 'sinkhorn':
        kwargs = {'reg': args.reg}

    print(f"\n{'='*60}")
    print(f"Method: {args.method}")
    print(f"{'='*60}")
    labels, exemplars, dist = run_method(joint, args.method, **kwargs)

    out = args.results_dir / f"Cluster_detail_results_4d_{args.method}.csv"
    save_labels(labels, timestamps, out, args.method)

    K     = len(np.unique(labels))
    sizes = sorted([(labels == c).sum() for c in np.unique(labels)], reverse=True)
    print(f"\n  K = {K}")
    print(f"  Cluster sizes: {sizes}")


if __name__ == '__main__':
    main()
