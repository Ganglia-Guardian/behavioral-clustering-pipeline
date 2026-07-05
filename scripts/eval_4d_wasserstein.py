"""
eval_4d_wasserstein.py
----------------------
Split-half AMI evaluation comparing 4D joint Wasserstein methods
against the current histogram CDF-L1 (1D marginal W1) baseline.

4D distance methods are in wasserstein_4d.py.
1D histogram CDF-L1 baseline is in clustering_pipeline.py.

Usage
-----
    # Fastest (~5s)
    python scripts/eval_4d_wasserstein.py --method sliced_w1 \\
        --harp-csv results/short_comparison_test/combined_harp_data_cleaned.csv

    # Medium (~5 min, full N)
    python scripts/eval_4d_wasserstein.py --method exact_emd ...

    # Slow — subsample recommended
    python scripts/eval_4d_wasserstein.py --method sinkhorn --max-windows 400 ...

    # Check runtime before committing
    python scripts/eval_4d_wasserstein.py --method sinkhorn --estimate-time ...
"""

import argparse
import sys
import time
import numpy as np
import scipy.spatial.distance as ssd
from pathlib import Path
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))

# ── 4D joint Wasserstein (new) ───────────────────────────────────────────────
import wasserstein_4d as w4d

# ── 1D marginal W1 via histogram CDF-L1 (existing pipeline) ─────────────────
from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, _build_cdf_features,
    ARENA_BIN_EDGES,
)

N_SPLITS  = 5
DAMPING   = 0.9
MAX_ITER  = 1000
CONV_ITER = 15


# ── AP from precomputed distance matrix ──────────────────────────────────────

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


# ── Split-half evaluation ────────────────────────────────────────────────────

def splithalfAP(data: np.ndarray, pw_fn, cx_fn,
                rng: np.random.Generator, label: str) -> tuple:
    """
    5-fold split-half stability test using AP-full.

    pw_fn(X)     -> (n, n) pairwise distance matrix
    cx_fn(X, Y)  -> (n_X, n_Y) cross distance matrix
    """
    N = data.shape[0]
    ami_scores = []

    for s in range(N_SPLITS):
        perm  = rng.permutation(N)
        idx_A, idx_B = perm[:N // 2], perm[N // 2:]
        print(f"\n  [{label}] split {s+1}/{N_SPLITS}  "
              f"N_A={len(idx_A)}  N_B={len(idx_B)}")

        print(f"    dist_A ({len(idx_A)}×{len(idx_A)})...")
        dist_A           = pw_fn(data[idx_A])
        labels_A, ex_loc = ap_from_dist(dist_A)
        K_A              = len(np.unique(labels_A))
        print(f"    K_A = {K_A}")

        print(f"    assigning B to {K_A} exemplars...")
        cross      = cx_fn(data[idx_A][ex_loc], data[idx_B])
        assigned_B = np.argmin(cross, axis=0)

        print(f"    dist_B ({len(idx_B)}×{len(idx_B)})...")
        dist_B    = pw_fn(data[idx_B])
        labels_B, _ = ap_from_dist(dist_B)
        K_B       = len(np.unique(labels_B))

        score = adjusted_mutual_info_score(assigned_B, labels_B)
        ami_scores.append(score)
        print(f"    K_A={K_A}  K_B={K_B}  AMI={score:.4f}")

    mean_ami = float(np.mean(ami_scores))
    std_ami  = float(np.std(ami_scores))
    print(f"\n  [{label}]  mean AMI = {mean_ami:.4f} ± {std_ami:.4f}")
    return mean_ami, std_ami


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--harp-csv',      type=Path, required=True)
    p.add_argument('--method',        choices=['sliced_w1', 'sinkhorn', 'exact_emd'],
                   default='sliced_w1')
    p.add_argument('--arena',         default='3d_wired')
    p.add_argument('--L',             type=int,   default=100,
                   help='Projection count for sliced_w1')
    p.add_argument('--reg',           type=float, default=0.05,
                   help='Entropy regularization for sinkhorn')
    p.add_argument('--max-windows',   type=int,   default=None,
                   help='Subsample to N windows (recommended for sinkhorn)')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--estimate-time', action='store_true',
                   help='Estimate runtime and exit (no evaluation)')
    args = p.parse_args()

    print(f"\nLoading: {args.harp_csv}")
    raw, ts, folders = load_cleaned_motion(str(args.harp_csv))
    sensor            = process_motion(raw)

    # 4D joint data
    joint = w4d.extract_joint(sensor)
    N     = joint.shape[0]
    print(f"Windows: {N}  |  joint shape: {joint.shape}")

    # Optional subsample
    idx = None
    if args.max_windows and N > args.max_windows:
        rng0 = np.random.default_rng(args.seed)
        idx   = rng0.choice(N, args.max_windows, replace=False)
        joint = joint[idx]
        N     = args.max_windows
        print(f"Subsampled to {N} windows")

    # Time estimation mode
    if args.estimate_time:
        kw = {'L': args.L, 'seed': args.seed} if args.method == 'sliced_w1' \
             else {'reg': args.reg} if args.method == 'sinkhorn' else {}
        w4d.estimate_time(joint, args.method, n_splits=N_SPLITS, **kw)
        return

    # 1D histogram CDF-L1 baseline features
    bin_edges  = ARENA_BIN_EDGES[args.arena]
    chan_sizes = [len(e) - 1 for e in bin_edges]
    hist_mat   = extract_histogram_features(sensor, bin_edges)
    hist_feats = _build_cdf_features(hist_mat, chan_sizes).astype(np.float32)
    if idx is not None:
        hist_feats = hist_feats[idx]

    # Distance function wrappers
    method_kw = {}
    if args.method == 'sliced_w1':
        method_kw = {'L': args.L, 'seed': args.seed}
    elif args.method == 'sinkhorn':
        method_kw = {'reg': args.reg}

    pw_4d = lambda X:    w4d.pairwise(X, args.method, **method_kw)
    cx_4d = lambda X, Y: w4d.cross(X, Y, args.method, **method_kw)

    pw_h  = lambda X:    ssd.cdist(X, X, metric='cityblock').astype(np.float64)
    cx_h  = lambda X, Y: ssd.cdist(X, Y, metric='cityblock').astype(np.float64)

    rng = np.random.default_rng(args.seed)

    # ── 4D evaluation ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"4D {args.method}  (N={N})")
    print(f"{'='*60}")
    t0 = time.time()
    mean_4d, std_4d = splithalfAP(joint, pw_4d, cx_4d,
                                  rng=np.random.default_rng(args.seed),
                                  label=args.method)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── 1D histogram baseline ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"1D Histogram CDF-L1 baseline  (N={N})")
    print(f"{'='*60}")
    t0 = time.time()
    mean_h, std_h = splithalfAP(hist_feats, pw_h, cx_h,
                                rng=np.random.default_rng(args.seed),
                                label='histogram')
    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    delta  = mean_4d - mean_h
    winner = args.method if delta > 0 else 'histogram'
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  1D Histogram CDF-L1:         AMI = {mean_h:.4f} ± {std_h:.4f}")
    print(f"  4D {args.method:<15}:    AMI = {mean_4d:.4f} ± {std_4d:.4f}")
    print(f"  delta AMI (4D - hist) = {delta:+.4f}   Winner: {winner}")


if __name__ == '__main__':
    main()
