"""
preference_sweep.py
-------------------
Sweep AP preference from min to median affinity, record K and split-half AMI.
Finds the preference value that maximises clustering stability.

Usage
-----
    python scripts/preference_sweep.py \
        --harp-csv results/short_comparison_test/combined_harp_data_cleaned.csv \
        --n-steps 12 --n-splits 3
"""
import argparse
import sys
import numpy as np
import scipy.spatial.distance as ssd
from pathlib import Path
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, _build_cdf_features, ARENA_BIN_EDGES,
)

DAMPING   = 0.9
MAX_ITER  = 1000
CONV_ITER = 15


def run_ap(affinity, pref):
    aff = affinity.copy()
    np.fill_diagonal(aff, pref)
    ap = AffinityPropagation(
        affinity='precomputed', preference=pref,
        damping=DAMPING, max_iter=MAX_ITER,
        convergence_iter=CONV_ITER, random_state=0,
    )
    labels = ap.fit_predict(aff)
    return labels, ap.cluster_centers_indices_


def split_half_ami(feats, affinity, pref, rng, n_splits):
    N = feats.shape[0]
    scores = []
    for _ in range(n_splits):
        perm  = rng.permutation(N)
        idx_A, idx_B = perm[:N//2], perm[N//2:]

        aff_A = affinity[np.ix_(idx_A, idx_A)]
        labels_A, ex_loc = run_ap(aff_A, pref)
        if len(ex_loc) == 0:
            continue
        ex_feats   = feats[idx_A][ex_loc]
        cross      = ssd.cdist(feats[idx_B], ex_feats, metric='cityblock')
        assigned_B = np.argmin(cross, axis=1)

        aff_B = affinity[np.ix_(idx_B, idx_B)]
        labels_B, _ = run_ap(aff_B, pref)

        scores.append(adjusted_mutual_info_score(assigned_B, labels_B))
    return float(np.mean(scores)) if scores else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--harp-csv', type=Path, required=True)
    p.add_argument('--arena',    default='3d_wired')
    p.add_argument('--n-steps',  type=int, default=12)
    p.add_argument('--n-splits', type=int, default=3)
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    print(f"Loading: {args.harp_csv}")
    raw, ts, _ = load_cleaned_motion(str(args.harp_csv))
    sensor     = process_motion(raw)
    bin_edges  = ARENA_BIN_EDGES[args.arena]
    chan_sizes  = [len(e)-1 for e in bin_edges]
    hist_mat   = extract_histogram_features(sensor, bin_edges)
    feats      = _build_cdf_features(hist_mat, chan_sizes).astype(np.float32)
    N          = feats.shape[0]
    print(f"Windows: {N}")

    print("Building full affinity matrix...")
    dist     = ssd.cdist(feats, feats, metric='cityblock').astype(np.float64)
    affinity = -(dist ** 2)

    pref_min    = float(affinity.min())
    pref_median = float(np.median(affinity))
    prefs       = np.linspace(pref_min, pref_median, args.n_steps)

    rng = np.random.default_rng(args.seed)

    print(f"\n{'Preference':>12} {'K':>5} {'Split-half AMI':>16}")
    print("-" * 38)

    best_ami, best_pref, best_K = -1, None, None
    results = []

    for pref in prefs:
        labels, _ = run_ap(affinity, pref)
        K         = len(np.unique(labels))
        ami       = split_half_ami(feats, affinity, pref, rng, args.n_splits)
        marker    = " ←" if ami > best_ami else ""
        print(f"{pref:>12.4f} {K:>5} {ami:>16.4f}{marker}")
        results.append((pref, K, ami))
        if ami > best_ami:
            best_ami, best_pref, best_K = ami, pref, K

    print(f"\nBest: preference={best_pref:.4f}  K={best_K}  AMI={best_ami:.4f}")

    out = Path("results") / f"preference_sweep_{args.harp_csv.parent.name}.csv"
    import pandas as pd
    pd.DataFrame(results, columns=["preference","K","split_half_ami"]).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
