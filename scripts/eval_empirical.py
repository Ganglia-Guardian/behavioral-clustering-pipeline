"""
eval_empirical.py
-----------------
Compare histogram CDF-L1 vs empirical W1 features using AP-full as reference.

Workflow
--------
1. Load existing ap_full Cluster_detail_results.csv as reference labels.
2. Load the matching combined_harp_data_cleaned.csv and extract BOTH feature sets:
     • histogram → CDF (30-dim, current method)
     • empirical → sorted quantiles (240-dim, proposed method)
3. Run KMeans with K = ap_full cluster count on both feature sets.
4. Report ARI vs ap_full for each — higher ARI means closer to the reference.

Why KMeans instead of re-running AP?
  AP takes 30+ seconds per run and its cluster count varies with features.
  KMeans fixes K to the ap_full count so the only variable is the features.

Usage
-----
    python scripts/eval_empirical.py \\
        --results-dir  results/comparison_test \\
        --harp-csv     results/comparison_test/combined_harp_data_cleaned.csv

    # Run on all datasets that have ap_full results:
    python scripts/eval_empirical.py --all
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, extract_empirical_features,
    _build_cdf_features, ARENA_BIN_EDGES,
)

RESULTS_ROOT = Path(__file__).parent.parent / "results"
N_PAIRS      = 2000   # pairs for Spearman distance comparison


# ── helpers ────────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df["ClusterIdx"].values.astype(int)


def run_kmeans(features: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=K, n_init=5, random_state=seed)
    return km.fit_predict(features)


def spearman_comparison(hist_feats: np.ndarray, emp_feats: np.ndarray,
                         rng: np.random.Generator) -> tuple[float, float]:
    """Spearman ρ between histogram and empirical pairwise distances."""
    N = hist_feats.shape[0]
    idx = rng.integers(0, N, size=(N_PAIRS, 2))
    hist_dists = np.sum(np.abs(hist_feats[idx[:, 0]] - hist_feats[idx[:, 1]]), axis=1)
    emp_dists  = np.sum(np.abs(emp_feats [idx[:, 0]] - emp_feats [idx[:, 1]]), axis=1)
    rho, pval  = spearmanr(hist_dists, emp_dists)
    # rank-flip rate
    flipped = total = 0
    for k in range(N_PAIRS):
        for l in range(k + 1, min(k + 50, N_PAIRS)):
            if (hist_dists[k] < hist_dists[l]) != (emp_dists[k] < emp_dists[l]):
                flipped += 1
            total += 1
    return float(rho), flipped / total


def evaluate_dataset(results_dir: Path, harp_csv: Path) -> dict:
    ap_full_csv = results_dir / "Cluster_detail_results_ap_full.csv"
    if not ap_full_csv.exists():
        return {"dataset": results_dir.name, "status": "no ap_full result"}

    print(f"\n{'='*60}")
    print(f"Dataset: {results_dir.name}")
    print(f"{'='*60}")

    ref_labels = load_labels(ap_full_csv)
    K          = len(np.unique(ref_labels))
    print(f"  ap_full: {len(ref_labels)} windows, {K} clusters")

    # Extract features
    print("  Extracting features...")
    raw, ts, folders = load_cleaned_motion(str(harp_csv))
    sensor      = process_motion(raw)
    bin_edges   = ARENA_BIN_EDGES["3d_wired"]
    chan_sizes  = [len(e) - 1 for e in bin_edges]

    hist_matrix = extract_histogram_features(sensor, bin_edges)
    emp_matrix  = extract_empirical_features(sensor)
    hist_feats  = _build_cdf_features(hist_matrix, chan_sizes).astype(np.float32)
    emp_feats   = emp_matrix.astype(np.float32)

    N = hist_feats.shape[0]
    N_ref = len(ref_labels)
    if N != N_ref:
        print(f"  Warning: window count mismatch ({N} vs {N_ref}), truncating")
        N = min(N, N_ref)
        hist_feats = hist_feats[:N]
        emp_feats  = emp_feats[:N]
        ref_labels = ref_labels[:N]

    # Spearman distance comparison
    print(f"  Computing Spearman ρ on {N_PAIRS} random pairs...")
    rng = np.random.default_rng(42)
    rho, flip_rate = spearman_comparison(hist_feats, emp_feats, rng)

    # KMeans clustering
    print(f"  Running KMeans (K={K}) on both feature sets...")
    labels_hist = run_kmeans(hist_feats, K)
    labels_emp  = run_kmeans(emp_feats,  K)

    ari_hist = adjusted_rand_score(ref_labels, labels_hist)
    ari_emp  = adjusted_rand_score(ref_labels, labels_emp)
    ari_h_e  = adjusted_rand_score(labels_hist, labels_emp)

    # Silhouette (subsample for speed)
    sub = rng.choice(N, min(2000, N), replace=False)
    sil_hist = silhouette_score(hist_feats[sub], labels_hist[sub], metric="l1")
    sil_emp  = silhouette_score(emp_feats[sub],  labels_emp[sub],  metric="l1")

    result = {
        "dataset":    results_dir.name,
        "n_windows":  N,
        "K":          K,
        "spearman_rho":   round(rho, 4),
        "flip_rate_pct":  round(flip_rate * 100, 1),
        "ari_histogram":  round(ari_hist, 4),
        "ari_empirical":  round(ari_emp,  4),
        "ari_hist_vs_emp": round(ari_h_e, 4),
        "sil_histogram":  round(sil_hist, 4),
        "sil_empirical":  round(sil_emp,  4),
        "winner":     "empirical" if ari_emp > ari_hist else "histogram",
        "delta_ari":  round(ari_emp - ari_hist, 4),
    }

    print(f"\n  ── Distance-level ──────────────────────────────")
    print(f"  Spearman ρ              = {rho:.4f}")
    print(f"  Rank-flip rate          = {flip_rate*100:.1f}%")
    print(f"\n  ── Clustering quality (vs ap_full) ─────────────")
    print(f"  ARI  histogram          = {ari_hist:.4f}")
    print(f"  ARI  empirical          = {ari_emp:.4f}   {'▲ better' if ari_emp > ari_hist else '▼ worse'}")
    print(f"  ARI  hist vs emp        = {ari_h_e:.4f}   (how different the two are)")
    print(f"\n  ── Silhouette (L1) ─────────────────────────────")
    print(f"  Silhouette histogram    = {sil_hist:.4f}")
    print(f"  Silhouette empirical    = {sil_emp:.4f}   {'▲ better' if sil_emp > sil_hist else '▼ worse'}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Single results directory (must contain ap_full CSV + harp CSV)")
    p.add_argument("--harp-csv", type=Path, default=None,
                   help="combined_harp_data_cleaned.csv for --results-dir")
    p.add_argument("--all", action="store_true",
                   help="Run on all datasets under results/ that have ap_full")
    return p


def main():
    args = _build_parser().parse_args()
    results = []

    if args.all:
        for d in sorted(RESULTS_ROOT.iterdir()):
            if not d.is_dir():
                continue
            harp = d / "combined_harp_data_cleaned.csv"
            if harp.exists():
                r = evaluate_dataset(d, harp)
                results.append(r)
    elif args.results_dir and args.harp_csv:
        r = evaluate_dataset(args.results_dir, args.harp_csv)
        results.append(r)
    else:
        print("Specify --results-dir + --harp-csv, or --all")
        sys.exit(1)

    # Summary table
    valid = [r for r in results if "ari_histogram" in r]
    if len(valid) > 1:
        print(f"\n\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Dataset':<30} {'ARI hist':>9} {'ARI emp':>9} {'ΔARI':>7} {'Winner':<10}")
        print("-" * 60)
        for r in valid:
            print(f"{r['dataset']:<30} {r['ari_histogram']:>9.4f} "
                  f"{r['ari_empirical']:>9.4f} {r['delta_ari']:>+7.4f} "
                  f"{r['winner']:<10}")

        avg_delta = np.mean([r["delta_ari"] for r in valid])
        print(f"\n  Average ΔARI (empirical − histogram) = {avg_delta:+.4f}")
        if avg_delta > 0:
            print("  → empirical features are on average closer to ap_full")
        else:
            print("  → histogram features are on average closer to ap_full")

    out_csv = RESULTS_ROOT / "eval_empirical_vs_histogram.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
