"""
compare_results.py
------------------
Loads Matlab AP-clustering results and Python HDBSCAN results for the same
recording and prints a side-by-side comparison.

Metrics reported
----------------
  - Number of windows processed
  - Number of unique clusters found
  - Cluster-size distribution (mean / std / min / max)
  - Adjusted Rand Index  (ARI, range −1..1, higher = more similar)
  - Normalized Mutual Information  (NMI, range 0..1, higher = more similar)

Usage (from repo root)
-----
    python3 Python_Pipeline/scripts/compare_results.py \
        --matlab   path/to/matlab_output.mat \
        --python   path/to/Cluster_detail_results.csv

Optional:
    --matlab-key   Clusters        # struct key inside .mat file (default: Clusters)
    --matlab-field idx             # field that holds per-window labels (default: idx)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_matlab(mat_path: str, key: str = "Clusters", field: str = "idx") -> np.ndarray:
    """
    Load per-window cluster labels from a .mat file saved by VPAPPAxes.m.

    The file must contain a struct stored in `key`; the per-window assignment
    vector lives in `struct.field` (default Clusters.idx), shape (N,1), dtype uint8.
    Returns a 1-D int array of length N.
    """
    mat = scipy.io.loadmat(mat_path)
    if key not in mat:
        raise KeyError(f"Key '{key}' not found in {mat_path}. "
                       f"Available keys: {[k for k in mat if not k.startswith('_')]}")
    struct = mat[key][0, 0]
    if field not in struct.dtype.names:
        raise KeyError(f"Field '{field}' not in struct. "
                       f"Available fields: {struct.dtype.names}")
    labels = struct[field].flatten().astype(int)
    print(f"[Matlab]  Loaded {len(labels):,} window labels from '{key}.{field}'")
    return labels


def load_python(csv_path: str) -> np.ndarray:
    """
    Load per-window cluster labels from Cluster_detail_results.csv produced by
    clustering_pipeline.py.  Expected column: 'ClusterIdx'.
    Returns a 1-D int array.
    """
    df = pd.read_csv(csv_path)
    if "ClusterIdx" not in df.columns:
        raise ValueError(f"Column 'ClusterIdx' not found. Columns: {df.columns.tolist()}")
    labels = df["ClusterIdx"].values.astype(int)
    print(f"[Python]  Loaded {len(labels):,} window labels from '{csv_path}'")
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Comparison helpers
# ──────────────────────────────────────────────────────────────────────────────

def cluster_stats(labels: np.ndarray, name: str) -> None:
    unique = np.unique(labels[labels >= 0])      # exclude HDBSCAN outlier label -1
    outliers = np.sum(labels == -1)
    sizes = [np.sum(labels == c) for c in unique]
    print(f"\n  [{name}]")
    print(f"    Windows total     : {len(labels):,}")
    print(f"    Clusters found    : {len(unique)}")
    if outliers:
        print(f"    Outlier windows   : {outliers:,}  (label = −1, not assigned to any cluster)")
    if sizes:
        print(f"    Cluster size      : mean={np.mean(sizes):.1f}  std={np.std(sizes):.1f}"
              f"  min={np.min(sizes)}  max={np.max(sizes)}")
        top5 = sorted(zip(unique, sizes), key=lambda x: -x[1])[:5]
        print(f"    Top-5 clusters    : {[(int(c), int(s)) for c, s in top5]}")


def compare(matlab_labels: np.ndarray, python_labels: np.ndarray) -> None:
    n_mat = len(matlab_labels)
    n_py  = len(python_labels)

    if n_mat != n_py:
        print(f"\n  ⚠  Window count mismatch: Matlab has {n_mat:,}, Python has {n_py:,}.")
        n = min(n_mat, n_py)
        print(f"     Comparing only the first {n:,} windows.")
        matlab_labels = matlab_labels[:n]
        python_labels = python_labels[:n]

    # Drop windows where either pipeline marked -1 (HDBSCAN outliers)
    valid = (matlab_labels >= 0) & (python_labels >= 0)
    n_valid = valid.sum()
    if n_valid < len(matlab_labels):
        print(f"     Dropping {len(matlab_labels) - n_valid:,} windows where at least one "
              f"pipeline returned −1 (outlier).")

    m = matlab_labels[valid]
    p = python_labels[valid]

    ari = adjusted_rand_score(m, p)
    nmi = normalized_mutual_info_score(m, p, average_method="arithmetic")

    print(f"\n  ── Similarity between the two clusterings ──")
    print(f"    Windows compared  : {n_valid:,}")
    print(f"    Adjusted Rand Index (ARI) : {ari:+.4f}  "
          f"  (1.0 = identical, 0.0 = random, −1.0 = maximally different)")
    print(f"    Normalized Mutual Info    : {nmi:.4f}  "
          f"  (1.0 = identical, 0.0 = no shared information)")

    if ari > 0.8:
        verdict = "Excellent agreement — the two algorithms found very similar structure."
    elif ari > 0.5:
        verdict = "Good agreement — major groupings are consistent."
    elif ari > 0.2:
        verdict = "Moderate agreement — some structure is shared but assignments differ."
    else:
        verdict = "Low agreement — the two algorithms found quite different groupings."
    print(f"\n  Interpretation: {verdict}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Matlab vs Python clustering results")
    parser.add_argument("--matlab",        required=True,  help="Path to Matlab .mat output file")
    parser.add_argument("--python",        required=True,  help="Path to Python Cluster_detail_results.csv")
    parser.add_argument("--matlab-key",    default="Clusters", help="Struct key in .mat file")
    parser.add_argument("--matlab-field",  default="idx",       help="Field with per-window labels")
    parser.add_argument("--python-label",  default="HDBSCAN + FAISS ANN",
                        help="Algorithm label for the Python results (e.g. 'Affinity Propagation')")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Clustering Comparison  —  Matlab (AP) vs Python ({args.python_label})")
    print("=" * 60)

    matlab_labels  = load_matlab(args.matlab, args.matlab_key, args.matlab_field)
    python_labels  = load_python(args.python)

    print("\n── Per-algorithm statistics ──")
    cluster_stats(matlab_labels, "Matlab  (Affinity Propagation)")
    cluster_stats(python_labels, f"Python  ({args.python_label})")

    compare(matlab_labels, python_labels)
    print()


if __name__ == "__main__":
    main()
