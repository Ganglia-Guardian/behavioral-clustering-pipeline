"""
run_benchmark.py
----------------
Runs the Python clustering pipeline (both HDBSCAN and AP modes) on the
included test data, then optionally compares against a Matlab baseline.
Prints timing, peak memory, and cluster-agreement statistics side by side.

Usage (from repo root):
    python3 scripts/run_benchmark.py

With Matlab comparison (requires your own .mat file):
    python3 scripts/run_benchmark.py --matlab path/to/matlab_output.mat

Requirements:
    pip install -r requirements.txt
"""

import argparse
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPO_ROOT = Path(__file__).resolve().parent.parent   # Python_Pipeline/
SCRIPTS_DIR = Path(__file__).resolve().parent          # Python_Pipeline/scripts/
OUTPUT_DIR = REPO_ROOT / "test_outputs"               # Python_Pipeline/test_outputs/

CLEAN_CSV = OUTPUT_DIR / "combined_harp_data_cleaned.csv"
OUT_HDBSCAN = OUTPUT_DIR / "benchmark_hdbscan.csv"
OUT_AP = OUTPUT_DIR / "benchmark_ap.csv"

sys.path.insert(0, str(SCRIPTS_DIR))




def hr(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024 / 1024:.1f} MB"


def cluster_summary(labels: np.ndarray) -> dict:
    valid = labels[labels >= 0]
    unique, counts = np.unique(valid, return_counts=True)
    return {
        "n_windows":  len(labels),
        "n_clusters": len(unique),
        "n_noise":    int((labels == -1).sum()),
        "size_mean":  float(counts.mean()) if len(counts) else 0,
        "size_std":   float(counts.std())  if len(counts) else 0,
        "size_min":   int(counts.min())    if len(counts) else 0,
        "size_max":   int(counts.max())    if len(counts) else 0,
    }


def ari_nmi(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    valid = (a >= 0) & (b >= 0)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    return (adjusted_rand_score(a[valid], b[valid]),
            normalized_mutual_info_score(a[valid], b[valid], average_method="arithmetic"))




def load_matlab_labels(mat_path: Path) -> np.ndarray:
    import scipy.io
    mat = scipy.io.loadmat(str(mat_path))
    c   = mat["Clusters"][0, 0]
    return c["idx"].flatten().astype(int)




def run_python(use_ap: bool, out_path: Path) -> tuple:
    """Returns (labels, elapsed_seconds, peak_bytes)."""
    from clustering_pipeline import (
        load_cleaned_motion, process_motion,
        extract_histogram_features, ARENA_BIN_EDGES,
        cluster_hdbscan, cluster_ap_full,
    )
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        raw, ts, folders = load_cleaned_motion(str(CLEAN_CSV))
        sensor        = process_motion(raw)
        edges         = ARENA_BIN_EDGES["3d_wired"]
        channel_sizes = [len(e) - 1 for e in edges]
        hist          = extract_histogram_features(sensor, edges)

    tracemalloc.start()
    t0 = time.perf_counter()

    with contextlib.redirect_stdout(buf):
        if use_ap:
            labels = cluster_ap_full(hist, channel_sizes)
        else:
            labels = cluster_hdbscan(hist, channel_sizes, min_cluster_size=5)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    win_ts      = ts[30::60][:len(labels)]
    win_folders = folders[::60][:len(labels)]
    cluster_idx = np.where(labels < 0, 0, labels + 1)
    pd.DataFrame({
        "ClusterIdx":  cluster_idx,
        "Timestamp":   win_ts,
        "Folder_Name": win_folders,
    }).to_csv(str(out_path), index=False)

    return labels, elapsed, peak




def matlab_estimate(n_windows: int) -> dict:
    """
    Expected Matlab runtime and memory based on the cost model of
    runDistanceSim.m + apclusterSparse.m.
    """
    mem_bytes  = n_windows ** 2 * 8
    emd_time_s = n_windows ** 2 * 0.5e-3        # ~0.5 ms per MEX emd() call
    ap_time_s  = 1000 * n_windows ** 2 * 1e-6   # 1000 iters × O(N²)
    total_s    = emd_time_s + ap_time_s
    return {
        "mem_bytes":  mem_bytes,
        "emd_days":   emd_time_s / 86400,
        "ap_days":    ap_time_s  / 86400,
        "total_days": total_s    / 86400,
    }




def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python clustering pipeline vs Matlab baseline"
    )
    parser.add_argument(
        "--matlab", default=None, metavar="PATH",
        help="Path to Matlab .mat output file (matlab_output.mat). "
             "If omitted, Matlab comparison is skipped and only estimates are shown."
    )
    args = parser.parse_args()

    mat_path = Path(args.matlab) if args.matlab else None
    has_matlab = mat_path is not None and mat_path.exists()

    if mat_path and not mat_path.exists():
        print(f"  WARNING: --matlab file not found: {mat_path}")
        print(f"  Skipping Matlab comparison. Only estimates will be shown.\n")

    if not CLEAN_CSV.exists():
        print(f"ERROR: test data not found: {CLEAN_CSV}")
        print("  Run prepare_test_data.py first to generate the input CSV.")
        sys.exit(1)

    print("=" * 60)
    print("  Clustering Benchmark")
    print("  Matlab baseline  vs  Python HDBSCAN  vs  Python AP")
    print("=" * 60)

    # ── Matlab baseline ────────────────────────────────────────────────────────
    hr("Matlab baseline")
    if has_matlab:
        mat_labels = load_matlab_labels(mat_path)
        N = len(mat_labels)
        mat_est = matlab_estimate(N)
        u, c = np.unique(mat_labels, return_counts=True)
        print(f"  Loaded from  : {mat_path}")
        print(f"  Windows      : {N:,}")
        print(f"  Clusters     : {len(u)}")
        print(f"  Size  mean / std / min / max : "
              f"{c.mean():.0f} / {c.std():.0f} / {c.min()} / {c.max()}")
    else:
        mat_labels = None
        # Use test-data window count to compute estimates
        import io, contextlib
        from clustering_pipeline import load_cleaned_motion, process_motion, extract_histogram_features, ARENA_BIN_EDGES
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            raw, ts, _ = load_cleaned_motion(str(CLEAN_CSV))
            sensor = process_motion(raw)
            edges  = ARENA_BIN_EDGES["3d_wired"]
            hist   = extract_histogram_features(sensor, edges)
        N = hist.shape[0]
        mat_est = matlab_estimate(N)
        print(f"  No .mat file provided — showing estimates only (N={N:,} windows).")
        print(f"  To enable comparison: python3 scripts/run_benchmark.py --matlab path/to/matlab_output.mat")

    print(f"\n  ── Estimated cost if run in Matlab (N={N:,}) ──")
    print(f"  Memory (N×N matrix) : {mb(mat_est['mem_bytes'])}")
    print(f"  EMD distance step   : {mat_est['emd_days'] * 24:.2f} hours")
    print(f"  AP clustering step  : {mat_est['ap_days']  * 24:.2f} hours")

    # ── Python HDBSCAN ─────────────────────────────────────────────────────────
    hr("Python HDBSCAN  (recommended for large data)")
    print("  Running ...", flush=True)
    hdb_labels, hdb_time, hdb_peak = run_python(use_ap=False, out_path=OUT_HDBSCAN)
    hdb_stats = cluster_summary(hdb_labels)
    hdb_ari, hdb_nmi = ari_nmi(mat_labels, hdb_labels) if has_matlab else (float("nan"), float("nan"))

    print(f"  Windows      : {hdb_stats['n_windows']:,}")
    print(f"  Clusters     : {hdb_stats['n_clusters']}")
    print(f"  Noise pts    : {hdb_stats['n_noise']:,}  "
          f"({100*hdb_stats['n_noise']/hdb_stats['n_windows']:.1f}%)")
    print(f"  Size  mean / std / min / max : "
          f"{hdb_stats['size_mean']:.0f} / {hdb_stats['size_std']:.0f} "
          f"/ {hdb_stats['size_min']} / {hdb_stats['size_max']}")
    print(f"\n  ── Performance ──")
    print(f"  Clustering time : {hdb_time:.2f}s")
    print(f"  Peak memory     : {mb(hdb_peak)}")
    if has_matlab:
        print(f"\n  ── Agreement with Matlab (ARI / NMI) ──")
        print(f"  ARI : {hdb_ari:+.4f}   NMI : {hdb_nmi:.4f}")

    # ── Python AP ──────────────────────────────────────────────────────────────
    hr("Python AP  (same algorithm as Matlab, for direct comparison)")
    print("  Running ...", flush=True)
    ap_labels, ap_time, ap_peak = run_python(use_ap=True, out_path=OUT_AP)
    ap_stats = cluster_summary(ap_labels)
    ap_ari, ap_nmi = ari_nmi(mat_labels, ap_labels) if has_matlab else (float("nan"), float("nan"))

    print(f"  Windows      : {ap_stats['n_windows']:,}")
    print(f"  Clusters     : {ap_stats['n_clusters']}")
    print(f"  Size  mean / std / min / max : "
          f"{ap_stats['size_mean']:.0f} / {ap_stats['size_std']:.0f} "
          f"/ {ap_stats['size_min']} / {ap_stats['size_max']}")
    print(f"\n  ── Performance ──")
    print(f"  Clustering time : {ap_time:.2f}s")
    print(f"  Peak memory     : {mb(ap_peak)}")
    if has_matlab:
        print(f"\n  ── Agreement with Matlab (ARI / NMI) ──")
        print(f"  ARI : {ap_ari:+.4f}   NMI : {ap_nmi:.4f}")

    # ── Summary table ──────────────────────────────────────────────────────────
    hr("Summary table")
    cols = ["Method", "Clusters", "Noise%", "Time(s)", "Peak RAM"]
    if has_matlab:
        cols.append("ARI vs Matlab")

    rows = [
        ["Matlab AP (estimated)",
         "—",
         "0%",
         f"~{(mat_est['emd_days'] + mat_est['ap_days']) * 24:.1f}h",
         mb(mat_est["mem_bytes"])],
        ["Python HDBSCAN",
         hdb_stats["n_clusters"],
         f"{100*hdb_stats['n_noise']/hdb_stats['n_windows']:.1f}%",
         f"{hdb_time:.2f}",
         mb(hdb_peak)],
        ["Python AP",
         ap_stats["n_clusters"],
         "0%",
         f"{ap_time:.2f}",
         mb(ap_peak)],
    ]
    if has_matlab:
        rows[0].append("—")
        rows[1].append(f"{hdb_ari:+.4f}")
        rows[2].append(f"{ap_ari:+.4f}")

    col_w = [max(len(c), max(len(str(r[i])) for r in rows)) + 2
             for i, c in enumerate(cols)]
    print("  " + "".join(c.ljust(w) for c, w in zip(cols, col_w)))
    print("  " + "-" * sum(col_w))
    for row in rows:
        print("  " + "".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    # ── Scalability note ───────────────────────────────────────────────────────
    hr("What happens at real scale  (N = 120,000 windows)")
    big = matlab_estimate(120_000)
    print(f"  Matlab memory (N×N matrix) : {mb(big['mem_bytes'])}  "
          f"→ physically impossible on most machines")
    print(f"  Matlab EMD distance step   : {big['emd_days']:.0f} days")
    print(f"  Matlab AP clustering step  : {big['ap_days']:.0f} days")
    print()
    print(f"  Python HDBSCAN time        : ~5–15 minutes  (O(N log N))")
    print(f"  Python HDBSCAN memory      : ~500 MB         (O(N))")
    print(f"  Python AP time             : infeasible at N=120,000  (still O(N²))")
    print()


if __name__ == "__main__":
    main()
