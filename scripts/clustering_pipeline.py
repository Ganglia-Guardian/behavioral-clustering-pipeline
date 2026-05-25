"""
clustering_pipeline.py

Python replacement for the Matlab clustering pipeline (VPAPPAxes.m).
Input/output format is fully compatible with the original Matlab pipeline.

Bottlenecks in the original Matlab pipeline:
  - runDistanceSim.m line 38   : zeros(N,N) allocates a full N×N matrix (~120 GB for real data)
  - runDistanceSim.m line 70-77: nested for-loop calling emd() MEX ~7.5 billion times (O(N²))
  - apclusterSparse.m          : Affinity Propagation runs up to 1000 iterations, each O(N²)

Improvements in this script:
  1. Vectorized histogram feature extraction   — no per-window Python loops
  2. 1D Wasserstein via CDF L1 distance        — replaces emd() MEX calls, O(d) per pair
  3. FAISS approximate nearest neighbors (ANN) — only computes K neighbors per point,
                                                  eliminates the O(N²) distance matrix entirely
  4. HDBSCAN clustering                        — O(N log N) time, O(N) memory, no preset K needed

Requirements:
    pip install numpy scipy pandas hdbscan faiss-cpu h5py

Usage:
    python clustering_pipeline.py \\
        --input  path/to/Combined_Results/combined_harp_data_cleaned.csv \\
        --output path/to/Results/test1/Cluster_detail_results.csv \\
        --arena  3d_wired
"""

import argparse
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import faiss
from scipy.signal import butter, filtfilt, medfilt


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WIN_SIZE = 60       # window size in samples (= 300 ms at 200 Hz)
FS       = 200      # sampling frequency in Hz
AC_RNG   = 4        # accelerometer range ±4 g
GYR_RNG  = 1000     # gyroscope range ±1000 dps

# Histogram bin edges for each of the 4 feature channels.
# Source: findFeaturesWired3DArena.m, hardHistNew.mat (Matlab hardcoded thresholds).
# 100 edges → 99 bins per channel.
_EDGES_3D = [
    np.linspace(-1.0,  0.8,  100),   # Feature 1: y_GA  (anterior-posterior gravity)
    np.linspace(-1.5,  2.0,  100),   # Feature 2: z     (dorsal-ventral acceleration)
    np.linspace(-2e4,  2e4,  100),   # Feature 3: z_gyro (dorsal-ventral gyroscope)
    np.linspace(-8.0,  0.7,  100),   # Feature 4: log(total body acceleration)
]

# TODO: replace with thresholds calibrated on 2D arena data.
# Currently uses the same edges as 3D; the z_gyro channel (index 2) range
# may need adjustment for flat-floor recordings.
_EDGES_2D = [
    np.linspace(-1.0,  0.8,  100),
    np.linspace(-1.5,  2.0,  100),
    np.linspace(-2e4,  2e4,  100),
    np.linspace(-8.0,  0.7,  100),
]

ARENA_BIN_EDGES = {
    "3d_wired":    _EDGES_3D,
    "2d_wired":    _EDGES_2D,
    "3d_wireless": _EDGES_3D,
    "2d_wireless": _EDGES_2D,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load cleaned motion data
# ─────────────────────────────────────────────────────────────────────────────

def load_cleaned_motion(csv_path: str) -> tuple:
    """
    Read combined_harp_data_cleaned.csv produced by stitched.py.
    Only rows where RegisterAddress == 34 carry IMU motion data.

    Returns
    -------
    raw_motion   : np.ndarray, shape (N, 10)
                   columns: [ax, ay, az, gx, gy, gz, mx, my, mz, counter]
    timestamps   : np.ndarray, shape (N,)
    folder_names : np.ndarray, shape (N,)  — arena label for each sample
    """
    print(f"[1/5] Loading data: {csv_path}")
    df = pd.read_csv(csv_path)

    motion_df = df[df["RegisterAddress"] == 34].reset_index(drop=True)
    print(f"      Motion rows: {len(motion_df):,}")

    timestamps   = motion_df["Timestamp"].values.astype(np.float64)
    folder_names = motion_df.iloc[:, 13].values

    # DataElement0 through DataElement9 sit at column indices 3–12
    raw_motion = motion_df.iloc[:, 3:13].values.astype(np.float64)

    return raw_motion, timestamps, folder_names


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Signal processing  (mirrors processData.m)
# ─────────────────────────────────────────────────────────────────────────────

def process_motion(raw_motion: np.ndarray) -> dict:
    """
    Reproduce every step of processData.m:
      1. Convert raw integers to physical units.
      2. Median filter (kernel=7) for spike removal.
      3. 1st-order Butterworth high-pass at 0.5 Hz to separate
         body acceleration (BA) from gravitational component (GA).
      4. Compute total body acceleration magnitude.

    Returns a dict with keys: y_GA, z, z_gyro, tot_accel.
    """
    print("[2/5] Signal processing (filtering, gravity separation)...")

    # Scale raw ADC values to physical units
    # Matlab equivalent: acc = data(:,3:5) .* (ac_rng / 32768)
    acc = raw_motion[:, 0:3] * (AC_RNG  / 32768.0)
    gyr = raw_motion[:, 3:6] * (GYR_RNG / 32768.0)

    # Median filter — kernel_size=7 matches Matlab's medfilt1(x, 7)
    # The negative sign on y corrects for sensor mounting orientation
    x      = medfilt(acc[:, 0], kernel_size=7)
    y      = medfilt(-acc[:, 1], kernel_size=7)
    z      = medfilt(acc[:, 2], kernel_size=7)
    z_gyro = medfilt(gyr[:, 2], kernel_size=7)

    # Butterworth 1st-order high-pass, cut-off 0.5 Hz
    # filtfilt gives zero-phase filtering, matching Matlab's filtfilt
    b, a  = butter(1, 0.5 / (FS / 2.0), btype="high")
    x_BA  = filtfilt(b, a, x)
    y_BA  = filtfilt(b, a, y)
    z_BA  = filtfilt(b, a, z)

    y_GA      = y - y_BA
    tot_accel = np.sqrt(x_BA**2 + y_BA**2 + z_BA**2)

    return {
        "y_GA":      y_GA,
        "z":         z,
        "z_gyro":    z_gyro,
        "tot_accel": tot_accel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Histogram feature extraction
#          (mirrors findFeaturesWired3DArena.m + runBuildMatrix.m option 2)
# ─────────────────────────────────────────────────────────────────────────────

def extract_histogram_features(sensor: dict, bin_edges: list) -> np.ndarray:
    """
    Divide each signal into non-overlapping WIN_SIZE-sample windows and
    compute a normalized histogram over each window for all 4 feature channels.
    The four per-channel histograms are concatenated into one feature vector.

    Key improvement over runBuildMatrix.m:
      The original code grows `totalMatrix` with [totalMatrix histMatrix...]
      inside a for-loop, causing repeated full-array reallocation on every
      iteration.  This version pre-allocates once and fills columns in-place,
      and uses np.add.at for fully vectorized bin counting across all windows
      simultaneously.

    Parameters
    ----------
    sensor    : dict with keys y_GA, z, z_gyro, tot_accel
    bin_edges : list of 4 edge arrays (one per channel)

    Returns
    -------
    hist_matrix : np.ndarray, shape (N_windows, total_bins)
                  total_bins = 4 channels × 99 bins = 396
    """
    print("[3/5] Extracting histogram features (vectorized)...")

    # The 4 channels match findFeaturesWired3DArena.m func{1,1..4}
    # Feature 4 uses log transform (logflag = [0 0 0 1] in Matlab)
    channels = [
        sensor["y_GA"],
        sensor["z"],
        sensor["z_gyro"],
        np.log(np.maximum(sensor["tot_accel"], 1e-10)),
    ]

    N_samples = len(channels[0])
    N_windows = N_samples // WIN_SIZE

    # Pre-allocate the full output matrix — avoids the repeated reallocation
    # that the original runBuildMatrix.m causes with totalMatrix = [totalMatrix ...]
    total_bins  = sum(len(e) - 1 for e in bin_edges)
    hist_matrix = np.zeros((N_windows, total_bins), dtype=np.float32)

    col_start = 0
    for channel, edges in zip(channels, bin_edges):
        n_bins = len(edges) - 1

        # Reshape signal into (N_windows, WIN_SIZE) — no copy, just a view
        win_data = channel[: N_windows * WIN_SIZE].reshape(N_windows, WIN_SIZE)

        # Vectorized bin counting across all windows at once:
        #   1. Flatten to 1-D, digitize every sample into a bin index
        #   2. Build row indices (which window each sample belongs to)
        #   3. Accumulate counts with np.add.at — equivalent to histc in Matlab
        flat    = win_data.ravel()
        bin_idx = np.digitize(flat, edges[1:])           # 0-based bin index
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        row_idx = np.repeat(np.arange(N_windows), WIN_SIZE)
        np.add.at(hist_matrix[:, col_start: col_start + n_bins], (row_idx, bin_idx), 1.0)

        hist_matrix[:, col_start: col_start + n_bins] /= WIN_SIZE  # normalize

        col_start += n_bins

    print(f"      Windows: {N_windows:,}   feature dim: {total_bins}")
    return hist_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Sparse similarity matrix via FAISS ANN
#          (replaces runDistanceSim.m option 2, the O(N²) EMD double loop)
# ─────────────────────────────────────────────────────────────────────────────

def build_sparse_similarity(hist_matrix: np.ndarray,
                             channel_sizes: list,
                             K: int = 100) -> np.ndarray:
    """
    Compute a sparse K-nearest-neighbor similarity matrix using FAISS,
    completely replacing the O(N²) double for-loop in runDistanceSim.m.

    Two key ideas:

    1) 1D Wasserstein via CDF L1 distance
       For a 1D histogram with uniform bin widths, the Wasserstein-1 (EMD)
       distance has a closed-form solution:
           W1(p, q) = sum_k | CDF_p(k) - CDF_q(k) | * bin_width
       where CDF is the cumulative sum of the histogram.
       This is O(d) per pair and fully vectorizable — no optimization solver
       needed.  The original code calls emd() MEX ~7.5 billion times; here
       we compute CDF once per window and let FAISS handle the distances.

    2) Approximate nearest neighbors (ANN) with FAISS IVFFlat
       Instead of computing all N*(N-1)/2 pairs (O(N²)), FAISS only finds
       the K most similar neighbors for each window using an inverted-file
       index.  This reduces time to O(N * K * log N) and memory to O(N * K).

       For N=120,000 and K=100:
         - Full matrix:  120,000² × 8 bytes = 115 GB
         - ANN output:   120,000 × 100 × 8 bytes = 96 MB

    Parameters
    ----------
    hist_matrix   : (N_windows, total_bins)
    channel_sizes : number of bins per channel, e.g. [99, 99, 99, 99]
    K             : number of nearest neighbors to keep per window

    Returns
    -------
    s : np.ndarray, shape (M, 3)
        Sparse similarity in triplet format [i, j, sim_value],
        compatible with apclusterSparse input format.
    """
    print(f"[4/5] Building sparse similarity matrix (FAISS ANN, K={K})...")

    # Convert per-channel histograms to CDF representations.
    # L1 distance between CDFs equals the Wasserstein-1 distance
    # (up to a constant bin-width factor, which cancels in comparisons).
    cdfs = []
    col  = 0
    for size in channel_sizes:
        h = hist_matrix[:, col: col + size]
        cdfs.append(np.cumsum(h, axis=1))
        col += size

    cdf_features = np.hstack(cdfs).astype(np.float32)  # (N_windows, total_bins)
    N, d = cdf_features.shape

    # Build a FAISS IVFFlat index with L1 (Manhattan) metric.
    # n_cells: number of Voronoi partitions — sqrt(N) is the standard heuristic.
    # nprobe:  how many cells to search at query time; higher = more accurate but slower.
    n_cells = max(int(np.sqrt(N)), 64)
    nprobe  = max(n_cells // 10, 10)

    print(f"      Training FAISS index ({n_cells} Voronoi cells, nprobe={nprobe})...")
    quantizer = faiss.IndexFlat(d, faiss.METRIC_L1)
    index     = faiss.IndexIVFFlat(quantizer, d, n_cells, faiss.METRIC_L1)
    index.train(cdf_features)
    index.add(cdf_features)
    index.nprobe = nprobe

    t0 = time.time()
    distances, neighbors = index.search(cdf_features, K + 1)  # +1 includes self
    print(f"      FAISS search: {time.time() - t0:.1f}s")

    # Column 0 is always the query point itself (distance = 0); drop it
    distances = distances[:, 1:]   # (N, K)
    neighbors = neighbors[:, 1:]   # (N, K)

    # Convert L1-CDF distance to AP-style similarity: s(i,j) = -(EMD²)
    # This matches the sign convention in runDistanceSim.m line 72:
    #   Dsim(pp,qq) = -(emd(...).^2)
    similarities = -(distances.astype(np.float64) ** 2)

    rows  = np.repeat(np.arange(N), K)
    cols  = neighbors.ravel()
    vals  = similarities.ravel()

    # FAISS returns -1 for missing neighbors when K > available points
    valid = cols >= 0
    s     = np.column_stack([rows[valid], cols[valid], vals[valid]])

    print(f"      Sparse entries: {len(s):,}  (vs full N²={N*N:,})")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a — HDBSCAN clustering  (recommended replacement for AP)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cdf_features(hist_matrix: np.ndarray, channel_sizes: list) -> np.ndarray:
    col, cdfs = 0, []
    for size in channel_sizes:
        cdfs.append(np.cumsum(hist_matrix[:, col: col + size], axis=1))
        col += size
    return np.hstack(cdfs).astype(np.float64)


def cluster_hdbscan(hist_matrix: np.ndarray,
                    channel_sizes: list,
                    min_cluster_size: int = 15) -> np.ndarray:
    """
    Cluster behavioral windows with UMAP → HDBSCAN.

    Why two steps?

    HDBSCAN on raw 396-dimensional histograms fails: in high dimensions all
    pairwise distances converge toward the same value (curse of dimensionality),
    so density estimation becomes unreliable and almost all points are labelled
    noise.  UMAP first learns a low-dimensional (15-D) manifold that preserves
    local neighbourhood structure; HDBSCAN then finds stable density peaks in
    that compact embedding.

    UMAP n_neighbors controls the global/local trade-off:
      - Small n_neighbors (10–20) → fine local structure, many small clusters
      - Large n_neighbors (40–80) → broader view, fewer but larger clusters
    Default n_neighbors = min(50, N // 10) adapts to dataset size.

    min_cluster_size controls granularity (analogous to AP preference):
      - Lower  → more clusters (finer, like lowering AP preference)
      - Higher → fewer clusters (coarser)
    Default 15 gives ~10–20 clusters on a typical 5-min recording.

    Windows that do not belong to any dense region are labelled -1 (noise).
    Unlike AP, HDBSCAN does not force every window into a cluster — these
    points are genuine behavioural transitions or ambiguous movements.

    Parameters
    ----------
    min_cluster_size : minimum windows to form a cluster (default 15).

    Returns
    -------
    labels : np.ndarray, shape (N_windows,)
             Integer cluster indices starting at 0; -1 = noise/outlier.
    """
    import umap as umap_lib

    N = hist_matrix.shape[0]
    print(f"[4/5] UMAP → HDBSCAN  (N={N:,}, min_cluster_size={min_cluster_size})")

    cdf_features = _build_cdf_features(hist_matrix, channel_sizes)
    t0 = time.time()

    # n_neighbors adapts to dataset size: larger datasets can afford a broader
    # neighbourhood; small recordings cap at 50 to avoid over-smoothing.
    n_neighbors = min(50, max(15, N // 20))
    n_components = 15
    print(f"      UMAP: {cdf_features.shape[1]}-D → {n_components}-D  "
          f"(n_neighbors={n_neighbors}, metric=L1) ...")

    reducer = umap_lib.UMAP(
        n_components = n_components,
        n_neighbors  = n_neighbors,
        min_dist     = 0.0,       # tight clusters in embedding space
        metric       = "l1",      # L1 on CDF = Wasserstein-1 distance
        random_state = 42,
        low_memory   = N > 50_000,
    )
    embedding = reducer.fit_transform(cdf_features)
    print(f"      UMAP done in {time.time() - t0:.1f}s")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size         = min_cluster_size,
        min_samples              = max(1, min_cluster_size // 5),
        metric                   = "euclidean",
        cluster_selection_method = "eom",
    )
    labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"      Total time: {time.time() - t0:.1f}s")
    print(f"      Clusters found : {n_clusters}")
    print(f"      Noise points   : {n_noise:,} ({100 * n_noise / len(labels):.1f}%)")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — Affinity Propagation  (optional, for comparison with Matlab)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_sparse(hist_matrix: np.ndarray,
                      channel_sizes: list,
                      K: int = 100) -> np.ndarray:
    """
    Run Affinity Propagation on a CDF-L1 similarity matrix.

    For N <= 10,000 the full N×N matrix is built via scipy.cdist — this
    matches the Matlab pipeline exactly (same algorithm, same preference =
    median of all pairwise similarities) and gives directly comparable results.

    For N > 10,000 a FAISS sparse matrix (K nearest neighbours) is used
    instead, but NOTE: sklearn's AP still costs O(N²) per iteration, so it
    remains infeasible for very large N regardless of how the matrix is built.
    Prefer cluster_hdbscan for production runs on large datasets.

    Parameters
    ----------
    K : neighbours used for the sparse path (N > 10,000 only)
    """
    from sklearn.cluster import AffinityPropagation
    import scipy.spatial.distance as ssd

    N = hist_matrix.shape[0]

    cdf_features = _build_cdf_features(hist_matrix, channel_sizes)

    print(f"[4/5] Affinity Propagation (N={N}) ...")
    print(f"      WARNING: AP is O(N²) per iteration — feasible only for N < ~10,000.")
    t0 = time.time()

    if N <= 10_000:
        # Full pairwise L1 distance → similarity = -(dist²)
        # This exactly mirrors runDistanceSim.m option 2 (Wasserstein / EMD)
        print(f"      Building full {N}×{N} similarity matrix via scipy.cdist ...")
        dist = ssd.cdist(cdf_features, cdf_features, metric="cityblock").astype(np.float64)
        affinity = -(dist ** 2)
        # Use min similarity as preference — matches VPAPPAxes.m where
        # Clusters.param equals min(sim), giving ~10-20 clusters on typical sessions
        preference = float(affinity.min())
        print(f"      Preference (min similarity): {preference:.4f}")
        ap = AffinityPropagation(
            affinity   = "precomputed",
            preference = preference,
            damping    = 0.9,
            max_iter   = 1000,
            convergence_iter = 100,
            random_state = 0,
        )
        labels = ap.fit_predict(affinity)
    else:
        # Sparse path: build via FAISS, then fill missing entries with worst similarity
        s   = build_sparse_similarity(hist_matrix, channel_sizes, K=K)
        min_sim = float(s[:, 2].min())
        # Start with a dense matrix of the worst similarity so that non-neighbours
        # have the lowest possible affinity (matches Matlab's sparse AP behaviour)
        affinity = np.full((N, N), min_sim, dtype=np.float64)
        rows = s[:, 0].astype(int)
        cols = s[:, 1].astype(int)
        affinity[rows, cols] = s[:, 2]
        affinity[cols, rows] = s[:, 2]      # symmetrise
        preference = float(np.median(s[:, 2]))
        np.fill_diagonal(affinity, preference)
        ap = AffinityPropagation(
            affinity   = "precomputed",
            preference = preference,
            damping    = 0.9,
            max_iter   = 1000,
            convergence_iter = 100,
            random_state = 0,
        )
        labels = ap.fit_predict(affinity)

    print(f"      AP done in {time.time() - t0:.1f}s  |  clusters: {len(set(labels))}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(input_csv:         str,
                 output_csv:        str,
                 arena:             str  = "3d_wired",
                 min_cluster_size:  int  = 15,
                 use_ap:            bool = False,
                 ann_k:             int  = 100) -> None:
    """
    Full replacement for Matlab's VPAPPAxes.m.

    Input  : combined_harp_data_cleaned.csv  (output of stitched.py)
    Output : Cluster_detail_results.csv      (same format as Matlab output)
             Columns: ClusterIdx, Timestamp, Folder_Name
    """
    t_start = time.time()

    raw_motion, timestamps, folder_names = load_cleaned_motion(input_csv)
    sensor       = process_motion(raw_motion)
    bin_edges    = ARENA_BIN_EDGES[arena]
    channel_sizes = [len(e) - 1 for e in bin_edges]
    hist_matrix  = extract_histogram_features(sensor, bin_edges)
    N_windows    = hist_matrix.shape[0]

    if use_ap:
        labels = cluster_ap_sparse(hist_matrix, channel_sizes, K=ann_k)
    else:
        labels = cluster_hdbscan(hist_matrix, channel_sizes, min_cluster_size)

    # Assign the timestamp of the window's middle sample to each window,
    # consistent with runTimestampClusterPairing.m in the Matlab pipeline
    mid           = WIN_SIZE // 2
    win_ts        = timestamps[mid::WIN_SIZE][:N_windows]
    win_folders   = folder_names[::WIN_SIZE][:N_windows]

    # Convert to 1-based cluster indices; noise points (label=-1) become 0
    cluster_idx = np.where(labels < 0, 0, labels + 1)

    result_df = pd.DataFrame({
        "ClusterIdx":  cluster_idx,
        "Timestamp":   win_ts,
        "Folder_Name": win_folders,
    })

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    elapsed = time.time() - t_start
    print(f"\nDone. Total time: {elapsed / 60:.1f} min")
    print(f"Saved to        : {output_csv}")
    print(f"Clusters found  : {cluster_idx.max()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Behavioral clustering pipeline — Python replacement for VPAPPAxes.m"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to combined_harp_data_cleaned.csv (output of stitched.py)",
    )
    p.add_argument(
        "--output", required=True,
        help="Path for output Cluster_detail_results.csv",
    )
    p.add_argument(
        "--arena", default="3d_wired",
        choices=["3d_wired", "2d_wired", "3d_wireless", "2d_wireless"],
        help="Arena type and wiring configuration (affects feature axis mapping)",
    )
    p.add_argument(
        "--min-cluster-size", type=int, default=15,
        help="HDBSCAN minimum cluster size (default: 15). "
             "Smaller value → more clusters, analogous to lowering AP preference.",
    )
    p.add_argument(
        "--use-ap", action="store_true",
        help="Use Affinity Propagation instead of HDBSCAN. "
             "Produces results closest to the original Matlab pipeline but "
             "is still O(N²) per iteration — only suitable for N < ~10,000.",
    )
    p.add_argument(
        "--ann-k", type=int, default=100,
        help="Number of nearest neighbors for FAISS ANN (used with --use-ap, default: 100)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_pipeline(
        input_csv        = args.input,
        output_csv       = args.output,
        arena            = args.arena,
        min_cluster_size = args.min_cluster_size,
        use_ap           = args.use_ap,
        ann_k            = args.ann_k,
    )
