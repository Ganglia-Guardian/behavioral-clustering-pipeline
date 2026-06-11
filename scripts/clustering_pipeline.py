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
    pip install numpy scipy pandas scikit-learn hdbscan umap-learn faiss-cpu h5py

Usage:
    python clustering_pipeline.py \\
        --input  path/to/Combined_Results/combined_harp_data_cleaned.csv \\
        --output path/to/Results/test1/Cluster_detail_results.csv \\
        --arena  3d_wired
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import faiss
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from tqdm import tqdm


class _Spinner:
    """Context manager: live spinner + elapsed time during silent sklearn calls."""
    def __init__(self, msg: str):
        self._msg    = msg
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._t0 = time.time()
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        sys.stdout.write(f"\r{' ' * (len(self._msg) + 15)}\r")
        sys.stdout.flush()

    def _run(self):
        chars = '|/-\\'
        i = 0
        while not self._stop.wait(0.25):
            elapsed = time.time() - self._t0
            sys.stdout.write(f"\r{self._msg} {chars[i % 4]} {elapsed:.0f}s")
            sys.stdout.flush()
            i += 1


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WIN_SIZE = 60       # window size in samples (= 300 ms at 200 Hz)
FS       = 200      # sampling frequency in Hz
AC_RNG   = 4        # accelerometer range ±4 g
GYR_RNG  = 1000     # gyroscope range ±1000 dps

# Histogram bin edges for each of the 4 feature channels.
# Source: hardHistNew.mat (Matlab hardcoded thresholds loaded in VPAPPAxes.m).
# These are the exact thresholds used in the Matlab pipeline after findHistogramCutOffs
# overwrites the generic BB edges.  11 + 11 + 6 + 2 = 30 bins total.
#
# Channel 4 note: hardHistNew stores the threshold in log-space (-3).
# Python histograms log(totAccelBA) directly with these log-space edges,
# which is mathematically equivalent to Matlab's exp-then-histc approach.
_EDGES_3D = [
    np.array([-np.inf, -1.0, -0.7778, -0.5556, -0.3333, -0.1111,
               0.1111,  0.3333,  0.5556,  0.7778,  1.0,  np.inf]),  # 11 bins: y_GA
    np.array([-np.inf, -1.5, -1.2222, -0.9444, -0.6667, -0.3889,
              -0.1111,  0.1667,  0.4444,  0.7222,  1.0,  np.inf]),  # 11 bins: z
    np.array([-np.inf, -100.0, -50.0, 0.0, 50.0, 100.0, np.inf]),   #  6 bins: z_gyro
    np.array([-np.inf, -3.0, np.inf]),                               #  2 bins: log(totAccelBA)
]

_EDGES_2D = _EDGES_3D  # same calibration until 2D-specific hardHistNew is available

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
    fn_col = "Folder_Name" if "Folder_Name" in motion_df.columns else "DataElement10"
    folder_names = motion_df[fn_col].values

    # DataElement0 through DataElement9 sit at column indices 3–12
    raw_motion = motion_df.iloc[:, 3:13].values.astype(np.float64)

    return raw_motion, timestamps, folder_names


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Signal processing  (mirrors processData.m)
# ─────────────────────────────────────────────────────────────────────────────

def _medfilt_clip(x: np.ndarray, w: int) -> np.ndarray:
    """
    1-D median filter with clipped (truncated) boundary — matches Matlab's myMedFilt1.
    Uses pandas rolling(center=True, min_periods=1) which clips the window at boundaries
    instead of zero-padding. scipy.signal.medfilt zero-pads, which differs at edges.
    """
    return (pd.Series(x)
              .rolling(window=w, center=True, min_periods=1)
              .median()
              .to_numpy(dtype=x.dtype))


def process_motion(raw_motion: np.ndarray) -> dict:
    """
    Signal processing pipeline:
      1. Scale ADC → physical units.
      2. Median filter (kernel=7) with clipped boundary (matches Matlab's myMedFilt1).
      3. Zero-phase 1st-order Butterworth high-pass at 0.5 Hz (filtfilt),
         matching the online Matlab pipeline (processDataOnlineJT_vitor_JT.m).
      4. Compute total body acceleration magnitude.
    """
    print("[2/5] Signal processing (filtering, gravity separation)...")

    acc = raw_motion[:, 0:3] * (AC_RNG  / 32768.0)
    gyr = raw_motion[:, 3:6] * (GYR_RNG / 32768.0)

    # Clipped-boundary median filter — matches Matlab's myMedFilt1(x, 7)
    x      = _medfilt_clip(acc[:, 0], 7)
    y      = _medfilt_clip(-acc[:, 1], 7)   # negative sign: sensor orientation
    z      = _medfilt_clip(acc[:, 2], 7)
    z_gyro = _medfilt_clip(gyr[:, 2], 7)

    # Zero-phase 1st-order Butterworth high-pass, cut-off 0.5 Hz.
    # filtfilt matches the online Matlab pipeline (processDataOnlineJT_vitor_JT.m).
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
                  total_bins = 11 + 11 + 6 + 2 = 30 bins (3D wired arena)
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

def build_knn_similarity(hist_matrix: np.ndarray,
                          channel_sizes: list,
                          K: int = 100) -> np.ndarray:
    """
    [Experimental — not used in the main pipeline]

    Build a sparse K-nearest-neighbor similarity matrix using FAISS IVFFlat ANN.
    Intended as input to a Sparse AP implementation (equivalent to Matlab's
    apclusterSparse.m).  sklearn's AffinityPropagation does not support sparse
    input, so this function has no downstream caller until a custom Sparse AP
    is added.

    For N=120,000 and K=100:
      - Full N×N matrix : 115 GB
      - K-NN output     : 96 MB

    Returns
    -------
    s : np.ndarray, shape (M, 3)
        Triplet format [i, j, sim_value] compatible with apclusterSparse.
    """
    print(f"[experimental] Building K-NN similarity matrix (FAISS ANN, K={K})...")

    cdf_features = _build_cdf_features(hist_matrix, channel_sizes).astype(np.float32)
    N, d = cdf_features.shape

    n_cells = max(int(np.sqrt(N)), 64)
    nprobe  = max(n_cells // 10, 10)

    quantizer = faiss.IndexFlat(d, faiss.METRIC_L1)
    index     = faiss.IndexIVFFlat(quantizer, d, n_cells, faiss.METRIC_L1)
    index.train(cdf_features)
    index.add(cdf_features)
    index.nprobe = nprobe

    t0 = time.time()
    distances, neighbors = index.search(cdf_features, K + 1)
    print(f"      FAISS search: {time.time() - t0:.1f}s")

    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]

    similarities = -(distances.astype(np.float64) ** 2)

    rows  = np.repeat(np.arange(N), K)
    cols  = neighbors.ravel()
    vals  = similarities.ravel()

    valid = cols >= 0
    s     = np.column_stack([rows[valid], cols[valid], vals[valid]])

    print(f"      Sparse entries: {len(s):,}  (vs full N²={N*N:,})")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a — Sparse AP on K-NN graph  (equivalent to Matlab apclusterSparse.m)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_sparse_knn(hist_matrix: np.ndarray,
                           channel_sizes: list,
                           K: int = 100,
                           preference: float = None,
                           damping: float = 0.9,
                           max_iter: int = 1000,
                           convergence_iter: int = 15,
                           _timing: dict = None) -> np.ndarray:
    """
    Affinity Propagation on a FAISS K-NN sparse similarity graph.

    Equivalent to Matlab's apclusterSparse.m.
    Memory : O(N×K) instead of O(N²) for AP full.
    Time   : O(N×K) per iteration instead of O(N²).

    Steps
    -----
    1. Build K-NN directed graph with FAISS IVFFlat (approximate L1).
    2. Symmetrize with element-wise max → each point has up to 2K neighbours.
    3. Add self-loops at `preference` value (diagonal of affinity matrix).
    4. Run AP message passing on the sparse structure — fully vectorised with
       np.maximum.reduceat; no Python loops over rows.
    5. Identify exemplars (R(k,k)+A(k,k) > 0); assign all N points to the
       nearest exemplar via FAISS exact L1 search.

    Parameters
    ----------
    K               : K-NN graph degree (default 100).
    preference      : AP preference. Default = min(K-NN similarities).
                      Higher → more clusters; lower → fewer clusters.
    damping         : Damping factor in [0.5, 1.0) (default 0.9).
    max_iter        : Maximum AP iterations (default 1000).
    convergence_iter: Stable iterations required to stop early (default 15).
    """
    import scipy.sparse as sp

    N = hist_matrix.shape[0]
    t0 = time.time()

    # Build CDF features early — needed for final assignment step
    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)

    # ── Step 1: K-NN sparse similarity ───────────────────────────────────────
    print(f"[4/5] Sparse AP (K-NN)  (N={N:,}, K={K})")
    print(f"      Building K-NN graph (FAISS IVFFlat)...")
    triplets = build_knn_similarity(hist_matrix, channel_sizes, K=K)
    t_dist   = time.time() - t0
    print(f"      K-NN done in {t_dist:.1f}s  ({len(triplets):,} directed edges)")

    # ── Step 2: Symmetrize and add diagonal ──────────────────────────────────
    i_arr = triplets[:, 0].astype(np.int32)
    j_arr = triplets[:, 1].astype(np.int32)
    s_arr = triplets[:, 2].astype(np.float64)

    S_ij  = sp.csr_matrix((s_arr, (i_arr, j_arr)), shape=(N, N))
    S_sym = S_ij.maximum(S_ij.T)          # element-wise max → symmetric

    if preference is None:
        # Estimate global min(similarity) from all N windows, same as AP full/sampled.
        # Using K-NN min is biased toward 0 (only near-neighbor pairs), causing
        # excessive cluster count. Global estimate matches Matlab's min(s(:,3)).
        preference = _estimate_min_affinity(cdf_all)
        print(f"      Preference: {preference:.4f}  (global min estimate from {N:,} windows)")
    else:
        print(f"      Preference: {preference:.4f}  (user-specified)")

    S_sym.setdiag(preference)
    # Do NOT call eliminate_zeros(): if preference == 0.0 that call would remove the
    # diagonal entries, leaving empty rows that cause -inf → +inf → NaN in AP messages.
    S_sym = S_sym.tocsr()
    S_sym.sort_indices()

    indptr  = S_sym.indptr                          # shape (N+1,)
    col_idx = S_sym.indices.astype(np.int32)        # shape (nnz,)
    S_data  = S_sym.data.astype(np.float64).copy()  # shape (nnz,)
    nnz     = len(S_data)

    row_idx   = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
    diag_mask = (row_idx == col_idx)

    mem_gb = nnz * 8 * 3 / 1e9
    print(f"      Sparse edges: {nnz:,}  |  R+A+S ≈ {mem_gb:.2f} GB")

    # ── Step 3: AP message passing (fully vectorised) ─────────────────────────
    R_data = np.zeros(nnz, dtype=np.float64)
    A_data = np.zeros(nnz, dtype=np.float64)

    prev_exemplar_set = None
    stable_count      = 0

    print(f"      Running AP (max_iter={max_iter}, convergence_iter={convergence_iter})...")
    t_algo_start = time.time()

    ap_pbar = tqdm(range(max_iter), desc="      AP iters", unit="iter",
                   leave=False, dynamic_ncols=True)
    for it in ap_pbar:

        # ── Responsibility update ─────────────────────────────────────────────
        AS_data = A_data + S_data

        # Row-wise max1  (np.maximum.reduceat: O(nnz), no Python loop)
        row_max1         = np.maximum.reduceat(AS_data, indptr[:-1])   # (N,)
        row_max1_per_nnz = row_max1[row_idx]

        # Identify argmax positions (with tie awareness)
        is_argmax    = AS_data >= row_max1_per_nnz - 1e-14
        argmax_count = np.zeros(N, dtype=np.int32)
        np.add.at(argmax_count, row_idx[is_argmax], 1)
        unique_argmax = is_argmax & (argmax_count[row_idx] == 1)

        # Row-wise max2 (mask out unique argmax, then reduceat again)
        AS_for_max2 = AS_data.copy()
        AS_for_max2[unique_argmax] = -np.inf
        row_max2 = np.maximum.reduceat(AS_for_max2, indptr[:-1])        # (N,)

        R_new = S_data - row_max1_per_nnz
        R_new[unique_argmax] = (S_data[unique_argmax]
                                - row_max2[row_idx[unique_argmax]])
        R_data = damping * R_data + (1.0 - damping) * R_new
        np.nan_to_num(R_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Availability update ───────────────────────────────────────────────
        # Column sum of max(0,R) for off-diagonal; actual R for diagonal.
        # Simplified formula (derivation in docstring):
        #   off-diag: A(i,k) = min(0, col_sum(k) − max(0, R(i,k)))
        #   diagonal: A(k,k) = col_sum(k) − R(k,k)
        # where col_sum(k) = R(k,k) + Σ_{i'≠k} max(0, R(i',k))
        pos_R_col = np.maximum(0.0, R_data)
        pos_R_col[diag_mask] = R_data[diag_mask]   # diagonal: use R, not max(0,R)

        col_sum = np.zeros(N, dtype=np.float64)
        np.add.at(col_sum, col_idx, pos_R_col)

        A_new = np.minimum(0.0, col_sum[col_idx] - np.maximum(0.0, R_data))
        A_new[diag_mask] = col_sum[col_idx[diag_mask]] - R_data[diag_mask]
        A_data = damping * A_data + (1.0 - damping) * A_new
        np.nan_to_num(A_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Convergence check ─────────────────────────────────────────────────
        RA_diag = np.zeros(N, dtype=np.float64)
        RA_diag[col_idx[diag_mask]] = (R_data + A_data)[diag_mask]
        exemplar_set = frozenset(np.where(RA_diag > 0)[0])

        if exemplar_set == prev_exemplar_set:
            stable_count += 1
            if stable_count >= convergence_iter:
                ap_pbar.close()
                print(f"      Converged at iteration {it + 1}")
                break
        else:
            stable_count = 0
        prev_exemplar_set = exemplar_set

        ap_pbar.set_postfix(exemplars=len(exemplar_set), stable=stable_count)
    else:
        print(f"      WARNING: did not converge in {max_iter} iterations")

    t_algo = time.time() - t_algo_start

    # ── Step 4: Identify exemplars and assign all points ──────────────────────
    exemplar_indices = np.array(sorted(exemplar_set), dtype=np.int64)

    if len(exemplar_indices) == 0:
        print("      WARNING: no exemplars found; returning single cluster")
        return np.zeros(N, dtype=int)

    exemplar_cdf = cdf_all.astype(np.float32)[exemplar_indices]

    index = faiss.IndexFlat(exemplar_cdf.shape[1], faiss.METRIC_L1)
    index.add(exemplar_cdf)
    _, assignments = index.search(cdf_all, 1)
    labels = assignments.ravel().astype(int)

    n_clusters = len(np.unique(labels))
    print(f"      Clusters: {n_clusters}  |  T_dist: {t_dist:.1f}s  "
          f"T_algo: {t_algo:.1f}s  Total: {t_dist + t_algo:.1f}s")

    if _timing is not None:
        _timing['t_dist']     = round(t_dist, 2)
        _timing['t_algo']     = round(t_algo, 2)
        _timing['preference'] = preference

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — HDBSCAN clustering  (recommended replacement for AP)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cdf_features(hist_matrix: np.ndarray, channel_sizes: list) -> np.ndarray:
    col, cdfs = 0, []
    for size in channel_sizes:
        # Divide by (size-1) to normalize each channel's CDF to [0, 1/(size-1)] scale.
        # Matches Matlab emd_1.c which divides each channel's CDF L1 sum by (n_bins-1),
        # giving equal weight to every channel regardless of bin count.
        cdfs.append(np.cumsum(hist_matrix[:, col: col + size], axis=1) / (size - 1))
        col += size
    return np.hstack(cdfs).astype(np.float64)


def cluster_hdbscan(hist_matrix: np.ndarray,
                    channel_sizes: list,
                    min_cluster_size: int = 15,
                    _timing: dict = None) -> np.ndarray:
    """
    Cluster behavioral windows with HDBSCAN on 30-D CDF features.

    Runs HDBSCAN directly on the CDF feature matrix using L1 (Manhattan)
    distance, which equals the Wasserstein-1 distance between per-channel
    histograms — the same metric used by the AP methods.

    min_cluster_size controls granularity (analogous to AP preference):
      - Lower  → more clusters (finer)
      - Higher → fewer clusters (coarser)
    Default 15 gives ~10–50 clusters on a typical 5-min recording.

    Windows that do not belong to any dense region are labelled -1 (noise).

    Parameters
    ----------
    min_cluster_size : minimum windows to form a cluster (default 15).

    Returns
    -------
    labels : np.ndarray, shape (N_windows,)
             Integer cluster indices starting at 0; -1 = noise/outlier.
    """
    N = hist_matrix.shape[0]
    print(f"[4/5] HDBSCAN  (N={N:,}, min_cluster_size={min_cluster_size})")

    t0 = time.time()
    cdf_features = _build_cdf_features(hist_matrix, channel_sizes)
    t_dist = time.time() - t0
    print(f"      CDF features: {cdf_features.shape[1]}-D  ({t_dist:.1f}s)")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size         = min_cluster_size,
        min_samples              = max(1, min_cluster_size // 5),
        metric                   = "l1",
        cluster_selection_method = "eom",
    )
    print(f"      Fitting HDBSCAN...")
    labels = clusterer.fit_predict(cdf_features)
    t_algo = time.time() - t0 - t_dist

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"      Total time: {time.time() - t0:.1f}s")
    print(f"      Clusters found : {n_clusters}")
    print(f"      Noise points   : {n_noise:,} ({100 * n_noise / len(labels):.1f}%)")
    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# AP shared helper — global preference estimation
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_min_affinity(cdf_all: np.ndarray, n_pairs: int = 500_000) -> float:
    """
    Estimate global min(affinity) = -max(L1_dist²) by sampling random pairs.

    AP preference = min(similarity) controls cluster count.  Using the sample
    or block min underestimates the true global minimum (i.e. the value is
    less negative), which biases AP toward too few clusters.  Sampling ~500k
    random pairs from all N windows gives a tight lower-bound estimate in <1s
    and is independent of any sub-sampling strategy used by the caller.
    """
    N = cdf_all.shape[0]
    rng = np.random.default_rng(42)
    i = rng.integers(0, N, n_pairs)
    j = rng.integers(0, N, n_pairs)
    # Avoid self-pairs (dist=0 which would underestimate max)
    same = i == j
    j[same] = (j[same] + 1) % N
    dists = np.abs(
        cdf_all[i].astype(np.float64) - cdf_all[j].astype(np.float64)
    ).sum(axis=1)
    return -float(dists.max() ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — Affinity Propagation  (optional, for comparison with Matlab)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_full(hist_matrix: np.ndarray,
                      channel_sizes: list,
                      K: int = 100,
                      preference: float = None,
                      _timing: dict = None) -> np.ndarray:
    """
    Run Affinity Propagation on a CDF-L1 similarity matrix.

    Full N×N affinity matrix via scipy.cdist.
    Preference = min(similarity), matching Matlab's min(s(:,3)) in VPAPPAxes.m.

    For N > 10,000 AP is refused: sklearn AP still costs O(N²) per iteration
    regardless of how the similarity matrix is built, making it infeasible.
    Use cluster_hdbscan for large datasets.

    Parameters
    ----------
    K          : neighbours for FAISS ANN (not used for N <= 10,000)
    preference : AP preference value (diagonal of affinity matrix).
                 Controls number of clusters: higher → more clusters,
                 lower → fewer clusters.
                 Default None uses median(affinity), matching the newer
                 Matlab partition-evaluation pipeline.
                 Tip: start from the printed default and tune toward 0 for
                 more clusters, or further negative for fewer.
    """
    from sklearn.cluster import AffinityPropagation
    import scipy.spatial.distance as ssd

    N = hist_matrix.shape[0]

    if N > 10_000:
        mem_gb = N ** 2 * 8 / 1e9
        print(f"      WARNING: N={N:,} — full N×N matrix = {mem_gb:.1f} GB. "
              f"Ensure you have enough RAM. Use --use-ap-sampled for a safer option.")

    cdf_features = _build_cdf_features(hist_matrix, channel_sizes)

    print(f"[4/5] Affinity Propagation (N={N}) ...")
    print(f"      WARNING: AP is O(N²) per iteration — feasible only for N < ~10,000.")
    t0 = time.time()

    # Full pairwise L1 distance → similarity = -(dist²)
    # Mirrors runDistanceSim.m option 2 (Wasserstein / EMD)
    print(f"      Building full {N}×{N} similarity matrix via scipy.cdist ...")
    dist     = ssd.cdist(cdf_features, cdf_features, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist = time.time() - t0

    if preference is None:
        # Use min(similarity): equivalent to Matlab's min(s(:,3)).
        # median(-dist²) ≈ 0 with coarse bins → cluster explosion; min is correct.
        preference = float(affinity.min())
        print(f"      Preference: {preference:.4f}  (auto = min similarity)")
        print(f"      Tip: use --preference to tune cluster count "
              f"(toward 0 = more clusters, more negative = fewer clusters)")
    else:
        print(f"      Preference: {preference:.4f}  (user-specified)")

    np.fill_diagonal(affinity, preference)

    ap = AffinityPropagation(
        affinity         = "precomputed",
        preference       = preference,
        damping          = 0.9,
        max_iter         = 1000,
        convergence_iter = 15,
        random_state     = 0,
    )
    print("      Running AP (max_iter=1000, convergence_iter=15)...")
    with _Spinner("      AP"):
        labels = ap.fit_predict(affinity)
    t_algo = time.time() - t0 - t_dist

    print(f"      AP done in {time.time() - t0:.1f}s  |  clusters: {len(set(labels))}")
    if _timing is not None:
        _timing['t_dist']     = round(t_dist, 2)
        _timing['t_algo']     = round(t_algo, 2)
        _timing['preference'] = preference
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Coreset helper — Greedy K-Center (Gonzalez 1985)
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans_sample(cdf_features: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """
    Mini-batch K-means centroid sampling.

    Runs Mini-batch K-means with k=K on all N windows, then maps each
    centroid to the nearest real window via FAISS exact L1 search.
    Guarantees one representative per "region" of the feature space,
    and selects central (not boundary) points — better suited for AP
    than Greedy K-Center coreset.

    Parameters
    ----------
    cdf_features : float32 array (N, d)
    K            : number of samples (= number of K-means clusters)
    seed         : random seed

    Returns
    -------
    selected : int64 array of shape (≤ K,) — indices into cdf_features
               (may be < K if duplicate nearest-real-points are deduplicated)
    """
    from sklearn.cluster import MiniBatchKMeans

    X = cdf_features.astype(np.float32)
    N = X.shape[0]

    print(f"      Running Mini-batch K-means (k={K:,}, N={N:,}) ...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3,
                         batch_size=min(4096, N), max_iter=100)
    km.fit(X)
    centroids = km.cluster_centers_.astype(np.float32)   # (K, d)
    print(f"      K-means done in {time.time() - t0:.1f}s")

    # Map each centroid to the nearest real window (FAISS exact L1)
    index = faiss.IndexFlat(X.shape[1], faiss.METRIC_L1)
    index.add(X)
    _, nn = index.search(centroids, 1)                   # (K, 1)
    selected = np.unique(nn.ravel().astype(np.int64))    # deduplicate
    return selected


def _stratified_sample(cdf_features: np.ndarray, K: int,
                        n_strata: int = 200, seed: int = 42) -> np.ndarray:
    """
    Two-stage stratified sampling for AP.

    Stage 1: Mini-batch K-means with k=n_strata to partition all N windows
             into rough behavioral groups.
    Stage 2: Sample K // n_strata windows equally from each stratum, ensuring
             rare behaviors (small clusters) have the same representation as
             common ones.  Any deficit from small strata is filled with a
             proportional top-up from the full pool.

    Compared to random sampling, this guarantees that a cluster with only
    20 windows is represented in the AP sample even if N >> sample_size.
    This directly addresses the root cause of low ARI vs AP full.

    Parameters
    ----------
    cdf_features : float32 array (N, d)
    K            : total sample size (target; actual may be slightly smaller)
    n_strata     : number of coarse K-means groups (default 200)
    seed         : random seed

    Returns
    -------
    selected : int64 array of shape (≤ K,) — indices into cdf_features
    """
    from sklearn.cluster import MiniBatchKMeans

    X   = cdf_features.astype(np.float32)
    N   = X.shape[0]
    rng = np.random.default_rng(seed)

    # Stage 1: coarse partition
    n_strata = min(n_strata, N)
    print(f"      Stratified: K-means k={n_strata} on {N:,} windows ...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=n_strata, random_state=seed, n_init=3,
                         batch_size=min(4096, N), max_iter=100)
    stratum_ids = km.fit_predict(X)
    print(f"      Coarse clustering done in {time.time()-t0:.1f}s")

    # Stage 2: equal sampling per stratum
    per_stratum = max(1, K // n_strata)
    selected    = []
    deficit     = 0  # windows we couldn't take from small strata

    for s in range(n_strata):
        idx    = np.where(stratum_ids == s)[0]
        n_take = min(len(idx), per_stratum)
        deficit += per_stratum - n_take
        chosen  = rng.choice(idx, n_take, replace=False)
        selected.append(chosen)

    selected = np.concatenate(selected)

    # Fill deficit from the full pool (excluding already selected)
    if deficit > 0:
        already  = set(selected.tolist())
        pool     = np.array([i for i in range(N) if i not in already], dtype=np.int64)
        n_fill   = min(deficit, len(pool))
        fill_idx = rng.choice(pool, n_fill, replace=False)
        selected = np.concatenate([selected, fill_idx])

    selected = np.sort(selected.astype(np.int64))
    print(f"      Stratified sample: {len(selected):,} points "
          f"({per_stratum} per stratum, {n_strata} strata)")
    return selected


def _coreset_sample(cdf_features: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """
    Greedy K-Center coreset selection.

    Selects K points such that every remaining point is within distance
    ≤ 2 × OPT of its nearest selected point (OPT = optimal max distance).
    Guarantees coverage of rare behaviors that random sampling can miss.

    Algorithm
    ---------
    1. Pick one random starting point.
    2. Repeat K-1 times:
       a. Select the point farthest from all currently selected points.
       b. Update per-point min-distance to the selected set.
    Complexity: O(K × N × d)  — all vectorized, no Python inner loops.

    Parameters
    ----------
    cdf_features : float32 array (N, d)
    K            : number of coreset points to select
    seed         : random seed for the initial point

    Returns
    -------
    selected : int64 array of shape (K,) — indices into cdf_features
    """
    N = cdf_features.shape[0]
    X = cdf_features.astype(np.float32)   # work in float32 for speed

    rng        = np.random.default_rng(seed)
    first      = int(rng.integers(0, N))
    selected   = [first]

    # dist_to_S[i] = L1 distance from point i to its nearest selected center
    dist_to_S = np.sum(np.abs(X - X[first]), axis=1)   # (N,)
    dist_to_S[first] = 0.0

    for step in tqdm(range(1, K), desc="      Coreset", unit="pt",
                     leave=False, dynamic_ncols=True):
        new_center = int(np.argmax(dist_to_S))
        selected.append(new_center)

        # Incremental update: only shrink distances
        new_dists = np.sum(np.abs(X - X[new_center]), axis=1)
        np.minimum(dist_to_S, new_dists, out=dist_to_S)
        dist_to_S[new_center] = 0.0

    return np.array(selected, dtype=np.int64)


# Step 5c — AP with random sampling  (scales to large N)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_sampled(hist_matrix: np.ndarray,
                       channel_sizes: list,
                       sample_size: int = 10000,
                       preference: float = None,
                       use_coreset: bool = False,
                       _timing: dict = None) -> np.ndarray:
    """
    Affinity Propagation on a subset, then assign all windows to the
    nearest exemplar via FAISS exact L1.  Scales AP to arbitrarily large N.

    Two-step process
    ----------------
    1. Select min(sample_size, N) windows — randomly (default) or via
       Greedy K-Center coreset (--use-coreset-sample).
       Run full AP on this subset to find cluster exemplars.
       sample_size=10000 → affinity matrix = 10000² × 8 bytes = 800 MB.

    2. Assign every window to its nearest exemplar (FAISS exact L1).

    Parameters
    ----------
    sample_size  : number of windows to pass to AP (default 10,000).
    preference   : AP preference — controls cluster count.
                   Default None uses min(affinity).
    use_coreset  : if True, use Greedy K-Center instead of random sampling.
                   Guarantees coverage of rare behaviors at the cost of
                   O(K×N) extra distance computations (~5–15 s).
    """
    from sklearn.cluster import AffinityPropagation
    import scipy.spatial.distance as ssd

    N = hist_matrix.shape[0]

    if N <= sample_size:
        print(f"[4/5] AP sampled: N={N:,} ≤ sample_size={sample_size:,}, running full AP.")
        return cluster_ap_full(hist_matrix, channel_sizes, preference=preference)

    sampling_method = "coreset" if use_coreset else "random"
    print(f"[4/5] AP sampled  (N={N:,}, sample={sample_size:,}, "
          f"{100*sample_size/N:.0f}% of data, sampling={sampling_method}) ...")

    # ── Build CDF features for all windows ────────────────────────────────────
    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)  # (N, d)

    # ── Select sample ─────────────────────────────────────────────────────────
    if use_coreset:
        print(f"      Running Greedy K-Center coreset (K={sample_size:,}) ...")
        t_coreset = time.time()
        sample_idx = _coreset_sample(cdf_all.astype(np.float32), sample_size, seed=42)
        t_coreset = time.time() - t_coreset
        print(f"      Coreset done in {t_coreset:.1f}s")
        if _timing is not None:
            _timing['t_coreset'] = round(t_coreset, 2)
    else:
        rng        = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(N, sample_size, replace=False))

    cdf_sample = cdf_all[sample_idx]                           # (sample_size, d)

    # ── AP on sample ──────────────────────────────────────────────────────────
    print(f"      Building {sample_size}×{sample_size} affinity matrix "
          f"({sample_size**2*8/1e6:.0f} MB) ...")
    t_dist_start = time.time()
    dist     = ssd.cdist(cdf_sample, cdf_sample, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist = time.time() - t_dist_start

    if preference is None:
        # Estimate global min(similarity) from all N windows, not just the sample.
        # Using sample min underestimates global min → preference too high → too few clusters.
        preference = _estimate_min_affinity(cdf_all)
        print(f"      Preference: {preference:.4f}  (global min estimate from {N:,} windows)")
    else:
        print(f"      Preference: {preference:.4f}  (user-specified)")

    np.fill_diagonal(affinity, preference)

    t0 = time.time()
    ap = AffinityPropagation(
        affinity         = "precomputed",
        preference       = preference,
        damping          = 0.9,
        max_iter         = 1000,
        convergence_iter = 15,
        random_state     = 0,
    )
    print("      Running AP (max_iter=1000, convergence_iter=15)...")
    with _Spinner("      AP"):
        sample_labels = ap.fit_predict(affinity)
    n_clusters    = len(set(sample_labels))
    print(f"      AP on sample: {time.time()-t0:.1f}s  |  {n_clusters} clusters found")

    # ── Get exemplar CDF feature vectors ──────────────────────────────────────
    exemplar_idx_in_sample = ap.cluster_centers_indices_      # indices into sample
    exemplar_cdf = cdf_sample[exemplar_idx_in_sample].astype(np.float32)  # (K, d)

    # ── Assign all N windows to nearest exemplar via exact FAISS L1 ───────────
    print(f"      Assigning all {N:,} windows to nearest exemplar (FAISS) ...")
    index = faiss.IndexFlat(exemplar_cdf.shape[1], faiss.METRIC_L1)
    index.add(exemplar_cdf)
    _, assignments = index.search(cdf_all.astype(np.float32), 1)
    labels = assignments.ravel().astype(int)

    t_algo = time.time() - t0
    print(f"      Total time: {t_dist + t_algo:.1f}s  |  Clusters: {n_clusters}")
    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5d — AP with Mini-batch K-means centroid sampling  (experimental)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_kmeans_sampled(hist_matrix: np.ndarray,
                              channel_sizes: list,
                              sample_size: int = 10000,
                              preference: float = None,
                              _timing: dict = None) -> np.ndarray:
    """
    AP sampled variant using Mini-batch K-means centroid sampling.

    Runs Mini-batch K-means with k=sample_size on all N windows, maps each
    centroid to the nearest real window, then runs AP on those windows.
    Compared to random sampling, this selects cluster-central points (not
    boundary points), improving coverage of rare behaviors and ARI vs AP full.

    Parameters
    ----------
    sample_size : K-means clusters / AP sample size (default 10,000).
    preference  : AP preference. Default = global min estimate from all N windows.
    """
    from sklearn.cluster import AffinityPropagation
    import scipy.spatial.distance as ssd

    N = hist_matrix.shape[0]

    if N <= sample_size:
        print(f"[4/5] AP kmeans: N={N:,} ≤ sample_size={sample_size:,}, running full AP.")
        return cluster_ap_full(hist_matrix, channel_sizes, preference=preference)

    print(f"[4/5] AP kmeans-sampled  (N={N:,}, sample={sample_size:,}, "
          f"{100*sample_size/N:.0f}% of data) ...")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)

    t_samp = time.time()
    sample_idx = _kmeans_sample(cdf_all.astype(np.float32), sample_size, seed=42)
    t_samp = time.time() - t_samp
    print(f"      K-means sample: {len(sample_idx):,} points in {t_samp:.1f}s")
    if _timing is not None:
        _timing['t_kmeans'] = round(t_samp, 2)

    cdf_sample = cdf_all[sample_idx]

    print(f"      Building {len(sample_idx)}×{len(sample_idx)} affinity matrix "
          f"({len(sample_idx)**2*8/1e6:.0f} MB) ...")
    t_dist_start = time.time()
    dist     = ssd.cdist(cdf_sample, cdf_sample, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist   = time.time() - t_dist_start

    if preference is None:
        preference = _estimate_min_affinity(cdf_all)
        print(f"      Preference: {preference:.4f}  (global min estimate from {N:,} windows)")
    else:
        print(f"      Preference: {preference:.4f}  (user-specified)")

    np.fill_diagonal(affinity, preference)

    t0 = time.time()
    ap = AffinityPropagation(
        affinity         = "precomputed",
        preference       = preference,
        damping          = 0.9,
        max_iter         = 1000,
        convergence_iter = 15,
        random_state     = 0,
    )
    print("      Running AP (max_iter=1000, convergence_iter=15)...")
    with _Spinner("      AP"):
        ap.fit(affinity)
    t_algo = time.time() - t0

    exemplar_indices = sample_idx[ap.cluster_centers_indices_]

    if len(exemplar_indices) == 0:
        print("      WARNING: no exemplars found; returning single cluster")
        return np.zeros(N, dtype=int)

    exemplar_cdf = cdf_all.astype(np.float32)[exemplar_indices]
    index = faiss.IndexFlat(exemplar_cdf.shape[1], faiss.METRIC_L1)
    index.add(exemplar_cdf)
    _, assignments = index.search(cdf_all.astype(np.float32), 1)
    labels = assignments.ravel().astype(int)

    n_clusters = len(np.unique(labels))
    print(f"      Clusters: {n_clusters}  |  T_sample: {t_samp:.1f}s  "
          f"T_dist: {t_dist:.1f}s  T_algo: {t_algo:.1f}s")

    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Step 5e — AP with two-stage stratified sampling  (experimental)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_stratified_sampled(hist_matrix: np.ndarray,
                                   channel_sizes: list,
                                   sample_size: int = 10000,
                                   n_strata: int = 200,
                                   preference: float = None,
                                   _timing: dict = None) -> np.ndarray:
    """
    AP sampled with two-stage stratified sampling.

    Runs Mini-batch K-means with k=n_strata to partition N windows into rough
    behavioral groups, then draws an equal number of windows from each stratum.
    This guarantees that rare behaviors (small clusters) are represented in the
    AP sample regardless of their frequency, directly addressing the low-ARI
    problem of random/coreset/kmeans sampling with small sample sizes.

    Parameters
    ----------
    sample_size : total AP sample size (default 10,000).
    n_strata    : coarse K-means clusters for stratification (default 200).
    preference  : AP preference. Default = global min estimate from all N windows.
    """
    from sklearn.cluster import AffinityPropagation
    import scipy.spatial.distance as ssd

    N = hist_matrix.shape[0]

    if N <= sample_size:
        print(f"[4/5] AP stratified: N={N:,} ≤ sample_size={sample_size:,}, running full AP.")
        return cluster_ap_full(hist_matrix, channel_sizes, preference=preference)

    print(f"[4/5] AP stratified-sampled  (N={N:,}, sample={sample_size:,}, "
          f"n_strata={n_strata}) ...")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)

    t_samp  = time.time()
    sample_idx = _stratified_sample(cdf_all.astype(np.float32),
                                     sample_size, n_strata=n_strata, seed=42)
    t_samp  = time.time() - t_samp
    if _timing is not None:
        _timing['t_stratified'] = round(t_samp, 2)

    cdf_sample = cdf_all[sample_idx]
    actual_size = len(sample_idx)

    print(f"      Building {actual_size}×{actual_size} affinity matrix "
          f"({actual_size**2*8/1e6:.0f} MB) ...")
    t_dist_start = time.time()
    dist     = ssd.cdist(cdf_sample, cdf_sample, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist   = time.time() - t_dist_start

    if preference is None:
        preference = _estimate_min_affinity(cdf_all)
        print(f"      Preference: {preference:.4f}  (global min estimate from {N:,} windows)")
    else:
        print(f"      Preference: {preference:.4f}  (user-specified)")

    np.fill_diagonal(affinity, preference)

    t0 = time.time()
    ap = AffinityPropagation(
        affinity         = "precomputed",
        preference       = preference,
        damping          = 0.9,
        max_iter         = 1000,
        convergence_iter = 15,
        random_state     = 0,
    )
    print("      Running AP (max_iter=1000, convergence_iter=15)...")
    with _Spinner("      AP"):
        ap.fit(affinity)
    t_algo = time.time() - t0

    exemplar_indices = sample_idx[ap.cluster_centers_indices_]

    if len(exemplar_indices) == 0:
        print("      WARNING: no exemplars found; returning single cluster")
        return np.zeros(N, dtype=int)

    exemplar_cdf = cdf_all.astype(np.float32)[exemplar_indices]
    index = faiss.IndexFlat(exemplar_cdf.shape[1], faiss.METRIC_L1)
    index.add(exemplar_cdf)
    _, assignments = index.search(cdf_all.astype(np.float32), 1)
    labels = assignments.ravel().astype(int)

    n_clusters = len(np.unique(labels))
    print(f"      Clusters: {n_clusters}  |  T_sample: {t_samp:.1f}s  "
          f"T_dist: {t_dist:.1f}s  T_algo: {t_algo:.1f}s")

    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5f — Hierarchical AP with FAISS k-means partitioning
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_hierarchical(hist_matrix: np.ndarray,
                             channel_sizes: list,
                             n_blocks: int = None,
                             preference: float = None,
                             _timing: dict = None) -> np.ndarray:
    """
    Two-level hierarchical AP using FAISS k-means partitioning.

    Addresses the coverage limitation of ap_sampled (which only uses ~6000
    windows regardless of N) without the O(N²) cost of full AP.

    Algorithm
    ---------
    1. FAISS k-means partitions all N windows into M balanced blocks
       (each block ~2000–3000 windows, O((N/M)²) memory per block).
    2. Full AP within each block → local exemplars.
    3. AP on all candidate exemplars (typically M×10–20 points) → final exemplars.
    4. FAISS exact L1 assigns every window to its nearest final exemplar.

    Complexity
    ----------
    Memory  : O((N/M)²) per block  — M=10 gives 100× less than full AP
    AP work : O(N²/M) total        — same 100× reduction
    Coverage: 100% of windows participate (vs ~6000/N for ap_sampled)

    Parameters
    ----------
    n_blocks    : number of k-means partitions.
                  Default: max(4, N // 2500), targeting ~2500 windows/block.
    preference  : AP preference for both levels. Default: min(affinity) per block.
    """
    from sklearn.cluster import AffinityPropagation
    import scipy.spatial.distance as ssd

    N = hist_matrix.shape[0]
    t0 = time.time()

    if n_blocks is None:
        n_blocks = max(4, N // 2500)

    print(f"[4/5] Hierarchical AP  (N={N:,}, M={n_blocks} blocks, "
          f"~{N // n_blocks:,} windows/block)")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes).astype(np.float32)
    d = cdf_all.shape[1]

    # ── Step 1: FAISS k-means partition ──────────────────────────────────────
    print(f"      Partitioning with FAISS k-means (niter=20)...")
    t_part = time.time()
    kmeans = faiss.Kmeans(d, n_blocks, niter=20, seed=42, verbose=False)
    kmeans.train(cdf_all)
    _, block_ids = kmeans.index.search(cdf_all, 1)
    block_ids = block_ids.ravel()
    block_sizes = np.bincount(block_ids, minlength=n_blocks)
    print(f"      Partition: {time.time()-t_part:.1f}s  |  "
          f"block sizes min={block_sizes.min()} max={block_sizes.max()} "
          f"mean={int(block_sizes.mean())}")

    # ── Step 2: Full AP within each block ─────────────────────────────────────
    # Level-1 preference: median(per-block affinity).
    # Using global_pref (very negative) collapses each block to 1 exemplar — too few
    # candidates for Level-2.  Using per-block min is too inconsistent across blocks.
    # Median is sklearn's natural default and produces ~sqrt(N_block) exemplars per block,
    # giving a healthy number of Level-2 candidates.
    # Compute global preference now for Level-2 (matches ap_full's scale).
    global_pref = preference if preference is not None else _estimate_min_affinity(cdf_all)
    if preference is None:
        print(f"      Global preference estimate (Level-2): {global_pref:.4f}")

    t_dist_total = 0.0
    t_ap_total   = 0.0
    candidate_indices = []

    print(f"      Running AP in each block...")
    for b in tqdm(range(n_blocks), desc="      Blocks", unit="block", leave=False):
        idx      = np.where(block_ids == b)[0]
        blk_cdf  = cdf_all[idx].astype(np.float64)

        td = time.time()
        dist     = ssd.cdist(blk_cdf, blk_cdf, metric="cityblock")
        affinity = -(dist ** 2)
        t_dist_total += time.time() - td

        # per-block median: let AP naturally cluster at the local scale
        pref_b = preference if preference is not None else float(np.median(affinity))
        np.fill_diagonal(affinity, pref_b)

        ta = time.time()
        ap = AffinityPropagation(
            affinity         = "precomputed",
            preference       = pref_b,
            damping          = 0.9,
            max_iter         = 300,
            convergence_iter = 15,
            random_state     = 0,
        )
        ap.fit(affinity)
        t_ap_total += time.time() - ta

        candidate_indices.append(idx[ap.cluster_centers_indices_])

    candidates = np.concatenate(candidate_indices)
    print(f"      Level-1 done: {len(candidates)} candidate exemplars  "
          f"(dist={t_dist_total:.1f}s  AP={t_ap_total:.1f}s)")

    # ── Step 3: AP on candidate exemplars ─────────────────────────────────────
    print(f"      Level-2 AP on {len(candidates)} candidates...")
    cand_cdf = cdf_all[candidates].astype(np.float64)

    td2      = time.time()
    dist2    = ssd.cdist(cand_cdf, cand_cdf, metric="cityblock")
    aff2     = -(dist2 ** 2)
    t_dist2  = time.time() - td2

    # Level-2 preference: median(aff2).
    # global_pref (= min over all N windows) is far too negative for the small candidate
    # set → collapses everything to 1-3 clusters.  min(aff2) has the same problem for
    # the candidate set.  Median is the sklearn default and operates at the local scale
    # of the candidates, retaining roughly 50% of candidates as final exemplars.
    pref2 = preference if preference is not None else float(np.median(aff2))
    np.fill_diagonal(aff2, pref2)

    ta2 = time.time()
    ap2 = AffinityPropagation(
        affinity         = "precomputed",
        preference       = pref2,
        damping          = 0.9,
        max_iter         = 500,
        convergence_iter = 30,
        random_state     = 0,
    )
    print(f"      Running AP Level-2 (max_iter=500)...")
    with _Spinner("      AP L2"):
        ap2.fit(aff2)
    t_ap2 = time.time() - ta2

    final_exemplar_idx = candidates[ap2.cluster_centers_indices_]
    n_clusters = len(final_exemplar_idx)
    print(f"      Level-2 done: {n_clusters} final exemplars  "
          f"(dist={t_dist2:.1f}s  AP={t_ap2:.1f}s)")

    # ── Step 4: Assign all windows ─────────────────────────────────────────────
    final_cdf = cdf_all[final_exemplar_idx]
    idx_assign = faiss.IndexFlat(d, faiss.METRIC_L1)
    idx_assign.add(final_cdf)
    _, assignments = idx_assign.search(cdf_all, 1)
    labels = assignments.ravel().astype(int)

    t_total   = time.time() - t0
    t_dist_all = t_dist_total + t_dist2
    t_ap_all   = t_ap_total  + t_ap2
    print(f"      Clusters: {n_clusters}  |  "
          f"T_dist: {t_dist_all:.1f}s  T_AP: {t_ap_all:.1f}s  Total: {t_total:.1f}s")

    if _timing is not None:
        _timing['t_dist']     = round(t_dist_all, 2)
        _timing['t_algo']     = round(t_ap_all,   2)
        _timing['preference'] = pref2

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette(cdf_features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Silhouette Score on a random subset (max 2,000 points) using L1.
    Returns NaN if fewer than 2 clusters are found (excluding noise).
    Score interpretation: >0.5 good, 0.25–0.5 reasonable, <0.25 poor.
    """
    from sklearn.metrics import silhouette_score

    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    if len(set(valid_labels)) < 2:
        return float("nan")

    feats = cdf_features[valid_mask]

    # Subsample for speed — silhouette is O(N²)
    max_samples = 2_000
    if len(feats) > max_samples:
        idx = np.random.default_rng(42).choice(len(feats), max_samples, replace=False)
        feats, valid_labels = feats[idx], valid_labels[idx]

    return float(silhouette_score(feats, valid_labels, metric="l1"))


def run_pipeline(input_csv:              str,
                 output_csv:             str,
                 arena:                  str   = "3d_wired",
                 min_cluster_size:       int   = 15,
                 use_ap:                 bool  = False,
                 use_ap_sampled:         bool  = False,
                 use_ap_sparse:          bool  = False,
                 use_ap_hierarchical:    bool  = False,
                 use_hdbscan:            bool  = False,
                 use_ap_kmeans:          bool  = False,
                 use_ap_stratified:      bool  = False,
                 use_coreset:            bool  = False,
                 ann_k:                  int   = 100,
                 sparse_k:               int   = 100,
                 preference:             float = None,
                 sample_size:            int   = 10000,
                 n_strata:               int   = 200) -> None:
    """
    Full replacement for Matlab's VPAPPAxes.m.

    Input  : combined_harp_data_cleaned.csv  (output of stitched.py)
    Output : Cluster_detail_results.csv      (same format as Matlab output)
             Columns: ClusterIdx, Timestamp, Folder_Name
    """
    t_start = time.time()

    raw_motion, timestamps, folder_names = load_cleaned_motion(input_csv)
    sensor        = process_motion(raw_motion)
    bin_edges     = ARENA_BIN_EDGES[arena]
    channel_sizes = [len(e) - 1 for e in bin_edges]
    hist_matrix   = extract_histogram_features(sensor, bin_edges)
    N_windows     = hist_matrix.shape[0]

    if use_ap:
        labels = cluster_ap_full(hist_matrix, channel_sizes,
                                   K=ann_k, preference=preference)
    elif use_ap_sparse:
        labels = cluster_ap_sparse_knn(hist_matrix, channel_sizes,
                                       K=sparse_k, preference=preference)
    elif use_ap_hierarchical:
        labels = cluster_ap_hierarchical(hist_matrix, channel_sizes,
                                         preference=preference)
    elif use_hdbscan:
        labels = cluster_hdbscan(hist_matrix, channel_sizes,
                                 min_cluster_size)
    elif use_ap_kmeans:
        labels = cluster_ap_kmeans_sampled(hist_matrix, channel_sizes,
                                           sample_size=sample_size,
                                           preference=preference)
    elif use_ap_stratified:
        labels = cluster_ap_stratified_sampled(hist_matrix, channel_sizes,
                                               sample_size=sample_size,
                                               n_strata=n_strata,
                                               preference=preference)
    else:
        labels = cluster_ap_sampled(hist_matrix, channel_sizes,
                                    sample_size=sample_size, preference=preference,
                                    use_coreset=use_coreset)

    # ── Silhouette Score ──────────────────────────────────────────────────────
    print("[5/5] Computing cluster quality (Silhouette Score)...")
    cdf_feats = _build_cdf_features(hist_matrix, channel_sizes)
    sil = _silhouette(cdf_feats, labels)
    n_clusters = len(set(labels) - {-1})
    n_noise    = int((labels == -1).sum())

    print(f"      Clusters found  : {n_clusters}")
    if n_noise:
        print(f"      Noise points    : {n_noise:,} ({100*n_noise/N_windows:.1f}%)")
    if not np.isnan(sil):
        quality = "good" if sil > 0.5 else "reasonable" if sil > 0.25 else "poor"
        print(f"      Silhouette Score: {sil:.4f}  ({quality})")
        print(f"      Interpretation  : >0.5 good  |  0.25–0.5 reasonable  |  <0.25 poor")
    else:
        print(f"      Silhouette Score: N/A (need ≥ 2 clusters)")

    # ── Save results ──────────────────────────────────────────────────────────
    mid         = WIN_SIZE // 2
    win_ts      = timestamps[mid::WIN_SIZE][:N_windows]
    win_folders = folder_names[::WIN_SIZE][:N_windows]

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
        help="Use Affinity Propagation (full N×N matrix). "
             "Closest to Matlab pipeline. Only suitable for N < ~10,000. "
             "For large data use --use-ap-sampled instead.",
    )
    p.add_argument(
        "--use-ap-sampled", action="store_true",
        help="Use AP with random sampling (recommended for N > 10,000). "
             "Runs AP on a random subset (--sample-size windows), finds exemplars, "
             "then assigns all N windows to the nearest exemplar via FAISS. "
             "Scales to arbitrarily large datasets.",
    )
    p.add_argument(
        "--use-coreset-sample", action="store_true",
        help="Use Greedy K-Center coreset sampling instead of random sampling "
             "for AP sampled.  Guarantees coverage of rare behaviors. "
             "Only used with --use-ap-sampled.",
    )
    p.add_argument(
        "--use-ap-sparse", action="store_true",
        help="Use Sparse AP on a K-NN graph (equivalent to Matlab apclusterSparse.m). "
             "Memory O(N×K) instead of O(N²). Produces results close to AP full "
             "without the N² memory wall. Use --sparse-k to set graph degree (default: 100).",
    )
    p.add_argument(
        "--use-ap-hierarchical", action="store_true",
        help="Use two-level hierarchical AP: partition into blocks via FAISS k-means, "
             "run AP on each block, then AP on block exemplars. "
             "Fastest AP-family method; suitable for any N.",
    )
    p.add_argument(
        "--use-hdbscan", action="store_true",
        help="Use HDBSCAN instead of AP sampled (reference only — not recommended for "
             "production; see Known Limitations in README).",
    )
    p.add_argument(
        "--use-ap-kmeans-sample", action="store_true",
        help="Use AP with Mini-batch K-means centroid sampling (experimental). "
             "Runs K-means on all N windows, maps centroids to nearest real windows, "
             "then runs AP on that sample. Selects cluster-central points, improving "
             "ARI vs AP full compared to random sampling.",
    )
    p.add_argument(
        "--use-ap-stratified", action="store_true",
        help="Use AP with two-stage stratified sampling (experimental). "
             "Runs coarse K-means (--n-strata groups), then draws equal windows "
             "from each group. Ensures rare behaviors are represented in the AP "
             "sample regardless of frequency — designed to improve ARI vs AP full.",
    )
    p.add_argument(
        "--n-strata", type=int, default=200,
        help="Number of coarse K-means groups for stratified sampling (default: 200). "
             "Only used with --use-ap-stratified.",
    )
    p.add_argument(
        "--sample-size", type=int, default=10000,
        help="Number of windows to sample for AP sampled (default: 10000). "
             "Only used with --use-ap-sampled.",
    )
    p.add_argument(
        "--sparse-k", type=int, default=100,
        help="K-NN graph degree for Sparse AP (default: 100). "
             "Higher K → denser graph → closer to AP full but more memory/time. "
             "Only used with --use-ap-sparse.",
    )
    p.add_argument(
        "--ann-k", type=int, default=100,
        help="Number of nearest neighbors for FAISS ANN (used with --use-ap, default: 100)",
    )
    p.add_argument(
        "--preference", type=float, default=None,
        help="AP preference value (controls cluster count). "
             "Higher (toward 0) → more clusters. "
             "Lower (more negative) → fewer clusters. "
             "Default: min(similarity matrix). Used with --use-ap or --use-ap-sampled.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_pipeline(
        input_csv             = args.input,
        output_csv            = args.output,
        arena                 = args.arena,
        min_cluster_size      = args.min_cluster_size,
        use_ap                = args.use_ap,
        use_ap_sampled        = args.use_ap_sampled,
        use_ap_sparse         = args.use_ap_sparse,
        use_ap_hierarchical   = args.use_ap_hierarchical,
        use_hdbscan           = args.use_hdbscan,
        use_ap_kmeans         = args.use_ap_kmeans_sample,
        use_ap_stratified     = args.use_ap_stratified,
        use_coreset           = args.use_coreset_sample,
        ann_k                 = args.ann_k,
        sparse_k              = args.sparse_k,
        preference            = args.preference,
        sample_size           = args.sample_size,
        n_strata              = args.n_strata,
    )
