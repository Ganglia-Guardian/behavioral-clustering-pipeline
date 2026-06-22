"""
clustering_pipeline.py  —  Python replacement for VPAPPAxes.m.

Output format is compatible with the original Matlab pipeline.
Replaces the O(N²) EMD double-loop with vectorized CDF-L1 distances and
several scalable AP variants (sampled, coreset, k-means, stratified, sparse,
two-level) plus HDBSCAN as a reference method.

Usage:
    python clustering_pipeline.py \\
        --input  path/to/combined_harp_data_cleaned.csv \\
        --output path/to/Cluster_detail_results.csv \\
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
import scipy.spatial.distance as ssd
from sklearn.cluster import AffinityPropagation, MiniBatchKMeans
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
from tqdm import tqdm


class _Spinner:
    """Context manager: live spinner + elapsed time during silent sklearn calls."""
    def __init__(self, msg: str):
        self._msg = msg
        self._stop = threading.Event()
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

# Bin edges from hardHistNew.mat (VPAPPAxes.m).  11 + 11 + 6 + 2 = 30 bins.
# Channel 4 threshold is in log-space (-3), matching Matlab's exp-then-histc.
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

    timestamps = motion_df["Timestamp"].values.astype(np.float64)
    fn_col = "Folder_Name" if "Folder_Name" in motion_df.columns else "DataElement10"
    folder_names = motion_df[fn_col].values

    # DataElement0–9 are at column indices 3–12
    raw_motion = motion_df.iloc[:, 3:13].values.astype(np.float64)

    return raw_motion, timestamps, folder_names


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Signal processing  (mirrors processData.m)
# ─────────────────────────────────────────────────────────────────────────────

def _medfilt_clip(x: np.ndarray, w: int) -> np.ndarray:
    """1-D median filter with clipped boundary, matching Matlab's myMedFilt1."""
    return (pd.Series(x)
              .rolling(window=w, center=True, min_periods=1)
              .median()
              .to_numpy(dtype=x.dtype))


def process_motion(raw_motion: np.ndarray) -> dict:
    """Scale ADC → physical units, median filter (k=7), 0.5 Hz high-pass (filtfilt)."""
    print("[2/5] Signal processing (filtering, gravity separation)...")

    acc = raw_motion[:, 0:3] * (AC_RNG / 32768.0)
    gyr = raw_motion[:, 3:6] * (GYR_RNG / 32768.0)

    # clipped-boundary median filter — matches Matlab's myMedFilt1(x, 7)
    x = _medfilt_clip(acc[:, 0], 7)
    y = _medfilt_clip(-acc[:, 1], 7)   # negative sign: sensor orientation
    z = _medfilt_clip(acc[:, 2], 7)
    z_gyro = _medfilt_clip(gyr[:, 2], 7)

    # zero-phase 1st-order Butterworth high-pass, cut-off 0.5 Hz
    b, a = butter(1, 0.5 / (FS / 2.0), btype="high")
    x_BA = filtfilt(b, a, x)
    y_BA = filtfilt(b, a, y)
    z_BA = filtfilt(b, a, z)

    y_GA = y - y_BA
    tot_accel = np.sqrt(x_BA**2 + y_BA**2 + z_BA**2)

    return {"y_GA": y_GA, "z": z, "z_gyro": z_gyro, "tot_accel": tot_accel}


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Histogram feature extraction
#          (mirrors findFeaturesWired3DArena.m + runBuildMatrix.m option 2)
# ─────────────────────────────────────────────────────────────────────────────

def extract_histogram_features(sensor: dict, bin_edges: list) -> np.ndarray:
    """
    Slice signals into non-overlapping WIN_SIZE windows, compute a normalised
    histogram per channel, and concatenate into one (N_windows, 30) feature matrix.
    """
    print("[3/5] Extracting histogram features (vectorized)...")

    channels = [
        sensor["y_GA"],
        sensor["z"],
        sensor["z_gyro"],
        np.log(np.maximum(sensor["tot_accel"], 1e-10)),
    ]

    N_samples = len(channels[0])
    N_windows = N_samples // WIN_SIZE

    total_bins = sum(len(e) - 1 for e in bin_edges)
    hist_matrix = np.zeros((N_windows, total_bins), dtype=np.float32)

    col_start = 0
    for channel, edges in zip(channels, bin_edges):
        n_bins = len(edges) - 1

        win_data = channel[: N_windows * WIN_SIZE].reshape(N_windows, WIN_SIZE)
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
    Build a sparse K-NN similarity matrix via FAISS IVFFlat (approximate L1).
    Returns triplets [i, j, sim] compatible with apclusterSparse.
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
    AP on a K-NN sparse similarity graph (equivalent to apclusterSparse.m).
    Memory O(NK) vs O(N²) for full AP.
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
        # Use global min (not K-NN min) to avoid bias toward near-neighbor pairs.
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

    # ── Step 3: AP message passing ───────────────────────────────────────────
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

        # Row-wise max1 and max2 (tie-aware, via reduceat — no Python loops)
        row_max1         = np.maximum.reduceat(AS_data, indptr[:-1])
        row_max1_per_nnz = row_max1[row_idx]

        is_argmax    = AS_data >= row_max1_per_nnz - 1e-14
        argmax_count = np.zeros(N, dtype=np.int32)
        np.add.at(argmax_count, row_idx[is_argmax], 1)
        unique_argmax = is_argmax & (argmax_count[row_idx] == 1)

        AS_for_max2 = AS_data.copy()
        AS_for_max2[unique_argmax] = -np.inf
        row_max2 = np.maximum.reduceat(AS_for_max2, indptr[:-1])

        R_new = S_data - row_max1_per_nnz
        R_new[unique_argmax] = (S_data[unique_argmax]
                                - row_max2[row_idx[unique_argmax]])
        R_data = damping * R_data + (1.0 - damping) * R_new
        np.nan_to_num(R_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Availability update ───────────────────────────────────────────────
        pos_R_col = np.maximum(0.0, R_data)
        pos_R_col[diag_mask] = R_data[diag_mask]   # diagonal uses R directly

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
        # print(f"  it={it}  exemplars={len(exemplar_set)}")
    else:
        print(f"      WARNING: did not converge in {max_iter} iterations")

    t_algo = time.time() - t_algo_start

    exemplar_indices = np.array(sorted(exemplar_set), dtype=np.int64)

    if len(exemplar_indices) == 0:
        print("      WARNING: no exemplars found; returning single cluster")
        return np.zeros(N, dtype=int)

    labels = _faiss_assign(cdf_all, exemplar_indices)

    n_clusters = len(np.unique(labels))
    print(f"      Clusters: {n_clusters}  |  T_dist: {t_dist:.1f}s  "
          f"T_algo: {t_algo:.1f}s  Total: {t_dist + t_algo:.1f}s")

    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
        _timing['preference'] = preference

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — HDBSCAN clustering  (recommended replacement for AP)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cdf_features(hist_matrix: np.ndarray, channel_sizes: list) -> np.ndarray:
    """CDF of each channel, divided by (n_bins-1) for equal channel weighting."""
    col, cdfs = 0, []
    for size in channel_sizes:
        cdfs.append(np.cumsum(hist_matrix[:, col: col + size], axis=1) / (size - 1))
        col += size
    return np.hstack(cdfs).astype(np.float64)


def cluster_hdbscan(hist_matrix: np.ndarray,
                    channel_sizes: list,
                    min_cluster_size: int = 15,
                    _timing: dict = None) -> np.ndarray:
    """HDBSCAN on CDF-L1 features. Labels: 0-based clusters, -1 = noise."""
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

def _faiss_assign(cdf_all: np.ndarray, exemplar_indices: np.ndarray) -> np.ndarray:
    """Assign every window to its nearest exemplar via FAISS exact L1 search."""
    exemplar_cdf = cdf_all.astype(np.float32)[exemplar_indices]
    idx = faiss.IndexFlat(exemplar_cdf.shape[1], faiss.METRIC_L1)
    idx.add(exemplar_cdf)
    _, nn = idx.search(cdf_all.astype(np.float32), 1)
    return nn.ravel().astype(int)


def _estimate_min_affinity(cdf_all: np.ndarray, n_pairs: int = 500_000) -> float:
    # 500k pairs is overkill for small datasets but it's fast enough that it's
    # not worth adding a conditional. tried 50k once and got slightly off preference.
    N = cdf_all.shape[0]
    rng = np.random.default_rng(42)
    i = rng.integers(0, N, n_pairs)
    j = rng.integers(0, N, n_pairs)
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
    Full AP on an N×N CDF-L1 similarity matrix.
    Preference = min(similarity), matching Matlab's min(s(:,3)).
    Skipped automatically for N > 20,000.
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

    print(f"      Building full {N}×{N} similarity matrix via scipy.cdist ...")
    dist     = ssd.cdist(cdf_features, cdf_features, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist = time.time() - t0

    if preference is None:
        # median causes cluster explosion with our coarse bins (learned this the hard way)
        preference = float(affinity.min())
        print(f"      Preference: {preference:.4f}  (auto = min similarity)")
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
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
        _timing['preference'] = preference
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Coreset helper — Greedy K-Center (Gonzalez 1985)
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans_sample(cdf_features: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """Mini-batch K-means (k=K), then map centroids to nearest real windows via FAISS."""
    X = cdf_features.astype(np.float32)
    N = X.shape[0]

    print(f"      Running Mini-batch K-means (k={K:,}, N={N:,}) ...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3,
                         batch_size=min(4096, N), max_iter=100)
    km.fit(X)
    centroids = km.cluster_centers_.astype(np.float32)   # (K, d)
    print(f"      K-means done in {time.time() - t0:.1f}s")

    index = faiss.IndexFlat(X.shape[1], faiss.METRIC_L1)
    index.add(X)
    _, nn = index.search(centroids, 1)                   # (K, 1)
    selected = np.unique(nn.ravel().astype(np.int64))    # deduplicate
    return selected


def _stratified_sample(cdf_features: np.ndarray, K: int,
                        n_strata: int = 200, seed: int = 42) -> np.ndarray:
    """
    Stratified sampling: partition into n_strata coarse groups via Mini-batch K-means,
    then draw K//n_strata windows from each stratum. Ensures rare behaviors are represented.
    """
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
    selected = []
    deficit = 0

    for s in range(n_strata):
        idx = np.where(stratum_ids == s)[0]
        n_take = min(len(idx), per_stratum)
        deficit += per_stratum - n_take
        chosen = rng.choice(idx, n_take, replace=False)
        selected.append(chosen)

    selected = np.concatenate(selected)

    # top up from the full pool if some strata were too small
    if deficit > 0:
        already = set(selected.tolist())
        pool = np.array([i for i in range(N) if i not in already], dtype=np.int64)
        n_fill = min(deficit, len(pool))
        fill_idx = rng.choice(pool, n_fill, replace=False)
        selected = np.concatenate([selected, fill_idx])

    selected = np.sort(selected.astype(np.int64))
    print(f"      Stratified sample: {len(selected):,} points "
          f"({per_stratum} per stratum, {n_strata} strata)")
    return selected


def _coreset_sample(cdf_features: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Greedy K-Center coreset: iteratively pick the point farthest from the selected set."""
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
    """AP on a random (or coreset) subset of sample_size windows, FAISS assigns the rest."""
    N = hist_matrix.shape[0]

    if N <= sample_size:
        print(f"[4/5] AP sampled: N={N:,} ≤ sample_size={sample_size:,}, running full AP.")
        return cluster_ap_full(hist_matrix, channel_sizes, preference=preference)

    sampling_method = "coreset" if use_coreset else "random"
    print(f"[4/5] AP sampled  (N={N:,}, sample={sample_size:,}, "
          f"{100*sample_size/N:.0f}% of data, sampling={sampling_method}) ...")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)

    if use_coreset:
        print(f"      Running Greedy K-Center coreset (K={sample_size:,}) ...")
        t_coreset = time.time()
        sample_idx = _coreset_sample(cdf_all.astype(np.float32), sample_size, seed=42)
        t_coreset = time.time() - t_coreset
        print(f"      Coreset done in {t_coreset:.1f}s")
        if _timing is not None:
            _timing['t_coreset'] = round(t_coreset, 2)
    else:
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(N, sample_size, replace=False))

    cdf_sample = cdf_all[sample_idx]

    print(f"      Building {sample_size}×{sample_size} affinity matrix "
          f"({sample_size**2*8/1e6:.0f} MB) ...")
    t_dist_start = time.time()
    dist     = ssd.cdist(cdf_sample, cdf_sample, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist = time.time() - t_dist_start

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
        sample_labels = ap.fit_predict(affinity)
    n_clusters    = len(set(sample_labels))
    print(f"      AP on sample: {time.time()-t0:.1f}s  |  {n_clusters} clusters found")

    exemplar_indices = sample_idx[ap.cluster_centers_indices_]
    labels = _faiss_assign(cdf_all, exemplar_indices)

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
    """AP on Mini-batch K-means centroids (mapped to nearest real windows)."""
    N = hist_matrix.shape[0]

    if N <= sample_size:
        print(f"[4/5] AP kmeans: N={N:,} ≤ sample_size={sample_size:,}, running full AP.")
        return cluster_ap_full(hist_matrix, channel_sizes, preference=preference)

    print(f"[4/5] AP kmeans-sampled  (N={N:,}, sample={sample_size:,}, "
          f"{100*sample_size/N:.0f}% of data) ...")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)

    t0_samp = time.time()
    sample_idx = _kmeans_sample(cdf_all.astype(np.float32), sample_size, seed=42)
    t_samp = time.time() - t0_samp
    print(f"      K-means sample: {len(sample_idx):,} points in {t_samp:.1f}s")
    if _timing is not None:
        _timing['t_kmeans'] = round(t_samp, 2)

    cdf_sample = cdf_all[sample_idx]
    n = len(sample_idx)

    print(f"      Building {n}×{n} affinity matrix ({n**2*8/1e6:.0f} MB) ...")
    t0_dist = time.time()
    dist = ssd.cdist(cdf_sample, cdf_sample, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist = time.time() - t0_dist

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

    labels = _faiss_assign(cdf_all, exemplar_indices)
    n_clusters = len(np.unique(labels))
    print(f"      Clusters: {n_clusters}  |  T_sample: {t_samp:.1f}s  "
          f"T_dist: {t_dist:.1f}s  T_algo: {t_algo:.1f}s")

    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5e — AP with two-stage stratified sampling
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_stratified_sampled(hist_matrix: np.ndarray,
                                   channel_sizes: list,
                                   sample_size: int = 10000,
                                   n_strata: int = 200,
                                   preference: float = None,
                                   _timing: dict = None) -> np.ndarray:
    """AP on a stratified sample drawn equally from n_strata coarse K-means groups."""

    N = hist_matrix.shape[0]

    if N <= sample_size:
        print(f"[4/5] AP stratified: N={N:,} ≤ sample_size={sample_size:,}, running full AP.")
        return cluster_ap_full(hist_matrix, channel_sizes, preference=preference)

    print(f"[4/5] AP stratified-sampled  (N={N:,}, sample={sample_size:,}, "
          f"n_strata={n_strata}) ...")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes)

    t0_samp = time.time()
    sample_idx = _stratified_sample(cdf_all.astype(np.float32),
                                    sample_size, n_strata=n_strata, seed=42)
    t_samp = time.time() - t0_samp
    if _timing is not None:
        _timing['t_stratified'] = round(t_samp, 2)

    cdf_sample = cdf_all[sample_idx]
    n = len(sample_idx)

    print(f"      Building {n}×{n} affinity matrix ({n**2*8/1e6:.0f} MB) ...")
    t0_dist = time.time()
    dist = ssd.cdist(cdf_sample, cdf_sample, metric="cityblock").astype(np.float64)
    affinity = -(dist ** 2)
    t_dist = time.time() - t0_dist

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

    labels = _faiss_assign(cdf_all, exemplar_indices)
    n_clusters = len(np.unique(labels))
    print(f"      Clusters: {n_clusters}  |  T_sample: {t_samp:.1f}s  "
          f"T_dist: {t_dist:.1f}s  T_algo: {t_algo:.1f}s")

    if _timing is not None:
        _timing['t_dist'] = round(t_dist, 2)
        _timing['t_algo'] = round(t_algo, 2)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 5f — Two-level AP with FAISS k-means partitioning
# ─────────────────────────────────────────────────────────────────────────────

def cluster_ap_twolevel(hist_matrix: np.ndarray,
                             channel_sizes: list,
                             n_blocks: int = None,
                             preference: float = None,
                             _timing: dict = None) -> np.ndarray:
    """
    Two-level AP: block AP on FAISS k-means partitions, then AP on candidate exemplars.
    All N windows participate; memory is O((N/M)²) per block instead of O(N²).
    """

    N = hist_matrix.shape[0]
    t0 = time.time()

    if n_blocks is None:
        n_blocks = max(4, N // 2500)
        # TODO: could auto-tune this based on available RAM instead of window count

    print(f"[4/5] Two-level AP  (N={N:,}, M={n_blocks} blocks, "
          f"~{N // n_blocks:,} windows/block)")

    cdf_all = _build_cdf_features(hist_matrix, channel_sizes).astype(np.float32)
    d = cdf_all.shape[1]

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

    # median preference per block: avoids collapsing each block to 1 exemplar
    global_pref = preference if preference is not None else _estimate_min_affinity(cdf_all)
    if preference is None:
        print(f"      Global preference estimate (Level-2): {global_pref:.4f}")

    t_dist_total = 0.0
    t_ap_total = 0.0
    candidate_indices = []

    print(f"      Running AP in each block...")
    for b in tqdm(range(n_blocks), desc="      Blocks", unit="block", leave=False):
        idx      = np.where(block_ids == b)[0]
        blk_cdf  = cdf_all[idx].astype(np.float64)

        td = time.time()
        dist     = ssd.cdist(blk_cdf, blk_cdf, metric="cityblock")
        affinity = -(dist ** 2)
        t_dist_total += time.time() - td

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

    candidates = np.unique(np.concatenate(candidate_indices))
    print(f"      Level-1 done: {len(candidates)} candidate exemplars  "
          f"(dist={t_dist_total:.1f}s  AP={t_ap_total:.1f}s)")

    print(f"      Level-2 AP on {len(candidates)} candidates...")
    cand_cdf = cdf_all[candidates].astype(np.float64)

    td2 = time.time()
    dist2 = ssd.cdist(cand_cdf, cand_cdf, metric="cityblock")
    aff2 = -(dist2 ** 2)
    t_dist2 = time.time() - td2

    # median preference: global_pref is too negative at the candidate scale
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

    labels = _faiss_assign(cdf_all, final_exemplar_idx)

    t_total   = time.time() - t0
    t_dist_all = t_dist_total + t_dist2
    t_ap_all   = t_ap_total  + t_ap2
    print(f"      Clusters: {n_clusters}  |  "
          f"T_dist: {t_dist_all:.1f}s  T_AP: {t_ap_all:.1f}s  Total: {t_total:.1f}s")

    if _timing is not None:
        _timing['t_dist'] = round(t_dist_all, 2)
        _timing['t_algo'] = round(t_ap_all, 2)
        _timing['preference'] = pref2

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _silhouette(cdf_features: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score (L1) on up to 2,000 subsampled non-noise points."""
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


def run_pipeline(input_csv: str,
                 output_csv: str,
                 arena: str = "3d_wired",
                 min_cluster_size: int = 15,
                 use_ap: bool = False,
                 use_ap_sampled: bool = False,
                 use_ap_sparse: bool = False,
                 use_ap_twolevel: bool = False,
                 use_hdbscan: bool = False,
                 use_ap_kmeans: bool = False,
                 use_ap_stratified: bool = False,
                 use_coreset: bool = False,
                 ann_k: int = 100,
                 sparse_k: int = 100,
                 preference: float = None,
                 sample_size: int = 10000,
                 n_strata: int = 200) -> None:
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
    elif use_ap_twolevel:
        labels = cluster_ap_twolevel(hist_matrix, channel_sizes,
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

    print("[5/5] Computing silhouette score...")
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
    else:
        print(f"      Silhouette Score: N/A (need ≥ 2 clusters)")

    mid = WIN_SIZE // 2
    win_ts = timestamps[mid::WIN_SIZE][:N_windows]
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
        "--use-ap-twolevel", action="store_true",
        help="Use two-level AP: partition into blocks via FAISS k-means, "
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
        input_csv=args.input,
        output_csv=args.output,
        arena=args.arena,
        min_cluster_size=args.min_cluster_size,
        use_ap=args.use_ap,
        use_ap_sampled=args.use_ap_sampled,
        use_ap_sparse=args.use_ap_sparse,
        use_ap_twolevel=args.use_ap_twolevel,
        use_hdbscan=args.use_hdbscan,
        use_ap_kmeans=args.use_ap_kmeans_sample,
        use_ap_stratified=args.use_ap_stratified,
        use_coreset=args.use_coreset_sample,
        ann_k=args.ann_k,
        sparse_k=args.sparse_k,
        preference=args.preference,
        sample_size=args.sample_size,
        n_strata=args.n_strata,
    )
