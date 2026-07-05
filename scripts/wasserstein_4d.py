"""
wasserstein_4d.py
-----------------
4D joint Wasserstein distance functions for IMU windows.

Each window is treated as 60 points in R^4:
    (y_GA[t], z[t], z_gyro[t], log_tot_accel[t])  for t = 0..59

Three methods:
  sliced_w1  : Sliced W1 — random projections to 1D, average W1  [O(L·N²·60 log 60)]
  exact_emd  : Exact W1 via LP (POT emd2) on 60×60 cost matrix  [O(N²·60³)]
  sinkhorn   : Sinkhorn-regularized OT (approximate)             [O(N²·60²·iters)]

Channel normalization (matches _EMPIRICAL_SCALES in clustering_pipeline.py):
  y_GA / 2.0,  z / 2.5,  z_gyro / 200.0,  log_tot / 3.0
"""

import time
import numpy as np
import scipy.spatial.distance as ssd

from clustering_pipeline import WIN_SIZE, _EMPIRICAL_SCALES


# ── Window extraction ────────────────────────────────────────────────────────

def extract_joint(sensor: dict) -> np.ndarray:
    """
    Build joint 4D windows from a sensor dict (output of process_motion).

    Returns
    -------
    joint : np.ndarray, shape (N_windows, WIN_SIZE, 4)
        Each row is one time-point in normalized R^4.
        Channel order: y_GA, z, z_gyro, log_tot_accel.
    """
    channels = [
        sensor["y_GA"],
        sensor["z"],
        sensor["z_gyro"],
        np.log(np.maximum(sensor["tot_accel"], 1e-10)),
    ]
    N = len(channels[0]) // WIN_SIZE
    joint = np.zeros((N, WIN_SIZE, 4), dtype=np.float32)
    for c, (ch, scale) in enumerate(zip(channels, _EMPIRICAL_SCALES)):
        joint[:, :, c] = ch[:N * WIN_SIZE].reshape(N, WIN_SIZE) / scale
    return joint


# ── Distance functions ───────────────────────────────────────────────────────

def sliced_w1(A: np.ndarray, B: np.ndarray,
              L: int = 100, seed: int = 42) -> np.ndarray:
    """
    Sliced Wasserstein-1 distance matrix.

    Parameters
    ----------
    A, B : np.ndarray, shape (N_A, WIN_SIZE, 4) and (N_B, WIN_SIZE, 4)
    L    : number of random 1D projections
    seed : RNG seed for reproducibility

    Returns
    -------
    dist : np.ndarray, shape (N_A, N_B)

    Algorithm
    ---------
    For each random unit vector theta in R^4:
        project A and B onto theta -> (N, 60) scalars
        sort each window's projection
        W1 for this direction = mean |sorted_A[i] - sorted_B[j]|  (cityblock / 60)
    Average over L directions.
    """
    rng  = np.random.default_rng(seed)
    dirs = rng.normal(size=(L, 4)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    dist = np.zeros((len(A), len(B)), dtype=np.float64)
    for l, theta in enumerate(dirs):
        proj_A = (A * theta[None, None, :]).sum(axis=2)   # (N_A, 60)
        proj_B = (B * theta[None, None, :]).sum(axis=2)   # (N_B, 60)
        dist  += ssd.cdist(np.sort(proj_A, axis=1),
                           np.sort(proj_B, axis=1),
                           metric='cityblock') / WIN_SIZE
        if (l + 1) % 25 == 0:
            print(f"      projection {l+1}/{L}")
    return dist / L


def exact_emd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Exact W1 via linear programming (POT emd2) on each 60×60 cost matrix.

    Parameters
    ----------
    A, B : np.ndarray, shape (N_A, WIN_SIZE, 4) and (N_B, WIN_SIZE, 4)

    Returns
    -------
    dist : np.ndarray, shape (N_A, N_B)

    Note: fast for small WIN_SIZE (60×60 LP ≈ 0.15 ms). Scales as O(N_A × N_B).
    """
    import ot
    N_A, N_B = len(A), len(B)
    w    = np.ones(WIN_SIZE, dtype=np.float64) / WIN_SIZE
    dist = np.zeros((N_A, N_B), dtype=np.float64)
    t0   = time.time()
    for i in range(N_A):
        for j in range(N_B):
            M = ssd.cdist(A[i].astype(np.float64),
                          B[j].astype(np.float64),
                          metric='cityblock')
            dist[i, j] = ot.emd2(w, w, M)
        done = (i + 1) * N_B
        if (i + 1) % max(1, N_A // 8) == 0:
            elapsed = time.time() - t0
            total   = N_A * N_B
            eta     = elapsed / done * (total - done) if done < total else 0
            print(f"      {done}/{total} pairs  {elapsed:.0f}s  ETA {eta:.0f}s")
    return dist


def sinkhorn(A: np.ndarray, B: np.ndarray, reg: float = 0.05) -> np.ndarray:
    """
    Sinkhorn-regularized OT distance (approximate W1).

    Parameters
    ----------
    A, B : np.ndarray, shape (N_A, WIN_SIZE, 4) and (N_B, WIN_SIZE, 4)
    reg  : entropy regularization coefficient (smaller = more accurate, slower)

    Returns
    -------
    dist : np.ndarray, shape (N_A, N_B)
    """
    import ot
    N_A, N_B = len(A), len(B)
    w    = np.ones(WIN_SIZE, dtype=np.float64) / WIN_SIZE
    dist = np.zeros((N_A, N_B), dtype=np.float64)
    t0   = time.time()
    for i in range(N_A):
        for j in range(N_B):
            M = ssd.cdist(A[i].astype(np.float64),
                          B[j].astype(np.float64),
                          metric='cityblock')
            result = ot.sinkhorn2(w, w, M, reg=reg)
            dist[i, j] = float(result) if np.isscalar(result) else float(result[0])
        done = (i + 1) * N_B
        if (i + 1) % max(1, N_A // 8) == 0:
            elapsed = time.time() - t0
            total   = N_A * N_B
            eta     = elapsed / done * (total - done) if done < total else 0
            print(f"      {done}/{total} pairs  {elapsed:.0f}s  ETA {eta:.0f}s")
    return dist


# ── Convenience dispatcher ───────────────────────────────────────────────────

def pairwise(joint: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Compute square pairwise distance matrix for one window set."""
    return cross(joint, joint, method, **kwargs)


def cross(A: np.ndarray, B: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Compute (N_A, N_B) distance matrix between two window sets."""
    if method == 'sliced_w1':
        return sliced_w1(A, B, **kwargs)
    elif method == 'exact_emd':
        return exact_emd(A, B)
    elif method == 'sinkhorn':
        return sinkhorn(A, B, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose sliced_w1, exact_emd, sinkhorn.")


# ── Time estimation ──────────────────────────────────────────────────────────

def estimate_time(joint: np.ndarray, method: str, n_splits: int = 5,
                  n_probe: int = 200, **kwargs):
    """
    Probe n_probe random pairs, then extrapolate to full split-half runtime.
    """
    N     = len(joint)
    rng   = np.random.default_rng(0)
    pairs = [(rng.integers(N), rng.integers(N)) for _ in range(n_probe)]

    t0 = time.time()
    if method == 'sliced_w1':
        # Probe is one cdist call per direction across n_probe pairs
        sliced_w1(joint[:n_probe], joint[:n_probe], **kwargs)
        elapsed = time.time() - t0
        per_pair_ms = elapsed / n_probe ** 2 * 1000
    else:
        import ot
        w = np.ones(WIN_SIZE, dtype=np.float64) / WIN_SIZE
        for i, j in pairs:
            M = ssd.cdist(joint[i].astype(np.float64),
                          joint[j].astype(np.float64),
                          metric='cityblock')
            if method == 'exact_emd':
                ot.emd2(w, w, M)
            else:
                ot.sinkhorn2(w, w, M, reg=kwargs.get('reg', 0.05))
        elapsed     = time.time() - t0
        per_pair_ms = elapsed / n_probe * 1000

    half_pairs  = (N // 2) * (N // 2 - 1) // 2
    total_pairs = n_splits * 2 * half_pairs
    eta_s       = per_pair_ms / 1000 * total_pairs

    print(f"\n  Time estimate ({method}):")
    print(f"    {n_probe} probes → {per_pair_ms:.2f} ms/pair")
    print(f"    {n_splits} splits × 2 halves × {half_pairs:,} pairs/half")
    print(f"    Estimated total: {eta_s/60:.1f} min")
