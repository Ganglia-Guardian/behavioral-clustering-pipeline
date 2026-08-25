"""
fig5f_downstream_agreement.py
------------------------------
Downstream behavioral agreement between AP sampled and AP full (paper Fig 5F).

ARI alone does not prove AP-sampled is a safe default: cluster ids are
arbitrary between two independent runs, cluster *sizes* can diverge even when
point-wise ARI is high, and occupancy/transition statistics -- not raw labels
-- are what actually get reported as behavioral findings. This script:

  1. matches AP-full clusters to AP-sampled clusters via Hungarian assignment
     on their contingency table (maximizing window overlap) -- cluster ids
     are arbitrary per-run, so this alignment step must happen before any of
     the comparisons below are meaningful,
  2. computes occupancy-vector correlation across matched cluster pairs,
  3. computes flattened transition-matrix correlation across matched pairs
     (transitions counted within-session only, via Folder_Name, to avoid
     spurious cross-session transition events),
  4. computes frame-level label agreement (single-session raster overlap),
  5. computes AP-full-exemplar -> nearest-AP-sampled-exemplar L1 distance
     (requires results/<dataset>/ap_exemplars.npz; run
     recompute_ap_exemplars.py first),
  6. bootstraps 95% CIs for (2) and (3).

Caveat on the transition-matrix bootstrap: it resamples flattened matrix
cells directly, which are not fully independent (rows sum to 1). Treat the
resulting CI as an approximate, first-pass estimate -- a session-level block
bootstrap would be more rigorous for the final manuscript numbers.

Usage
-----
    python3 scripts/fig5f_downstream_agreement.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

# Only datasets where AP full is feasible (N <= AP_FULL_MAX_N = 25,000) support
# this comparison. moving_test (N=47,028) still exceeds that and is excluded.
DATASETS = ["short_comparison_test", "comparison_test", "control_mouse_1_jul",
            "control_mouse_1_oct", "mp_mouse_1_jul", "mp_mouse_1_oct", "still_test"]
N_BOOTSTRAP = 1000
RNG_SEED = 0


def _load_bundle(npz_path: Path):
    """Load the self-consistent (labels_full, labels_sampled, exemplars, cdf,
    timestamps, folders) bundle written by recompute_ap_exemplars.py. Using
    this instead of the on-disk Cluster_detail_results CSVs guarantees AP full
    and AP sampled came from the same run with aligned preference -- see
    recompute_ap_exemplars.py's docstring for why the CSVs alone cannot be
    trusted to be a matched pair for every dataset."""
    data = np.load(npz_path, allow_pickle=True)
    return data


def match_clusters(labels_a: np.ndarray, labels_b: np.ndarray):
    """Hungarian matching on the contingency table, maximizing window overlap.

    Cluster ids from two independent AP runs are not comparable by number --
    "cluster 5" in AP full and "cluster 5" in AP sampled are unrelated labels
    that happened to get the same integer. This finds the one-to-one pairing
    between AP-full ids and AP-sampled ids that maximizes shared windows.

    Returns (pairs, (ids_a, ids_b, contingency_table)). pairs has at most
    min(K_a, K_b) entries; a pair with zero overlap is dropped (unmatched).
    """
    ids_a = np.unique(labels_a[labels_a >= 0])
    ids_b = np.unique(labels_b[labels_b >= 0])
    cont = np.zeros((len(ids_a), len(ids_b)), dtype=np.int64)
    for i, a in enumerate(ids_a):
        mask_a = labels_a == a
        for j, b in enumerate(ids_b):
            cont[i, j] = np.sum(mask_a & (labels_b == b))
    row_ind, col_ind = linear_sum_assignment(-cont)
    pairs = [(ids_a[r], ids_b[c]) for r, c in zip(row_ind, col_ind) if cont[r, c] > 0]
    return pairs, (ids_a, ids_b, cont)


def occupancy(labels: np.ndarray, ids) -> np.ndarray:
    N = len(labels)
    return np.array([np.sum(labels == i) / N for i in ids])


def transition_matrix(labels, timestamps, folders, ids) -> np.ndarray:
    """First-order transition probabilities among `ids`.

    Consecutive windows are defined by timestamp order; a transition is only
    counted when both windows share the same Folder_Name (session), so
    session boundaries never create a spurious transition event.
    """
    order = np.argsort(timestamps, kind="stable")
    labels_o = labels[order]
    folders_o = folders[order]

    id_to_idx = {cid: k for k, cid in enumerate(ids)}
    K = len(ids)
    counts = np.zeros((K, K), dtype=np.float64)

    for t in range(len(labels_o) - 1):
        if folders_o[t] != folders_o[t + 1]:
            continue
        a, b = labels_o[t], labels_o[t + 1]
        if a in id_to_idx and b in id_to_idx:
            counts[id_to_idx[a], id_to_idx[b]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)
    return probs


def bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP,
                    seed: int = RNG_SEED):
    """Point estimate + 95% bootstrap CI for Pearson correlation, resampling
    paired (x, y) entries with replacement."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return float("nan"), (float("nan"), float("nan"))
    r0 = float(np.corrcoef(x, y)[0, 1])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        boots[b] = np.nan if (np.std(xb) == 0 or np.std(yb) == 0) \
            else np.corrcoef(xb, yb)[0, 1]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return r0, (float(lo), float(hi))


def raster_agreement(labels_full, labels_sampled, pairs) -> float:
    """Fraction of windows where AP-sampled's label, translated through the
    matched-cluster pairing, equals AP-full's label at that same window.
    An AP-sampled cluster with no match (novel/unmatched) always counts as
    disagreement."""
    b_to_a = {b: a for a, b in pairs}
    translated = np.array([b_to_a.get(b, -999) for b in labels_sampled])
    return float(np.mean(translated == labels_full))


def exemplar_overlap(data):
    cdf = data["cdf"].astype(np.float64)
    ex_full = data["exemplar_idx_full"]
    ex_sampled = data["exemplar_idx_sampled"]

    cdf_full = cdf[ex_full]
    cdf_sampled = cdf[ex_sampled]

    dists = np.array([np.abs(cdf_sampled - v).sum(axis=1).min() for v in cdf_full])

    return {
        "n_exemplars_full": int(len(ex_full)),
        "n_exemplars_sampled": int(len(ex_sampled)),
        "mean_dist": float(dists.mean()),
        "median_dist": float(np.median(dists)),
        "frac_identical": float(np.mean(dists == 0.0)),
    }


def run_dataset(name: str):
    npz_path = RESULTS_DIR / name / "ap_exemplars.npz"
    if not npz_path.exists():
        print(f"[{name}] SKIP: {npz_path.name} not found "
              f"(run recompute_ap_exemplars.py first)")
        return None

    data = _load_bundle(npz_path)
    labels_full = data["labels_full"]
    labels_sampled = data["labels_sampled"]
    ts_full = ts_sampled = data["timestamps"]
    folders_full = folders_sampled = data["folders"]

    pairs, (ids_a, ids_b, cont) = match_clusters(labels_full, labels_sampled)
    K_full, K_sampled, K_matched = len(ids_a), len(ids_b), len(pairs)
    print(f"[{name}] K_full={K_full}  K_sampled={K_sampled}  matched={K_matched}")

    matched_a = [a for a, b in pairs]
    matched_b = [b for a, b in pairs]

    occ_full = occupancy(labels_full, matched_a)
    occ_sampled = occupancy(labels_sampled, matched_b)
    occ_r, occ_ci = bootstrap_corr(occ_full, occ_sampled)

    tm_full = transition_matrix(labels_full, ts_full, folders_full, matched_a)
    tm_sampled = transition_matrix(labels_sampled, ts_sampled, folders_sampled, matched_b)
    tm_r, tm_ci = bootstrap_corr(tm_full.ravel(), tm_sampled.ravel())

    raster_agree = raster_agreement(labels_full, labels_sampled, pairs)

    ex = exemplar_overlap(data)

    row = {
        "dataset": name,
        "K_full": K_full,
        "K_sampled": K_sampled,
        "K_matched": K_matched,
        "occupancy_r": round(occ_r, 4),
        "occupancy_ci_lo": round(occ_ci[0], 4),
        "occupancy_ci_hi": round(occ_ci[1], 4),
        "transition_r": round(tm_r, 4),
        "transition_ci_lo": round(tm_ci[0], 4),
        "transition_ci_hi": round(tm_ci[1], 4),
        "raster_agreement": round(raster_agree, 4),
        "exemplar_mean_dist": round(ex["mean_dist"], 4),
        "exemplar_median_dist": round(ex["median_dist"], 4),
        "exemplar_frac_identical": round(ex["frac_identical"], 4),
    }
    return row


def main():
    rows = []
    for name in DATASETS:
        row = run_dataset(name)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No datasets produced results.")
        return

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "fig5f_downstream_agreement.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("Fig 5F -- AP sampled vs AP full downstream behavioral agreement")
    print("=" * 78)
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
