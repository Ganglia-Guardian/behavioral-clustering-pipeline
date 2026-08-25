"""
recompute_ap_exemplars.py
--------------------------
Re-run AP full and AP sampled (preference aligned) on every dataset where AP
full is feasible (N <= AP_FULL_MAX_N), capturing which windows were chosen as
exemplars and saving a single self-consistent (labels_full, labels_sampled,
exemplars, cdf) bundle per dataset.

Why this exists: batch_compare.py only saves final per-window cluster labels,
not which windows AP selected as exemplars -- that's computed internally by
cluster_ap_full/cluster_ap_sampled but discarded. This script re-runs both
with return_exemplars=True (added to clustering_pipeline.py) to recover it.

Why it does NOT just trust the cached Cluster_detail_results_ap_sampled.csv:
batch_compare.py only aligns AP sampled's preference to AP full's exact value
when AP full is *freshly computed in the same session, immediately before* AP
sampled. That is not guaranteed across the repo's run history -- e.g. for
control_mouse_1_jul/control_mouse_1_oct/mp_mouse_1_oct/still_test, AP full was
originally SKIPPED (N exceeded the old AP_FULL_MAX_N=20,000) when AP sampled's
cached CSV was produced, so that cached AP sampled used an independent,
unaligned preference estimate; AP full was only added later in a separate
`--force ap_full` run after AP_FULL_MAX_N was raised to 25,000. The two cached
files are therefore not a matched pair for those 4 datasets. Confirmed
empirically: re-running AP sampled with the now-available AP full preference
disagrees with the cached CSV on 68-97% of windows for those datasets.

This script is the fix: for each dataset it recomputes AP full AND AP sampled
together, in the same process, with sampled's preference aligned to full's --
producing one guaranteed-consistent pair -- and OVERWRITES
Cluster_detail_results_ap_sampled.csv with this aligned result wherever it
differs from what's on disk, so the repo's saved results stop mixing aligned
and unaligned runs. (AP full's own cached CSV is left untouched; it already
matched on every dataset tested.)

Output per dataset: results/<dataset>/ap_exemplars.npz containing
    labels_full / labels_sampled     -- per-window cluster labels (this run)
    exemplar_idx_full / _sampled     -- window indices chosen as exemplars
    cdf                              -- (N, 30) CDF-L1 feature matrix
    timestamps / folders             -- per-window Timestamp / Folder_Name,
                                         aligned index-for-index with labels

Usage
-----
    python3 scripts/recompute_ap_exemplars.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_pipeline import (
    load_cleaned_motion, process_motion, extract_histogram_features,
    ARENA_BIN_EDGES, _build_cdf_features,
    cluster_ap_full, cluster_ap_sampled,
)
from batch_compare import save_result_csv

# Only datasets where AP full is feasible (N <= AP_FULL_MAX_N = 25,000) support
# this comparison. moving_test (N=47,028) still exceeds that and is excluded.
DATASETS = ["short_comparison_test", "comparison_test", "control_mouse_1_jul",
            "control_mouse_1_oct", "mp_mouse_1_jul", "mp_mouse_1_oct", "still_test"]
AP_SAMPLE_SIZE = 10_000
WIN_SIZE = 60


def _labels_from_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    cluster_idx = df["ClusterIdx"].values.astype(int)
    return np.where(cluster_idx == 0, -1, cluster_idx - 1)


def main():
    for name in DATASETS:
        out_dir = RESULTS_DIR / name
        clean_csv = out_dir / "combined_harp_data_cleaned.csv"
        if not clean_csv.exists():
            print(f"[{name}] SKIP: {clean_csv} not found")
            continue

        print(f"[{name}] loading + featurizing ...")
        raw, ts, folders = load_cleaned_motion(str(clean_csv))
        sensor = process_motion(raw)
        edges = ARENA_BIN_EDGES["3d_wired"]
        channel_sizes = [len(e) - 1 for e in edges]
        hist = extract_histogram_features(sensor, edges)
        cdf = _build_cdf_features(hist, channel_sizes)
        N = hist.shape[0]
        print(f"[{name}] N={N:,} windows")

        print(f"[{name}] running AP full (return_exemplars=True) ...")
        timing_full = {}
        labels_full, exemplar_idx_full = cluster_ap_full(
            hist, channel_sizes, _timing=timing_full, return_exemplars=True)

        cached_full = _labels_from_csv(out_dir / "Cluster_detail_results_ap_full.csv")
        if not np.array_equal(labels_full, cached_full):
            print(f"[{name}] NOTE: AP full re-run differs from cached CSV "
                  f"({np.sum(labels_full != cached_full):,} / {N:,} windows) -- "
                  f"overwriting cache with this run for self-consistency.")
        else:
            print(f"[{name}] AP full re-run matches cached labels exactly.")

        pref = timing_full.get("preference")
        print(f"[{name}] running AP sampled (preference aligned to AP full = {pref:.4f}) ...")
        timing_sampled = {}
        labels_sampled, exemplar_idx_sampled = cluster_ap_sampled(
            hist, channel_sizes, sample_size=AP_SAMPLE_SIZE, preference=pref,
            _timing=timing_sampled, return_exemplars=True)

        sampled_csv = out_dir / "Cluster_detail_results_ap_sampled.csv"
        cached_sampled = _labels_from_csv(sampled_csv)
        if not np.array_equal(labels_sampled, cached_sampled):
            n_diff = np.sum(labels_sampled != cached_sampled)
            print(f"[{name}] Cached AP sampled was NOT preference-aligned to AP full "
                  f"({n_diff:,} / {N:,} windows differ) -- overwriting {sampled_csv.name} "
                  f"with the aligned result.")
            save_result_csv(labels_sampled, ts, folders, N, sampled_csv)
        else:
            print(f"[{name}] AP sampled re-run matches cached labels exactly.")

        # Per-window timestamp/folder, same downsampling save_result_csv uses,
        # so timestamps/folders here line up index-for-index with labels_full
        # and labels_sampled.
        mid = WIN_SIZE // 2
        win_ts = ts[mid::WIN_SIZE][:N]
        win_folders = folders[::WIN_SIZE][:N]

        out_path = out_dir / "ap_exemplars.npz"
        np.savez(out_path,
                 labels_full=labels_full,
                 labels_sampled=labels_sampled,
                 exemplar_idx_full=exemplar_idx_full,
                 exemplar_idx_sampled=exemplar_idx_sampled,
                 cdf=cdf.astype(np.float32),
                 timestamps=win_ts,
                 folders=np.array(win_folders, dtype=object))
        print(f"[{name}] saved {out_path}  "
              f"(K_full={len(exemplar_idx_full)}, K_sampled={len(exemplar_idx_sampled)})\n")


if __name__ == "__main__":
    main()
