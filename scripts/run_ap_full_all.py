"""
run_ap_full_all.py
------------------
Run AP full on every dataset and save results.
Skips datasets where the output CSV already exists.

Usage (from repo root):
    python3 scripts/run_ap_full_all.py
    python3 scripts/run_ap_full_all.py --include-moving-test   # also run N=47k dataset
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
LAB_DATA    = REPO_ROOT.parent / "lab_data"

sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, ARENA_BIN_EDGES,
    cluster_ap_sparse, _build_cdf_features, _silhouette, WIN_SIZE,
)
from prepare_test_data import convert as prepare_csv


DATASETS = [
    {
        "name":    "short_comparison_test",
        "raw_csv": LAB_DATA / "short_comparison_test" / "harp_data_cut.csv",
        "label":   "short_test",
        "arena":   "3d_wired",
    },
    {
        "name":    "comparison_test",
        "raw_csv": LAB_DATA / "comparison_test" / "Harp-Motion2025-04-07T08_22_29.csv",
        "label":   "comparison_test",
        "arena":   "3d_wired",
    },
    {
        "name":    "mp_mouse_1_jul",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-07-08T08_36_38.csv",
        "label":   "mp_jul",
        "arena":   "3d_wired",
    },
    {
        "name":    "control_mouse_1_jul",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-07-07T15_17_33.csv",
        "label":   "control_jul",
        "arena":   "3d_wired",
    },
    {
        "name":    "control_mouse_1_oct",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-10-13T15_04_32.csv",
        "label":   "control_oct",
        "arena":   "3d_wired",
    },
    {
        "name":    "mp_mouse_1_oct",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-10-13T07_36_52.csv",
        "label":   "mp_oct",
        "arena":   "3d_wired",
    },
    {
        "name":    "still_test",
        "raw_csv": LAB_DATA / "still_test" / "Harp-Motion2025-09-15T08_33_43.csv",
        "label":   "still_test",
        "arena":   "3d_wired",
    },
    {
        "name":    "moving_test",
        "raw_csv": LAB_DATA / "moving_test" / "IMU" / "Harp-Motion2025-05-23T12_08_47.csv",
        "label":   "moving_test",
        "arena":   "3d_wired",
        "skip_by_default": True,   # N=47,028 → ~17.7 GB distance matrix
    },
]


def run_one(cfg: dict, out_dir: Path) -> dict:
    out_csv   = out_dir / "Cluster_detail_results_ap_full.csv"
    clean_csv = out_dir / "combined_harp_data_cleaned.csv"

    if out_csv.exists():
        print(f"  [skip] {cfg['name']} — result already exists: {out_csv.name}")
        existing = pd.read_csv(str(out_csv))
        n = len(existing)
        n_clusters = existing["ClusterIdx"].nunique()
        return {"name": cfg["name"], "status": "cached", "n_windows": n, "n_clusters": n_clusters}

    # Prepare data if needed
    if not clean_csv.exists():
        print(f"  Preparing data ...")
        prepare_csv(str(cfg["raw_csv"]), str(clean_csv), cfg["label"])

    raw, ts, folders = load_cleaned_motion(str(clean_csv))
    sensor    = process_motion(raw)
    edges     = ARENA_BIN_EDGES[cfg["arena"]]
    hist      = extract_histogram_features(sensor, edges)
    N         = hist.shape[0]
    mem_gb    = N ** 2 * 8 / 1e9
    ch_sizes  = [len(e) - 1 for e in edges]

    print(f"\n  {cfg['name']}  (N={N:,}, matrix ≈ {mem_gb:.1f} GB)")
    print(f"  ─────────────────────────────────────────────────")

    _timing = {}
    t0 = time.perf_counter()
    labels = cluster_ap_sparse(hist, ch_sizes, _timing=_timing)
    elapsed = time.perf_counter() - t0

    cdf = _build_cdf_features(hist, ch_sizes)
    sil = _silhouette(cdf, labels)

    # Save CSV
    mid         = WIN_SIZE // 2
    win_ts      = ts[mid::WIN_SIZE][:N]
    win_folders = folders[::WIN_SIZE][:N]
    cluster_idx = np.where(labels < 0, 0, labels + 1)
    pd.DataFrame({
        "ClusterIdx":  cluster_idx,
        "Timestamp":   win_ts,
        "Folder_Name": win_folders,
    }).to_csv(str(out_csv), index=False)

    n_clusters = len(set(labels))
    quality    = "good" if sil > 0.5 else "reasonable" if sil > 0.25 else "poor"
    pref       = _timing.get("preference", float("nan"))

    print(f"  Clusters   : {n_clusters}")
    print(f"  Silhouette : {sil:.4f}  ({quality})")
    print(f"  Preference : {pref:.4f}")
    print(f"  T_dist     : {_timing.get('t_dist', 0):.1f}s")
    print(f"  T_algo     : {_timing.get('t_algo', 0):.1f}s")
    print(f"  Total      : {elapsed:.1f}s")
    print(f"  Saved      : {out_csv}")

    return {
        "name":       cfg["name"],
        "status":     "ok",
        "n_windows":  N,
        "n_clusters": n_clusters,
        "silhouette": round(sil, 4),
        "preference": round(pref, 4),
        "t_dist_s":   _timing.get("t_dist", float("nan")),
        "t_algo_s":   _timing.get("t_algo", float("nan")),
        "t_s":        round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-moving-test", action="store_true",
                        help="Also run moving_test (N=47,028, needs ~17.7 GB RAM)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AP full — all datasets")
    print("=" * 60)

    summary = []
    for cfg in DATASETS:
        if cfg.get("skip_by_default") and not args.include_moving_test:
            n_approx = 47028
            mem_gb   = n_approx ** 2 * 8 / 1e9
            print(f"\n  [skip] {cfg['name']}  (N≈{n_approx:,}, matrix ≈ {mem_gb:.1f} GB)")
            print(f"         Pass --include-moving-test to run this dataset.")
            summary.append({"name": cfg["name"], "status": "skipped_memory"})
            continue

        out_dir = RESULTS_DIR / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_one(cfg, out_dir)
        summary.append(result)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    cols = ["name", "status", "n_windows", "n_clusters", "silhouette", "t_s"]
    header = f"  {'Dataset':<30}  {'Status':<8}  {'N':>7}  {'Clusters':>8}  {'Sil':>6}  {'Time(s)':>8}"
    print(header)
    print("  " + "-" * 68)
    for r in summary:
        name   = r.get("name", "")[:30]
        status = r.get("status", "")
        n      = r.get("n_windows", "")
        nc     = r.get("n_clusters", "")
        sil    = r.get("silhouette", "")
        t      = r.get("t_s", "")
        n_str  = f"{n:,}" if isinstance(n, int) else str(n)
        nc_str = str(nc) if nc != "" else "—"
        sil_str = f"{sil:.4f}" if isinstance(sil, float) else "—"
        t_str  = f"{t:.1f}" if isinstance(t, float) else "—"
        print(f"  {name:<30}  {status:<8}  {n_str:>7}  {nc_str:>8}  {sil_str:>6}  {t_str:>8}")


if __name__ == "__main__":
    main()
