"""
batch_compare.py
----------------
Runs all three clustering methods on every lab dataset and saves a
side-by-side comparison.

Methods compared
----------------
  1. HDBSCAN       — UMAP + HDBSCAN (current default, scalable)
  2. AP full       — Full N×N affinity matrix + AP (exact, but memory-heavy)
  3. AP sampled    — AP on 6,000-window subset, then assign all via FAISS

Results are saved to:
  results/<dataset>/Cluster_detail_results_hdbscan.csv
  results/<dataset>/Cluster_detail_results_ap_full.csv
  results/<dataset>/Cluster_detail_results_ap_sampled.csv
  results/method_comparison.csv      (machine-readable summary)
  results/method_comparison_report.txt  (human-readable report)

Usage (from repo root):
    python3 scripts/batch_compare.py
"""

import sys
import time
import traceback
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
    cluster_hdbscan, cluster_ap_sparse, cluster_ap_sampled,
    _build_cdf_features, _silhouette, WIN_SIZE,
)
from prepare_test_data import convert as prepare_csv


# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = [
    {
        "name":    "short_comparison_test",
        "raw_csv": LAB_DATA / "short_comparison_test" / "harp_data_cut.csv",
        "label":   "short_test",
        "arena":   "3d_wired",
        "mcs":     15,      # HDBSCAN min_cluster_size
        "ap_mcs":  15,      # AP preference auto
    },
    {
        "name":    "comparison_test",
        "raw_csv": LAB_DATA / "comparison_test" / "Harp-Motion2025-04-07T08_22_29.csv",
        "label":   "comparison_test",
        "arena":   "3d_wired",
        "mcs":     50,
    },
    {
        "name":    "control_mouse_1_jul",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-07-07T15_17_33.csv",
        "label":   "control_jul",
        "arena":   "3d_wired",
        "mcs":     80,
    },
    {
        "name":    "control_mouse_1_oct",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-10-13T15_04_32.csv",
        "label":   "control_oct",
        "arena":   "3d_wired",
        "mcs":     80,
    },
    {
        "name":    "mp_mouse_1_jul",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-07-08T08_36_38.csv",
        "label":   "mp_jul",
        "arena":   "3d_wired",
        "mcs":     70,
    },
    {
        "name":    "mp_mouse_1_oct",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-10-13T07_36_52.csv",
        "label":   "mp_oct",
        "arena":   "3d_wired",
        "mcs":     80,
    },
    {
        "name":    "moving_test",
        "raw_csv": LAB_DATA / "moving_test" / "IMU" / "Harp-Motion2025-05-23T12_08_47.csv",
        "label":   "moving_test",
        "arena":   "3d_wired",
        "mcs":     150,
    },
    {
        "name":    "still_test",
        "raw_csv": LAB_DATA / "still_test" / "Harp-Motion2025-09-15T08_33_43.csv",
        "label":   "still_test",
        "arena":   "3d_wired",
        "mcs":     80,
    },
]

AP_SAMPLE_SIZE = 6_000   # windows used for AP sampled
AP_FULL_MAX_N  = 15_000  # only run AP full when N <= this (memory/time limit)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(title: str, width: int = 62) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print('─' * width)


def cluster_stats(labels: np.ndarray) -> dict:
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return {"n_clusters": 0, "n_noise": len(labels),
                "mean": 0, "std": 0, "min": 0, "max": 0}
    unique, counts = np.unique(valid, return_counts=True)
    return {
        "n_clusters": len(unique),
        "n_noise":    int((labels == -1).sum()),
        "mean":       float(counts.mean()),
        "std":        float(counts.std()),
        "min":        int(counts.min()),
        "max":        int(counts.max()),
    }


def save_result_csv(labels, timestamps, folder_names, n_windows, out_path):
    mid         = WIN_SIZE // 2
    win_ts      = timestamps[mid::WIN_SIZE][:n_windows]
    win_folders = folder_names[::WIN_SIZE][:n_windows]
    cluster_idx = np.where(labels < 0, 0, labels + 1)
    pd.DataFrame({
        "ClusterIdx":  cluster_idx,
        "Timestamp":   win_ts,
        "Folder_Name": win_folders,
    }).to_csv(str(out_path), index=False)


# ── Load or prepare data ──────────────────────────────────────────────────────

def load_data(cfg: dict, out_dir: Path) -> tuple:
    """Prepare CSV if needed, then load into feature matrix. Returns (hist, ts, folders, cdf, channel_sizes)."""
    clean_csv = out_dir / "combined_harp_data_cleaned.csv"
    if not clean_csv.exists():
        print(f"  Preparing data ...")
        prepare_csv(str(cfg["raw_csv"]), str(clean_csv), cfg["label"])

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        raw, ts, folders = load_cleaned_motion(str(clean_csv))
        sensor = process_motion(raw)
        edges  = ARENA_BIN_EDGES[cfg["arena"]]

    channel_sizes = [len(e) - 1 for e in edges]
    with contextlib.redirect_stdout(buf):
        hist = extract_histogram_features(sensor, edges)

    cdf = _build_cdf_features(hist, channel_sizes)
    return hist, ts, folders, cdf, channel_sizes


# ── Run one method ────────────────────────────────────────────────────────────

def run_method(method: str, hist, cdf, channel_sizes, mcs: int, out_path: Path,
               ts, folders) -> dict:
    """Run one clustering method, save CSV, return metrics dict."""
    import io, contextlib
    buf = io.StringIO()
    N   = hist.shape[0]

    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            if method == "hdbscan":
                labels = cluster_hdbscan(hist, channel_sizes, min_cluster_size=mcs)
            elif method == "ap_full":
                labels = cluster_ap_sparse(hist, channel_sizes)
            elif method == "ap_sampled":
                labels = cluster_ap_sampled(hist, channel_sizes,
                                             sample_size=AP_SAMPLE_SIZE)
            else:
                raise ValueError(f"Unknown method: {method}")

        elapsed = time.perf_counter() - t0
        sil     = _silhouette(cdf, labels)
        stats   = cluster_stats(labels)
        save_result_csv(labels, ts, folders, N, out_path)

        return {
            "method":     method,
            "status":     "ok",
            "n_windows":  N,
            "n_clusters": stats["n_clusters"],
            "n_noise":    stats["n_noise"],
            "noise_pct":  round(100 * stats["n_noise"] / N, 2),
            "sil":        round(sil, 4) if not np.isnan(sil) else float("nan"),
            "mean":       round(stats["mean"], 1),
            "std":        round(stats["std"], 1),
            "min":        stats["min"],
            "max":        stats["max"],
            "t_s":        round(elapsed, 2),
        }
    except MemoryError:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ MemoryError after {elapsed:.1f}s")
        return {"method": method, "status": "oom",
                "n_windows": N, "t_s": round(elapsed, 2)}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  ✗ Error: {e}")
        traceback.print_exc()
        return {"method": method, "status": "error", "error": str(e),
                "n_windows": N, "t_s": round(elapsed, 2)}


# ── Print result row ──────────────────────────────────────────────────────────

def print_result(m: dict) -> None:
    if m["status"] != "ok":
        print(f"    {m['method']:<14} STATUS={m['status']}")
        return
    sil = m["sil"]
    sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
    quality = ("good" if sil > 0.5 else
               "reasonable" if sil > 0.25 else "poor") if not np.isnan(sil) else "N/A"
    print(f"    {m['method']:<14}  clusters={m['n_clusters']:>3}  "
          f"noise={m['noise_pct']:.1f}%  "
          f"sil={sil_str} ({quality})  "
          f"time={m['t_s']:.1f}s")


# ── Summary report ────────────────────────────────────────────────────────────

def write_comparison_report(all_rows: list, out_path: Path) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("  Method Comparison Report")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 72)
    lines.append(f"  Methods: HDBSCAN  |  AP full  |  AP sampled (n={AP_SAMPLE_SIZE:,})")
    lines.append(f"  Silhouette: >0.5 good | 0.25–0.5 reasonable | <0.25 poor")

    datasets = sorted(set(r["dataset"] for r in all_rows))
    methods  = ["hdbscan", "ap_full", "ap_sampled"]

    for ds in datasets:
        lines.append("")
        lines.append("─" * 72)
        lines.append(f"  {ds}")
        lines.append("─" * 72)
        rows = {r["method"]: r for r in all_rows if r["dataset"] == ds}

        cols  = ["Method", "Clusters", "Noise%", "Silhouette", "Quality", "Time(s)"]
        col_w = [16, 10, 8, 12, 12, 9]
        lines.append("  " + "".join(c.ljust(w) for c, w in zip(cols, col_w)))
        lines.append("  " + "-" * sum(col_w))

        for method in methods:
            m = rows.get(method, {"method": method, "status": "not_run"})
            if m.get("status") == "skipped":
                row = [method, "—", "—", "—", "skipped (N too large)", "—"]
            elif m.get("status") != "ok":
                row = [method, "—", "—", "—", m.get("status","—"), "—"]
            else:
                sil = m.get("sil", float("nan"))
                sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
                quality = ("good" if sil > 0.5 else
                           "reasonable" if sil > 0.25 else
                           "poor") if not np.isnan(sil) else "N/A"
                row = [method,
                       str(m["n_clusters"]),
                       f"{m['noise_pct']:.1f}%",
                       sil_str, quality,
                       str(m["t_s"])]
            lines.append("  " + "".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    lines.append("")
    lines.append("=" * 72)
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Method Comparison: HDBSCAN vs AP full vs AP sampled")
    print("=" * 62)

    all_rows = []
    methods  = ["hdbscan", "ap_full", "ap_sampled"]

    for cfg in DATASETS:
        hr(cfg["name"])
        out_dir = RESULTS_DIR / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        print("  Loading / preparing data ...")
        try:
            hist, ts, folders, cdf, channel_sizes = load_data(cfg, out_dir)
            N = hist.shape[0]
            print(f"  Windows: {N:,}")
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        for method in methods:
            out_csv = out_dir / f"Cluster_detail_results_{method}.csv"

            # Skip AP full for large datasets
            if method == "ap_full" and N > AP_FULL_MAX_N:
                print(f"\n  → {method}  SKIPPED  "
                      f"(N={N:,} > AP_FULL_MAX_N={AP_FULL_MAX_N:,}, "
                      f"would need {N**2*8/1e9:.1f} GB)")
                all_rows.append({
                    "dataset": cfg["name"], "method": method,
                    "status": "skipped", "n_windows": N,
                    "note": f"N={N:,} exceeds AP_FULL_MAX_N={AP_FULL_MAX_N:,}",
                })
                continue

            print(f"\n  → {method} ...")
            m = run_method(method, hist, cdf, channel_sizes,
                           cfg["mcs"], out_csv, ts, folders)
            m["dataset"] = cfg["name"]
            all_rows.append(m)
            print_result(m)

    # ── Save comparison CSV ───────────────────────────────────────────────────
    csv_rows = []
    for r in all_rows:
        csv_rows.append({
            "dataset":    r["dataset"],
            "method":     r["method"],
            "status":     r.get("status", ""),
            "n_windows":  r.get("n_windows", ""),
            "n_clusters": r.get("n_clusters", ""),
            "noise_pct":  r.get("noise_pct", ""),
            "silhouette": r.get("sil", ""),
            "cluster_mean": r.get("mean", ""),
            "cluster_min":  r.get("min", ""),
            "cluster_max":  r.get("max", ""),
            "t_s":        r.get("t_s", ""),
        })
    pd.DataFrame(csv_rows).to_csv(
        RESULTS_DIR / "method_comparison.csv", index=False)

    # ── Write report ──────────────────────────────────────────────────────────
    hr("Full Comparison Report")
    write_comparison_report(all_rows, RESULTS_DIR / "method_comparison_report.txt")

    print(f"\n  Saved: {RESULTS_DIR / 'method_comparison.csv'}")
    print(f"  Saved: {RESULTS_DIR / 'method_comparison_report.txt'}")


if __name__ == "__main__":
    main()
