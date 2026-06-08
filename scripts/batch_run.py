"""
batch_run.py
------------
Runs the clustering pipeline on all available lab datasets and generates
a comprehensive summary report with timing, cluster statistics, and
quality metrics.

Usage (from repo root):
    python3 scripts/batch_run.py

Results are saved to:
    results/<dataset_name>/Cluster_detail_results.csv
    results/summary_report.txt
    results/summary_metrics.csv
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
LAB_DATA    = REPO_ROOT.parent / "lab_data"

sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, ARENA_BIN_EDGES,
    cluster_ap_full, cluster_hdbscan,
    _build_cdf_features, _silhouette,
    WIN_SIZE,
)
from prepare_test_data import convert as prepare_csv, _LOSS_CUTOFF


# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = [
    {
        "name":       "short_comparison_test",
        "raw_csv":    LAB_DATA / "short_comparison_test" / "harp_data_cut.csv",
        "label":      "short_test",
        "arena":      "3d_wired",
        "use_ap":     True,
        "preference": None,
        "mcs":        15,
        "note":       "6-min test clip. AP used (N < 10,000).",
    },
    {
        "name":       "comparison_test",
        "raw_csv":    LAB_DATA / "comparison_test" / "Harp-Motion2025-04-07T08_22_29.csv",
        "label":      "comparison_test",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        50,
        "note":       "~69-min 2D/3D arena comparison recording.",
    },
    {
        "name":       "control_mouse_1_jul",
        "raw_csv":    LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-07-07T15_17_33.csv",
        "label":      "control_jul",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        80,
        "note":       "Control mouse, session 1 (July 2025).",
    },
    {
        "name":       "control_mouse_1_oct",
        "raw_csv":    LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-10-13T15_04_32.csv",
        "label":      "control_oct",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        80,
        "note":       "Control mouse, session 2 (October 2025, 3 months later).",
    },
    {
        "name":       "mp_mouse_1_jul",
        "raw_csv":    LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-07-08T08_36_38.csv",
        "label":      "mp_jul",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        70,
        "note":       "Mito Park mouse (Parkinson model), session 1 (July 2025).",
    },
    {
        "name":       "mp_mouse_1_oct",
        "raw_csv":    LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-10-13T07_36_52.csv",
        "label":      "mp_oct",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        80,
        "note":       "Mito Park mouse (Parkinson model), session 2 (October 2025).",
    },
    {
        "name":       "moving_test",
        "raw_csv":    LAB_DATA / "moving_test" / "IMU" / "Harp-Motion2025-05-23T12_08_47.csv",
        "label":      "moving_test",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        150,
        "note":       "~238-min baseline recording: mouse in motion.",
    },
    {
        "name":       "still_test",
        "raw_csv":    LAB_DATA / "still_test" / "Harp-Motion2025-09-15T08_33_43.csv",
        "label":      "still_test",
        "arena":      "3d_wired",
        "use_ap":     False,
        "mcs":        80,
        "note":       "~120-min baseline recording: mouse relatively still.",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(title: str, width: int = 62) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print('─' * width)


def cluster_stats(labels: np.ndarray) -> dict:
    valid  = labels[labels >= 0]
    unique, counts = np.unique(valid, return_counts=True)
    if len(counts) == 0:
        return {"n_clusters": 0, "n_noise": int((labels==-1).sum()),
                "mean": 0, "std": 0, "min": 0, "max": 0, "top3": []}
    order = np.argsort(-counts)
    top3  = [(int(unique[i]), int(counts[i])) for i in order[:3]]
    return {
        "n_clusters": len(unique),
        "n_noise":    int((labels == -1).sum()),
        "mean":       float(counts.mean()),
        "std":        float(counts.std()),
        "min":        int(counts.min()),
        "max":        int(counts.max()),
        "top3":       top3,
    }


def run_one(cfg: dict, out_dir: Path) -> dict:
    """Run pipeline on one dataset, return metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_csv  = out_dir / "combined_harp_data_cleaned.csv"
    result_csv = out_dir / "Cluster_detail_results.csv"

    metrics = {
        "name":      cfg["name"],
        "algorithm": "AP" if cfg["use_ap"] else "HDBSCAN",
        "note":      cfg.get("note", ""),
        "status":    "ok",
        "error":     "",
    }

    t_total = time.perf_counter()

    # ── Step 0: Prepare data ──────────────────────────────────────────────────
    print(f"  [0/5] Preparing data (detecting packet loss, cleaning windows) ...")
    t0 = time.perf_counter()
    import io, contextlib
    buf = io.StringIO()
    prep_stats = prepare_csv(str(cfg["raw_csv"]), str(clean_csv), cfg["label"])
    metrics["t_prepare_s"]      = round(time.perf_counter() - t0, 2)
    metrics["n_missing_packets"] = prep_stats.get("n_inserted", 0)
    metrics["loss_pct"]          = prep_stats.get("loss_pct", 0.0)
    metrics["n_discarded_wins"]  = prep_stats.get("n_discarded_wins", 0)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print(f"  [1/5] Loading ...")
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        raw_motion, timestamps, folder_names = load_cleaned_motion(str(clean_csv))
    metrics["t_load_s"]    = round(time.perf_counter() - t0, 2)
    metrics["n_samples"]   = len(raw_motion)
    metrics["duration_min"]= round(len(raw_motion) / 200 / 60, 1)

    # ── Step 2: Signal processing ─────────────────────────────────────────────
    print(f"  [2/5] Signal processing ...")
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        sensor = process_motion(raw_motion)
    metrics["t_signal_s"] = round(time.perf_counter() - t0, 2)

    # ── Step 3: Feature extraction ────────────────────────────────────────────
    print(f"  [3/5] Feature extraction ...")
    t0 = time.perf_counter()
    bin_edges     = ARENA_BIN_EDGES[cfg["arena"]]
    channel_sizes = [len(e) - 1 for e in bin_edges]
    with contextlib.redirect_stdout(buf):
        hist_matrix = extract_histogram_features(sensor, bin_edges)
    metrics["t_features_s"] = round(time.perf_counter() - t0, 2)
    metrics["n_windows"]    = hist_matrix.shape[0]

    # ── Step 4: Clustering ────────────────────────────────────────────────────
    print(f"  [4/5] Clustering ({metrics['algorithm']}, "
          f"N={metrics['n_windows']:,}) ...")
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        if cfg["use_ap"]:
            labels = cluster_ap_full(hist_matrix, channel_sizes,
                                       preference=cfg.get("preference"))
        else:
            labels = cluster_hdbscan(hist_matrix, channel_sizes,
                                     min_cluster_size=cfg["mcs"])
    metrics["t_cluster_s"] = round(time.perf_counter() - t0, 2)

    # ── Step 5: Silhouette Score ──────────────────────────────────────────────
    print(f"  [5/5] Quality metrics ...")
    cdf_feats = _build_cdf_features(hist_matrix, channel_sizes)
    sil = _silhouette(cdf_feats, labels)
    metrics["silhouette"]  = round(sil, 4) if not np.isnan(sil) else float("nan")

    stats = cluster_stats(labels)
    metrics.update(stats)

    # ── Save results ──────────────────────────────────────────────────────────
    mid         = WIN_SIZE // 2
    N_windows   = hist_matrix.shape[0]
    win_ts      = timestamps[mid::WIN_SIZE][:N_windows]
    win_folders = folder_names[::WIN_SIZE][:N_windows]
    cluster_idx = np.where(labels < 0, 0, labels + 1)

    pd.DataFrame({
        "ClusterIdx":  cluster_idx,
        "Timestamp":   win_ts,
        "Folder_Name": win_folders,
    }).to_csv(str(result_csv), index=False)

    metrics["t_total_s"] = round(time.perf_counter() - t_total, 2)
    metrics["result_csv"] = str(result_csv)
    return metrics


def print_dataset_report(m: dict) -> None:
    sil = m.get("silhouette", float("nan"))
    sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
    quality = ("good" if sil > 0.5 else
               "reasonable" if sil > 0.25 else
               "poor") if not np.isnan(sil) else "N/A"

    print(f"  Dataset          : {m['name']}")
    print(f"  Note             : {m['note']}")
    print(f"  Algorithm        : {m['algorithm']}"
          + (f"  (min_cluster_size not tracked for AP)" if m['algorithm']=='AP' else ""))
    print(f"  Duration         : {m.get('duration_min', '?')} min")
    print(f"  ── Data Quality ──")
    print(f"  Missing packets  : {m.get('n_missing_packets', 0):,}  "
          f"({m.get('loss_pct', 0.0):.2f}% packet loss)")
    print(f"  Discarded windows: {m.get('n_discarded_wins', 0):,}  "
          f"(>{_LOSS_CUTOFF}% NaN)")
    print(f"  Windows          : {m.get('n_windows', 0):,}")
    print(f"  Clusters found   : {m.get('n_clusters', 0)}")
    print(f"  Noise points     : {m.get('n_noise', 0):,}  "
          f"({100*m.get('n_noise',0)/max(m.get('n_windows',1),1):.1f}%)")
    print(f"  Cluster sizes    : mean={m.get('mean',0):.0f}  "
          f"std={m.get('std',0):.0f}  "
          f"min={m.get('min',0)}  max={m.get('max',0)}")
    print(f"  Top-3 clusters   : {m.get('top3', [])}")
    print(f"  Silhouette Score : {sil_str}  ({quality})")
    print(f"\n  ── Timing ──")
    print(f"  Prepare data     : {m.get('t_prepare_s','?')}s")
    print(f"  Load             : {m.get('t_load_s','?')}s")
    print(f"  Signal processing: {m.get('t_signal_s','?')}s")
    print(f"  Feature extraction: {m.get('t_features_s','?')}s")
    print(f"  Clustering       : {m.get('t_cluster_s','?')}s")
    print(f"  Total            : {m.get('t_total_s','?')}s  "
          f"({m.get('t_total_s',0)/60:.1f} min)")


def write_summary(all_metrics: list, out_path: Path) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("  Clustering Pipeline — Batch Run Summary")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # Summary table
    cols = ["Dataset", "Windows", "Algo", "Clusters", "Noise%",
            "Loss%", "Silhouette", "Quality", "Total(s)"]
    col_w = [28, 9, 8, 9, 7, 7, 11, 11, 9]

    lines.append("")
    lines.append("  " + "".join(c.ljust(w) for c, w in zip(cols, col_w)))
    lines.append("  " + "-" * sum(col_w))

    for m in all_metrics:
        if m["status"] != "ok":
            row = [m["name"], "—", "—", "—", "—", "—", "—", "ERROR", "—"]
        else:
            sil = m.get("silhouette", float("nan"))
            sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
            quality = ("good" if sil > 0.5 else
                       "reasonable" if sil > 0.25 else
                       "poor") if not np.isnan(sil) else "N/A"
            noise_pct = f"{100*m.get('n_noise',0)/max(m.get('n_windows',1),1):.1f}%"
            loss_pct  = f"{m.get('loss_pct', 0.0):.2f}%"
            row = [
                m["name"][:27],
                f"{m.get('n_windows',0):,}",
                m["algorithm"],
                str(m.get("n_clusters", 0)),
                noise_pct,
                loss_pct,
                sil_str,
                quality,
                f"{m.get('t_total_s',0):.1f}",
            ]
        lines.append("  " + "".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    lines.append("")
    lines.append("  Note: Silhouette Score — >0.5 good | 0.25–0.5 reasonable | <0.25 poor")

    # Detailed timing table
    lines.append("")
    lines.append("─" * 70)
    lines.append("  Timing breakdown (seconds)")
    lines.append("─" * 70)
    tcols = ["Dataset", "Prepare", "Load", "Signal", "Features", "Cluster", "Total"]
    tcol_w = [28, 9, 7, 8, 10, 9, 7]
    lines.append("  " + "".join(c.ljust(w) for c, w in zip(tcols, tcol_w)))
    lines.append("  " + "-" * sum(tcol_w))
    for m in all_metrics:
        if m["status"] != "ok":
            continue
        row = [
            m["name"][:27],
            str(m.get("t_prepare_s", "?")),
            str(m.get("t_load_s", "?")),
            str(m.get("t_signal_s", "?")),
            str(m.get("t_features_s", "?")),
            str(m.get("t_cluster_s", "?")),
            str(m.get("t_total_s", "?")),
        ]
        lines.append("  " + "".join(str(v).ljust(w) for v, w in zip(row, tcol_w)))

    # Per-dataset details
    for m in all_metrics:
        lines.append("")
        lines.append("─" * 70)
        lines.append(f"  {m['name']}")
        lines.append("─" * 70)
        if m["status"] != "ok":
            lines.append(f"  ERROR: {m['error']}")
            continue
        sil = m.get("silhouette", float("nan"))
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        quality = ("good" if sil > 0.5 else
                   "reasonable" if sil > 0.25 else
                   "poor") if not np.isnan(sil) else "N/A"
        lines += [
            f"  Note             : {m['note']}",
            f"  Algorithm        : {m['algorithm']}",
            f"  Duration         : {m.get('duration_min','?')} min",
            f"  Missing packets  : {m.get('n_missing_packets',0):,}  ({m.get('loss_pct',0.0):.2f}% packet loss)",
            f"  Discarded windows: {m.get('n_discarded_wins',0):,}",
            f"  Windows          : {m.get('n_windows',0):,}",
            f"  Clusters found   : {m.get('n_clusters',0)}",
            f"  Noise points     : {m.get('n_noise',0):,}  "
            f"({100*m.get('n_noise',0)/max(m.get('n_windows',1),1):.1f}%)",
            f"  Cluster sizes    : mean={m.get('mean',0):.0f}  "
            f"std={m.get('std',0):.0f}  "
            f"min={m.get('min',0)}  max={m.get('max',0)}",
            f"  Top-3 clusters   : {m.get('top3',[])}",
            f"  Silhouette Score : {sil_str}  ({quality})",
            f"  Timing (s)       : prepare={m.get('t_prepare_s','?')}  "
            f"load={m.get('t_load_s','?')}  "
            f"signal={m.get('t_signal_s','?')}  "
            f"features={m.get('t_features_s','?')}  "
            f"cluster={m.get('t_cluster_s','?')}  "
            f"total={m.get('t_total_s','?')}",
            f"  Result CSV       : {m.get('result_csv','')}",
        ]

    lines.append("")
    lines.append("=" * 70)
    report = "\n".join(lines)
    out_path.write_text(report)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Clustering Pipeline — Batch Run")
    print("=" * 62)
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Datasets to process: {len(DATASETS)}")

    all_metrics = []

    pbar = tqdm(DATASETS, desc="Datasets", unit="dataset")
    for cfg in pbar:
        pbar.set_postfix_str(cfg["name"])
        hr(f"[{DATASETS.index(cfg)+1}/{len(DATASETS)}]  {cfg['name']}")
        tqdm.write(f"  {cfg['note']}")
        out_dir = RESULTS_DIR / cfg["name"]

        try:
            m = run_one(cfg, out_dir)
            all_metrics.append(m)
            print_dataset_report(m)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            traceback.print_exc()
            all_metrics.append({
                "name": cfg["name"], "status": "error",
                "error": str(e), "note": cfg.get("note", ""),
                "algorithm": "AP" if cfg.get("use_ap") else "HDBSCAN",
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    hr("Summary")
    report = write_summary(all_metrics, RESULTS_DIR / "summary_report.txt")

    # Save machine-readable CSV
    rows = []
    for m in all_metrics:
        rows.append({
            "dataset":      m["name"],
            "status":       m.get("status", "error"),
            "algorithm":    m.get("algorithm", ""),
            "duration_min": m.get("duration_min", ""),
            "n_windows":    m.get("n_windows", ""),
            "n_clusters":   m.get("n_clusters", ""),
            "n_noise":      m.get("n_noise", ""),
            "noise_pct":    round(100*m.get("n_noise",0)/max(m.get("n_windows",1),1), 2),
            "silhouette":   m.get("silhouette", ""),
            "cluster_mean": round(m.get("mean", 0), 1),
            "cluster_std":  round(m.get("std", 0), 1),
            "cluster_min":  m.get("min", ""),
            "cluster_max":  m.get("max", ""),
            "t_prepare_s":  m.get("t_prepare_s", ""),
            "t_load_s":     m.get("t_load_s", ""),
            "t_signal_s":   m.get("t_signal_s", ""),
            "t_features_s": m.get("t_features_s", ""),
            "t_cluster_s":  m.get("t_cluster_s", ""),
            "t_total_s":    m.get("t_total_s", ""),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    print(report)
    print(f"\n  Saved to: {RESULTS_DIR / 'summary_report.txt'}")
    print(f"  Saved to: {RESULTS_DIR / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
