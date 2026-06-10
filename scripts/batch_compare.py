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
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
LAB_DATA    = REPO_ROOT.parent / "lab_data"

sys.path.insert(0, str(SCRIPTS_DIR))

from clustering_pipeline import (
    load_cleaned_motion, process_motion,
    extract_histogram_features, ARENA_BIN_EDGES,
    cluster_hdbscan, cluster_ap_full, cluster_ap_sampled,
    cluster_ap_sparse_knn, cluster_ap_hierarchical,
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
        "mcs":     5,       # HDBSCAN min_cluster_size
        "ap_mcs":  15,      # AP preference auto
    },
    {
        "name":    "comparison_test",
        "raw_csv": LAB_DATA / "comparison_test" / "Harp-Motion2025-04-07T08_22_29.csv",
        "label":   "comparison_test",
        "arena":   "3d_wired",
        "mcs":     15,
    },
    {
        "name":    "control_mouse_1_jul",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-07-07T15_17_33.csv",
        "label":   "control_jul",
        "arena":   "3d_wired",
        "mcs":     15,
    },
    {
        "name":    "control_mouse_1_oct",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "control_mouse_1" / "Harp-Motion2025-10-13T15_04_32.csv",
        "label":   "control_oct",
        "arena":   "3d_wired",
        "mcs":     15,
    },
    {
        "name":    "mp_mouse_1_jul",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-07-08T08_36_38.csv",
        "label":   "mp_jul",
        "arena":   "3d_wired",
        "mcs":     15,
    },
    {
        "name":    "mp_mouse_1_oct",
        "raw_csv": LAB_DATA / "mito_park_progression_test" / "mp_mouse_1" / "Harp-Motion2025-10-13T07_36_52.csv",
        "label":   "mp_oct",
        "arena":   "3d_wired",
        "mcs":     15,
    },
    {
        "name":    "moving_test",
        "raw_csv": LAB_DATA / "moving_test" / "IMU" / "Harp-Motion2025-05-23T12_08_47.csv",
        "label":   "moving_test",
        "arena":   "3d_wired",
        "mcs":     15,
    },
    {
        "name":    "still_test",
        "raw_csv": LAB_DATA / "still_test" / "Harp-Motion2025-09-15T08_33_43.csv",
        "label":   "still_test",
        "arena":   "3d_wired",
        "mcs":     15,
    },
]

AP_SAMPLE_SIZE  = 6_000   # windows used for AP sampled
AP_FULL_MAX_N   = 20_000  # only run AP full when N <= this (~3×N²×8 bytes needed for AP internals)
AP_SPARSE_K     = 1_000   # K-NN graph degree for Sparse AP


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


# ── Load cached result ────────────────────────────────────────────────────────

def load_cached(out_path: Path, cdf: np.ndarray, N: int) -> dict | None:
    """Read existing result CSV and recompute metrics. Returns None if unreadable."""
    try:
        df = pd.read_csv(out_path)
        if len(df) != N:
            return None
        cluster_idx = df["ClusterIdx"].values.astype(int)
        labels = np.where(cluster_idx == 0, -1, cluster_idx - 1)
        stats  = cluster_stats(labels)
        sil    = _silhouette(cdf, labels)
        return {
            "status":    "ok",
            "n_windows": N,
            "n_clusters": stats["n_clusters"],
            "n_noise":    stats["n_noise"],
            "noise_pct":  round(100 * stats["n_noise"] / N, 2),
            "sil":        round(sil, 4) if not np.isnan(sil) else float("nan"),
            "mean":       round(stats["mean"], 1),
            "std":        round(stats["std"], 1),
            "min":        stats["min"],
            "max":        stats["max"],
            "t_dist":     float("nan"),
            "t_algo":     float("nan"),
            "t_s":        float("nan"),
            "preference": None,
            "labels":     labels,
            "cached":     True,
        }
    except Exception:
        return None


# ── Run one method ────────────────────────────────────────────────────────────

def run_method(method: str, hist, cdf, channel_sizes, mcs: int, out_path: Path,
               ts, folders, preference: float = None) -> dict:
    """Run one clustering method, save CSV, return metrics dict."""
    import io, contextlib
    buf = io.StringIO()
    N   = hist.shape[0]

    _timing = {}
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            if method == "hdbscan":
                labels = cluster_hdbscan(hist, channel_sizes, min_cluster_size=mcs,
                                         _timing=_timing)
            elif method == "ap_full":
                labels = cluster_ap_full(hist, channel_sizes, _timing=_timing)
            elif method == "ap_sampled":
                labels = cluster_ap_sampled(hist, channel_sizes,
                                             sample_size=AP_SAMPLE_SIZE, _timing=_timing,
                                             preference=preference)
            elif method == "ap_coreset":
                labels = cluster_ap_sampled(hist, channel_sizes,
                                             sample_size=AP_SAMPLE_SIZE, _timing=_timing,
                                             preference=preference, use_coreset=True)
            elif method == "ap_sparse":
                labels = cluster_ap_sparse_knn(hist, channel_sizes,
                                                K=AP_SPARSE_K, _timing=_timing)
            elif method == "ap_hierarchical":
                labels = cluster_ap_hierarchical(hist, channel_sizes, _timing=_timing)
            else:
                raise ValueError(f"Unknown method: {method}")

        elapsed = time.perf_counter() - t0
        sil     = _silhouette(cdf, labels)
        stats   = cluster_stats(labels)
        save_result_csv(labels, ts, folders, N, out_path)

        return {
            "method":     method,
            "status":     "ok",
            "n_windows":       N,
            "n_clusters":      stats["n_clusters"],
            "n_noise":         stats["n_noise"],
            "noise_pct":       round(100 * stats["n_noise"] / N, 2),
            "sil":             round(sil, 4) if not np.isnan(sil) else float("nan"),
            "mean":            round(stats["mean"], 1),
            "std":             round(stats["std"], 1),
            "min":             stats["min"],
            "max":             stats["max"],
            "t_dist":          _timing.get("t_dist", float("nan")),
            "t_algo":          _timing.get("t_algo", float("nan")),
            "t_s":             round(elapsed, 2),
            "preference":      _timing.get("preference", None),
            "labels":          labels,
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

def _format_result(m: dict) -> str:
    if m.get("status") != "ok":
        return f"{m['method']:<14} STATUS={m.get('status','?')}"
    sil = m["sil"]
    sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
    quality = ("good" if sil > 0.5 else
               "reasonable" if sil > 0.25 else "poor") if not np.isnan(sil) else "N/A"
    t_dist = m.get("t_dist", float("nan"))
    t_algo = m.get("t_algo", float("nan"))
    t_dist_str = f"{t_dist:.1f}s" if not np.isnan(t_dist) else "—"
    t_algo_str = f"{t_algo:.1f}s" if not np.isnan(t_algo) else "—"
    return (f"{m['method']:<14}  clusters={m['n_clusters']:>3}  "
            f"noise={m['noise_pct']:.1f}%  sil={sil_str} ({quality})  "
            f"dist={t_dist_str}  algo={t_algo_str}  total={m['t_s']:.1f}s")


def print_result(m: dict) -> None:
    print("    " + _format_result(m))


# ── RI / ARI vs AP full ───────────────────────────────────────────────────────

def _compute_rand_indices(dataset_results: list) -> None:
    """Compute RI and ARI vs ap_full for each method. Modifies dicts in-place."""
    from sklearn.metrics import adjusted_rand_score
    try:
        from sklearn.metrics import rand_score
    except ImportError:
        rand_score = None

    ref = next((m for m in dataset_results
                if m["method"] == "ap_full" and m.get("status") == "ok"), None)

    for m in dataset_results:
        if m["method"] == "ap_full" or m.get("status") != "ok" or ref is None:
            m.setdefault("ri", float("nan"))
            m.setdefault("ari", float("nan"))
            m.setdefault("ri_n_excluded", 0)
            continue

        ref_labels  = np.array(ref["labels"])
        pred_labels = np.array(m["labels"])

        valid      = pred_labels >= 0
        n_excluded = int((~valid).sum())

        if valid.sum() < 2 or len(set(pred_labels[valid])) < 2:
            m["ri"] = m["ari"] = float("nan")
            m["ri_n_excluded"] = n_excluded
            continue

        ref_v, pred_v = ref_labels[valid], pred_labels[valid]
        m["ari"]           = round(float(adjusted_rand_score(ref_v, pred_v)), 4)
        m["ri"]            = round(float(rand_score(ref_v, pred_v)), 4) if rand_score else float("nan")
        m["ri_n_excluded"] = n_excluded


# ── Summary report ────────────────────────────────────────────────────────────

def write_comparison_report(all_rows: list, out_path: Path) -> None:
    lines = []
    lines.append("=" * 85)
    lines.append("  Method Comparison Report")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 85)
    lines.append(f"  AP preference: min(similarity)  [matches Matlab AccelCluster, min(s(:,3))]")
    lines.append(f"  Silhouette: >0.5 good | 0.25–0.5 reasonable | <0.25 poor")
    lines.append(f"  ARI reference: ap_full")

    datasets = sorted(set(r["dataset"] for r in all_rows))
    display_order = ["hdbscan", "ap_full", "ap_sampled", "ap_coreset", "ap_sparse", "ap_hierarchical"]

    for ds in datasets:
        lines.append("")
        lines.append("─" * 85)
        ds_rows = {r["method"]: r for r in all_rows if r["dataset"] == ds}
        n_windows = next((r["n_windows"] for r in ds_rows.values() if r.get("n_windows")), "?")
        n_str = f"{n_windows:,}" if isinstance(n_windows, int) else str(n_windows)
        lines.append(f"  {ds}  (N = {n_str} windows)")
        lines.append("─" * 85)

        cols  = ["Method", "Clusters", "Noise%", "Silhouette", "Quality", "ARI vs ap_full", "Total(s)"]
        col_w = [14, 10, 8, 12, 12, 16, 9]
        lines.append("  " + "".join(c.ljust(w) for c, w in zip(cols, col_w)))
        lines.append("  " + "-" * sum(col_w))

        for key in display_order:
            m = ds_rows.get(key, {"method": key, "status": "not_run"})
            if m.get("status") == "skipped":
                row = [key, "—", "—", "—", "skipped", "—", "—"]
            elif m.get("status") != "ok":
                row = [key, "—", "—", "—", m.get("status", "—"), "—", "—"]
            else:
                sil = m.get("sil", float("nan"))
                sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
                quality = ("good" if sil > 0.5 else
                           "reasonable" if sil > 0.25 else
                           "poor") if not np.isnan(sil) else "N/A"
                ari = m.get("ari", float("nan"))
                ari_str = f"{ari:.4f}" if not np.isnan(ari) else "—"
                row = [key,
                       str(m["n_clusters"]),
                       f"{m['noise_pct']:.1f}%",
                       sil_str, quality, ari_str,
                       str(m["t_s"])]
            lines.append("  " + "".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    lines.append("")
    lines.append("=" * 85)
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", nargs="*", metavar="METHOD",
                        help="Force re-run. No args = re-run all. "
                             "Specific methods: --force hdbscan ap_sampled")
    args = parser.parse_args()

    if args.force is None:
        force_methods = set()          # cache mode: skip nothing
    elif len(args.force) == 0:
        force_methods = None           # None = force everything
    else:
        force_methods = set(args.force)

    print("=" * 62)
    print("  Method Comparison: HDBSCAN | AP full | AP sampled | AP sparse")
    print("=" * 62)
    if force_methods is None:
        print("  Cache: DISABLED (--force, re-running all methods)")
    elif force_methods:
        print(f"  Cache: force re-run for: {', '.join(sorted(force_methods))}")
    else:
        print("  Cache: ENABLED (use --force to re-run)")

    all_rows = []
    methods  = ["hdbscan", "ap_full", "ap_sampled", "ap_coreset", "ap_sparse", "ap_hierarchical"]

    dataset_pbar = tqdm(DATASETS, desc="Datasets", unit="dataset", position=0)
    for cfg in dataset_pbar:
        dataset_pbar.set_postfix_str(cfg["name"])
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

        dataset_results = []
        ap_full_preference = None

        ap_pbar = tqdm(methods, desc="  Methods", unit="method",
                       position=1, leave=False)
        for method in ap_pbar:
            ap_pbar.set_postfix_str(method)
            out_csv = out_dir / f"Cluster_detail_results_{method}.csv"

            if method == "ap_full" and N > AP_FULL_MAX_N:
                tqdm.write(f"\n  → {method}  SKIPPED  "
                           f"(N={N:,} > AP_FULL_MAX_N={AP_FULL_MAX_N:,}, "
                           f"would need {N**2*8/1e9:.1f} GB)")
                dataset_results.append({
                    "dataset": cfg["name"], "method": method,
                    "status": "skipped", "n_windows": N,
                    "note": f"N={N:,} exceeds AP_FULL_MAX_N={AP_FULL_MAX_N:,}",
                })
                continue

            use_cache = (
                force_methods is not None           # not --force (all)
                and method not in (force_methods or set())  # not in forced list
                and out_csv.exists()
            )

            if use_cache:
                cached = load_cached(out_csv, cdf, N)
                if cached is not None:
                    cached["method"]  = method
                    cached["dataset"] = cfg["name"]
                    if method == "ap_full":
                        ap_full_preference = cached.get("preference")
                    dataset_results.append(cached)
                    tqdm.write(f"\n  → {method} ... (cached)")
                    tqdm.write("    " + _format_result(cached))
                    continue

            tqdm.write(f"\n  → {method} ...")
            pref = ap_full_preference if (method == "ap_sampled" and ap_full_preference is not None) else None
            if pref is not None:
                tqdm.write(f"      (preference aligned to AP full: {pref:.4f})")
            m = run_method(method, hist, cdf, channel_sizes,
                           cfg["mcs"], out_csv, ts, folders, preference=pref)
            m["dataset"] = cfg["name"]
            if method == "ap_full" and m.get("status") == "ok":
                ap_full_preference = m.get("preference")
            dataset_results.append(m)
            tqdm.write("    " + _format_result(m))

        _compute_rand_indices(dataset_results)
        if any(m["method"] == "ap_full" and m.get("status") == "ok"
               for m in dataset_results):
            print("\n  Agreement with AP full (Matlab reference):")
            for m in dataset_results:
                if m["method"] == "ap_full" or m.get("status") != "ok":
                    continue
                ri   = m.get("ri",  float("nan"))
                ari  = m.get("ari", float("nan"))
                excl = m.get("ri_n_excluded", 0)
                ri_str  = f"{ri:.4f}"  if not np.isnan(ri)  else "N/A"
                ari_str = f"{ari:.4f}" if not np.isnan(ari) else "N/A"
                note = f"  (excl. {excl:,} noise pts)" if excl > 0 else ""
                print(f"    {m['method']:<14}  RI={ri_str}  ARI={ari_str}{note}")

        for m in dataset_results:
            m.pop("labels", None)
        all_rows.extend(dataset_results)

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
            "t_dist_s":   r.get("t_dist", ""),
            "t_algo_s":   r.get("t_algo", ""),
            "t_s":        r.get("t_s", ""),
            "ri_vs_ap_full":  r.get("ri",  ""),
            "ari_vs_ap_full": r.get("ari", ""),
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
