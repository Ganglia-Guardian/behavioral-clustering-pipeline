# Python Clustering Pipeline

Python implementation of the Matlab behavioral clustering pipeline (`VPAPPAxes.m`).  
Processes IMU sensor data from mice, segments movement into 300 ms windows, and clusters them into behavioral motifs.

Output format matches the original Matlab pipeline — all existing visualization notebooks still work.

---

## Folder Structure

```
Python_Pipeline/
├── scripts/
│   ├── clustering_pipeline.py   # Main pipeline
│   ├── batch_run.py             # Run all datasets, generate summary report
│   ├── batch_compare.py         # Compare all methods side by side
│   ├── prepare_test_data.py     # Convert raw Harp CSV to pipeline-ready format
│   └── compare_results.py       # Compare Python output against Matlab baseline
├── test_outputs/
│   ├── combined_harp_data_cleaned.csv   # Preprocessed input
│   ├── inspect_stage1_raw.csv           # Raw sensor values
│   ├── inspect_stage2_processed.csv     # Filtered signals (y_GA, z, z_gyro, tot_accel)
│   └── inspect_stage3_features.csv      # Histogram feature matrix (N × 30)
└── results/
    ├── method_comparison.csv            # Side-by-side method comparison
    ├── method_comparison_report.txt     # Human-readable comparison report
    └── <dataset_name>/
        ├── Cluster_detail_results_hdbscan.csv
        ├── Cluster_detail_results_ap_full.csv
        ├── Cluster_detail_results_ap_sampled.csv
        └── Cluster_detail_results_ap_sparse.csv
```

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below assume you are in `Python_Pipeline/` and use `.venv/bin/python3`.

---

## How to Run

### Prepare raw data

```bash
.venv/bin/python3 scripts/prepare_test_data.py \
    --input  path/to/Harp-Motion.csv \
    --output path/to/combined_harp_data_cleaned.csv \
    --label  session_name
```

Skip this if you already have `combined_harp_data_cleaned.csv`.

### Run clustering

```bash
# HDBSCAN (default)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv

# AP full (closest to Matlab; N ≤ 20,000 only)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap

# AP sampled
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled

# Sparse AP
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sparse
```

**Method guide:**

| Method | Flag | Notes |
|---|---|---|
| HDBSCAN | *(default)* | Fastest; handles noise points |
| AP full | `--use-ap` | Closest to Matlab; N ≤ 20,000 only |
| AP sampled | `--use-ap-sampled` | Best AP-compatible option for large N |
| Sparse AP | `--use-ap-sparse` | See [Known Limitations](#known-limitations) |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN — smaller → more clusters |
| `--sample-size` | `6000` | AP sampled — windows passed to AP |
| `--sparse-k` | `100` | Sparse AP — K-NN graph degree |
| `--preference` | auto | AP methods — higher (toward 0) → more clusters |

### Batch runs

```bash
.venv/bin/python3 scripts/batch_run.py       # all datasets, summary report
.venv/bin/python3 scripts/batch_compare.py   # all methods, side-by-side comparison
```

---

## Pipeline Overview

```
Raw Harp CSV
    │
    ▼  prepare_test_data.py
    │   Detect missing packets (Counter 0–127 wrap)
    │   Discard windows with > 10% NaN; interpolate remaining NaN
    │
    ▼  process_motion()
    │   Scale ADC → physical units (±4 g, ±1000 dps)
    │   Median filter (kernel=7) — spike removal
    │   Butterworth 1st-order high-pass 0.5 Hz via filtfilt (zero-phase)
    │   y_GA = y − y_BA  (gravity/posture),  tot_accel = √(xBA²+yBA²+zBA²)
    │
    ▼  extract_histogram_features()
    │   Slice into 60-sample (300 ms) non-overlapping windows
    │   Histogram per channel with hardHistNew.mat bin edges (30 bins total):
    │
    │     # 11 bins: y_GA
    │     [-1.0, -0.7778, -0.5556, -0.3333, -0.1111, 0.1111, 0.3333, 0.5556, 0.7778, 1.0, np.inf]
    │     # 11 bins: z
    │     [-1.5, -1.2222, -0.9444, -0.6667, -0.3889, -0.1111, 0.1667, 0.4444, 0.7222, 1.0, np.inf]
    │     # 6 bins: z_gyro
    │     [-100.0, -50.0, 0.0, 50.0, 100.0, np.inf]
    │     # 2 bins: log(totAccelBA)
    │     [-3.0, np.inf]
    │
    │   Normalize each window histogram to sum = 1 → N × 30 feature matrix
    │
    ▼  _build_cdf_features()
    │   cumsum per channel → L1 distance = 1D Wasserstein (EMD)
    │
    ▼  clustering
    │   cluster_hdbscan()        UMAP (15-D, L1) → HDBSCAN
    │   cluster_ap_full()        N×N affinity → AP  (preference = min)
    │   cluster_ap_sampled()     6,000-sample → AP → FAISS assign
    │   cluster_ap_sparse_knn()  FAISS K-NN graph → sparse AP → FAISS assign
    │
    ▼
Cluster_detail_results.csv  (ClusterIdx | Timestamp | Folder_Name)
```

**AP preference** is set to `min(similarity)`, matching Matlab `AccelCluster` (`min(s(:,3))`).

---

## Results

AP full is the Matlab-equivalent reference (skipped when N > 20,000).

### Cluster counts and Silhouette scores

| Dataset | N | HDBSCAN k | sil | AP full k | sil | AP sampled k | sil | AP sparse k | sil |
|---|---|---|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 3 | 0.13 | 9 | 0.17 | 9 | **0.17** | 21 | 0.14 |
| comparison_test | 13,799 | 8 | 0.16 | 44 | 0.16 | 26 | 0.17 | 132 | 0.13 |
| mp_mouse_1_jul | 19,714 | 25 | −0.04 | 45 | 0.24 | 20 | **0.25** | 136 | 0.17 |
| control_mouse_1_jul | 23,902 | 5 | 0.18 | — | — | 21 | 0.17 | 211 | 0.08 |
| control_mouse_1_oct | 23,906 | 13 | 0.07 | — | — | 21 | 0.16 | 203 | 0.10 |
| mp_mouse_1_oct | 23,866 | 24 | 0.05 | — | — | 20 | **0.32** | 145 | 0.24 |
| still_test | 23,877 | 5 | 0.24 | — | — | 20 | **0.32** | 139 | 0.19 |
| moving_test | 47,028 | 57 | 0.08 | — | — | 25 | **0.34** | 414 | 0.16 |

### Agreement with AP full (ARI)

ARI = 1.0 → identical; ARI ≈ 0 → no better than chance. Only available for N ≤ 20,000.

| Dataset | HDBSCAN ARI | AP sampled ARI | AP sparse ARI |
|---|---|---|---|
| short_comparison_test | 0.021 | **1.000** | 0.373 |
| comparison_test | 0.083 | **0.456** | 0.282 |
| mp_mouse_1_jul | 0.082 | **0.430** | 0.374 |

### Timing

| Dataset | N | HDBSCAN | AP full | AP sampled | AP sparse |
|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 6.9 s | 0.6 s | 0.6 s | 0.2 s |
| comparison_test | 13,799 | 12.7 s | 138 s | 15.8 s | 19.8 s |
| mp_mouse_1_jul | 19,714 | 14.1 s | 396 s | 17.6 s | 36.5 s |
| control_mouse_1_jul | 23,902 | 15.2 s | — | 19.3 s | 41.1 s |
| control_mouse_1_oct | 23,906 | 14.1 s | — | 26.8 s | 57.4 s |
| mp_mouse_1_oct | 23,866 | 17.6 s | — | 15.4 s | 54.3 s |
| still_test | 23,877 | 15.7 s | — | 24.1 s | 43.0 s |
| moving_test | 47,028 | 61.5 s | — | 20.5 s | 375 s |

---

## Known Limitations

**AP sparse over-clustering:** each node only communicates with its K nearest neighbours, so distant windows never merge. This structurally produces more clusters than AP full regardless of preference — it is not a tuning issue. Use AP sampled instead for large-N datasets.

**AP full memory:** requires an N×N affinity matrix (~14 GB at N=24,000). Automatically skipped when N > 20,000.
