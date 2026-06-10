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
        ├── Cluster_detail_results_ap_coreset.csv
        └── Cluster_detail_results_ap_sparse.csv
```

---

## Setup

**Python 3.10 or later required** (faiss-cpu compatibility).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below assume you are in `Python_Pipeline/` and use `.venv/bin/python3`.

---

## Input Data Format

The pipeline reads `combined_harp_data_cleaned.csv` — a concatenated Harp device log with one extra column.

Required columns:

| Column | Type | Description |
|---|---|---|
| `Command` | int | Harp message type |
| `RegisterAddress` | int | **Must be 34** for IMU rows |
| `Timestamp` | float | Seconds |
| `DataElement0` | int | Accelerometer X (raw ADC) |
| `DataElement1` | int | Accelerometer Y |
| `DataElement2` | int | Accelerometer Z |
| `DataElement3` | int | Gyroscope X |
| `DataElement4` | int | Gyroscope Y |
| `DataElement5` | int | Gyroscope Z |
| `DataElement6` | int | Magnetometer X |
| `DataElement7` | int | Magnetometer Y |
| `DataElement8` | int | Magnetometer Z |
| `DataElement9` | int | Sample counter (0–127) |
| `Folder_Name` | str | Session label (e.g. `mouse1_jul`) |

Only rows where `RegisterAddress == 34` are processed; all other rows are ignored.

Use `prepare_test_data.py` to convert a raw Harp-Motion CSV into this format:

```bash
.venv/bin/python3 scripts/prepare_test_data.py \
    --input  path/to/Harp-Motion.csv \
    --output path/to/combined_harp_data_cleaned.csv \
    --label  session_name
```

Multiple sessions can be concatenated into one file (pipeline handles them together).

---

## How to Run

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

# AP sampled (random subset)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled

# AP coreset (Greedy K-Center subset — better coverage than random)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled --use-coreset-sample

# AP hierarchical (two-level AP; experimental)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-hierarchical

# Sparse AP
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sparse
```

**Method guide:**

| Method | Flag | Notes |
|---|---|---|
| HDBSCAN | *(default)* | Fastest; detects noise points; runs directly on 30-D CDF features with L1 |
| AP full | `--use-ap` | Closest to Matlab; N ≤ 20,000 only |
| AP sampled | `--use-ap-sampled` | Random 6,000-window subset → AP → FAISS assignment |
| AP coreset | `--use-ap-sampled --use-coreset-sample` | Greedy K-Center subset; better rare-behavior coverage than random |
| AP hierarchical | `--use-ap-hierarchical` | Two-level AP; experimental — results may vary |
| Sparse AP | `--use-ap-sparse` | See [Known Limitations](#known-limitations) |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | Arena type — determines bin edges (see below) |
| `--min-cluster-size` | `15` | HDBSCAN — smaller → more clusters |
| `--sample-size` | `6000` | AP sampled/coreset — windows passed to AP |
| `--sparse-k` | `100` | Sparse AP — K-NN graph degree |
| `--preference` | auto | AP methods — higher (toward 0) → more clusters |

**Arena types:**

| Value | Description |
|---|---|
| `3d_wired` | 3D terrain arena, wired IMU *(default; matches Matlab reference)* |
| `3d_wireless` | 3D terrain arena, wireless IMU *(same bin edges as 3d_wired)* |
| `2d_wired` | Flat arena, wired IMU |
| `2d_wireless` | Flat arena, wireless IMU |

### Batch runs

```bash
# Run all datasets with default method, generate summary report
.venv/bin/python3 scripts/batch_run.py

# Compare all methods side by side
.venv/bin/python3 scripts/batch_compare.py

# Re-run only specific methods (other methods loaded from cache)
.venv/bin/python3 scripts/batch_compare.py --force ap_sampled ap_coreset
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
    │   cluster_hdbscan()        30-D CDF features, L1 metric → HDBSCAN (no dimensionality reduction)
    │   cluster_ap_full()        N×N affinity → AP  (preference = global min)
    │   cluster_ap_sampled()     6,000-sample → AP → FAISS assign  (preference = global min estimate)
    │   cluster_ap_sparse_knn()  FAISS K-NN graph → sparse AP → FAISS assign
    │
    ▼
Cluster_detail_results.csv  (ClusterIdx | Timestamp | Folder_Name)
```

**AP preference** is set to `min(similarity)` = −max(L1 dist²), matching Matlab `AccelCluster` (`min(s(:,3))`).  
For AP sampled/coreset, preference is estimated from 500,000 random pairs across all N windows (not just the subset), ensuring consistent scale with AP full.

---

## Results

AP full is the Matlab-equivalent reference (automatically skipped when N > 20,000).  
Silhouette scores are computed on 30-D CDF features with L1 distance.

### Cluster counts and Silhouette scores

| Dataset | N | HDBSCAN k | sil | AP full k | sil | AP sampled k | sil | AP coreset k | sil | AP sparse k | sil |
|---|---|---|---|---|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 2 | 0.30 | 9 | 0.15 | 9 | 0.15 | 9 | 0.15 | 22 | 0.14 |
| comparison_test | 13,799 | 7 | 0.26 | 33 | 0.15 | 23 | 0.13 | 24 | 0.13 | 88 | 0.12 |
| mp_mouse_1_jul | 19,714 | 82 | 0.07 | 41 | 0.19 | 25 | 0.18 | 25 | **0.22** | 136 | −0.02 |
| control_mouse_1_jul | 23,902 | 3 | **0.41** | — | — | 22 | 0.15 | 25 | 0.16 | 160 | 0.08 |
| control_mouse_1_oct | 23,906 | 17 | 0.12 | — | — | 21 | 0.16 | 26 | 0.15 | 169 | 0.07 |
| mp_mouse_1_oct | 23,866 | 207 | 0.36 | — | — | 24 | **0.24** | 25 | 0.18 | 149 | −0.06 |
| still_test | 23,877 | 221 | 0.32 | — | — | 20 | 0.20 | 26 | 0.20 | 147 | −0.11 |
| moving_test | 47,028 | 473 | **0.58** | — | — | 22 | **0.29** | 32 | **0.29** | 445 | 0.01 |

### Agreement with AP full (ARI)

ARI = 1.0 → identical to AP full; ARI ≈ 0 → no better than chance.  
Only computed for datasets where AP full runs (N ≤ 20,000).

| Dataset | HDBSCAN | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|
| short_comparison_test | 0.001 | **1.000** | **1.000** | 0.342 |
| comparison_test | 0.004 | **0.389** | 0.338 | 0.325 |
| mp_mouse_1_jul | 0.074 | **0.395** | 0.353 | 0.302 |

### Timing (clustering step only)

| Dataset | N | HDBSCAN | AP full | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | < 0.1 s | 0.6 s | 0.3 s | 0.2 s | 0.2 s |
| comparison_test | 13,799 | 2.3 s | ~135 s | 9.9 s | 9.8 s | ~20 s |
| mp_mouse_1_jul | 19,714 | 1.7 s | ~390 s | 15.9 s | 10.3 s | ~36 s |
| control_mouse_1_jul | 23,902 | 3.9 s | — | 7.6 s | 10.1 s | ~40 s |
| control_mouse_1_oct | 23,906 | 2.8 s | — | 14.9 s | 12.0 s | ~58 s |
| mp_mouse_1_oct | 23,866 | 1.7 s | — | 9.6 s | 14.0 s | ~54 s |
| still_test | 23,877 | 1.7 s | — | 6.5 s | 8.8 s | ~43 s |
| moving_test | 47,028 | 5.7 s | — | 16.6 s | 10.5 s | ~370 s |

---

## Known Limitations

**AP sparse over-clustering:** each node only communicates with its K nearest neighbours, so distant windows never merge. This structurally produces far more clusters than AP full regardless of preference. Negative silhouette scores on some datasets confirm poor partition quality. Use AP sampled or AP coreset instead for large-N datasets.

**HDBSCAN noise sensitivity:** HDBSCAN assigns noise points (−1) and its cluster count is highly sensitive to `--min-cluster-size`. The default `mcs=15` produces hundreds of clusters and noise rates above 50% on large datasets (e.g. `moving_test`: 473 clusters, 54% noise; `still_test`: 221 clusters, 66% noise). No single `mcs` value works well across all dataset sizes. HDBSCAN is retained as a reference but AP methods are preferred for consistent behavioral segmentation.

**AP full memory:** requires an N×N distance matrix and an N×N affinity matrix simultaneously (~6 GB at the N=20,000 threshold). Automatically skipped when N > 20,000.
