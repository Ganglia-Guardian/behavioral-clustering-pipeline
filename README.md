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
        ├── Cluster_detail_results_ap_twolevel.csv
        ├── Cluster_detail_results_ap_sparse.csv
        ├── Cluster_detail_results_ap_kmeans.csv
        └── Cluster_detail_results_ap_stratified.csv
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
# AP sampled (default — recommended for all N)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv

# AP full (closest to Matlab; N ≤ 20,000 only)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap

# AP sampled (explicit flag)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled

# AP coreset (Greedy K-Center subset — better coverage than random)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled --use-coreset-sample

# AP two-level (two-level AP; experimental)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-twolevel

# AP K-Means sampled (Mini-batch K-Means centroids as AP input)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-kmeans-sample

# AP stratified sampled (stratified sampling to preserve rare behaviors)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-stratified

# Sparse AP
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sparse

# HDBSCAN (reference only — see Known Limitations)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-hdbscan
```

**Method guide:**

| Method | Flag | Notes |
|---|---|---|
| AP sampled | *(default)* | Recommended for all N; highest ARI vs AP full |
| AP full | `--use-ap` | Closest to Matlab; N ≤ 20,000 only |
| AP coreset | `--use-ap-sampled --use-coreset-sample` | Greedy K-Center subset; best silhouette on large N |
| AP two-level | `--use-ap-twolevel` | Two-level AP; cluster count closest to AP full |
| AP K-Means | `--use-ap-kmeans-sample` | Mini-batch K-Means centroids as AP sample; highest ARI on comparison_test |
| AP stratified | `--use-ap-stratified` | Stratified sampling across 200 coarse clusters; better rare-behavior coverage |
| Sparse AP | `--use-ap-sparse` | See [Known Limitations](#known-limitations) |
| HDBSCAN | `--use-hdbscan` | Reference only — not recommended for production; see [Known Limitations](#known-limitations) |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN — smaller → more clusters |
| `--sample-size` | `10000` | AP sampled/coreset/kmeans/stratified — windows passed to AP |
| `--sparse-k` | `100` | Sparse AP — K-NN graph degree |
| `--preference` | auto | AP methods — higher (toward 0) → more clusters |

### Batch runs

```bash
.venv/bin/python3 scripts/batch_compare.py                          # all methods, side-by-side comparison
.venv/bin/python3 scripts/batch_compare.py --force ap_sampled       # re-run specific methods; others load from cache
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
    │   cluster_hdbscan()                  30-D CDF features, L1 metric → HDBSCAN
    │   cluster_ap_full()                  N×N affinity → AP  (preference = global min)
    │   cluster_ap_sampled()               10,000-sample → AP → FAISS assign  (random or coreset)
    │   cluster_ap_kmeans_sampled()        Mini-batch K-Means centroids → AP → FAISS assign
    │   cluster_ap_stratified_sampled()    Stratified K-Means sample → AP → FAISS assign
    │   cluster_ap_sparse_knn()            FAISS K-NN graph → sparse AP → FAISS assign
    │   cluster_ap_twolevel()              Block AP → candidate exemplars → second AP → FAISS assign
    │
    ▼
Cluster_detail_results.csv  (ClusterIdx | Timestamp | Folder_Name)
```

**AP preference** is set to `min(similarity)` = −max(L1²), matching Matlab `AccelCluster` (`min(s(:,3))`).
For AP sampled/coreset/kmeans/stratified, preference is estimated from 500,000 random pairs across all N windows (not just the subset), keeping the scale consistent with AP full.

---

## Results

AP full is the Matlab-equivalent reference (skipped when N > 20,000).

### Cluster counts and Silhouette scores

`k` = number of clusters found; `sil` = Silhouette score (higher is better; >0.5 good, 0.25–0.5 reasonable, <0.25 poor).

| Dataset | N | HDBSCAN k | sil | AP full k | sil | AP sampled k | sil | AP coreset k | sil | AP two-level k | sil | AP sparse k | sil | AP K-Means k | sil | AP stratified k | sil |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 2 | 0.30 | 9 | 0.15 | 9 | 0.15 | 9 | 0.15 | 11 | 0.16 | 19 | 0.13 | 9 | 0.15 | 9 | 0.15 |
| comparison_test | 13,799 | 7 | 0.26 | 33 | 0.15 | 30 | 0.14 | 30 | 0.13 | 33 | 0.15 | 75 | 0.12 | 29 | 0.14 | 34 | 0.15 |
| mp_mouse_1_jul | 19,714 | 82 | 0.07 | 41 | 0.19 | 28 | 0.18 | 32 | **0.22** | 38 | 0.20 | 108 | 0.15 | 27 | **0.22** | 33 | 0.20 |
| control_mouse_1_jul | 23,902 | 3 | **0.41** | — | — | 30 | 0.14 | 34 | 0.13 | 48 | 0.13 | 134 | 0.08 | 29 | 0.13 | 35 | 0.15 |
| control_mouse_1_oct | 23,906 | 17 | 0.12 | — | — | 29 | 0.14 | 32 | **0.16** | 46 | 0.13 | 142 | 0.06 | 28 | 0.15 | 36 | 0.13 |
| mp_mouse_1_oct | 23,866 | 207 | 0.36 | — | — | 30 | 0.22 | 34 | 0.21 | 33 | 0.21 | 145 | 0.13 | 31 | 0.21 | 38 | **0.22** |
| still_test | 23,877 | 221 | 0.32 | — | — | 29 | 0.19 | 30 | 0.19 | 36 | 0.17 | 118 | 0.18 | 28 | 0.19 | 32 | 0.19 |
| moving_test | 47,028 | 473 | **0.58** | — | — | 30 | 0.28 | 42 | **0.31** | 32 | 0.23 | 222 | 0.20 | 32 | 0.26 | 42 | 0.26 |

### Agreement with AP full (ARI)

ARI = 1.0 → identical; ARI ≈ 0 → no better than chance. Only available for N ≤ 20,000.

| Dataset | HDBSCAN ARI | AP sampled ARI | AP coreset ARI | AP two-level ARI | AP sparse ARI | AP K-Means ARI | AP stratified ARI |
|---|---|---|---|---|---|---|---|
| short_comparison_test | 0.001 | **1.000** | **1.000** | 0.548 | 0.395 | **1.000** | **1.000** |
| comparison_test | 0.004 | 0.425 | 0.403 | 0.347 | 0.350 | **0.458** | 0.444 |
| mp_mouse_1_jul | 0.073 | **0.475** | 0.423 | 0.437 | 0.389 | 0.459 | 0.462 |

### Timing

| Dataset | N | HDBSCAN | AP full | AP sampled | AP coreset | AP two-level | AP sparse |
|---|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | < 0.1 s | 0.6 s | 0.3 s | 0.2 s | 0.2 s | 0.3 s |
| comparison_test | 13,799 | 2.3 s | ~135 s | 30.3 s | 24.6 s | 16.8 s | 16 s |
| mp_mouse_1_jul | 19,714 | 1.7 s | ~390 s | 33.1 s | 29.2 s | 25.6 s | 61 s |
| control_mouse_1_jul | 23,902 | 3.9 s | — | 18.3 s | 26.3 s | 28.5 s | 81 s |
| control_mouse_1_oct | 23,906 | 2.8 s | — | 19.8 s | 24.7 s | 35.1 s | 113 s |
| mp_mouse_1_oct | 23,866 | 1.7 s | — | 34.1 s | 25.5 s | 28.1 s | 190 s |
| still_test | 23,877 | 1.7 s | — | 24.4 s | 30.5 s | 26.0 s | 54 s |
| moving_test | 47,028 | 5.7 s | — | 51.4 s | 35.1 s | 70.5 s | 172 s |

---

## Known Limitations

**AP sparse over-clustering:** each node only communicates with its K nearest neighbours, so distant windows never merge. This structurally produces more clusters than AP full (typically 4–10×) regardless of preference — it is not a tuning issue. Use AP sampled or AP coreset instead for large-N datasets.

**HDBSCAN noise sensitivity:** cluster count is highly sensitive to `--min-cluster-size` and no single value works well across all dataset sizes. Large datasets can produce hundreds of clusters with majority noise points (e.g. moving_test: 473 clusters / 54% noise). Retained as a reference method; AP methods are preferred for consistent segmentation.

**AP full memory:** requires 4 simultaneous N×N matrices (distance, affinity, and sklearn's internal R/A message matrices) — approximately 12 GB at N=20,000. Automatically skipped when N > 20,000.
