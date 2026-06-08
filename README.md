# Python Clustering Pipeline

Python replacement for the Matlab behavioral clustering pipeline (`VPAPPAxes.m`).  
Processes IMU sensor data from mice, segments movement into behavioral windows, and clusters them into behavioral motifs.

The output format is identical to the original Matlab pipeline, so all existing visualization notebooks still work.

---

## Folder Structure

```
Python_Pipeline/          ← repo root
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── clustering_pipeline.py   # Main pipeline (replacement for VPAPPAxes.m)
│   ├── batch_run.py             # Batch runner: processes all datasets, generates summary report
│   ├── batch_compare.py         # Compares all three clustering methods side by side
│   ├── run_ap_full_all.py       # Runs AP full on each dataset individually and saves results
│   ├── prepare_test_data.py     # Converts raw Harp CSV to pipeline-ready format
│   ├── compare_results.py       # Compares Python output against Matlab baseline
│   └── run_benchmark.py         # Benchmark: timing and memory comparison
├── test_outputs/
│   ├── combined_harp_data_cleaned.csv   # Preprocessed input (pipeline-ready format)
│   ├── inspect_stage1_raw.csv           # Raw sensor values
│   ├── inspect_stage2_processed.csv     # Physical-unit signals after filtering
│   ├── inspect_stage3_features.csv      # Histogram feature matrix (N_windows × 396)
│   └── python_ap_results.csv            # AP clustering output (test data)
└── results/
    ├── summary_report.txt               # Batch run report (timing, clusters, quality)
    ├── summary_metrics.csv              # Machine-readable summary of all datasets
    ├── method_comparison.csv            # Side-by-side comparison of all three methods
    ├── method_comparison_report.txt     # Human-readable method comparison report
    └── <dataset_name>/
        ├── Cluster_detail_results_hdbscan.csv    # HDBSCAN results
        ├── Cluster_detail_results_ap_sampled.csv # AP sampled results
        └── Cluster_detail_results_ap_full.csv    # AP full results (N ≤ 20,000 only)
```

---

## Requirements

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below assume you are in the `Python_Pipeline/` root directory and use `.venv/bin/python3`.

---

## How to Run

### Step 1 — Prepare raw data

Convert a raw Harp-Motion CSV to the pipeline-ready format:

```bash
.venv/bin/python3 scripts/prepare_test_data.py \
    --input  path/to/Harp-Motion.csv \
    --output path/to/combined_harp_data_cleaned.csv \
    --label  session_name
```

This detects missing packets (Counter column, 0–127 wrapping), discards 300ms windows with > 10% NaN, and interpolates remaining NaN sensor values.

If you already have a `combined_harp_data_cleaned.csv` from `stitched.py`, skip this step.

---

### Step 2 — Run the clustering pipeline

Three methods are available:

```bash
# HDBSCAN (default — recommended for large datasets)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv

# AP sampled (recommended for cross-session comparison)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled

# AP full (closest to Matlab; only practical for N ≤ 20,000)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap
```

**Method selection guide:**

| Method | When to use |
|---|---|
| HDBSCAN *(default)* | Any size; best Silhouette score in most cases |
| AP sampled `--use-ap-sampled` | Cross-session comparison; produces consistent cluster counts (~20) |
| AP full `--use-ap` | N ≤ 20,000; closest to original Matlab output |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | Arena type: `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN only — smaller → more clusters |
| `--n-neighbors` | auto | HDBSCAN only — UMAP n_neighbors; lower → more clusters |
| `--sample-size` | `6000` | AP sampled only — windows passed to AP |
| `--preference` | auto | AP full / AP sampled — controls cluster count; higher → more clusters |

Every run prints a Silhouette Score:
```
Silhouette Score: 0.3261  (reasonable)
Interpretation  : >0.5 good  |  0.25–0.5 reasonable  |  <0.25 poor
```

---

### Step 3 — Compare against Matlab results (optional)

```bash
.venv/bin/python3 scripts/compare_results.py \
    --matlab       path/to/matlab_output.mat \
    --python       path/to/Cluster_detail_results.csv \
    --python-label "AP sampled"
```

---

### Batch run on all datasets

```bash
# Run the pipeline on every dataset and save a summary report
.venv/bin/python3 scripts/batch_run.py

# Run all three methods and save a side-by-side comparison
.venv/bin/python3 scripts/batch_compare.py
```

AP full is automatically skipped when N > 20,000 (AP internals require ~3×N²×8 bytes).  
Results saved to `results/method_comparison_report.txt` and `results/method_comparison.csv`.

---

## What Changed vs. the Original Matlab Pipeline

### 1. Feature extraction — vectorized

**Matlab:** grows `totalMatrix` inside a loop → O(N²) memory copies.  
**Python:** pre-allocates once, fills with `np.add.at` → O(N) memory.

### 2. Distance calculation — analytic formula

**Matlab:** calls `emd()` MEX solver N² times.  
**Python:** uses the closed-form 1D Wasserstein distance via CDF L1 norm:

```
W₁(p, q) = Σ |CDF_p(k) − CDF_q(k)|
```

### 3. Clustering — three scalable options

| Method | Flag | Peak memory at N=24,000 | Time at N=24,000 |
|---|---|---|---|
| HDBSCAN | *(default)* | ~500 MB (UMAP) | ~20 s |
| AP sampled | `--use-ap-sampled` | ~288 MB (6,000² affinity) | ~30 s |
| AP full | `--use-ap` | ~14 GB (N² affinity) | ~10 min (est.) |

---

## Lab Dataset Results

AP full is used as a Matlab-equivalent reference and is skipped when N > 20,000.

### Cluster counts and Silhouette scores

| Dataset | N | HDBSCAN k | HDBSCAN sil | AP full k | AP full sil | AP sampled k | AP sampled sil |
|---|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 23 | 0.10 | 7 | 0.18 | 7 | 0.18 |
| comparison_test | 13,799 | 3 | **0.50** | 36 | 0.16 | 22 | 0.18 |
| mp_mouse_1_jul | 19,714 | 11 | **0.33** | 40 | 0.20 | 18 | 0.22 |
| control_mouse_1_jul | 23,902 | 2 | 0.23 | — | — | 18 | 0.16 |
| control_mouse_1_oct | 23,906 | 4 | **0.46** | — | — | 23 | 0.19 |
| mp_mouse_1_oct | 23,866 | 8 | 0.07 | — | — | 22 | 0.22 |
| still_test | 23,877 | 2 | **0.52** | — | — | 19 | 0.19 |
| moving_test | 47,028 | 42 | 0.05 | — | — | 22 | **0.28** |

### Agreement with AP full (Matlab reference)

RI and ARI measure agreement with AP full labels. ARI = 1.0 means identical clustering; ARI ≈ 0 means no better than chance. HDBSCAN noise points (label = −1) are excluded.

| Dataset | HDBSCAN RI | HDBSCAN ARI | AP sampled RI | AP sampled ARI |
|---|---|---|---|---|
| short_comparison_test | 0.866 | 0.329 | **1.000** | **1.000** |
| comparison_test | 0.065 | 0.002 | 0.945 | 0.392 |
| mp_mouse_1_jul | 0.221 | 0.012 | 0.937 | 0.360 |

Datasets with N > 20,000 do not have RI/ARI (AP full was skipped).  
Full results with timing in `results/method_comparison_report.txt`.

---

## Processing Pipeline Overview

```
Raw Harp CSV (Harp-Motion.csv)
        │
        ▼  prepare_test_data.py
        │   Detect missing packets (Counter column)
        │   Discard windows with >10% NaN
        │   Interpolate remaining NaN
        │
combined_harp_data_cleaned.csv
        │
        ▼  Step 1: load_cleaned_motion()
        │   Filter RegisterAddress == 34 rows
        │
        ▼  Step 2: process_motion()          ← mirrors processData.m
        │   Scale to physical units (g, dps)
        │   Median filter kernel=7
        │   Butterworth high-pass 0.5 Hz
        │   → 4 channels: y_GA, z, z_gyro, log(tot_accel)
        │
        ▼  Step 3: extract_histogram_features()
        │   Slice into 60-sample (300ms) windows
        │   Histogram each channel into 99 bins
        │   → feature matrix: N_windows × 396
        │
        ▼  Step 4: CDF transform
        │   cumsum per channel → L1 distance = 1D Wasserstein/EMD
        │
        ▼  Step 5: cluster_hdbscan() / cluster_ap_full() / cluster_ap_sampled()
        │
        ▼
Cluster_detail_results.csv
        ClusterIdx | Timestamp | Folder_Name
```
