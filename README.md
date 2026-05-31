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
│   ├── prepare_test_data.py     # Converts raw Harp CSV to pipeline-ready format
│   ├── compare_results.py       # Compares Python output against Matlab baseline
│   └── run_benchmark.py         # Benchmark: runs all methods and prints comparison table
├── test_outputs/
│   ├── combined_harp_data_cleaned.csv   # Preprocessed input (pipeline-ready format)
│   ├── inspect_stage1_raw.csv           # Raw sensor values (71,760 rows)
│   ├── inspect_stage2_processed.csv     # Physical-unit signals after filtering
│   ├── inspect_stage3_features.csv      # Histogram feature matrix (1,196 windows × 396 features)
│   └── python_ap_results.csv            # AP clustering output (7 clusters, test data)
└── results/
    ├── summary_report.txt               # Batch run report (timing, clusters, quality)
    ├── summary_metrics.csv              # Machine-readable summary of all datasets
    ├── method_comparison.csv            # Side-by-side comparison of all three methods
    ├── method_comparison_report.txt     # Human-readable method comparison report
    └── <dataset_name>/
        ├── Cluster_detail_results_hdbscan.csv    # HDBSCAN results
        ├── Cluster_detail_results_ap_sampled.csv # AP sampled results
        └── Cluster_detail_results_ap_full.csv    # AP full results (small datasets only)
```

---

## Requirements

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> **Python used throughout this README:**  
> `.venv/bin/python3`  
> All commands below assume you are in the `Python_Pipeline/` root directory.

---

## How to Run

### Step 1 — Prepare raw data

The pipeline reads `combined_harp_data_cleaned.csv`.  
Run this adapter to convert a raw Harp-Motion CSV:

```bash
.venv/bin/python3 scripts/prepare_test_data.py \
    --input  path/to/Harp-Motion.csv \
    --output path/to/combined_harp_data_cleaned.csv \
    --label  session_name
```

The adapter now performs full data quality processing:
- Detects missing packets via the Counter column (0–127 wrapping)
- Discards 300ms windows with > 10% NaN (packet loss too high)
- Linearly interpolates remaining NaN sensor values

If you already have a `combined_harp_data_cleaned.csv` from `stitched.py`, skip this step.

---

### Step 2 — Run the clustering pipeline

Three clustering methods are available:

```bash
# Default: HDBSCAN (recommended for N > 10,000 windows)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv

# AP sampled (recommended for large N, consistent cluster counts)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sampled

# AP full (closest to Matlab, only for N < ~10,000)
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap
```

**All parameters:**

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to `combined_harp_data_cleaned.csv` |
| `--output` | *(required)* | Path for output CSV |
| `--arena` | `3d_wired` | Arena type: `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--use-ap` | off | AP full (N×N matrix). Closest to Matlab. Only for N < ~10,000. |
| `--use-ap-sampled` | off | AP with random sampling. Recommended for N > 10,000. |
| `--sample-size` | `6000` | Windows sampled for AP sampled. Only with `--use-ap-sampled`. |
| `--preference` | auto | AP preference (controls cluster count). Higher → more clusters. Used with `--use-ap` or `--use-ap-sampled`. |
| `--min-cluster-size` | `15` | HDBSCAN min cluster size. Smaller → more clusters. Default only. |
| `--n-neighbors` | auto | UMAP n_neighbors. Lower → more clusters. Default only. |

**Choosing a method:**

| Data size | Recommended | Reason |
|---|---|---|
| N < 10,000 | `--use-ap` | Closest to Matlab, stable |
| N > 10,000, cross-session comparison | `--use-ap-sampled` | Consistent ~20 clusters across sessions |
| N > 10,000, cluster quality priority | *(default HDBSCAN)* | Higher Silhouette in some cases |

Every run prints a **Silhouette Score** after clustering:
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

Runs the pipeline on every dataset and writes results + a summary report:

```bash
.venv/bin/python3 scripts/batch_run.py
```

### Compare all three methods

Runs HDBSCAN, AP full, and AP sampled on every dataset and saves a side-by-side comparison:

```bash
.venv/bin/python3 scripts/batch_compare.py
```

AP full is automatically skipped for datasets where N > 15,000 (memory/time constraint).  
Results are saved to `results/method_comparison_report.txt` and `results/method_comparison.csv`.

---

### One-command benchmark (optional)

```bash
# Without Matlab file (estimates only)
.venv/bin/python3 scripts/run_benchmark.py

# With Matlab file (full comparison)
.venv/bin/python3 scripts/run_benchmark.py --matlab path/to/matlab_output.mat
```

---

## What Changed vs. the Original Matlab Pipeline

Signal processing and feature extraction are **identical** to Matlab.  
The changes are in the three bottleneck steps.

### 1. Feature extraction — vectorized

**Matlab:** `totalMatrix = [totalMatrix histMatrix]` inside a loop → O(N²) memory copies.  
**Python:** pre-allocate once, fill with `np.add.at` → O(N) memory.

### 2. Distance calculation — analytic formula

**Matlab:** calls `emd()` MEX solver N² times (~83 days for N=120,000).  
**Python:** CDF-L1 = W₁ closed-form solution + FAISS K-NN search.

```
W₁(p, q) = Σ |CDF_p(k) − CDF_q(k)|
```

| | Matlab | Python |
|---|---|---|
| Memory at N=120,000 | **109 GB** | **96 MB** |
| Time at N=120,000 | ~83 days | ~2 minutes |

### 3. Clustering — three options

| Method | Flag | When to use | Time at N=120,000 |
|---|---|---|---|
| HDBSCAN | *(default)* | Large data, cluster quality | ~5–15 min |
| AP sampled | `--use-ap-sampled` | Large data, cross-session comparison | ~2–5 min |
| AP full | `--use-ap` | Small data (N < 10,000), Matlab comparison | ~seconds |

---

## Lab Dataset Results

Results from running all three methods on the available lab recordings:

| Dataset | N | HDBSCAN clusters | HDBSCAN sil | AP sampled clusters | AP sampled sil |
|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 23 | 0.10 | 7 | 0.18 |
| comparison_test | 13,799 | 3 | **0.50** | 22 | 0.17 |
| control_mouse_1_jul | 23,902 | 2 | 0.23 | 18 | 0.16 |
| control_mouse_1_oct | 23,906 | 4 | **0.46** | 23 | 0.19 |
| mp_mouse_1_jul | 19,714 | 11 | **0.33** | 21 | 0.22 |
| mp_mouse_1_oct | 23,866 | 8 | 0.07 | 22 | 0.22 |
| moving_test | 47,028 | 42 | 0.05 | 22 | **0.28** |
| still_test | 23,877 | 2 | **0.52** | 19 | 0.19 |

Full results and timing in `results/method_comparison_report.txt`.

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
        ▼  Step 2: process_motion()          ← identical to processData.m
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
        ▼  Step 4: build_sparse_similarity()
        │   CDF transform, FAISS K-NN search
        │
        ▼  Step 5: cluster_hdbscan() / cluster_ap_sparse() / cluster_ap_sampled()
        │
        ▼
Cluster_detail_results.csv
        ClusterIdx | Timestamp | Folder_Name
```
