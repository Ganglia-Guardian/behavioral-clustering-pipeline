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
│   ├── batch_compare.py         # Compares all four clustering methods side by side
│   ├── run_ap_full_all.py       # Runs AP full on each dataset individually and saves results
│   ├── prepare_test_data.py     # Converts raw Harp CSV to pipeline-ready format
│   ├── compare_results.py       # Compares Python output against Matlab baseline
│   └── run_benchmark.py         # Benchmark: timing and memory comparison
├── test_outputs/
│   ├── combined_harp_data_cleaned.csv   # Preprocessed input (pipeline-ready format)
│   ├── inspect_stage1_raw.csv           # Raw sensor values (10 columns: ax ay az gx gy gz mx my mz counter)
│   ├── inspect_stage2_processed.csv     # Physical-unit signals after filtering (y_GA, z, z_gyro, tot_accel)
│   ├── inspect_stage3_features.csv      # Histogram feature matrix (N_windows × 30)
│   └── python_ap_results.csv            # AP clustering output (test data)
└── results/
    ├── summary_report.txt               # Batch run report (timing, clusters, quality)
    ├── summary_metrics.csv              # Machine-readable summary of all datasets
    ├── method_comparison.csv            # Side-by-side comparison of all four methods
    ├── method_comparison_report.txt     # Human-readable method comparison report
    └── <dataset_name>/
        ├── Cluster_detail_results.csv             # Default method result (HDBSCAN or AP)
        ├── Cluster_detail_results_hdbscan.csv     # HDBSCAN results
        ├── Cluster_detail_results_ap_full.csv     # AP full results (N ≤ 20,000 only)
        ├── Cluster_detail_results_ap_sampled.csv  # AP sampled results
        └── Cluster_detail_results_ap_sparse.csv   # Sparse AP results
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

This detects missing packets (Counter column, 0–127 wrapping), discards 300 ms windows with > 10% NaN, and interpolates remaining NaN sensor values.

If you already have a `combined_harp_data_cleaned.csv` from `stitched.py`, skip this step.

---

### Step 2 — Run the clustering pipeline

Four methods are available:

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

# Sparse AP (K-NN graph; O(N×K) memory instead of O(N²))
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv \
    --use-ap-sparse
```

**Method selection guide:**

| Method | When to use |
|---|---|
| HDBSCAN *(default)* | Any size; fastest; handles noise points |
| AP sampled `--use-ap-sampled` | Cross-session comparison; best agreement with AP full at scale |
| AP full `--use-ap` | N ≤ 20,000; closest to original Matlab output |
| Sparse AP `--use-ap-sparse` | Reference/comparison; note: structurally produces more clusters than AP full (see [Known Limitations](#known-limitations)) |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | Arena type: `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN only — smaller → more clusters |
| `--n-neighbors` | auto | HDBSCAN only — UMAP n_neighbors; lower → more clusters |
| `--sample-size` | `6000` | AP sampled only — windows passed to AP |
| `--sparse-k` | `100` | Sparse AP only — K-NN graph degree; higher → denser graph |
| `--preference` | auto | AP methods — controls cluster count; higher (toward 0) → more clusters |

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

# Run all four methods and save a side-by-side comparison
.venv/bin/python3 scripts/batch_compare.py
```

AP full is automatically skipped when N > 20,000 (AP internals require ~3×N²×8 bytes).  
Results saved to `results/method_comparison_report.txt` and `results/method_comparison.csv`.

---

## What Changed vs. the Original Matlab Pipeline

### 1. Histogram bin configuration — corrected to match Matlab

**Matlab:** `hardHistNew.mat` defines 30 bins total: 11 + 11 + 6 + 2, one set per feature channel.  
**Old Python (incorrect):** used 99 uniform bins per channel, producing a 396-dimensional feature vector.  
**Current Python:** uses the exact same 30-bin edges loaded from `hardHistNew.mat`, producing a 30-dimensional feature vector that matches the Matlab pipeline.

```
Channel      Bins  Edge range
──────────────────────────────────────────────
y_GA          11   [−1.0, …, 1.0, +∞]
z             11   [−1.5, …, 1.0, +∞]
z_gyro         6   [−100, −50, 0, 50, 100, +∞]
log(totAccel)  2   [−3.0, +∞]
──────────────────────────────────────────────
Total         30
```

### 2. Signal processing — zero-phase filtering

**Matlab (new `partition-evaluation` version):** uses `filtfilt` for zero-phase Butterworth high-pass filtering.  
**Python:** also uses `scipy.signal.filtfilt` — matches the new Matlab version exactly.

The processing chain for each IMU recording:

```
Raw ADC values  →  scale to g / dps
                →  median filter (kernel=7)       removes spike noise
                →  Butterworth high-pass 0.5 Hz   separates body accel (BA) from gravity (GA)
                →  y_GA = y − y_BA                gravity component (posture)
                →  tot_accel = √(xBA² + yBA² + zBA²)  movement magnitude
```

### 3. AP preference — min(similarity)

AP preference controls how many exemplars (cluster centers) emerge.  
**Python uses `min(similarity)`**, which matches the Matlab `AccelCluster` pipeline (`min(s(:,3))`).  
Using `median(similarity)` — as in the newer `partition-evaluation` Matlab version — causes a cluster explosion with the 30-bin coarse feature space (similarities cluster near 0, so median ≈ 0 → every point becomes its own exemplar).

### 4. Feature extraction — vectorized

**Matlab:** grows `totalMatrix` inside a loop → O(N²) memory copies.  
**Python:** pre-allocates once, fills with `np.add.at` → O(N) memory.

### 5. Distance calculation — analytic formula

**Matlab:** calls `emd()` MEX solver N² times.  
**Python:** uses the closed-form 1D Wasserstein distance via CDF L1 norm:

```
W₁(p, q) = Σ |CDF_p(k) − CDF_q(k)|
```

### 6. Clustering — four scalable options

| Method | Flag | Peak memory at N=24,000 | Time at N=24,000 |
|---|---|---|---|
| HDBSCAN | *(default)* | ~500 MB (UMAP embedding) | ~15 s |
| AP sampled | `--use-ap-sampled` | ~288 MB (6,000² affinity) | ~20–27 s |
| AP full | `--use-ap` | ~14 GB (N² affinity) | ~7 min (measured at N=19,714) |
| Sparse AP | `--use-ap-sparse` | ~115 MB (N×K sparse, K=1,000) | ~40–60 s |

---

## Lab Dataset Results

AP full is used as the Matlab-equivalent reference; it is skipped when N > 20,000.  
All results use the corrected 30-bin feature configuration from `hardHistNew.mat`, generated 2026-06-08.

### Cluster counts and Silhouette scores

| Dataset | N | HDBSCAN k | HDBSCAN sil | AP full k | AP full sil | AP sampled k | AP sampled sil | AP sparse k | AP sparse sil |
|---|---|---|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 3 | 0.13 | 9 | 0.17 | 9 | **0.17** | 21 | 0.14 |
| comparison_test | 13,799 | 8 | 0.16 | 44 | 0.16 | 26 | 0.17 | 132 | 0.13 |
| mp_mouse_1_jul | 19,714 | 25 | −0.04 | 45 | 0.24 | 20 | **0.25** | 136 | 0.17 |
| control_mouse_1_jul | 23,902 | 5 | 0.18 | — | — | 21 | 0.17 | 211 | 0.08 |
| control_mouse_1_oct | 23,906 | 13 | 0.07 | — | — | 21 | 0.16 | 203 | 0.10 |
| mp_mouse_1_oct | 23,866 | 24 | 0.05 | — | — | 20 | **0.32** | 145 | 0.24 |
| still_test | 23,877 | 5 | 0.24 | — | — | 20 | **0.32** | 139 | 0.19 |
| moving_test | 47,028 | 57 | 0.08 | — | — | 25 | **0.34** | 414 | 0.16 |

### Agreement with AP full (Matlab reference)

RI and ARI measure agreement with AP full labels. ARI = 1.0 means identical clustering; ARI ≈ 0 means no better than chance. HDBSCAN noise points (label = −1) are excluded from RI/ARI.

| Dataset | HDBSCAN RI | HDBSCAN ARI | AP sampled RI | AP sampled ARI | AP sparse RI | AP sparse ARI |
|---|---|---|---|---|---|---|
| short_comparison_test | 0.2293 | 0.0209 | **1.0000** | **1.0000** | 0.8878 | 0.3734 |
| comparison_test | 0.6400 | 0.0831 | 0.9598 | **0.4564** | 0.9729 | 0.2817 |
| mp_mouse_1_jul | 0.6724 | 0.0824 | 0.9474 | **0.4295** | 0.9744 | 0.3742 |

Datasets with N > 20,000 have no RI/ARI (AP full was skipped).  
Full results with timing in `results/method_comparison_report.txt`.

### Timing summary

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

### AP sparse over-clustering

Sparse AP uses a K-NN graph (default K=1,000) so each node only communicates with its K nearest neighbours. This limits global message propagation: distant windows form isolated "local echo chambers" and never merge. The result is structurally more clusters than AP full regardless of the preference setting — this is an inherent property of sparse AP on a K-NN graph, not a tuning issue.

**Recommendation:** use AP sampled for large-N datasets that need AP-compatible results.

### AP full memory wall

AP full requires an N×N affinity matrix (~3×N²×8 bytes for AP internals). At N=24,000 this is ~14 GB. The pipeline automatically skips AP full when N > 20,000.

---

## Processing Pipeline Overview

```
Raw Harp CSV (Harp-Motion.csv)
        │
        ▼  prepare_test_data.py
        │   Detect missing packets (Counter column, wraps 0–127)
        │   Discard 300 ms windows with > 10% NaN samples
        │   Interpolate remaining NaN sensor values
        │
combined_harp_data_cleaned.csv
        │
        ▼  Step 1: load_cleaned_motion()
        │   Filter rows where RegisterAddress == 34
        │   Extract columns DataElement0–9 (ax ay az gx gy gz mx my mz counter)
        │
        ▼  Step 2: process_motion()          ← mirrors processData.m
        │   Scale raw ADC → physical units (±4 g, ±1000 dps)
        │   Median filter (kernel=7)  — spike removal
        │   Butterworth 1st-order high-pass 0.5 Hz via filtfilt  — zero-phase
        │   y_GA = y − y_BA                 (gravity / posture channel)
        │   tot_accel = √(xBA² + yBA² + zBA²)  (movement magnitude)
        │   log transform on tot_accel before binning
        │
        ▼  Step 3: extract_histogram_features()
        │   Slice into 60-sample (300 ms) non-overlapping windows
        │   Histogram each channel using hardHistNew.mat bin edges:
        │     y_GA: 11 bins, z: 11 bins, z_gyro: 6 bins, log(tot_accel): 2 bins
        │   Normalize each window histogram to sum = 1
        │   → feature matrix: N_windows × 30
        │
        ▼  Step 4: CDF transform  (_build_cdf_features)
        │   cumsum per channel → L1 distance = 1D Wasserstein / EMD
        │   → CDF feature matrix: N_windows × 30
        │
        ▼  Step 5: clustering
        │   cluster_hdbscan()        UMAP (15-D, L1) → HDBSCAN
        │   cluster_ap_full()        full N×N affinity → sklearn AP
        │   cluster_ap_sampled()     random 6,000-sample → AP → FAISS assign
        │   cluster_ap_sparse_knn()  FAISS K-NN graph → sparse AP → FAISS assign
        │
        ▼
Cluster_detail_results.csv
        ClusterIdx | Timestamp | Folder_Name
```
