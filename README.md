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
│   ├── prepare_test_data.py     # Converts raw Harp CSV to pipeline-ready format
│   ├── compare_results.py       # Compares Python output against Matlab baseline
│   └── run_benchmark.py         # One-command benchmark: runs all methods and prints comparison table
└── test_outputs/
    ├── combined_harp_data_cleaned.csv   # Preprocessed input (pipeline-ready format)
    ├── inspect_stage1_raw.csv           # Raw sensor values (71,760 rows)
    ├── inspect_stage2_processed.csv     # Physical-unit signals after filtering
    ├── inspect_stage3_features.csv      # Histogram feature matrix (1,196 windows × 396 features)
    └── python_ap_results.csv            # AP clustering output (7 clusters, test data)
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

### Step 1 — Prepare raw data (first time only)

The pipeline reads `combined_harp_data_cleaned.csv`, which is the output of `stitched.py`.  
If you are starting from a raw Harp-Motion CSV instead, run this adapter first:

```bash
.venv/bin/python3 scripts/prepare_test_data.py \
    --input  path/to/harp_data_cut.csv \
    --output test_outputs/combined_harp_data_cleaned.csv \
    --label  session_1
```

| Argument | Description |
|---|---|
| `--input` | Path to raw `harp_data_cut.csv` (or any Harp-Motion CSV) |
| `--output` | Where to write the pipeline-ready CSV |
| `--label` | Session/folder name written into each row (any string) |

If you already have a `combined_harp_data_cleaned.csv` from `stitched.py`, skip this step entirely.  
The `test_outputs/` folder already contains one for the included 6-minute test recording.

---

### Step 2 — Run the clustering pipeline

```bash
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  test_outputs/combined_harp_data_cleaned.csv \
    --output test_outputs/my_results.csv \
    --arena  3d_wired \
    --use-ap
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to `combined_harp_data_cleaned.csv` |
| `--output` | *(required)* | Path for output `Cluster_detail_results.csv` |
| `--arena` | `3d_wired` | Arena type: `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--use-ap` | off | Use Affinity Propagation (same algorithm as Matlab). Recommended for N < 10,000 windows. |
| `--min-cluster-size` | `15` | HDBSCAN minimum cluster size (only used without `--use-ap`) |

**Choosing between AP and HDBSCAN:**

| Data size | Recommended | Reason |
|---|---|---|
| N < 10,000 windows | `--use-ap` | Closest to Matlab results, stable |
| N > 10,000 windows | *(default HDBSCAN)* | AP is O(N²) and infeasible at scale |

**Output file columns** (same as Matlab's `Cluster_detail_results.csv`):

| Column | Description |
|---|---|
| `ClusterIdx` | Cluster number (1-based). 0 = noise/unassigned (HDBSCAN only). |
| `Timestamp` | Harp timestamp of the window's midpoint (seconds) |
| `Folder_Name` | Session label this window came from |

---

### Step 3 — Compare against Matlab results (optional)

Requires a `.mat` file produced by the original Matlab pipeline.

```bash
.venv/bin/python3 scripts/compare_results.py \
    --matlab       path/to/matlab_output.mat \
    --python       test_outputs/my_results.csv \
    --python-label "Affinity Propagation"
```

| Argument | Default | Description |
|---|---|---|
| `--matlab` | *(required)* | Path to Matlab `.mat` output file |
| `--python` | *(required)* | Path to Python `Cluster_detail_results.csv` |
| `--python-label` | `HDBSCAN + FAISS ANN` | Label shown in output (e.g. `"Affinity Propagation"`) |
| `--matlab-key` | `Clusters` | Struct key inside `.mat` file |
| `--matlab-field` | `idx` | Field that holds per-window labels |

Prints cluster count, size distribution, and agreement metrics (ARI, NMI) side by side.

---

### One-command benchmark (optional)

Runs both HDBSCAN and AP on the test data and prints a full performance table.  
Pass `--matlab` to include Matlab comparison; omit it to show estimates only.

```bash
# Without Matlab file (estimates only)
.venv/bin/python3 scripts/run_benchmark.py

# With Matlab file (full comparison)
.venv/bin/python3 scripts/run_benchmark.py --matlab path/to/matlab_output.mat
```

---

## What Changed vs. the Original Matlab Pipeline

The signal processing and feature extraction steps are **identical** to Matlab.  
The changes are in the three bottleneck steps that made the Matlab pipeline infeasible for large recordings.

### 1. Feature extraction — vectorized (was: growing loop)

**Matlab (`runBuildMatrix.m`):**
```matlab
for kk = 1:N_windows
    tmp = [];
    for i = 1:N_channels
        tempHist = histc(...)
        tmp = [tmp tempHist]   % copies the whole matrix on every iteration
    end
    newWin(kk,:) = tmp;
end
```
`totalMatrix = [totalMatrix histMatrix]` inside a loop copies the entire matrix on every iteration — N × C full copies in total.

**Python (`extract_histogram_features`):**
```python
hist_matrix = np.zeros((N_windows, total_bins))   # allocate once
np.add.at(hist_matrix, (row_idx, bin_idx), 1.0)   # fill all windows at once
```
Pre-allocates the full matrix once, then fills all windows simultaneously with vectorized indexing. No copies, no Python loops over windows.

---

### 2. Distance calculation — analytic formula (was: EMD solver called N² times)

**Matlab (`runDistanceSim.m` option 2):**
```matlab
for pp = 1:N
    for qq = 1:pp
        Dsim(pp,qq) = -(emd(hist_pp, hist_qq, boundaries).^2)
    end
end
```
Calls the `emd()` MEX solver for every pair of windows. For N = 120,000 windows this is **7.5 billion calls** (~83 days).

**Python (`build_sparse_similarity`):**
```python
cdf = np.cumsum(histogram, axis=1)          # CDF of each histogram, O(N·d)
distances, neighbors = faiss_index.search(cdf, K=100)   # K-NN search
```
For a 1D normalized histogram, the Wasserstein-1 distance (= EMD) has a closed-form solution:

```
W₁(p, q) = Σ |CDF_p(k) − CDF_q(k)|
```

Computing the CDF once and taking the L1 distance is O(d) per pair — no solver needed.  
FAISS then finds the K=100 nearest neighbors per window without computing all N² pairs.

| | Matlab | Python |
|---|---|---|
| Distance calls | N² (full matrix) | N × K (sparse) |
| Memory | N² × 8 bytes = **109 GB** at N=120,000 | N × K × 8 bytes = **96 MB** at N=120,000 |
| Time (N=120,000) | ~83 days | ~2 minutes |

---

### 3. Clustering — HDBSCAN / AP options (was: AP only, O(N²) per iteration)

**Matlab (`apclusterSparse.m`):**  
Affinity Propagation runs up to 1,000 iterations. Each iteration updates messages for all N points — O(N²) per iteration, O(N² × 1,000) total. For N = 120,000 this takes an estimated **167 days**.

**Python — two options:**

| Mode | Flag | When to use | Time at N=120,000 |
|---|---|---|---|
| Affinity Propagation | `--use-ap` | N < 10,000 windows, best match to Matlab results | ~hours (still O(N²)) |
| UMAP + HDBSCAN | *(default)* | N > 10,000 windows | ~5–15 minutes |

UMAP reduces the 396-dimensional feature space to 15 dimensions before HDBSCAN runs, solving the curse-of-dimensionality problem that causes HDBSCAN to fail on raw histogram features.

---

## Processing Pipeline Overview

```
Raw Harp CSV (harp_data_cut.csv)
        │
        ▼  prepare_test_data.py  (if not from stitched.py)
combined_harp_data_cleaned.csv
        │
        ▼  Step 1: load_cleaned_motion()
        │   Filter RegisterAddress == 34 rows
        │   71,760 rows × 10 columns  (AccXYZ, GyroXYZ, MagXYZ, counter)
        │
        ▼  Step 2: process_motion()          ← identical to processData.m
        │   Scale to physical units (g, dps)
        │   Median filter kernel=7 (spike removal)
        │   Butterworth high-pass 0.5 Hz (gravity separation)
        │   → 4 channels: y_GA, z, z_gyro, log(tot_accel)
        │
        ▼  Step 3: extract_histogram_features()  ← identical to runBuildMatrix.m
        │   Slice into 60-sample (300 ms) windows
        │   Histogram each channel into 99 bins
        │   → feature matrix: N_windows × 396
        │
        ▼  Step 4: build_sparse_similarity()     ← replaces runDistanceSim.m
        │   CDF transform per channel
        │   FAISS K-NN search (L1 metric)
        │   → sparse similarity matrix: N × K triplets
        │
        ▼  Step 5: cluster_ap_sparse() or cluster_hdbscan()
        │   AP  → same algorithm as Matlab, feasible for N < 10,000
        │   HDBSCAN → UMAP 396-D→15-D, then density clustering
        │
        ▼
Cluster_detail_results.csv
        ClusterIdx | Timestamp | Folder_Name
```

---

## Test Data

The `test_outputs/` folder contains results from a 6-minute test recording (1,196 behavioral windows):

| File | Description |
|---|---|
| `combined_harp_data_cleaned.csv` | Pipeline-ready input (output of stitched.py) |
| `inspect_stage1_raw.csv` | Raw ADC values from the sensor (71,760 rows) |
| `inspect_stage2_processed.csv` | Signals in physical units after filtering |
| `inspect_stage3_features.csv` | Histogram feature matrix (1,196 rows × 396 features) |
| `python_ap_results.csv` | AP clustering result: 7 clusters |
