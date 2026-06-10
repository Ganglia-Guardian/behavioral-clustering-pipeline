# Python Clustering Pipeline

A Python reimplementation of the Matlab behavioral clustering pipeline (`VPAPPAxes.m`).

Takes raw IMU sensor data from mice, chops it into 300 ms movement windows, and groups those windows into behavioral motifs using histogram features + Affinity Propagation or HDBSCAN. Output CSV format is identical to the original Matlab pipeline — all downstream visualization notebooks work without changes.

---

## Setup

> Python 3.10+ required (faiss-cpu compatibility)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below assume you are inside `Python_Pipeline/` and use `.venv/bin/python3`.

---

## Quick Start

```bash
# Default: HDBSCAN clustering
.venv/bin/python3 scripts/clustering_pipeline.py \
    --input  path/to/combined_harp_data_cleaned.csv \
    --output path/to/Cluster_detail_results.csv

# Compare all methods on all datasets
.venv/bin/python3 scripts/batch_compare.py

# Re-run only specific methods (others load from cache)
.venv/bin/python3 scripts/batch_compare.py --force ap_sampled ap_coreset
```

---

## Clustering Methods

| Method | Flag | Description |
|---|---|---|
| **HDBSCAN** | *(default)* | Fastest. Runs directly on 30-D CDF features with L1 metric. Marks outliers as noise. |
| **AP full** | `--use-ap` | Closest to the original Matlab pipeline. Requires computing an N×N similarity matrix — only feasible for N ≤ 20,000. |
| **AP sampled** | `--use-ap-sampled` | Randomly samples 6,000 windows, runs AP on the subset, then assigns all N windows via nearest-exemplar search. Scales to any N. |
| **AP coreset** | `--use-ap-sampled --use-coreset-sample` | Same as AP sampled but uses Greedy K-Center sampling instead of random — better coverage of rare behaviors. |
| **AP hierarchical** | `--use-ap-hierarchical` | Experimental two-level AP. Results may vary. |
| **Sparse AP** | `--use-ap-sparse` | Not recommended — see [Known Limitations](#known-limitations). |

**Tunable parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `--arena` | `3d_wired` | Bin edges for histogram features: `3d_wired` · `3d_wireless` · `2d_wired` · `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN sensitivity — lower value → more (smaller) clusters |
| `--sample-size` | `6000` | Number of windows sampled for AP sampled / coreset |
| `--preference` | auto | AP cluster count control — closer to 0 → more clusters, more negative → fewer |

---

## How It Works

### Input

The pipeline reads `combined_harp_data_cleaned.csv` — a concatenated Harp IMU log with one extra column (`Folder_Name`) identifying the recording session. Only rows with `RegisterAddress == 34` (IMU samples) are used. Accelerometer channels are `DataElement0–2`, gyroscope channels `DataElement3–5`.

Use `scripts/prepare_test_data.py` to convert a raw Harp-Motion CSV into this format.

### Processing steps

**1. Signal processing**
- Scale raw ADC values to physical units (±4 g accelerometer, ±1000 dps gyroscope)
- Remove spikes with a median filter (kernel size 7)
- Remove gravity/DC drift with a zero-phase Butterworth high-pass filter at 0.5 Hz
- Derive: `y_GA` (posture component) and `tot_accel` (total body acceleration magnitude)

**2. Feature extraction**
- Slice the signal into non-overlapping 60-sample (300 ms) windows
- For each window, compute a 30-bin normalized histogram across 4 channels:

  | Channel | Bins | Edges |
  |---|---|---|
  | y_GA | 11 | −1.0, −0.78, −0.56, −0.33, −0.11, 0.11, 0.33, 0.56, 0.78, 1.0, ∞ |
  | z | 11 | −1.5, −1.22, −0.94, −0.67, −0.39, −0.11, 0.17, 0.44, 0.72, 1.0, ∞ |
  | z_gyro | 6 | −100, −50, 0, 50, 100, ∞ |
  | log(tot_accel) | 2 | −3.0, ∞ |

  Bin edges match `hardHistNew.mat` from the original Matlab pipeline.

**3. Distance metric**
- Apply a CDF transform (cumsum) to each channel's histogram
- L1 distance on CDF vectors = 1D Wasserstein distance (Earth Mover's Distance)

**4. Clustering → output**
- Run the chosen algorithm on the N × 30 CDF feature matrix
- Write `Cluster_detail_results.csv` with columns: `ClusterIdx`, `Timestamp`, `Folder_Name`

AP preference is set to `min(similarity)` = −max(L1²), matching Matlab's `AccelCluster`. For AP sampled/coreset, preference is estimated from 500,000 random window pairs drawn from the full dataset — not just the subset — to keep the scale consistent with AP full.

---

## Results

Silhouette score interpretation: **> 0.5** good · **0.25–0.5** reasonable · **< 0.25** poor  
AP full is skipped automatically when N > 20,000 (memory limit).

### Cluster counts

| Dataset | N | HDBSCAN | AP full | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 2 | 9 | 9 | 9 | 22 |
| comparison_test | 13,799 | 7 | 33 | 23 | 24 | 88 |
| mp_mouse_1_jul | 19,714 | 82 | 41 | 25 | 25 | 136 |
| control_mouse_1_jul | 23,902 | 3 | — | 22 | 25 | 160 |
| control_mouse_1_oct | 23,906 | 17 | — | 21 | 26 | 169 |
| mp_mouse_1_oct | 23,866 | 207 | — | 24 | 25 | 149 |
| still_test | 23,877 | 221 | — | 20 | 26 | 147 |
| moving_test | 47,028 | 473 | — | 22 | 32 | 445 |

### Silhouette scores

| Dataset | HDBSCAN | AP full | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|---|
| short_comparison_test | 0.30 | 0.15 | 0.15 | 0.15 | 0.14 |
| comparison_test | 0.26 | 0.15 | 0.13 | 0.13 | 0.12 |
| mp_mouse_1_jul | 0.07 | 0.19 | 0.18 | **0.22** | −0.02 |
| control_mouse_1_jul | **0.41** | — | 0.15 | 0.16 | 0.08 |
| control_mouse_1_oct | 0.12 | — | 0.16 | 0.15 | 0.07 |
| mp_mouse_1_oct | 0.36 | — | **0.24** | 0.18 | −0.06 |
| still_test | 0.32 | — | 0.20 | 0.20 | −0.11 |
| moving_test | **0.58** | — | **0.29** | **0.29** | 0.01 |

### Agreement with AP full — ARI (N ≤ 20,000 only)

ARI = 1.0 means identical to AP full; ARI ≈ 0 means no better than chance.

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
| mp_mouse_1_oct | 23,866 | 1.7 s | — | 9.6 s | 14.0 s | ~54 s |
| moving_test | 47,028 | 5.7 s | — | 16.6 s | 10.5 s | ~370 s |

---

## Known Limitations

**AP sparse over-clusters.** Each window only communicates with its K nearest neighbors, so globally distant windows can never merge — regardless of how the preference is tuned. This structurally inflates cluster counts and produces poor silhouette scores (negative on several datasets). Use AP sampled or AP coreset instead.

**HDBSCAN is inconsistent across dataset sizes.** Its cluster count is highly sensitive to `--min-cluster-size`, and no single value generalises well. With the default `mcs=15`, large recordings often fragment into hundreds of clusters with majority noise (moving_test: 473 clusters / 54% noise; still_test: 221 clusters / 66% noise). HDBSCAN is retained as a reference but AP methods are preferred for consistent behavioral segmentation.

**AP full has a memory limit.** It builds an N×N distance matrix and an N×N affinity matrix in memory simultaneously (~6 GB at N=20,000). The pipeline automatically skips AP full and prints a warning when N > 20,000.
