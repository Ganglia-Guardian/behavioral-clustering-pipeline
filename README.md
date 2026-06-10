# Python Clustering Pipeline

Python implementation of the Matlab behavioral clustering pipeline (`VPAPPAxes.m`).  
Processes Harp IMU data from mice, segments movement into 300 ms windows, and clusters them into behavioral motifs.  
Output format matches the original Matlab pipeline — all existing visualization notebooks still work.

---

## Setup

**Python 3.10+ required** (faiss-cpu compatibility).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All commands below assume you are in `Python_Pipeline/` and use `.venv/bin/python3`.

---

## Folder Structure

```
Python_Pipeline/
├── scripts/
│   ├── clustering_pipeline.py   # Main pipeline
│   ├── batch_run.py             # Run all datasets, generate summary report
│   ├── batch_compare.py         # Compare all methods side by side
│   └── prepare_test_data.py     # Convert raw Harp CSV to pipeline-ready format
└── results/
    ├── method_comparison.csv
    ├── method_comparison_report.txt
    └── <dataset_name>/
        ├── Cluster_detail_results_hdbscan.csv
        ├── Cluster_detail_results_ap_full.csv
        ├── Cluster_detail_results_ap_sampled.csv
        ├── Cluster_detail_results_ap_coreset.csv
        └── Cluster_detail_results_ap_sparse.csv
```

---

## How to Run

```bash
# HDBSCAN (default)
.venv/bin/python3 scripts/clustering_pipeline.py --input data.csv --output results.csv

# AP full — closest to Matlab; N ≤ 20,000 only
.venv/bin/python3 scripts/clustering_pipeline.py --input data.csv --output results.csv --use-ap

# AP sampled — random 6,000-window subset
.venv/bin/python3 scripts/clustering_pipeline.py --input data.csv --output results.csv --use-ap-sampled

# AP coreset — Greedy K-Center subset (better rare-behavior coverage)
.venv/bin/python3 scripts/clustering_pipeline.py --input data.csv --output results.csv --use-ap-sampled --use-coreset-sample

# AP hierarchical — two-level AP; experimental
.venv/bin/python3 scripts/clustering_pipeline.py --input data.csv --output results.csv --use-ap-hierarchical

# Sparse AP
.venv/bin/python3 scripts/clustering_pipeline.py --input data.csv --output results.csv --use-ap-sparse
```

**Method summary:**

| Method | Flag | Notes |
|---|---|---|
| HDBSCAN | *(default)* | Fastest; detects noise points; no N limit |
| AP full | `--use-ap` | Matlab-equivalent reference; N ≤ 20,000 only |
| AP sampled | `--use-ap-sampled` | Fast AP approximation for any N |
| AP coreset | `--use-ap-sampled --use-coreset-sample` | Better rare-behavior coverage than random sampling |
| AP hierarchical | `--use-ap-hierarchical` | Experimental — results may vary |
| Sparse AP | `--use-ap-sparse` | Not recommended; see Known Limitations |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | Arena type: `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN minimum cluster size |
| `--sample-size` | `6000` | Windows passed to AP sampled/coreset |
| `--preference` | auto | AP preference — toward 0 → more clusters |

**Batch comparison:**

```bash
.venv/bin/python3 scripts/batch_compare.py                         # all methods
.venv/bin/python3 scripts/batch_compare.py --force ap_sampled      # re-run specific methods; others load from cache
```

---

## Pipeline

**Input:** `combined_harp_data_cleaned.csv` — concatenated Harp logs (`RegisterAddress == 34` rows only) with a `Folder_Name` column. Key sensor columns: `DataElement0–2` (accelerometer XYZ), `DataElement3–5` (gyroscope XYZ), `DataElement9` (sample counter 0–127).  
Use `scripts/prepare_test_data.py` to convert a raw Harp-Motion CSV to this format.

**Processing steps:**

```
Raw CSV → filter RegisterAddress == 34
        → scale ADC to ±4 g / ±1000 dps
        → median filter (kernel=7) for spike removal
        → Butterworth high-pass 0.5 Hz (zero-phase)
        → y_GA = y − y_BA  (posture),  tot_accel = √(xBA²+yBA²+zBA²)
        → 60-sample (300 ms) non-overlapping windows
        → histogram per channel (30 bins total, hardHistNew.mat edges):
             y_GA   : 11 bins  [-1.0, -0.78, -0.56, -0.33, -0.11, 0.11, 0.33, 0.56, 0.78, 1.0, ∞]
             z      : 11 bins  [-1.5, -1.22, -0.94, -0.67, -0.39, -0.11, 0.17, 0.44, 0.72, 1.0, ∞]
             z_gyro :  6 bins  [-100, -50, 0, 50, 100, ∞]
             log(tot_accel) : 2 bins  [-3.0, ∞]
        → CDF transform per channel → N × 30 feature matrix
        → L1 distance on CDF = 1D Wasserstein / EMD
        → clustering → Cluster_detail_results.csv  (ClusterIdx | Timestamp | Folder_Name)
```

Feature channels match `findFeaturesWired3DArena.m`. AP preference = `min(similarity)` = −max(L1²), matching Matlab `AccelCluster`. For AP sampled/coreset, preference is estimated from 500,000 random pairs across all N windows (not just the subset) to stay consistent with AP full's scale.

---

## Results

Silhouette scored on 30-D CDF features with L1. AP full skipped when N > 20,000.

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

### ARI vs AP full (N ≤ 20,000 only)

| Dataset | HDBSCAN | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|
| short_comparison_test | 0.001 | **1.000** | **1.000** | 0.342 |
| comparison_test | 0.004 | **0.389** | 0.338 | 0.325 |
| mp_mouse_1_jul | 0.074 | **0.395** | 0.353 | 0.302 |

### Timing (clustering step only)

| Dataset | N | HDBSCAN | AP full | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | <0.1 s | 0.6 s | 0.3 s | 0.2 s | 0.2 s |
| comparison_test | 13,799 | 2.3 s | ~135 s | 9.9 s | 9.8 s | ~20 s |
| mp_mouse_1_jul | 19,714 | 1.7 s | ~390 s | 15.9 s | 10.3 s | ~36 s |
| control_mouse_1_jul | 23,902 | 3.9 s | — | 7.6 s | 10.1 s | ~40 s |
| mp_mouse_1_oct | 23,866 | 1.7 s | — | 9.6 s | 14.0 s | ~54 s |
| moving_test | 47,028 | 5.7 s | — | 16.6 s | 10.5 s | ~370 s |

---

## Known Limitations

**AP sparse:** structurally over-clusters — distant windows never communicate through the K-NN graph, so they can never merge regardless of preference. Use AP sampled or coreset instead.

**HDBSCAN:** cluster count is highly sensitive to `--min-cluster-size` and no single value generalises across dataset sizes. Produces hundreds of clusters and >50% noise on large recordings (moving_test: 473 clusters / 54% noise; still_test: 221 / 66%). Retained as a reference method only.

**AP full memory:** builds N×N distance and affinity matrices simultaneously (~6 GB at N=20,000). Automatically skipped when N > 20,000.
