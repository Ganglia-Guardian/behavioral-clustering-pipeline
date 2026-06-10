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

| Method | Flag | Best for |
|---|---|---|
| HDBSCAN | *(default)* | Quick exploration; detects noise |
| AP full | `--use-ap` | Matlab-equivalent reference (small datasets) |
| AP sampled | `--use-ap-sampled` | Large N; fast AP approximation |
| AP coreset | `--use-ap-sampled --use-coreset-sample` | Large N; better coverage of rare behaviors |
| AP hierarchical | `--use-ap-hierarchical` | Experimental |
| Sparse AP | `--use-ap-sparse` | Not recommended — see Known Limitations |

**Key parameters:**

| Argument | Default | Description |
|---|---|---|
| `--arena` | `3d_wired` | Arena type: `3d_wired`, `3d_wireless`, `2d_wired`, `2d_wireless` |
| `--min-cluster-size` | `15` | HDBSCAN minimum cluster size |
| `--sample-size` | `6000` | Windows passed to AP sampled/coreset |
| `--preference` | auto | AP preference — toward 0 → more clusters |

**Batch comparison:**

```bash
.venv/bin/python3 scripts/batch_compare.py                        # all methods
.venv/bin/python3 scripts/batch_compare.py --force ap_sampled     # re-run specific methods only
```

---

## Pipeline

```
Raw Harp CSV (RegisterAddress == 34 rows only)
    ↓ Scale ADC → ±4 g / ±1000 dps · median filter (k=7) · high-pass 0.5 Hz
    ↓ 60-sample (300 ms) windows → 30-bin histogram (y_GA×11, z×11, z_gyro×6, log_accel×2)
    ↓ CDF transform per channel → L1 = Wasserstein-1 distance
    ↓ clustering → Cluster_detail_results.csv  (ClusterIdx | Timestamp | Folder_Name)
```

Feature channels match `findFeaturesWired3DArena.m`. AP preference = `min(similarity)` = −max(L1²), matching Matlab `AccelCluster`. For AP sampled/coreset, preference is estimated from 500,000 random pairs across all N windows to avoid underestimating the global scale.

**Input:** `combined_harp_data_cleaned.csv` — concatenated Harp logs with a `Folder_Name` column added.  
Use `scripts/prepare_test_data.py` to convert raw Harp-Motion CSV to this format.

---

## Results

Silhouette scored on 30-D CDF features with L1. AP full skipped when N > 20,000.

| Dataset | N | HDBSCAN k / sil | AP full k / sil | AP sampled k / sil | AP coreset k / sil | AP sparse k / sil |
|---|---|---|---|---|---|---|
| short_comparison_test | 1,195 | 2 / 0.30 | 9 / 0.15 | 9 / 0.15 | 9 / 0.15 | 22 / 0.14 |
| comparison_test | 13,799 | 7 / 0.26 | 33 / 0.15 | 23 / 0.13 | 24 / 0.13 | 88 / 0.12 |
| mp_mouse_1_jul | 19,714 | 82 / 0.07 | 41 / 0.19 | 25 / 0.18 | 25 / **0.22** | 136 / −0.02 |
| control_mouse_1_jul | 23,902 | 3 / **0.41** | — | 22 / 0.15 | 25 / 0.16 | 160 / 0.08 |
| control_mouse_1_oct | 23,906 | 17 / 0.12 | — | 21 / 0.16 | 26 / 0.15 | 169 / 0.07 |
| mp_mouse_1_oct | 23,866 | 207 / 0.36 | — | 24 / **0.24** | 25 / 0.18 | 149 / −0.06 |
| still_test | 23,877 | 221 / 0.32 | — | 20 / 0.20 | 26 / 0.20 | 147 / −0.11 |
| moving_test | 47,028 | 473 / **0.58** | — | 22 / **0.29** | 32 / **0.29** | 445 / 0.01 |

**ARI vs AP full** (datasets with N ≤ 20,000 only):

| Dataset | HDBSCAN | AP sampled | AP coreset | AP sparse |
|---|---|---|---|---|
| short_comparison_test | 0.001 | **1.000** | **1.000** | 0.342 |
| comparison_test | 0.004 | **0.389** | 0.338 | 0.325 |
| mp_mouse_1_jul | 0.074 | **0.395** | 0.353 | 0.302 |

---

## Known Limitations

**AP sparse:** structurally over-clusters — distant windows never communicate through the K-NN graph. Not tunable. Use AP sampled or coreset instead.

**HDBSCAN:** cluster count is highly sensitive to `--min-cluster-size` and no single value works across all dataset sizes. Produces hundreds of clusters and >50% noise on large datasets (moving_test: 473 clusters / 54% noise; still_test: 221 / 66%). Retained as a reference method only.

**AP full memory:** builds N×N distance and affinity matrices simultaneously (~6 GB at N=20,000). Automatically skipped when N > 20,000.
