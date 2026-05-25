"""
prepare_test_data.py
--------------------
Converts a raw Harp-Motion CSV file (harp_data_cut.csv format) into the
combined_harp_data_cleaned.csv format expected by clustering_pipeline.py.

The only transformation is padding every row to at least 14 columns and
filling column 13 with a folder/session label so that load_cleaned_motion()
can read it correctly.

Usage (from repo root):
    python3 Python_Pipeline/scripts/prepare_test_data.py \
        --input  path/to/harp_data_cut.csv \
        --output path/to/combined_harp_data_cleaned.csv \
        --label  session_name
"""

import argparse
import pandas as pd
import numpy as np


def convert(input_path: str, output_path: str, label: str) -> None:
    print(f"Reading {input_path} ...")

    # Read raw CSV; column count varies per row (RegisterAddress==34 rows have more)
    raw_rows = []
    max_cols = 0
    with open(input_path, "r") as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            parts = line.strip().split(",")
            raw_rows.append(parts)
            max_cols = max(max_cols, len(parts))

    print(f"  Total rows: {len(raw_rows)}, max columns per row: {max_cols}")

    # Pad every row to max_cols with empty string, then to at least 14 columns
    target_cols = max(max_cols, 14)
    padded = [r + [""] * (target_cols - len(r)) for r in raw_rows]

    col_names = ["Command", "RegisterAddress", "Timestamp"]
    col_names += [f"DataElement{i}" for i in range(target_cols - 3)]

    df = pd.DataFrame(padded, columns=col_names)

    # Cast numeric columns
    df["Command"] = pd.to_numeric(df["Command"], errors="coerce")
    df["RegisterAddress"] = pd.to_numeric(df["RegisterAddress"], errors="coerce")
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    for c in col_names[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Trim to multiple of 60 so windows never straddle session boundaries
    motion_mask = df["RegisterAddress"] == 34
    n_motion = motion_mask.sum()
    keep_motion = (n_motion // 60) * 60
    print(f"  RegisterAddress==34 rows: {n_motion:,} → keeping {keep_motion:,} (multiple of 60)")

    # Rebuild: keep only the first keep_motion motion rows + all non-motion rows
    motion_idx = df.index[motion_mask][:keep_motion]
    non_motion_idx = df.index[~motion_mask]
    keep_idx = sorted(list(motion_idx) + list(non_motion_idx))
    df = df.loc[keep_idx].reset_index(drop=True)

    # Set folder label at column index 13
    col13 = col_names[13]
    df[col13] = label
    df.loc[df["RegisterAddress"] != 34, col13] = ""

    print(f"  Writing {len(df):,} rows to {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare raw Harp CSV for clustering_pipeline.py")
    parser.add_argument("--input",  required=True,  help="Path to raw harp_data_cut.csv")
    parser.add_argument("--output", required=True,  help="Path to write combined_harp_data_cleaned.csv")
    parser.add_argument("--label",  default="session_1",
                        help="Folder/session label written to column 13 (default: session_1)")
    args = parser.parse_args()
    convert(args.input, args.output, args.label)
