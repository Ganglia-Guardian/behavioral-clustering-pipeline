"""
prepare_test_data.py  —  convert a raw Harp-Motion CSV into the
combined_harp_data_cleaned.csv format that clustering_pipeline.py expects.

Handles missing-packet detection (Counter 0-127 wrap), window-level NaN
filtering (>10% loss → discard), linear interpolation, and label assignment.

Usage:
    python3 scripts/prepare_test_data.py \\
        --input  path/to/Harp-Motion.csv \\
        --output path/to/combined_harp_data_cleaned.csv \\
        --label  session_name
"""

import argparse
import pandas as pd
import numpy as np


# counter wraps 0 → 127 → 0
_COUNTER_COL = 12
_COUNTER_MAX = 128
_WIN_SIZE = 60
_LOSS_CUTOFF = 10.0  # max % NaN per window before we throw it away



def _read_and_pad(input_path: str) -> tuple:
    """Read raw variable-width CSV, pad to 14+ columns, cast to numeric."""
    raw_rows = []
    max_cols = 0
    with open(input_path, "r") as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.strip().split(",")
            raw_rows.append(parts)
            max_cols = max(max_cols, len(parts))

    target_cols = max(max_cols, 14)
    col_names = ["Command", "RegisterAddress", "Timestamp"]
    col_names += [f"DataElement{i}" for i in range(target_cols - 3)]
    padded = [r + [""] * (target_cols - len(r)) for r in raw_rows]

    df = pd.DataFrame(padded, columns=col_names)
    df["Command"] = pd.to_numeric(df["Command"], errors="coerce")
    df["RegisterAddress"] = pd.to_numeric(df["RegisterAddress"], errors="coerce")
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    for c in col_names[3:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, col_names


def _fill_missing_packets(df: pd.DataFrame) -> tuple:
    """Insert NaN placeholder rows for dropped packets detected via counter gaps."""
    motion_mask = df["RegisterAddress"] == 34
    motion_df = df[motion_mask].reset_index(drop=True)
    other_df = df[~motion_mask]

    new_rows = []
    n_inserted = 0

    for i in range(len(motion_df) - 1):
        curr = motion_df.iloc[i]
        nxt = motion_df.iloc[i + 1]
        new_rows.append(curr)

        c_curr = curr.iloc[_COUNTER_COL]
        c_next = nxt.iloc[_COUNTER_COL]

        if pd.notna(c_curr) and pd.notna(c_next):
            diff = int((c_next - c_curr) % _COUNTER_MAX)
            if diff > 1:
                t_start = curr.iloc[2]
                t_end = nxt.iloc[2]
                t_step = (t_end - t_start) / diff
                for j in range(1, diff):
                    nan_row = pd.Series([np.nan] * len(curr), index=curr.index)
                    nan_row.iloc[0] = 3    # Command
                    nan_row.iloc[1] = 34   # RegisterAddress
                    nan_row.iloc[2] = round(t_start + j * t_step, 4)
                    nan_row.iloc[_COUNTER_COL] = (c_curr + j) % _COUNTER_MAX
                    new_rows.append(nan_row)
                n_inserted += diff - 1

    new_rows.append(motion_df.iloc[-1])
    filled = pd.DataFrame(new_rows, columns=df.columns).reset_index(drop=True)
    return filled, other_df, n_inserted


def _clean_and_interpolate(motion_df: pd.DataFrame) -> tuple:
    """Discard windows with >_LOSS_CUTOFF% NaN, interpolate the rest."""
    n_windows = len(motion_df) // _WIN_SIZE
    kept = []
    n_discarded = 0

    for i in range(n_windows):
        window = motion_df.iloc[i * _WIN_SIZE : (i + 1) * _WIN_SIZE]
        # only check sensor cols (3-11); col 13 is Folder_Name placeholder
        n_nan = window.iloc[:, 3:12].isna().any(axis=1).sum()
        pct_nan = 100.0 * n_nan / _WIN_SIZE

        if pct_nan <= _LOSS_CUTOFF:
            kept.append(window)
        else:
            n_discarded += 1

    if not kept:
        return pd.DataFrame(columns=motion_df.columns), n_discarded

    cleaned = pd.concat(kept).reset_index(drop=True)

    # Interpolate sensor columns: DataElement0–8 (indices 3–11)
    cleaned.iloc[:, 3:12] = (
        cleaned.iloc[:, 3:12]
        .interpolate(method="linear", axis=0)
        .round()
    )

    return cleaned, n_discarded


def convert(input_path: str, output_path: str, label: str) -> dict:
    """Run full preprocessing for one Harp-Motion CSV. Returns summary stats."""
    print(f"  Preparing: {input_path}")

    df, col_names = _read_and_pad(input_path)
    n_raw_motion = int((df["RegisterAddress"] == 34).sum())

    filled_motion, other_df, n_inserted = _fill_missing_packets(df)

    cleaned_motion, n_discarded = _clean_and_interpolate(filled_motion)

    n_keep = (len(cleaned_motion) // _WIN_SIZE) * _WIN_SIZE
    cleaned_motion = cleaned_motion.iloc[:n_keep].reset_index(drop=True)

    col13 = col_names[13]
    cleaned_motion[col13] = cleaned_motion[col13].astype(object)
    cleaned_motion[col13] = label

    combined = pd.concat([other_df, cleaned_motion], ignore_index=True)
    combined = combined.rename(columns={col13: "Folder_Name"})
    combined = combined.sort_values("Timestamp").reset_index(drop=True)
    combined.to_csv(output_path, index=False)

    n_final_motion = len(cleaned_motion)
    n_final_windows = n_final_motion // _WIN_SIZE
    loss_pct = 100.0 * n_inserted / max(n_raw_motion + n_inserted, 1)

    print(f"  Raw motion rows: {n_raw_motion:,}")
    if n_inserted > 0:
        print(f"  Missing packets: {n_inserted:,}  ({loss_pct:.2f}%)")
    if n_discarded > 0:
        print(f"  Windows discarded: {n_discarded:,}  (>{_LOSS_CUTOFF}% NaN)")
    print(f"  Final: {n_final_motion:,} rows  ({n_final_windows:,} windows)  → {output_path}")

    return {
        "n_raw_motion": n_raw_motion,
        "n_inserted": n_inserted,
        "loss_pct": round(loss_pct, 2),
        "n_discarded_wins": n_discarded,
        "n_final_windows": n_final_windows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a raw Harp-Motion CSV for clustering_pipeline.py"
    )
    parser.add_argument("--input", required=True, help="Path to raw Harp-Motion CSV")
    parser.add_argument("--output", required=True, help="Output path for combined_harp_data_cleaned.csv")
    parser.add_argument("--label", default="session_1",
                        help="Session label for Folder_Name column (default: session_1)")
    args = parser.parse_args()
    convert(args.input, args.output, args.label)
