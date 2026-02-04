import os
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def infer_rfs_columns(df: pd.DataFrame) -> list[str]:
    # Detect columns that store Relative Frequency Shift (RFS) values per mode
    rfs_cols = [c for c in df.columns if c.lower().startswith("rfs")]
    if not rfs_cols:
        raise ValueError("No RFS columns found (expected columns like 'RFS_mode1', ...).")
    return rfs_cols


def get_constant_severity(df: pd.DataFrame) -> float:
    # Each single-crack CSV is expected to contain one constant severity value
    if "severity" not in df.columns:
        raise ValueError("Expected a 'severity' column in the CSV.")

    sev_vals = df["severity"].unique()
    if len(sev_vals) != 1:
        print(f"Warning: severity column has {len(sev_vals)} unique values; using the first.")
    return float(sev_vals[0])


def build_two_crack_dataset(file1: str, file2: str, out_path: str, enforce_x2_gt_x1: bool = True) -> None:
    # Read both single-crack CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Basic schema check
    if "x" not in df1.columns or "x" not in df2.columns:
        raise ValueError("Each CSV must include an 'x' column for crack position.")

    # Identify RFS mode columns in both files (must match)
    rfs_cols1 = infer_rfs_columns(df1)
    rfs_cols2 = infer_rfs_columns(df2)

    if set(rfs_cols1) != set(rfs_cols2):
        raise ValueError(
            "RFS columns mismatch between files:\n"
            f"{os.path.basename(file1)}: {rfs_cols1}\n"
            f"{os.path.basename(file2)}: {rfs_cols2}"
        )

    # Keep a stable ordering like RFS_mode1 ... RFS_mode10
    rfs_cols = sorted(rfs_cols1, key=lambda s: (len(s), s))

    # Extract constant severity for each file
    sev1 = get_constant_severity(df1)
    sev2 = get_constant_severity(df2)

    # Keep only position + RFS columns
    df1s = df1[["x"] + rfs_cols].copy()
    df2s = df2[["x"] + rfs_cols].copy()

    # Convert to NumPy arrays for fast cartesian combination
    x1 = df1s["x"].to_numpy()
    x2 = df2s["x"].to_numpy()
    rfs1 = df1s[rfs_cols].to_numpy()  # (n1, m)
    rfs2 = df2s[rfs_cols].to_numpy()  # (n2, m)

    # Build all (x1, x2) pairs
    X1 = np.repeat(x1, len(x2))
    X2 = np.tile(x2, len(x1))

    # Combine RFS (mode-wise sum) for every pair
    combined_rfs = (rfs1[:, None, :] + rfs2[None, :, :]).reshape(-1, rfs1.shape[1])

    # Optionally keep only pairs with x2 > x1 (avoids duplicates and equal positions)
    if enforce_x2_gt_x1:
        mask = X2 > X1
        X1 = X1[mask]
        X2 = X2[mask]
        combined_rfs = combined_rfs[mask]

    # Build output table
    out_df = pd.DataFrame(combined_rfs, columns=rfs_cols)
    out_df.insert(0, "x2", X2)
    out_df.insert(0, "x1", X1)
    out_df["severity1"] = sev1
    out_df["severity2"] = sev2

    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  (rows={len(out_df):,}, modes={len(rfs_cols)})")


def main():
    # Fixed folders (as requested)
    input_dir = Path("single_crack_csvs")
    output_dir = Path("two_crack_csvs")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Build 10 two-crack datasets from 20 single-crack CSV files (cantilever beam RFS)."
    )
    parser.add_argument(
        "--allow_equal_positions",
        action="store_true",
        help="Keep all (x1, x2) pairs (including x2 <= x1). Default keeps only x2 > x1.",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help=(
            "Optional explicit file pairs (filenames only, from single_crack_csvs). "
            "Format: fileA.csv,fileB.csv fileC.csv,fileD.csv ..."
        ),
    )
    args = parser.parse_args()

    # Load all single-crack CSV files from the fixed input directory, sorted alphabetically
    files = sorted(glob.glob(str(input_dir / "*.csv")))
    if len(files) < 2:
        raise ValueError(f"Need at least two CSV files in: {input_dir.resolve()}")

    # Decide which 10 file pairs to combine
    if args.pairs:
        pairs = []
        for p in args.pairs:
            a, b = p.split(",")
            fa = input_dir / a.strip()
            fb = input_dir / b.strip()
            if not fa.exists() or not fb.exists():
                raise FileNotFoundError(f"Pair file not found: {fa} or {fb}")
            pairs.append((str(fa), str(fb)))
        if len(pairs) != 10:
            print(f"Warning: you provided {len(pairs)} pairs; expected 10. Will process all provided pairs.")
    else:
        # Default: consecutive pairing after sorting -> (file 0 with file 1), (file 2 with file 3), ... produces up to 10 datasets
        if len(files) < 20:
            print(f"Warning: found {len(files)} files; expected 20. Pairing consecutively anyway.")
        pairs = [(files[i], files[i + 1]) for i in range(0, min(len(files) - 1, 20), 2)]
        pairs = pairs[:10]

    enforce = not args.allow_equal_positions

    # Build and save the 10 two-crack CSV files
    for idx, (f1, f2) in enumerate(pairs, start=1):
        out_name = f"two_crack_dataset_{idx:02d}.csv"
        out_path = output_dir / out_name
        build_two_crack_dataset(f1, f2, str(out_path), enforce_x2_gt_x1=enforce)


if __name__ == "__main__":
    main()
