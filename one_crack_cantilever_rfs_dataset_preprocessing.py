import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def find_rfs_columns(df):
    """
    Identify RFS feature columns using the naming convention: RFS_mode*
    """
    return [c for c in df.columns if c.startswith("RFS_mode")]


def rfs_combine_split_standardize(
        input_dir,
        output_dir,
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
):
    """
    Combine CSV files, split into train/val/test using scikit-learn,
    fit StandardScaler on TRAIN only, and apply to val/test.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing single-crack CSV files
    output_dir : str or Path
        Directory where train/val/test CSVs and scaler will be saved
    test_size : float
        Fraction of data reserved for the test set
    val_size : float
        Fraction of data reserved for the validation set
    random_seed : int
        Random seed for reproducibility
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load and combine all CSV files
    # --------------------------------------------------
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    combined = pd.concat(
        [pd.read_csv(f) for f in csv_files],
        ignore_index=True
    )

    # --------------------------------------------------
    # Identify RFS columns and ensure float type
    # --------------------------------------------------
    rfs_cols = find_rfs_columns(combined)
    if not rfs_cols:
        raise ValueError("No RFS_mode* columns found.")

    combined[rfs_cols] = combined[rfs_cols].astype(float)

    # --------------------------------------------------
    # Split into train+val and test
    # --------------------------------------------------
    train_val, test = train_test_split(
        combined,
        test_size=test_size,
        shuffle=True,
        random_state=random_seed,
    )

    # Validation fraction relative to remaining data
    val_ratio = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        shuffle=True,
        random_state=random_seed,
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # --------------------------------------------------
    # Fit StandardScaler on TRAIN only
    # --------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(train[rfs_cols])

    # --------------------------------------------------
    # Apply scaler to all splits
    # --------------------------------------------------
    train.loc[:, rfs_cols] = scaler.transform(train[rfs_cols])
    val.loc[:, rfs_cols] = scaler.transform(val[rfs_cols])
    test.loc[:, rfs_cols] = scaler.transform(test[rfs_cols])

    # --------------------------------------------------
    # Format x column cleanly (optional, cosmetic)
    # --------------------------------------------------
    for df_part in (train, val, test):
        if "x" in df_part.columns:
            df_part["x"] = df_part["x"].astype(float).map(lambda v: f"{v:.3f}")

    # --------------------------------------------------
    # Save datasets and scaler
    # --------------------------------------------------
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    scaler_path = output_dir / "scaler.joblib"

    train.to_csv(train_path, index=False, float_format="%.12e")
    val.to_csv(val_path, index=False, float_format="%.12e")
    test.to_csv(test_path, index=False, float_format="%.12e")

    joblib.dump(scaler, scaler_path)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("Preprocessing complete.")
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Train / Val / Test sizes: {len(train)} / {len(val)} / {len(test)}")
    print(f"Standardized RFS columns: {len(rfs_cols)}")
    print(f"Scaler saved to: {scaler_path.resolve()}")

    return train, val, test, scaler


# ======================================================
# Main entry point
# ======================================================
if __name__ == "__main__":
    """
    Example usage:
    Place all single-crack CSV files in the folder 'single_crack_csvs/'
    and run this script directly.
    """

    rfs_combine_split_standardize(
        input_dir="single_crack_csvs",
        output_dir="cantilever_one_crack_rfs_combined_split_standardized",
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
    )
