import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# Column finders
# ============================================================

def find_fdam_columns(df):
    """
    Identify damaged-frequency feature columns using the naming convention: fdam_mode*
    """
    return [c for c in df.columns if c.startswith("fdam_mode")]


def find_deltaf_columns(df):
    """
    Identify delta-frequency feature columns using the naming convention: deltaf_mode*
    """
    return [c for c in df.columns if c.startswith("deltaf_mode")]


# ============================================================
# Preprocessing: damaged frequency dataset
# ============================================================

def fdam_combine_split_standardize(
        input_dir,
        output_dir,
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
):
    """
    Combine damaged-frequency CSV files, split into train/val/test using scikit-learn,
    fit StandardScaler on TRAIN only, and apply to val/test.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing damaged-frequency CSV files (one per severity)
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

    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # --------------------------------------------------
    # Identify damaged-frequency columns and ensure float type
    # --------------------------------------------------
    fdam_cols = find_fdam_columns(combined)
    if not fdam_cols:
        raise ValueError("No fdam_mode* columns found.")

    combined[fdam_cols] = combined[fdam_cols].astype(float)

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
    scaler.fit(train[fdam_cols])

    # --------------------------------------------------
    # Apply scaler to all splits
    # --------------------------------------------------
    train.loc[:, fdam_cols] = scaler.transform(train[fdam_cols])
    val.loc[:, fdam_cols] = scaler.transform(val[fdam_cols])
    test.loc[:, fdam_cols] = scaler.transform(test[fdam_cols])

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
    print("Damaged-frequency preprocessing complete.")
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Train / Val / Test sizes: {len(train)} / {len(val)} / {len(test)}")
    print(f"Standardized fdam columns: {len(fdam_cols)}")
    print(f"Scaler saved to: {scaler_path.resolve()}")

    return train, val, test, scaler


# ============================================================
# Preprocessing: delta frequency dataset
# ============================================================

def deltaf_combine_split_standardize(
        input_dir,
        output_dir,
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
):
    """
    Combine delta-frequency CSV files, split into train/val/test using scikit-learn,
    fit StandardScaler on TRAIN only, and apply to val/test.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing delta-frequency CSV files (one per severity)
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

    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # --------------------------------------------------
    # Identify delta-frequency columns and ensure float type
    # --------------------------------------------------
    deltaf_cols = find_deltaf_columns(combined)
    if not deltaf_cols:
        raise ValueError("No deltaf_mode* columns found.")

    combined[deltaf_cols] = combined[deltaf_cols].astype(float)

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
    scaler.fit(train[deltaf_cols])

    # --------------------------------------------------
    # Apply scaler to all splits
    # --------------------------------------------------
    train.loc[:, deltaf_cols] = scaler.transform(train[deltaf_cols])
    val.loc[:, deltaf_cols] = scaler.transform(val[deltaf_cols])
    test.loc[:, deltaf_cols] = scaler.transform(test[deltaf_cols])

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
    print("Delta-frequency preprocessing complete.")
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Train / Val / Test sizes: {len(train)} / {len(val)} / {len(test)}")
    print(f"Standardized deltaf columns: {len(deltaf_cols)}")
    print(f"Scaler saved to: {scaler_path.resolve()}")

    return train, val, test, scaler


# ======================================================
# Main entry point
# ======================================================
if __name__ == "__main__":
    """
    Example usage:
    Place all per-severity damaged-frequency CSV files in 'single_crack_fdam_csvs/'
    and all per-severity delta-frequency CSV files in 'single_crack_deltaf_csvs/'
    and run this script directly.
    """

    # Preprocess damaged-frequency dataset
    fdam_combine_split_standardize(
        input_dir="single_crack_fdam_csvs",
        output_dir="cantilever_one_crack_fdam_combined_split_standardized",
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
    )

    # Preprocess delta-frequency dataset
    deltaf_combine_split_standardize(
        input_dir="single_crack_deltaf_csvs",
        output_dir="cantilever_one_crack_deltaf_combined_split_standardized",
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
    )
