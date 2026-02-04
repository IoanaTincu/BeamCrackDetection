import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def find_dlc_columns(df):
    """
    Identify DLC feature columns using the naming convention: DLC_mode*
    """
    return [c for c in df.columns if c.startswith("DLC_mode")]


def dlc_split_standardize(
        input_csv,
        output_dir,
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
):
    """
    Preprocess a DLC dataset from a single CSV file:
    - Load dataset
    - Split into train/val/test using scikit-learn
    - Fit StandardScaler on TRAIN only (DLC columns)
    - Apply scaler to val/test
    - Save train/val/test splits and the scaler

    Parameters
    ----------
    input_csv : str or Path
        Path to the DLC CSV file (generated with x_step=0.0001)
    output_dir : str or Path
        Directory where train/val/test CSVs and scaler will be saved
    test_size : float
        Fraction of data reserved for the test set
    val_size : float
        Fraction of data reserved for the validation set
    random_seed : int
        Random seed for reproducibility
    """

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load the DLC CSV file
    # --------------------------------------------------
    if not input_csv.exists():
        raise FileNotFoundError(f"DLC CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # --------------------------------------------------
    # Identify DLC columns and ensure float type
    # --------------------------------------------------
    dlc_cols = find_dlc_columns(df)
    if not dlc_cols:
        raise ValueError("No DLC_mode* columns found.")

    df[dlc_cols] = df[dlc_cols].astype(float)

    # --------------------------------------------------
    # Split into train+val and test
    # --------------------------------------------------
    train_val, test = train_test_split(
        df,
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
    scaler.fit(train[dlc_cols])

    # --------------------------------------------------
    # Apply scaler to all splits
    # --------------------------------------------------
    train.loc[:, dlc_cols] = scaler.transform(train[dlc_cols])
    val.loc[:, dlc_cols] = scaler.transform(val[dlc_cols])
    test.loc[:, dlc_cols] = scaler.transform(test[dlc_cols])

    # --------------------------------------------------
    # Format x column cleanly (optional, cosmetic)
    # For step size 0.0001 -> 4 decimals
    # --------------------------------------------------
    if "x" in df.columns:
        for df_part in (train, val, test):
            df_part["x"] = df_part["x"].astype(float).map(lambda v: f"{v:.4f}")

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
    print("DLC preprocessing complete.")
    print(f"Input file: {input_csv.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Train / Val / Test sizes: {len(train)} / {len(val)} / {len(test)}")
    print(f"Standardized DLC columns: {len(dlc_cols)}")
    print(f"Scaler saved to: {scaler_path.resolve()}")

    return train, val, test, scaler


# ======================================================
# Main entry point
# ======================================================
if __name__ == "__main__":
    """
    Example usage:
    Provide the DLC CSV generated earlier with x_step=0.0001 and run directly.
    """

    dlc_split_standardize(
        input_csv="dlc_dataset/cantilever_beam_dlc_modes1to10_step_0p0001.csv",
        output_dir="cantilever_one_crack_dlc_split_standardized",
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
    )
