from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def load_splits(split_dir: str | Path):
    """
    Load train/val/test.csv from a directory created by your preprocessing pipeline.
    Expected files:
      - train.csv
      - val.csv
      - test.csv

    Returns
    -------
    (train_df, val_df, test_df) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    split_dir = Path(split_dir)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    missing = [p for p in (train_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files in {split_dir}: {missing}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def to_float_series(s: pd.Series) -> pd.Series:
    """
    Coerce a Series to float safely.
    """
    return pd.to_numeric(s, errors="coerce").astype(float)


class TabularDataset(Dataset):
    """
    Dataset for tabular regression.

    Inputs
    ------
    df : pd.DataFrame
    feature_cols : list[str]
        Which columns to use as input features.
    target_col : str
        Regression target column. For your one-crack experiments, target_col="x".

    Notes
    -----
    - We reshape y to (N, 1) so that it matches the neural network output shape
      for single-output regression.
    - We defensively drop any rows containing NaNs after coercion.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], target_col: str = "x"):
        X = df[feature_cols].apply(to_float_series).to_numpy(dtype=np.float32)
        y = to_float_series(df[target_col]).to_numpy(dtype=np.float32).reshape(-1, 1)

        # Drop rows with NaNs/infs (should not happen after your preprocessing, but safe)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        self.X = X[mask]
        self.y = y[mask]

        if self.X.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got shape {self.X.shape}")
        if self.y.ndim != 2 or self.y.shape[1] != 1:
            raise ValueError(f"Expected y to be shape (N,1), got shape {self.y.shape}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
