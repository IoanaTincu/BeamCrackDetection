"""
Feature relevance analysis for one-crack datasets (no forward selection).

Works with your FOUR normalized datasets:

1) RFS dataset            -> features: RFS_mode1..RFS_mode10
2) DLC dataset (0.1 mm)   -> features: DLC_mode1..DLC_mode10
3) Damaged frequency      -> features: fdam_mode1..fdam_mode10
4) Delta frequency (Δf)   -> features: deltaf_mode1..deltaf_mode10

Target (default): crack position x (regression)

Metrics computed per feature (mode):
- Pearson correlation
- Spearman correlation
- Mutual Information
- Linear regression coefficients
- LASSO coefficients (LassoCV)
- Random Forest permutation importance
- Single-feature CV R^2

Also computes a "consensus ranking" that combines several metrics into one
overall ranking to reduce reliance on any single metric.

This version is corrected for newer scikit-learn:
- LassoCV: do not pass alphas=None (deprecated)
- RandomForestRegressor: do not use max_features='auto' (invalid); use 'sqrt'
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# ============================================================
# Utilities
# ============================================================

def detect_feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """
    Detect feature columns by one of the provided prefixes.

    Example prefixes:
      ("RFS_mode",) or ("DLC_mode",) or ("fdam_mode",) or ("deltaf_mode",)
    """
    cols: list[str] = []
    for p in prefixes:
        cols.extend([c for c in df.columns if c.startswith(p)])

    # Keep stable ordering by mode number if possible
    def mode_key(name: str):
        import re
        m = re.search(r"(\d+)$", name)
        return int(m.group(1)) if m else 10 ** 9

    return sorted(set(cols), key=mode_key)


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure selected columns are numeric floats; coerce invalid strings to NaN.
    """
    out = df.copy()
    out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
    return out


def load_split_csvs(split_dir: str | Path) -> pd.DataFrame:
    """
    Load train/val/test.csv from a directory (your standardized split outputs),
    and return one combined DataFrame with a 'split' column.
    """
    split_dir = Path(split_dir)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    missing = [p for p in (train_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files: {missing}")

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)


def get_target_x(df: pd.DataFrame) -> np.ndarray:
    """
    Extract target x as float array.
    """
    if "x" not in df.columns:
        raise ValueError("Target column 'x' not found in dataset.")

    return pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)


# ============================================================
# Feature relevance metrics
# ============================================================

def correlations_with_target(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Pearson correlation:
      - Question it asks:
          "Is this feature linearly related to the target?"
      - What it measures:
          Strength and direction of a *linear* relationship (range [-1, 1]).
          |r| close to 1 => strong linear association.

    Spearman correlation:
      - Question it asks:
          "Is this feature *monotonically* related to the target?"
      - What it measures:
          Strength and direction of a monotonic relationship using ranks
          (range [-1, 1]). Captures monotonic but nonlinear patterns.
    """
    pearson = {}
    spearman = {}

    y_series = pd.Series(y, name="y")

    for col in X.columns:
        x_series = pd.to_numeric(X[col], errors="coerce")
        pearson[col] = x_series.corr(y_series, method="pearson")
        spearman[col] = x_series.corr(y_series, method="spearman")

    return pd.DataFrame({
        "pearson_r": pd.Series(pearson),
        "spearman_r": pd.Series(spearman),
    })


def mutual_information(X: pd.DataFrame, y: np.ndarray, random_state=42) -> pd.Series:
    """
    Mutual Information (MI):
      - Question it asks:
          "How much information does this feature contain about the target,
           regardless of whether the relationship is linear?"
      - What it measures:
          A nonnegative value where higher means stronger dependency.
          MI can capture nonlinear relationships that correlations miss.
    """
    X_np = X.to_numpy(dtype=float)
    mi = mutual_info_regression(X_np, y, random_state=random_state)
    return pd.Series(mi, index=X.columns, name="mutual_info")


def linear_and_lasso_coeffs(X: pd.DataFrame, y: np.ndarray, random_state=42) -> pd.DataFrame:
    """
    Linear regression coefficients:
      - Question it asks:
          "If we fit a simple linear model x ≈ sum(w_i * feature_i),
           which features get the largest weights?"
      - What it measures:
          Coefficients w_i. With standardized features, |w_i| is a rough
          measure of importance under a linear assumption.

    LASSO coefficients (L1-regularized linear regression):
      - Question it asks:
          "Can we explain the target well using a *small subset* of features?"
      - What it measures:
          Coefficients after forcing sparsity. Many coefficients become exactly 0.
          LassoCV chooses the regularization strength (alpha) via cross-validation.

    Notes:
      - Coefficients can be unstable when features are highly correlated.
      - Signs indicate direction, magnitudes indicate strength (for standardized X).
    """
    # Linear regression
    lin = LinearRegression()
    lin.fit(X, y)
    lin_coef = pd.Series(lin.coef_, index=X.columns, name="lin_coef")

    # LASSO with CV (future-proof: do not set alphas=None)
    lasso = LassoCV(
        cv=5,
        random_state=random_state,
        max_iter=20000,
    )
    lasso.fit(X, y)
    lasso_coef = pd.Series(lasso.coef_, index=X.columns, name="lasso_coef")

    out = pd.DataFrame({
        "lin_coef": lin_coef,
        "lasso_coef": lasso_coef,
    })
    out.attrs["lasso_alpha"] = float(lasso.alpha_)
    return out


def rf_permutation_importance(X: pd.DataFrame, y: np.ndarray, random_state=42) -> pd.Series:
    """
    Random Forest permutation importance (R^2 drop):
      - Question it asks:
          "If we destroy (permute) this feature, how much does model performance drop?"
      - What it measures:
          The average decrease in R^2 after randomly shuffling a feature column.
          Larger drop => the model relied on that feature more.

    Why this is useful:
      - Captures nonlinear effects and feature interactions better than linear models.
      - More robust than using RF's built-in impurity importances.

    Caveat:
      - If multiple features contain redundant information, importance can be shared
        among them (each may look less important individually).
    """
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        max_features="sqrt",  # 'auto' is invalid in newer scikit-learn
    )
    rf.fit(X, y)

    perm = permutation_importance(
        rf,
        X,
        y,
        n_repeats=20,
        random_state=random_state,
        n_jobs=-1,
        scoring="r2",
    )

    return pd.Series(
        perm.importances_mean,
        index=X.columns,
        name="perm_importance_r2_drop",
    )


def single_feature_cv_scores(X: pd.DataFrame, y: np.ndarray, cv_splits=5, random_state=42) -> pd.Series:
    """
    Single-feature CV R^2 (using LinearRegression):
      - Question it asks:
          "If I only use this ONE feature, how well can I predict the target?"
      - What it measures:
          Cross-validated R^2 score using only that feature.
          Higher => better standalone predictive power (a proxy for identifiability).

    Why this matters:
      - It answers a practical question: "Which single mode is most informative by itself?"
      - It is less sensitive to multicollinearity than coefficients from multi-feature models.
    """
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()

    scores = {}
    for col in X.columns:
        r2 = cross_val_score(model, X[[col]], y, cv=cv, scoring="r2")
        scores[col] = float(np.mean(r2))

    return pd.Series(scores, name="single_feature_cv_r2")


# ============================================================
# Main analysis runner
# ============================================================

def analyze_dataset(
        name: str,
        df: pd.DataFrame,
        feature_prefixes: tuple[str, ...],
        output_dir: str | Path,
        use_split: str | None = "train",
        random_state=42,
):
    """
    Run feature relevance analysis on a dataset DataFrame.

    Parameters
    ----------
    name : str
        Dataset label (e.g., "RFS", "DLC", "FDAM", "DELTAF")
    df : DataFrame
        Data containing features and 'x' (and 'split' if filtering)
    feature_prefixes : tuple
        Column prefixes to detect features
    output_dir : str or Path
        Where to save results
    use_split : str or None
        Analyze only this split to avoid leakage (recommended: "train").
        If None, analyzes all rows.
    """
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze only the training split to avoid leakage (recommended)
    if use_split is not None:
        if "split" not in df.columns:
            raise ValueError("Requested use_split but no 'split' column found.")
        df_use = df[df["split"] == use_split].copy()
    else:
        df_use = df.copy()

    # Detect features and coerce numeric
    feature_cols = detect_feature_columns(df_use, feature_prefixes)
    if not feature_cols:
        raise ValueError(f"[{name}] No feature columns found for prefixes {feature_prefixes}")

    # Use only the first N modes (e.g., 6)
    N_MODES_TO_USE = 6
    feature_cols = feature_cols[:N_MODES_TO_USE]

    df_use = ensure_numeric(df_use, feature_cols + ["x"])
    df_use = df_use.dropna(subset=feature_cols + ["x"])

    X = df_use[feature_cols]
    y = get_target_x(df_use)

    # --------------------------------------------------
    # Compute relevance metrics
    # --------------------------------------------------
    corr_df = correlations_with_target(X, y)
    mi = mutual_information(X, y, random_state=random_state)
    coef_df = linear_and_lasso_coeffs(X, y, random_state=random_state)
    perm_imp = rf_permutation_importance(X, y, random_state=random_state)
    single_r2 = single_feature_cv_scores(X, y, cv_splits=5, random_state=random_state)

    # --------------------------------------------------
    # Combine per-feature metrics into one table
    # --------------------------------------------------
    metrics = corr_df.join(mi).join(coef_df).join(perm_imp).join(single_r2)

    # Helpful absolute-value columns (direction removed, strength retained)
    metrics["abs_pearson_r"] = metrics["pearson_r"].abs()
    metrics["abs_spearman_r"] = metrics["spearman_r"].abs()
    metrics["abs_lin_coef"] = metrics["lin_coef"].abs()
    metrics["abs_lasso_coef"] = metrics["lasso_coef"].abs()

    # --------------------------------------------------
    # Consensus ranking
    # --------------------------------------------------
    """
    Consensus ranking:
      - What it is:
          A combined score that aggregates several different importance metrics
          into one overall ranking (lower score = better rank).
      - Why we use it:
          Each metric has biases:
            * Correlations focus on linear/monotonic relationships
            * MI captures nonlinear dependence but can be noisy
            * Permutation importance depends on the chosen model
            * Single-feature R^2 reflects standalone predictability
          By combining them, we reduce reliance on any single metric and look for
          features that are consistently important across multiple "views".
      - How to interpret it:
          A feature with a good consensus rank is robustly informative, not just
          "important under one particular method".
    """

    # Simple consensus score (sum of ranks; lower is better)
    metrics["consensus_score"] = (
            metrics["mutual_info"].rank(ascending=False) +
            metrics["perm_importance_r2_drop"].rank(ascending=False) +
            metrics["single_feature_cv_r2"].rank(ascending=False) +
            metrics["abs_spearman_r"].rank(ascending=False)
    )

    metrics = metrics.sort_values("consensus_score")

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    metrics_path = output_dir / "feature_metrics.csv"
    summary_path = output_dir / "summary.txt"

    metrics.to_csv(metrics_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Split analyzed: {use_split}\n")
        f.write(f"Rows analyzed: {len(df_use)}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"LASSO alpha: {coef_df.attrs.get('lasso_alpha', None)}\n\n")

        f.write("Top 4 features by consensus_score (lower is better):\n")
        metrics_top = metrics.head(4)
        for feat in metrics_top.index:
            f.write(
                f"  {feat}: "
                f"MI={metrics_top.loc[feat, 'mutual_info']:.6f}, "
                f"PermImp={metrics_top.loc[feat, 'perm_importance_r2_drop']:.6f}, "
                f"SingleR2={metrics_top.loc[feat, 'single_feature_cv_r2']:.6f}, "
                f"|Spearman|={metrics_top.loc[feat, 'abs_spearman_r']:.6f}\n"
            )

    print(f"\n[{name}] Analysis complete.")
    print(f"  Rows analyzed: {len(df_use)} | Features: {len(feature_cols)} | Split: {use_split}")
    print(f"  Saved: {metrics_path}")
    print(f"  Top features: {list(metrics.head(4).index)}")

    return metrics


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    """
    Configure paths to your FOUR normalized split directories.

    Expected structure per directory:
      - train.csv
      - val.csv
      - test.csv

    Recommended:
      Analyze TRAIN split only (use_split='train') for feature ranking,
      to avoid leakage from val/test.
    """

    # Update these if your directories differ
    RFS_SPLIT_DIR = "cantilever_one_crack_rfs_combined_split_standardized"
    DLC_SPLIT_DIR = "cantilever_one_crack_dlc_split_standardized"
    FDAM_SPLIT_DIR = "cantilever_one_crack_fdam_combined_split_standardized"
    DELTAF_SPLIT_DIR = "cantilever_one_crack_deltaf_combined_split_standardized"

    OUT_DIR = "feature_analysis_results"

    # Load datasets (train/val/test combined with 'split' label)
    rfs_df = load_split_csvs(RFS_SPLIT_DIR)
    dlc_df = load_split_csvs(DLC_SPLIT_DIR)
    fdam_df = load_split_csvs(FDAM_SPLIT_DIR)
    deltaf_df = load_split_csvs(DELTAF_SPLIT_DIR)

    # Run analysis (train split only)
    analyze_dataset(
        name="RFS",
        df=rfs_df,
        feature_prefixes=("RFS_mode",),
        output_dir=OUT_DIR,
        use_split="train",
        random_state=42,
    )

    analyze_dataset(
        name="DLC",
        df=dlc_df,
        feature_prefixes=("DLC_mode",),
        output_dir=OUT_DIR,
        use_split="train",
        random_state=42,
    )

    analyze_dataset(
        name="FDAM",
        df=fdam_df,
        feature_prefixes=("fdam_mode",),
        output_dir=OUT_DIR,
        use_split="train",
        random_state=42,
    )

    analyze_dataset(
        name="DELTAF",
        df=deltaf_df,
        feature_prefixes=("deltaf_mode",),
        output_dir=OUT_DIR,
        use_split="train",
        random_state=42,
    )

    print("\nAll analyses complete. See:", Path(OUT_DIR).resolve())
