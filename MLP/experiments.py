from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from data import load_splits
from train_utils import TrainConfig, train_one_run, save_checkpoint


# ============================================================
# Feature selection helpers
# ============================================================

def modes_1_to_k(prefix: str, k: int) -> list[str]:
    """
    Convenience helper:
      prefix="RFS_mode", k=6 -> ["RFS_mode1", ..., "RFS_mode6"]
    """
    return [f"{prefix}{i}" for i in range(1, k + 1)]


def load_top_features_from_feature_analysis(
        feature_metrics_csv: str | Path,
        top_k: int,
) -> list[str]:
    """
    Load the top-k features from your feature analysis CSV.

    Expected:
    - The CSV contains 'consensus_score' (lower is better), and a feature-name column.
    - When pandas saves a DataFrame with an index, the feature names often become
      the first column called 'Unnamed: 0'. We handle that case.

    Returns
    -------
    list[str]
        Top-k feature names (best-to-worst).
    """
    df = pd.read_csv(feature_metrics_csv)

    # Feature names likely saved as index -> "Unnamed: 0"
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "feature"})

    if "feature" not in df.columns:
        raise ValueError(
            f"Could not find 'feature' column in {feature_metrics_csv}. "
            f"Ensure the first column is the feature name (often 'Unnamed: 0')."
        )

    if "consensus_score" in df.columns:
        df = df.sort_values("consensus_score", ascending=True)

    feats = df["feature"].astype(str).tolist()
    return feats[:top_k]


# ============================================================
# Experiment definitions
# ============================================================

@dataclass
class DatasetSpec:
    """
    One dataset to run experiments on.
    """
    name: str
    split_dir: str | Path
    feature_prefix: str
    feature_analysis_csv: str | Path


def run_experiments_for_dataset(
        spec: DatasetSpec,
        cfg: TrainConfig,
        out_dir: str | Path,
        feature_sets: dict[str, list[str]],
        target_col: str = "x",
) -> pd.DataFrame:
    """
    Run a set of experiments for one dataset and save trained models.

    Parameters
    ----------
    spec : DatasetSpec
        Dataset specification (name, split_dir, etc.)
    cfg : TrainConfig
        Training configuration (fixed hidden layer size, etc.)
    out_dir : str | Path
        Output root folder, e.g. "runs/"
    feature_sets : dict[str, list[str]]
        A mapping:
            experiment_name -> list of feature columns to use
        This is passed in so experiment definitions are not hard-coded inside
        this function.
    target_col : str
        Regression target column (for one-crack: "x")

    Returns
    -------
    pd.DataFrame
        Results table sorted by test_rmse.
    """
    out_dir = Path(out_dir)
    models_dir = out_dir / "models" / spec.name
    results_dir = out_dir / "results" / spec.name

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(spec.split_dir)

    rows = []
    for exp_name, feats in feature_sets.items():
        model, metrics = train_one_run(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feats,
            target_col=target_col,
            cfg=cfg,
        )

        # Save checkpoint (weights + metadata)
        ckpt_path = models_dir / f"{exp_name}.pt"
        metadata = {
            "dataset": spec.name,
            "split_dir": str(Path(spec.split_dir).resolve()),
            "experiment": exp_name,
            "feature_cols": feats,
            "target_col": target_col,
            "train_config": {
                "hidden_layers": list(cfg.hidden_layers),
                "activation": cfg.activation,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "patience": cfg.patience,
                "seed": cfg.seed,
                "device": cfg.device,
            },
            "metrics": metrics,
        }
        save_checkpoint(ckpt_path, model, metadata)

        row = {"dataset": spec.name, "experiment": exp_name}
        row.update(metrics)
        rows.append(row)

        print(f"[{spec.name}] {exp_name}: test_rmse={metrics['test_rmse']:.6f} | saved {ckpt_path}")

    results_df = pd.DataFrame(rows).sort_values("test_rmse", ascending=True)
    results_csv = results_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)

    print(f"[{spec.name}] Saved results table to: {results_csv}")
    return results_df


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    """
    This script runs experiments and saves trained models + results tables.

    Plotting has been intentionally removed per your request.
    """

    RUNS_DIR = Path("runs")

    # Fixed architecture for fair comparison (one hidden layer, fixed width)
    cfg = TrainConfig(
        hidden_layers=(64,),  # fixed hidden neurons
        activation="relu",
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=1000,
        patience=30,
        seed=42,
        device=None,  # auto-select
    )

    # NOTE: split_dir paths updated to use relative paths with "../" as requested.
    specs = [
        DatasetSpec(
            name="RFS",
            split_dir="../cantilever_one_crack_rfs_combined_split_standardized",
            feature_prefix="RFS_mode",
            feature_analysis_csv="../feature_analysis_results/RFS/feature_metrics.csv",
        ),
        DatasetSpec(
            name="DLC",
            split_dir="../cantilever_one_crack_dlc_split_standardized",
            feature_prefix="DLC_mode",
            feature_analysis_csv="../feature_analysis_results/DLC/feature_metrics.csv",
        ),
        DatasetSpec(
            name="FDAM",
            split_dir="../cantilever_one_crack_fdam_combined_split_standardized",
            feature_prefix="fdam_mode",
            feature_analysis_csv="../feature_analysis_results/FDAM/feature_metrics.csv",
        ),
        DatasetSpec(
            name="DELTAF",
            split_dir="../cantilever_one_crack_deltaf_combined_split_standardized",
            feature_prefix="deltaf_mode",
            feature_analysis_csv="../feature_analysis_results/DELTAF/feature_metrics.csv",
        ),
    ]

    all_results = []
    for spec in specs:
        # Define feature sets OUTSIDE the runner, then pass them in.
        feature_sets = {
            # Single input neuron: mode 1 only
            "mode1_only": [f"{spec.feature_prefix}1"],

            # First six features (modes 1..6)
            "modes1_to_6": modes_1_to_k(spec.feature_prefix, 6),

            # Top 4 features from your feature analysis consensus ranking
            "top4_from_feature_analysis": load_top_features_from_feature_analysis(
                spec.feature_analysis_csv,
                top_k=4,
            ),
        }

        df_res = run_experiments_for_dataset(
            spec=spec,
            cfg=cfg,
            out_dir=RUNS_DIR,
            feature_sets=feature_sets,
            target_col="x",
        )
        all_results.append(df_res)

    # Save combined summary across all datasets
    combined = pd.concat(all_results, ignore_index=True)
    combined_csv = RUNS_DIR / "results" / "combined_results.csv"
    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)

    print(f"\nSaved combined results to: {combined_csv.resolve()}")
