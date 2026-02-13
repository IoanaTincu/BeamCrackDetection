from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # tqdm progress bar (auto chooses notebook/terminal)

from data import TabularDataset
from model import MLP


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    """
    Set seeds for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error:
      "On average, how large is the squared error, expressed in the original unit?"
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error:
      "On average, how far are predictions from true values (absolute distance)?"
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 score:
      "How much variance in the target does the model explain?"
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (sse / sst if sst > 0 else np.nan))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a loader, returning (y_true, y_pred) as numpy arrays.
    """
    model.eval()
    ys, preds = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yhat = model(Xb).cpu().numpy()
        preds.append(yhat)
        ys.append(yb.numpy())
    return np.vstack(ys), np.vstack(preds)


# ============================================================
# Training
# ============================================================

@dataclass
class TrainConfig:
    """
    Training hyperparameters.
    """
    hidden_layers: tuple[int, ...] = (64,)  # one hidden layer fixed width
    activation: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 30
    seed: int = 42
    device: str | None = None


def train_one_run(
        train_df,
        val_df,
        test_df,
        feature_cols: list[str],
        target_col: str,
        cfg: TrainConfig,
):
    """
    Train one model for a given feature subset and return:
    - model (trained, best-by-validation RMSE)
    - results dict with val/test metrics

    Update in this version:
    - Adds epoch progress visualization using tqdm.
    - Shows current val RMSE and best val RMSE in the progress bar.
    """

    set_seed(cfg.seed)

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets/loaders
    train_ds = TabularDataset(train_df, feature_cols=feature_cols, target_col=target_col)
    val_ds = TabularDataset(val_df, feature_cols=feature_cols, target_col=target_col)
    test_ds = TabularDataset(test_df, feature_cols=feature_cols, target_col=target_col)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = MLP(
        input_dim=len(feature_cols),
        output_dim=1,
        hidden_layers=cfg.hidden_layers,
        activation=cfg.activation,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None
    bad_epochs = 0

    # --------------------------------------------------------
    # Epoch progress bar
    # --------------------------------------------------------
    pbar = tqdm(
        range(1, cfg.max_epochs + 1),
        desc=f"Training ({len(feature_cols)} features)",
        unit="epoch",
        leave=True,
    )

    for epoch in pbar:
        # -----------------------------
        # Train for one epoch
        # -----------------------------
        model.train()
        running_loss = 0.0
        n_batches = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yhat = model(Xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)

        # -----------------------------
        # Validation RMSE
        # -----------------------------
        yv, pv = predict(model, val_loader, device=device)
        val_rmse = rmse(yv, pv)

        # -----------------------------
        # Early stopping logic
        # -----------------------------
        improved = val_rmse < (best_val - 1e-7)
        if improved:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Update progress bar display
        pbar.set_postfix({
            "train_loss": f"{train_loss:.3e}",
            "val_rmse": f"{val_rmse:.6f}",
            "best_val": f"{best_val:.6f}",
            "pat": f"{bad_epochs}/{cfg.patience}",
        })

        if bad_epochs >= cfg.patience:
            pbar.set_postfix({
                "train_loss": f"{train_loss:.3e}",
                "val_rmse": f"{val_rmse:.6f}",
                "best_val": f"{best_val:.6f}",
                "stopped": "early",
            })
            break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on val/test using best model
    yv, pv = predict(model, val_loader, device=device)
    yt, pt = predict(model, test_loader, device=device)

    results = {
        "n_features": int(len(feature_cols)),
        "features": ",".join(feature_cols),
        "val_rmse": rmse(yv, pv),
        "val_mae": mae(yv, pv),
        "val_r2": r2_score(yv, pv),
        "test_rmse": rmse(yt, pt),
        "test_mae": mae(yt, pt),
        "test_r2": r2_score(yt, pt),
    }

    return model, results


# ============================================================
# Checkpoint saving/loading
# ============================================================

def save_checkpoint(
        path: str | Path,
        model: nn.Module,
        metadata: dict,
):
    """
    Save model weights + metadata in a single file.

    We store:
    - model_state_dict
    - metadata (json-serializable dict): feature list, dataset name, metrics, etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, input_dim: int, cfg: TrainConfig):
    """
    Load a saved checkpoint and reconstruct the model architecture
    using (input_dim, cfg.hidden_layers, cfg.activation).
    """
    payload = torch.load(path, map_location="cpu")
    model = MLP(
        input_dim=input_dim,
        output_dim=1,
        hidden_layers=cfg.hidden_layers,
        activation=cfg.activation,
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload.get("metadata", {})
