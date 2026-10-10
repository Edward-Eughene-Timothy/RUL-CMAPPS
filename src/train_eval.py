"""
Training loop, metrics (RMSE + PHM08 score), and evaluation helpers.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
from typing import Dict, Tuple


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 asymmetric scoring function.
    Early predictions (positive error) are penalized less than late ones.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 8,
) -> Dict:
    """
    Train with early stopping on validation RMSE (or train RMSE if no val).
    Returns history dict.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    history = {"train_loss": [], "val_rmse": [], "best_epoch": 0}
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
            val_rmse = rmse(y_val, val_pred)
            history["val_rmse"].append(val_rmse)
            scheduler.step(val_rmse)

            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                history["best_epoch"] = epoch
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            history["val_rmse"].append(None)

        if epoch % 5 == 0 or epoch == 1:
            val_str = f"{history['val_rmse'][-1]:.3f}" if history["val_rmse"][-1] is not None else "N/A"
            print(f"  Epoch {epoch:3d} | train_loss={avg_loss:.4f} | val_rmse={val_str}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str = "cpu") -> Dict:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(device)).cpu().numpy()
    # Clip predictions to reasonable range
    preds = np.clip(preds, 0, 150)
    return {
        "rmse": rmse(y, preds),
        "mae": mae(y, preds),
        "score": phm08_score(y, preds),
        "predictions": preds,
        "y_true": y,
    }


def split_train_val(X, y, val_ratio=0.15, seed=42):
    """Simple random split (for speed). For strict GroupKFold one can group by unit."""
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
