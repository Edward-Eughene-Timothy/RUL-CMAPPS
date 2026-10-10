"""
Main experiment runner for CMAPSS RUL prediction (FD001, FD002, FD003).
Uses CNN-LSTM deep learning model.
Saves models, metrics, predictions and generates visualization-ready data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocess import full_preprocess
from model import get_model
from train_eval import train_model, evaluate, split_train_val, rmse, mae, phm08_score

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
VIZ_DIR = PROJECT_ROOT / "visualizations"

for d in [RESULTS_DIR, MODELS_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

SEQUENCE_LENGTH = 30
EPOCHS = 35
BATCH_SIZE = 256
LR = 1e-3
MODEL_NAME = "cnn_lstm"


def plot_training_history(history, subset, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="Train Loss")
    ax[0].set_title(f"{subset} – Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    val = [v for v in history["val_rmse"] if v is not None]
    if val:
        ax[1].plot(val, label="Val RMSE", color="orange")
        ax[1].axvline(history["best_epoch"] - 1, color="red", linestyle="--", label="Best epoch")
        ax[1].set_title(f"{subset} – Validation RMSE")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("RMSE")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curve → {save_path}")


def plot_predictions(y_true, y_pred, subset, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    axes[0].plot([0, max(y_true.max(), y_pred.max())], [0, max(y_true.max(), y_pred.max())], "r--", lw=2)
    axes[0].set_xlabel("True RUL")
    axes[0].set_ylabel("Predicted RUL")
    axes[0].set_title(f"{subset} – Predicted vs True")
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = y_pred - y_true
    axes[1].hist(errors, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Prediction Error (Pred - True)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{subset} – Error Distribution")
    axes[1].grid(True, alpha=0.3)

    # Sorted RUL comparison
    idx = np.argsort(y_true)
    axes[2].plot(y_true[idx], label="True RUL", marker="o", markersize=3, alpha=0.7)
    axes[2].plot(y_pred[idx], label="Predicted RUL", marker="x", markersize=3, alpha=0.7)
    axes[2].set_xlabel("Engine (sorted by true RUL)")
    axes[2].set_ylabel("RUL")
    axes[2].set_title(f"{subset} – Sorted Comparison")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction plots → {save_path}")


def run_one_subset(subset: str):
    print(f"\n{'='*60}")
    print(f"  Processing {subset}")
    print(f"{'='*60}")

    data = full_preprocess(DATA_DIR, subset, sequence_length=SEQUENCE_LENGTH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    n_features = X_train.shape[-1]

    print(f"  Train sequences: {X_train.shape}")
    print(f"  Test sequences : {X_test.shape}  (one per engine)")
    print(f"  Features used  : {n_features} → {data['feature_cols']}")

    # Train / val split
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, val_ratio=0.12)

    model = get_model(MODEL_NAME, n_features, SEQUENCE_LENGTH)
    print(f"  Model: {MODEL_NAME} | params: {sum(p.numel() for p in model.parameters()):,}")

    history = train_model(
        model,
        X_tr, y_tr,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        device=DEVICE,
        patience=7,
    )

    # Save model
    model_path = MODELS_DIR / f"{subset}_{MODEL_NAME}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved → {model_path}")

    # Evaluate on test
    results = evaluate(model, X_test, y_test, device=DEVICE)
    print(f"\n  === {subset} TEST RESULTS ===")
    print(f"  RMSE  : {results['rmse']:.3f}")
    print(f"  MAE   : {results['mae']:.3f}")
    print(f"  Score : {results['score']:.1f}")

    # Save metrics & predictions
    metrics = {
        "subset": subset,
        "model": MODEL_NAME,
        "rmse": results["rmse"],
        "mae": results["mae"],
        "score": results["score"],
        "n_train_seq": int(X_train.shape[0]),
        "n_test_engines": int(X_test.shape[0]),
        "sequence_length": SEQUENCE_LENGTH,
        "features": data["feature_cols"],
        "best_epoch": history["best_epoch"],
    }
    with open(RESULTS_DIR / f"{subset}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "unit": data["test_units"],
        "true_RUL": results["y_true"],
        "pred_RUL": results["predictions"],
        "error": results["predictions"] - results["y_true"],
    })
    pred_df.to_csv(RESULTS_DIR / f"{subset}_predictions.csv", index=False)

    # Visualizations
    plot_training_history(history, subset, VIZ_DIR / f"{subset}_training_curve.png")
    plot_predictions(results["y_true"], results["predictions"], subset, VIZ_DIR / f"{subset}_predictions.png")

    return metrics, results


def main():
    all_metrics = []
    # Focus on FD001 → FD003 as requested. If FD003 is weak we still report it;
    # user said "if FD003 not producing good result then just FD002".
    subsets = ["FD001", "FD002", "FD003"]

    for subset in subsets:
        try:
            metrics, _ = run_one_subset(subset)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  ERROR on {subset}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    if all_metrics:
        summary = pd.DataFrame(all_metrics)[["subset", "rmse", "mae", "score", "n_test_engines"]]
        summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
        print("\n" + "=" * 60)
        print("  FINAL SUMMARY")
        print("=" * 60)
        print(summary.to_string(index=False))
        print(f"\nResults saved under: {RESULTS_DIR}")
        print(f"Visualizations under: {VIZ_DIR}")
        print(f"Models under       : {MODELS_DIR}")


if __name__ == "__main__":
    main()
