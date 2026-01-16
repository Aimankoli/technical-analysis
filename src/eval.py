"""
Evaluation script for the fusion model.

Computes: AUC, accuracy, balanced accuracy, Brier score, calibration curve.

Usage:
    python -m src.eval --config configs/config.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import FusionModel, create_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(config: dict, device: torch.device) -> FusionModel:
    """Load trained model from checkpoint."""
    model_path = Path("data/models/best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Create model with saved config or current config
    saved_config = checkpoint.get("config", config)
    model = create_model(saved_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def get_predictions(
    model: FusionModel,
    dataloader,
    device: torch.device,
    config: dict,
) -> dict:
    """Get model predictions on a dataset."""
    all_labels = []
    all_probs = []
    all_forward_returns = []
    all_end_dates = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            image = batch["image"].to(device)
            text_embedding = batch["text_embedding"].to(device)
            numeric_features = batch["numeric_features"].to(device)

            logits = model(
                image=image,
                text_embedding=text_embedding if config.get("use_text_embeddings", True) else None,
                numeric_features=numeric_features if config.get("use_numeric_features", True) else None,
            )

            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_forward_returns.extend(batch["forward_return"].numpy().tolist())
            all_end_dates.extend(batch["end_date"])
            all_sample_ids.extend(batch["sample_id"].tolist())

    return {
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "forward_returns": np.array(all_forward_returns),
        "end_dates": all_end_dates,
        "sample_ids": all_sample_ids,
    }


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive evaluation metrics."""
    preds = (probs > threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "brier_score": brier_score_loss(labels, probs),
    }

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = cm.tolist()

    # Classification report
    metrics["classification_report"] = classification_report(
        labels, preds, target_names=["Down", "Up"], output_dict=True
    )

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
        metrics["calibration"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
    except ValueError:
        metrics["calibration"] = None

    return metrics


def print_metrics(metrics: dict, split_name: str = "Test"):
    """Print metrics in a formatted way."""
    print(f"\n{'='*50}")
    print(f"{split_name} Set Evaluation Results")
    print(f"{'='*50}")

    print(f"\nOverall Metrics:")
    print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Brier Score:       {metrics['brier_score']:.4f}")

    print(f"\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  Predicted:    Down    Up")
    print(f"  Actual Down:  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"  Actual Up:    {cm[1, 0]:5d}  {cm[1, 1]:5d}")

    cr = metrics["classification_report"]
    print(f"\nPer-Class Metrics:")
    print(f"  Class    Precision  Recall  F1-Score  Support")
    for cls in ["Down", "Up"]:
        c = cr[cls]
        print(f"  {cls:8} {c['precision']:.4f}     {c['recall']:.4f}   {c['f1-score']:.4f}    {c['support']:5.0f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    args = parser.parse_args()

    config = load_config(args.config)

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print("\nLoading model...")
    model = load_model(config, device)

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = get_predictions(model, test_loader, device, config)
    metrics = compute_metrics(predictions["labels"], predictions["probs"], args.threshold)
    print_metrics(metrics, "Test")

    # Also evaluate on validation set for comparison
    print("\nEvaluating on validation set...")
    val_predictions = get_predictions(model, val_loader, device, config)
    val_metrics = compute_metrics(val_predictions["labels"], val_predictions["probs"], args.threshold)
    print_metrics(val_metrics, "Validation")

    # Save detailed results
    results_dir = Path("data/models")
    results_path = results_dir / "eval_results.json"

    results = {
        "test": metrics,
        "validation": val_metrics,
        "threshold": args.threshold,
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # Save predictions for backtesting
    predictions_path = results_dir / "test_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump({
            "sample_ids": predictions["sample_ids"],
            "end_dates": predictions["end_dates"],
            "probs": predictions["probs"].tolist(),
            "labels": predictions["labels"].tolist(),
            "forward_returns": predictions["forward_returns"].tolist(),
        }, f, indent=2)
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
