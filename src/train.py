"""
Training script for the fusion model.

Usage:
    python -m src.train --config configs/config.yaml
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import FusionModel, count_parameters, create_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: FusionModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move to device
        image = batch["image"].to(device)
        text_embedding = batch["text_embedding"].to(device)
        numeric_features = batch["numeric_features"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(
            image=image,
            text_embedding=text_embedding if config.get("use_text_embeddings", True) else None,
            numeric_features=numeric_features if config.get("use_numeric_features", True) else None,
        )

        # Loss and backward
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    # Compute metrics
    avg_loss = total_loss / len(dataloader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    accuracy = np.mean(
        (np.array(all_probs) > 0.5).astype(int) == np.array(all_labels)
    )

    return {
        "loss": avg_loss,
        "auc": auc,
        "accuracy": accuracy,
    }


def evaluate(
    model: FusionModel,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            image = batch["image"].to(device)
            text_embedding = batch["text_embedding"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            logits = model(
                image=image,
                text_embedding=text_embedding if config.get("use_text_embeddings", True) else None,
                numeric_features=numeric_features if config.get("use_numeric_features", True) else None,
            )

            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    accuracy = np.mean(
        (np.array(all_probs) > 0.5).astype(int) == np.array(all_labels)
    )

    return {
        "loss": avg_loss,
        "auc": auc,
        "accuracy": accuracy,
        "labels": all_labels,
        "probs": all_probs,
    }


def train(config: dict) -> dict:
    """Main training function."""
    # Set seed
    set_seed(config.get("seed", 42))

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model
    print("\nCreating model...")
    model = create_model(config)
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function with optional class weighting
    # Compute class weights from training data
    train_labels = [batch["label"].numpy() for batch in train_loader]
    train_labels = np.concatenate(train_labels)
    pos_weight = (1 - train_labels.mean()) / train_labels.mean()
    print(f"Class distribution: {train_labels.mean():.2%} positive")
    print(f"Using pos_weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # Training loop
    epochs = config.get("epochs", 10)
    patience = config.get("early_stop_patience", 3)
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Update scheduler
        scheduler.step(val_metrics["auc"])

        # Early stopping
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  New best model! Val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, config)
    print(f"Test - Loss: {test_metrics['loss']:.4f}, AUC: {test_metrics['auc']:.4f}, Acc: {test_metrics['accuracy']:.4f}")

    # Save model
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Save metrics
    metrics = {
        "best_val_auc": best_val_auc,
        "test_loss": test_metrics["loss"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
    }
    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
