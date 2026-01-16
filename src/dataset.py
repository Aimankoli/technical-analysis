"""
Fusion Dataset for training the multi-modal model.

Combines:
- Chart images
- FinBERT text embeddings
- Numeric features

Usage:
    from src.dataset import FusionDataset, get_dataloaders
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Numeric features to extract from indicators
NUMERIC_FEATURES = [
    "return_1d",
    "return_5d",
    "bb_percent_b",
    "distance_to_sma20",
    "rsi14",
    "trend_slope_10d",
    "volatility_20d",
    "volume_zscore",
]


class FusionDataset(Dataset):
    """
    Multi-modal dataset combining images, text embeddings, and numeric features.
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
        images_dir: str,
        embeddings_path: Optional[str] = None,
        embedding_mapping_path: Optional[str] = None,
        use_text_embeddings: bool = True,
        use_numeric_features: bool = True,
        image_size: int = 112,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            samples_df: DataFrame with sample metadata
            indicators_df: DataFrame with indicators for all dates
            images_dir: Directory containing chart images
            embeddings_path: Path to finbert embeddings .npy file
            embedding_mapping_path: Path to sample_id to index mapping JSON
            use_text_embeddings: Whether to include text embeddings
            use_numeric_features: Whether to include numeric features
            image_size: Expected image size
            transform: Image transforms (default: normalize for ImageNet)
        """
        self.samples_df = samples_df.reset_index(drop=True)
        self.indicators_df = indicators_df
        self.images_dir = Path(images_dir)
        self.use_text_embeddings = use_text_embeddings
        self.use_numeric_features = use_numeric_features
        self.image_size = image_size

        # Default transform: normalize for pretrained ImageNet models
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

        # Load embeddings if enabled
        self.embeddings = None
        self.embedding_mapping = None
        if use_text_embeddings and embeddings_path:
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                if embedding_mapping_path and os.path.exists(embedding_mapping_path):
                    with open(embedding_mapping_path, "r") as f:
                        # Convert keys back to int
                        self.embedding_mapping = {
                            int(k): v for k, v in json.load(f).items()
                        }

        # Compute feature statistics for normalization (from training set only!)
        self._feature_means = None
        self._feature_stds = None

    def set_feature_stats(self, means: np.ndarray, stds: np.ndarray):
        """Set normalization statistics (computed from training set)."""
        self._feature_means = means
        self._feature_stds = stds

    def compute_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and std of numeric features from this dataset."""
        features = []
        for idx in range(len(self)):
            end_idx = self.samples_df.iloc[idx]["end_idx"]
            row = self.indicators_df.iloc[end_idx]
            feat = [row.get(f, 0.0) for f in NUMERIC_FEATURES]
            features.append(feat)

        features = np.array(features)
        # Replace NaN with 0 for stats computation
        features = np.nan_to_num(features, nan=0.0)

        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero

        return means, stds

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns dict with:
            - image: (3, H, W) tensor
            - text_embedding: (768,) tensor if enabled
            - numeric_features: (n_features,) tensor if enabled
            - label: scalar tensor (0 or 1)
            - forward_return: scalar tensor (float)
            - end_date: string
            - sample_id: int
        """
        sample = self.samples_df.iloc[idx]
        sample_id = sample["sample_id"]

        # Load image
        image_path = self.images_dir / f"{sample_id}.png"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Get text embedding
        text_embedding = torch.zeros(768)
        if self.use_text_embeddings and self.embeddings is not None:
            if self.embedding_mapping and sample_id in self.embedding_mapping:
                emb_idx = self.embedding_mapping[sample_id]
                text_embedding = torch.from_numpy(
                    self.embeddings[emb_idx].astype(np.float32)
                )

        # Get numeric features
        numeric_features = torch.zeros(len(NUMERIC_FEATURES))
        if self.use_numeric_features:
            end_idx = sample["end_idx"]
            row = self.indicators_df.iloc[end_idx]
            feat = [row.get(f, 0.0) for f in NUMERIC_FEATURES]
            feat = np.array(feat, dtype=np.float32)
            feat = np.nan_to_num(feat, nan=0.0)

            # Normalize if stats are set
            if self._feature_means is not None and self._feature_stds is not None:
                feat = (feat - self._feature_means) / self._feature_stds

            numeric_features = torch.from_numpy(feat)

        return {
            "image": image,
            "text_embedding": text_embedding,
            "numeric_features": numeric_features,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "forward_return": torch.tensor(sample["forward_return"], dtype=torch.float32),
            "end_date": str(sample["end_date"]),
            "sample_id": sample_id,
        }


def get_dataloaders(
    config: dict,
    batch_size: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.

    Args:
        config: Configuration dictionary
        batch_size: Override batch size from config

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    ticker = config["ticker"]
    batch_size = batch_size or config.get("batch_size", 64)

    # Load data
    samples_df = pd.read_parquet("data/samples/samples.parquet")
    indicators_df = pd.read_csv(f"data/raw/{ticker}_indicators.csv", parse_dates=["Date"])

    # Split by split column
    train_df = samples_df[samples_df["split"] == "train"]
    val_df = samples_df[samples_df["split"] == "val"]
    test_df = samples_df[samples_df["split"] == "test"]

    print(f"Dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Paths
    images_dir = "data/images"
    embeddings_path = "data/embeddings/finbert.npy"
    embedding_mapping_path = "data/embeddings/sample_id_to_idx.json"

    # Create datasets
    train_dataset = FusionDataset(
        samples_df=train_df,
        indicators_df=indicators_df,
        images_dir=images_dir,
        embeddings_path=embeddings_path,
        embedding_mapping_path=embedding_mapping_path,
        use_text_embeddings=config.get("use_text_embeddings", True),
        use_numeric_features=config.get("use_numeric_features", True),
        image_size=config.get("image_size", 112),
    )

    # Compute normalization stats from training set
    train_means, train_stds = train_dataset.compute_feature_stats()
    train_dataset.set_feature_stats(train_means, train_stds)

    # Create val/test datasets with same normalization
    val_dataset = FusionDataset(
        samples_df=val_df,
        indicators_df=indicators_df,
        images_dir=images_dir,
        embeddings_path=embeddings_path,
        embedding_mapping_path=embedding_mapping_path,
        use_text_embeddings=config.get("use_text_embeddings", True),
        use_numeric_features=config.get("use_numeric_features", True),
        image_size=config.get("image_size", 112),
    )
    val_dataset.set_feature_stats(train_means, train_stds)

    test_dataset = FusionDataset(
        samples_df=test_df,
        indicators_df=indicators_df,
        images_dir=images_dir,
        embeddings_path=embeddings_path,
        embedding_mapping_path=embedding_mapping_path,
        use_text_embeddings=config.get("use_text_embeddings", True),
        use_numeric_features=config.get("use_numeric_features", True),
        image_size=config.get("image_size", 112),
    )
    test_dataset.set_feature_stats(train_means, train_stds)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
