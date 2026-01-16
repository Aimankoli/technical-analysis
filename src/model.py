"""
Fusion Model for multi-modal stock prediction.

Combines:
- CNN backbone (ResNet18) for chart images
- MLP for FinBERT text embeddings
- MLP for numeric features

Usage:
    from src.model import FusionModel
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageBranch(nn.Module):
    """CNN branch for processing chart images."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        output_dim: int = 512,
    ):
        super().__init__()

        # Load pretrained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features  # 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features  # 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features  # 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove the final classification layer
        self.backbone.fc = nn.Identity()

        # Project to output dimension if needed
        if in_features != output_dim:
            self.projection = nn.Linear(in_features, output_dim)
        else:
            self.projection = nn.Identity()

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, H, W)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        features = self.backbone(x)
        return self.projection(features)


class TextBranch(nn.Module):
    """MLP branch for processing FinBERT text embeddings."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Text embedding tensor of shape (B, 768)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        return self.mlp(x)


class NumericBranch(nn.Module):
    """MLP branch for processing numeric features."""

    def __init__(
        self,
        input_dim: int = 8,  # Number of numeric features
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Numeric features tensor of shape (B, input_dim)

        Returns:
            Feature tensor of shape (B, output_dim)
        """
        return self.mlp(x)


class FusionModel(nn.Module):
    """
    Multi-modal fusion model combining image, text, and numeric branches.

    Architecture:
        - Image: ResNet18 -> 512-dim
        - Text: MLP 768 -> 256 -> 128
        - Numeric: MLP 8 -> 64 -> 32
        - Concat -> MLP -> sigmoid output
    """

    def __init__(
        self,
        use_image: bool = True,
        use_text: bool = True,
        use_numeric: bool = True,
        cnn_backbone: str = "resnet18",
        num_numeric_features: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.use_image = use_image
        self.use_text = use_text
        self.use_numeric = use_numeric

        # Initialize enabled branches
        fused_dim = 0

        if use_image:
            self.image_branch = ImageBranch(
                backbone=cnn_backbone,
                pretrained=True,
                output_dim=512,
            )
            fused_dim += self.image_branch.output_dim

        if use_text:
            self.text_branch = TextBranch(
                input_dim=768,
                hidden_dim=256,
                output_dim=128,
                dropout=dropout,
            )
            fused_dim += self.text_branch.output_dim

        if use_numeric:
            self.numeric_branch = NumericBranch(
                input_dim=num_numeric_features,
                hidden_dim=64,
                output_dim=32,
                dropout=dropout,
            )
            fused_dim += self.numeric_branch.output_dim

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Single output for binary classification
        )

    def forward(
        self,
        image: torch.Tensor = None,
        text_embedding: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass through fusion model.

        Args:
            image: (B, 3, H, W) tensor
            text_embedding: (B, 768) tensor
            numeric_features: (B, num_features) tensor

        Returns:
            Logits tensor of shape (B, 1)
        """
        features = []

        if self.use_image and image is not None:
            img_feat = self.image_branch(image)
            features.append(img_feat)

        if self.use_text and text_embedding is not None:
            text_feat = self.text_branch(text_embedding)
            features.append(text_feat)

        if self.use_numeric and numeric_features is not None:
            num_feat = self.numeric_branch(numeric_features)
            features.append(num_feat)

        # Concatenate all features
        fused = torch.cat(features, dim=1)

        # Classification
        logits = self.head(fused)
        return logits

    def get_probs(
        self,
        image: torch.Tensor = None,
        text_embedding: torch.Tensor = None,
        numeric_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(image, text_embedding, numeric_features)
        return torch.sigmoid(logits)


def create_model(config: dict) -> FusionModel:
    """Create fusion model from config."""
    from src.dataset import NUMERIC_FEATURES

    return FusionModel(
        use_image=True,  # Always use image
        use_text=config.get("use_text_embeddings", True),
        use_numeric=config.get("use_numeric_features", True),
        cnn_backbone=config.get("cnn_backbone", "resnet18"),
        num_numeric_features=len(NUMERIC_FEATURES),
        dropout=config.get("dropout", 0.3),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
