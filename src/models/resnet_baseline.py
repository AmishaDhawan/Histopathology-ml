"""
ResNet-50 fine-tuned for histopathology tile classification.

This is the Phase 1 baseline model. ResNet-50 is chosen because:
1. Strong ImageNet transfer learning — low/mid-level features (edges,
   textures, color patterns) transfer well to histopathology
2. Well-understood architecture — easy to debug and interpret
3. Fast training — establishes a performance floor quickly
4. Sufficient capacity for 4-class tile classification

The model serves as the baseline that Phase 2 (ViT) aims to surpass.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 fine-tuned for histopathology tile classification.

    Architecture decisions:
    - Initialize from ImageNet weights (torchvision IMAGENET1K_V1)
    - Fine-tune ALL layers (not just classifier head):
        Reason: histopathology features are domain-specific enough that
        freezing the backbone underperforms. Low-level features (edges,
        textures) transfer from ImageNet but mid/high-level features need
        domain adaptation.
    - Replace final FC layer with: Dropout(0.3) -> Linear(2048, num_classes)
    - Dropout added to reduce overfitting on a ~2000 sample dataset

    Why NOT a foundation model at this stage:
        Domain-specific pathology foundation models (UNI, Virchow) were
        trained on H&E staining. Our data is multiplexed fluorescence.
        ImageNet-pretrained ResNet-50 transfers more cleanly as a baseline.
        More importantly: start simple. Establish the baseline first.

    Args:
        num_classes: number of output classes (default 4)
        pretrained: whether to load ImageNet pretrained weights
        dropout_rate: dropout probability before final linear layer
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet50(weights=weights)

        # Get the feature dimension from the original FC layer
        in_features = self.backbone.fc.in_features  # 2048 for ResNet-50

        # Replace the final FC layer with dropout + linear
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input tensor of shape (N, 3, H, W)

        Returns:
            logits of shape (N, num_classes)
        """
        return self.backbone(x)

    def get_feature_extractor(self) -> nn.Module:
        """
        Return the backbone without the final FC layer.

        Useful for extracting features for downstream tasks
        (e.g., feeding into the multimodal VLM in Phase 3).
        """
        # Create a copy without the FC layer
        import copy
        feature_extractor = copy.deepcopy(self.backbone)
        feature_extractor.fc = nn.Identity()
        return feature_extractor
