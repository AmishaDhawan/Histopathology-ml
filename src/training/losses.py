"""
Loss functions for class-imbalanced histopathology classification.

In histopathology datasets, class imbalance is the norm:
- Stroma dominates (~55% of tiles)
- Necrosis is rare (~10%)
- Standard cross-entropy is dominated by easy majority-class examples

Focal loss addresses this by down-weighting well-classified examples
and focusing the gradient on hard, misclassified samples.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter. gamma=0 reduces to cross-entropy.
               gamma=2 is standard for class imbalance.
        alpha: class weights tensor. If None, uses uniform weights.
        reduction: 'mean' or 'sum'

    Why focal loss over cross-entropy:
        In imbalanced datasets, standard cross-entropy is dominated by
        easy majority-class examples. Their loss is small but their COUNT
        is large, so they dominate the gradient. Focal loss multiplies the
        loss by (1 - p_t)^gamma, which down-weights easy examples (high p_t)
        and focuses the gradient on hard, misclassified examples.

    When to use focal loss:
        - Class imbalance ratio > 3:1 (ours is ~5.5:1 stroma vs necrosis)
        - When minority class performance is critical (e.g., detecting tumor)
        - When you observe the model converging to always predict the majority class

    When cross-entropy may be sufficient:
        - Balanced datasets
        - When combined with proper sampling strategies (WeightedRandomSampler)
        - When class-weighted cross-entropy achieves adequate minority class recall
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: raw logits of shape (N, C) where C = num_classes
            targets: ground truth labels of shape (N,) with class indices

        Returns:
            Scalar loss value
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Gather the probability of the true class for each sample
        # p_t shape: (N,)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy term: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
