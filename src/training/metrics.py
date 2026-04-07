"""
Evaluation metrics for histopathology classification.

Why these specific metrics:

- **Macro-F1**: The primary metric. Averages F1 across all classes equally,
  so minority classes (tumor, necrosis) count as much as majority classes
  (stroma). Unlike accuracy, a model that always predicts "stroma" would
  get 0 F1 on tumor/immune/necrosis → low macro-F1.

- **AUROC**: Threshold-independent metric. Measures how well the model
  ranks positive examples above negative ones, regardless of the
  classification threshold. Useful for comparing models without
  committing to a specific operating point.

- **Confusion Matrix**: Visual diagnostic tool. Reveals which classes
  are confused with each other (e.g., tumor ↔ immune confusion suggests
  the model struggles with nuclear density patterns).
"""

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix,
)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score across all classes.

    Macro-F1 computes F1 for each class independently, then takes the
    unweighted mean. This treats all classes equally regardless of their
    frequency, making it the right metric for imbalanced datasets where
    minority class performance matters.

    Args:
        y_true: ground truth labels, shape (N,)
        y_pred: predicted labels, shape (N,)

    Returns:
        Macro-averaged F1 score in [0, 1]
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_auroc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_classes: int = 4,
) -> float:
    """
    Compute macro-averaged AUROC using one-vs-rest strategy.

    AUROC is threshold-independent: it measures the probability that a
    randomly chosen positive example is ranked higher than a randomly
    chosen negative example. A value of 0.5 means random, 1.0 means
    perfect ranking.

    Why AUROC over accuracy:
    - Accuracy depends on a fixed threshold (0.5 by default)
    - AUROC evaluates the model across ALL possible thresholds
    - Better for comparing models when the optimal threshold differs

    Args:
        y_true: ground truth labels, shape (N,)
        y_scores: predicted probabilities, shape (N, num_classes)
        num_classes: number of classes

    Returns:
        Macro-averaged AUROC in [0, 1]
    """
    # One-hot encode y_true
    y_true_onehot = np.zeros((len(y_true), num_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    try:
        return float(roc_auc_score(
            y_true_onehot, y_scores, average="macro", multi_class="ovr"
        ))
    except ValueError:
        # Can happen if a class has no samples in the batch
        return 0.0


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a confusion matrix as a seaborn heatmap, normalized by row.

    Row normalization shows recall per class: for each true class, what
    fraction was predicted as each class. This is more informative than
    raw counts for imbalanced datasets because it shows per-class
    performance independent of class frequency.

    Args:
        y_true: ground truth labels, shape (N,)
        y_pred: predicted labels, shape (N,)
        class_names: list of class name strings
        save_path: if provided, save figure to this path
        normalize: if True, normalize by row (true class)

    Returns:
        matplotlib Figure object
    """
    cm = sk_confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        annot_data = cm_normalized
        fmt = ".2f"
        title = "Confusion Matrix (Row-Normalized)"
    else:
        annot_data = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        annot_data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


class MetricTracker:
    """
    Tracks training and validation metrics across epochs.

    Stores loss, macro-F1, and AUROC for both train and validation sets,
    and provides plotting functionality for training curves.

    Usage:
        tracker = MetricTracker()
        for epoch in range(num_epochs):
            tracker.update("train", epoch, loss=0.5, macro_f1=0.7, auroc=0.85)
            tracker.update("val", epoch, loss=0.6, macro_f1=0.65, auroc=0.82)
        tracker.plot_training_curves("training_curves.png")
    """

    def __init__(self):
        self.history = {
            "train": {"loss": [], "macro_f1": [], "auroc": []},
            "val": {"loss": [], "macro_f1": [], "auroc": []},
        }
        self.epochs = []

    def update(
        self,
        phase: str,
        epoch: int,
        loss: float,
        macro_f1: float,
        auroc: float = 0.0,
    ):
        """Record metrics for a given phase and epoch."""
        if phase == "train" and epoch not in self.epochs:
            self.epochs.append(epoch)
        self.history[phase]["loss"].append(loss)
        self.history[phase]["macro_f1"].append(macro_f1)
        self.history[phase]["auroc"].append(auroc)

    def plot_training_curves(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot loss and macro-F1 training curves.

        Shows train vs validation curves to diagnose:
        - Underfitting: both curves are high/flat
        - Overfitting: train curve improves but val curve diverges
        - Good fit: both curves converge

        Args:
            save_path: if provided, save figure to this path

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        axes[0].plot(self.epochs, self.history["train"]["loss"], "b-", label="Train")
        axes[0].plot(self.epochs, self.history["val"]["loss"], "r-", label="Val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Macro-F1 curves
        axes[1].plot(self.epochs, self.history["train"]["macro_f1"], "b-", label="Train")
        axes[1].plot(self.epochs, self.history["val"]["macro_f1"], "r-", label="Val")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].set_title("Macro-F1")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AUROC curves
        axes[2].plot(self.epochs, self.history["train"]["auroc"], "b-", label="Train")
        axes[2].plot(self.epochs, self.history["val"]["auroc"], "r-", label="Val")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUROC")
        axes[2].set_title("AUROC")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("Training Curves", y=1.02, fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
