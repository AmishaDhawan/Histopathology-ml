#!/usr/bin/env python3
"""
CLI: Train ResNet-50 for histopathology tile classification.

Usage:
    python scripts/train_resnet.py \
        --config configs/resnet_config.yaml \
        --data_dir data/ \
        --output_dir outputs/resnet/ \
        --device cuda

    # Quick test run (2 epochs, CPU):
    python scripts/train_resnet.py \
        --config configs/resnet_config.yaml \
        --data_dir data/ \
        --output_dir outputs/resnet/ \
        --device cpu \
        --epochs 2
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import HistoDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.resnet_baseline import ResNet50Classifier
from src.training.losses import FocalLoss
from src.training.metrics import (
    compute_macro_f1,
    compute_auroc,
    plot_confusion_matrix,
    MetricTracker,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    macro_f1 = compute_macro_f1(all_labels, all_preds)
    auroc = compute_auroc(all_labels, all_probs)

    return epoch_loss, macro_f1, auroc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    macro_f1 = compute_macro_f1(all_labels, all_preds)
    auroc = compute_auroc(all_labels, all_probs)

    return epoch_loss, macro_f1, auroc, all_preds, all_labels


def main():
    args = parse_args()
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Override epochs if specified via CLI
    num_epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    image_size = config["data"]["image_size"]
    patience = config["training"]["early_stopping_patience"]

    # Build datasets
    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    train_dataset = HistoDataset(
        manifest_path, split="train", transform=get_train_transforms(image_size)
    )
    val_dataset = HistoDataset(
        manifest_path, split="val", transform=get_val_transforms(image_size)
    )
    test_dataset = HistoDataset(
        manifest_path, split="test", transform=get_val_transforms(image_size)
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Weighted random sampler for class imbalance
    sample_weights = train_dataset.get_class_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    # Use fewer workers on CPU to avoid overhead
    num_workers = min(config["data"].get("num_workers", 4), 2) if args.device == "cpu" else config["data"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=config["data"].get("pin_memory", True) and args.device != "cpu",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"].get("pin_memory", True) and args.device != "cpu",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config["data"].get("pin_memory", True) and args.device != "cpu",
    )

    # Build model
    model = ResNet50Classifier(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout_rate"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function
    class_weights = train_dataset.get_class_weight_tensor().to(device)
    if config["loss"]["type"] == "focal":
        criterion = FocalLoss(
            gamma=config["loss"]["gamma"],
            alpha=class_weights,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # LR scheduler
    if config["training"]["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
    else:
        scheduler = None

    # Training loop
    tracker = MetricTracker()
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    print(f"\nTraining for {num_epochs} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Train F1':>8} | {'Val F1':>6} | {'Val AUROC':>9} | {'LR':>10}")
    print("-" * 80)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_f1, train_auroc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_f1, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        # Track metrics
        tracker.update("train", epoch, train_loss, train_f1, train_auroc)
        tracker.update("val", epoch, val_loss, val_f1, val_auroc)

        elapsed = time.time() - start_time
        print(
            f"{epoch + 1:>5d} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
            f"{train_f1:>8.4f} | {val_f1:>6.4f} | {val_auroc:>9.4f} | {current_lr:>10.2e} | {elapsed:.1f}s"
        )

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_auroc": val_auroc,
                    "config": config,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    # Save final checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        os.path.join(args.output_dir, "final_model.pt"),
    )

    # Plot training curves
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    tracker.plot_training_curves(curves_path)
    print(f"\nTraining curves saved to {curves_path}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    # Load best model for test evaluation
    best_ckpt = torch.load(
        os.path.join(args.output_dir, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_f1, test_auroc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    # Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names=test_dataset.class_names,
        save_path=cm_path,
    )
    print(f"Confusion matrix saved to {cm_path}")

    # Final summary
    accuracy = (test_preds == test_labels).mean()
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Metric':<15} | {'Train':>8} | {'Val':>8} | {'Test':>8}")
    print("-" * 50)
    print(f"{'Loss':<15} | {tracker.history['train']['loss'][-1]:>8.4f} | {tracker.history['val']['loss'][-1]:>8.4f} | {test_loss:>8.4f}")
    print(f"{'Macro-F1':<15} | {tracker.history['train']['macro_f1'][-1]:>8.4f} | {tracker.history['val']['macro_f1'][-1]:>8.4f} | {test_f1:>8.4f}")
    print(f"{'AUROC':<15} | {tracker.history['train']['auroc'][-1]:>8.4f} | {tracker.history['val']['auroc'][-1]:>8.4f} | {test_auroc:>8.4f}")
    print(f"{'Accuracy':<15} | {'—':>8} | {'—':>8} | {accuracy:>8.4f}")
    print("=" * 60)
    print(f"Best val macro-F1: {best_val_f1:.4f} (epoch {best_ckpt['epoch'] + 1})")
    print("Done!")


if __name__ == "__main__":
    main()
