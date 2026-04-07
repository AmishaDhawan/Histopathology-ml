#!/usr/bin/env python3
"""
CLI: Train ViT-Base with FlashAttention for histopathology tile classification.

Usage:
    # Single GPU/CPU training:
    python scripts/train_vit.py \
        --config configs/vit_config.yaml \
        --data_dir data/ \
        --output_dir outputs/vit/ \
        --use_flash_attention true \
        --device cuda

    # Quick test run (2 epochs, CPU):
    python scripts/train_vit.py \
        --config configs/vit_config.yaml \
        --data_dir data/ \
        --output_dir outputs/vit/ \
        --device cpu \
        --epochs 2

    # Distributed training (DDP):
    python -m torch.distributed.launch --nproc_per_node=4 \
        scripts/train_vit.py \
        --config configs/vit_config.yaml \
        --data_dir data/ \
        --output_dir outputs/vit/ \
        --distributed
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import HistoDataset
from src.models.vit_model import ViTHistoClassifier, measure_attention_memory
from src.training.losses import FocalLoss
from src.training.metrics import (
    MetricTracker,
    compute_auroc,
    compute_macro_f1,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT-Base with FlashAttention")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument(
        "--use_flash_attention",
        type=str,
        default=None,
        help="Override flash attention setting (true/false)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training with DDP",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by torch.distributed.launch)",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_memory(step_name, device):
    """Log memory usage at a given step (GPU via torch.cuda, CPU via psutil RSS)."""
    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"  [Memory @ {step_name}] RSS: {rss_mb:.1f} MB", end="")
    if torch.cuda.is_available() and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        print(f" | GPU Allocated: {allocated:.1f} MB | GPU Reserved: {reserved:.1f} MB", end="")
    print()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, warmup_epochs, base_lr):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Linear warmup for ViT stability
        if epoch < warmup_epochs:
            warmup_factor = (epoch * len(loader) + batch_idx + 1) / (warmup_epochs * len(loader))
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * warmup_factor

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for ViT training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

        # Log memory every 10 steps
        if batch_idx % 10 == 0:
            log_memory(f"epoch {epoch + 1}, step {batch_idx}", device)

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


def save_attention_maps(model, val_dataset, device, output_dir, class_names):
    """
    Save attention map visualizations for 4 validation samples (one per class).

    Attention maps show WHAT the model attends to. For histopathology,
    we expect high attention on cell nuclei and tissue boundaries, not background.
    """
    print("\nGenerating attention map visualizations...")

    # Find one sample per class
    class_samples = {}
    for idx in range(len(val_dataset)):
        image, label = val_dataset[idx]
        if label not in class_samples and len(class_samples) < 4:
            class_samples[label] = (image, label)
        if len(class_samples) == 4:
            break

    if not class_samples:
        print("  Warning: No validation samples found for attention maps")
        return

    fig, axes = plt.subplots(len(class_samples), 2, figsize=(10, 5 * len(class_samples)))
    if len(class_samples) == 1:
        axes = axes.reshape(1, -1)

    for row, (label_idx, (image, label)) in enumerate(sorted(class_samples.items())):
        # Get attention maps
        input_tensor = image.unsqueeze(0).to(device)
        attn_maps = model.get_attention_maps(input_tensor)

        if attn_maps is None:
            print("  Warning: Could not extract attention maps")
            return

        # Average across attention heads: (num_heads, N+1, N+1) -> (N+1, N+1)
        attn_avg = attn_maps[0].mean(dim=0).cpu().numpy()

        # Get CLS token attention to all patches (row 0, skip CLS column)
        cls_attn = attn_avg[0, 1:]  # (N,)

        # Reshape to 2D grid
        grid_size = int(cls_attn.shape[0] ** 0.5)
        cls_attn_2d = cls_attn.reshape(grid_size, grid_size)

        # Denormalize image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        # Plot original image
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Original: {class_names[label_idx]}")
        axes[row, 0].axis("off")

        # Plot attention heatmap overlay
        axes[row, 1].imshow(img_np)
        # Resize attention map to image size for overlay
        from PIL import Image as PILImage
        attn_resized = np.array(
            PILImage.fromarray(
                (cls_attn_2d * 255).astype(np.uint8)
            ).resize((img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR)
        ) / 255.0
        axes[row, 1].imshow(attn_resized, cmap="jet", alpha=0.5)
        axes[row, 1].set_title(f"Attention: {class_names[label_idx]}")
        axes[row, 1].axis("off")

    plt.suptitle(
        "ViT Attention Maps (CLS token → patch attention, averaged across heads)",
        fontsize=12,
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "attention_maps.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Attention maps saved to {save_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Override settings from CLI
    num_epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    image_size = config["data"]["image_size"]
    patience = config["training"]["early_stopping_patience"]
    warmup_epochs = config["training"].get("warmup_epochs", 10)
    dropout = config["training"].get("dropout", 0.1)

    use_flash = config["model"].get("use_flash_attention", True)
    if args.use_flash_attention is not None:
        use_flash = args.use_flash_attention.lower() == "true"

    # Print memory analysis before model creation
    print("\n--- FlashAttention Memory Analysis ---")
    for seq_len in [197, 1025, 4097]:
        mem = measure_attention_memory(seq_len)
        print(
            f"  seq_len={seq_len:>5d}: Standard={mem['standard_attention_mb']:.2f} MB, "
            f"Flash={mem['flash_attention_mb']:.2f} MB, "
            f"Reduction={mem['reduction_factor']:.1f}x"
        )
    print("--------------------------------------\n")

    # Generate synthetic data if manifest doesn't exist
    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    if not os.path.exists(manifest_path):
        print("Generating synthetic data...")
        download_script = os.path.join(args.data_dir, "download_patches.py")
        if os.path.exists(download_script):
            import subprocess
            subprocess.run([sys.executable, download_script], check=True)
        else:
            raise FileNotFoundError(
                f"manifest.csv not found at {manifest_path} and no download script available"
            )

    # Build datasets
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
    num_workers = (
        min(config["data"].get("num_workers", 4), 2)
        if args.device == "cpu"
        else config["data"].get("num_workers", 4)
    )

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
    log_memory("before model creation", device)

    model = ViTHistoClassifier(
        num_classes=config["model"]["num_classes"],
        use_flash_attention=use_flash,
        patch_size=config["model"]["patch_size"],
        image_size=image_size,
        pretrained=config["model"]["pretrained"],
        dropout=dropout,
    ).to(device)

    log_memory("after model creation", device)

    # Print parameter breakdown
    param_info = model.count_parameters()
    print("\nViT-Base Model Parameters:")
    for key, value in param_info.items():
        print(f"  {key}: {value:,}")
    print(f"  FlashAttention: {'enabled' if use_flash else 'disabled'}")

    # Loss function
    class_weights = train_dataset.get_class_weight_tensor().to(device)
    if config["loss"]["type"] == "focal":
        criterion = FocalLoss(
            gamma=config["loss"]["gamma"],
            alpha=class_weights,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer: AdamW with higher weight decay for ViT
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # LR scheduler: cosine annealing (applied after warmup)
    if config["training"]["lr_scheduler"] == "cosine" and num_epochs > warmup_epochs:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs
        )
    else:
        scheduler = None

    # Training loop
    tracker = MetricTracker()
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    print(f"\nTraining ViT-Base for {num_epochs} epochs...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | "
          f"{'Train F1':>8} | {'Val F1':>6} | {'Val AUROC':>9} | {'LR':>10} | {'Time':>6}")
    print("-" * 85)

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_f1, train_auroc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch, warmup_epochs, lr,
        )

        # Validate
        val_loss, val_f1, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Update scheduler (after warmup period)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()

        # Track metrics
        tracker.update("train", epoch, train_loss, train_f1, train_auroc)
        tracker.update("val", epoch, val_loss, val_f1, val_auroc)

        elapsed = time.time() - start_time
        print(
            f"{epoch + 1:>5d} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
            f"{train_f1:>8.4f} | {val_f1:>6.4f} | {val_auroc:>9.4f} | "
            f"{current_lr:>10.2e} | {elapsed:.1f}s"
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
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(no improvement for {patience} epochs)")
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

    # Save attention maps from the last layer for 4 validation samples
    save_attention_maps(model, val_dataset, device, args.output_dir, test_dataset.class_names)

    # Final summary
    accuracy = (test_preds == test_labels).mean()
    print("\n" + "=" * 65)
    print("FINAL RESULTS — ViT-Base + FlashAttention")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Train':>8} | {'Val':>8} | {'Test':>8}")
    print("-" * 55)
    print(
        f"{'Loss':<15} | {tracker.history['train']['loss'][-1]:>8.4f} | "
        f"{tracker.history['val']['loss'][-1]:>8.4f} | {test_loss:>8.4f}"
    )
    print(
        f"{'Macro-F1':<15} | {tracker.history['train']['macro_f1'][-1]:>8.4f} | "
        f"{tracker.history['val']['macro_f1'][-1]:>8.4f} | {test_f1:>8.4f}"
    )
    print(
        f"{'AUROC':<15} | {tracker.history['train']['auroc'][-1]:>8.4f} | "
        f"{tracker.history['val']['auroc'][-1]:>8.4f} | {test_auroc:>8.4f}"
    )
    print(f"{'Accuracy':<15} | {'---':>8} | {'---':>8} | {accuracy:>8.4f}")
    print("=" * 65)
    print(f"Best val macro-F1: {best_val_f1:.4f} (epoch {best_ckpt['epoch'] + 1})")
    print(f"FlashAttention: {'enabled' if use_flash else 'disabled'}")
    log_memory("end of training", device)
    print("Done!")


if __name__ == "__main__":
    main()
