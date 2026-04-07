#!/usr/bin/env python3
"""
Synthetic Histopathology Dataset Generator

Generates 2,000 synthetic 1024x1024 RGB PNG tiles mimicking histopathology
tissue classes: tumor, immune, stroma, necrosis.

Each class has visually distinct color/texture characteristics designed to
approximate real H&E-stained tissue appearance. The class distribution is
intentionally imbalanced to mirror real pathology datasets.

Usage:
    python data/download_patches.py [--output_dir data/] [--num_tiles 2000]
"""

import argparse
import os
import csv
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# Class distribution mirrors real pathology data imbalance
CLASS_CONFIG = {
    "tumor": 0.15,     # Dense purple-stained nuclei
    "immune": 0.20,    # Scattered small dark circular nuclei
    "stroma": 0.55,    # Fibrous pink texture
    "necrosis": 0.10,  # Pale ghost-cell appearance
}

TILE_SIZE = 1024
NUM_CHANNELS = 3
SEED = 42


def generate_tumor_tile(rng):
    """
    Tumor: dense purple-stained nuclei pattern.
    Pink background with dense purple circular blobs using PIL draw for speed.
    HSV: H=270-310, S=0.6-0.9, V=0.3-0.6 overlay on pink background.
    """
    bg_r = int(rng.integers(210, 240))
    bg_g = int(rng.integers(180, 210))
    bg_b = int(rng.integers(200, 225))
    img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    num_blobs = int(rng.integers(200, 500))
    for _ in range(num_blobs):
        cx = int(rng.integers(0, TILE_SIZE))
        cy = int(rng.integers(0, TILE_SIZE))
        r = int(rng.integers(8, 25))
        cr = int(rng.integers(80, 140))
        cg = int(rng.integers(30, 80))
        cb = int(rng.integers(100, 170))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(cr, cg, cb))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return np.array(img)


def generate_immune_tile(rng):
    """
    Immune: scattered small dark circular nuclei.
    Light pinkish background with sparse small dark blue-purple dots.
    HSV: H=240-270, small radius (r=3-8px).
    """
    bg_r = int(rng.integers(220, 245))
    bg_g = int(rng.integers(210, 235))
    bg_b = int(rng.integers(220, 240))
    img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    num_blobs = int(rng.integers(80, 200))
    for _ in range(num_blobs):
        cx = int(rng.integers(0, TILE_SIZE))
        cy = int(rng.integers(0, TILE_SIZE))
        r = int(rng.integers(3, 9))
        cr = int(rng.integers(30, 70))
        cg = int(rng.integers(20, 60))
        cb = int(rng.integers(80, 140))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(cr, cg, cb))

    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return np.array(img)


def generate_stroma_tile(rng):
    """
    Stroma: fibrous pink texture.
    Use superposed sine waves in pink range (HSV: H=330-360, S=0.2-0.5).
    """
    base_r = int(rng.integers(200, 230))
    base_g = int(rng.integers(160, 190))
    base_b = int(rng.integers(170, 200))
    tile = np.full((TILE_SIZE, TILE_SIZE, NUM_CHANNELS),
                   [base_r, base_g, base_b], dtype=np.float32)

    yy, xx = np.mgrid[0:TILE_SIZE, 0:TILE_SIZE].astype(np.float32)

    num_waves = int(rng.integers(5, 15))
    for _ in range(num_waves):
        angle = float(rng.uniform(0, np.pi))
        freq = float(rng.uniform(0.005, 0.03))
        phase = float(rng.uniform(0, 2 * np.pi))
        amplitude = float(rng.uniform(15, 40))
        rotated = xx * np.cos(angle) + yy * np.sin(angle)
        wave = amplitude * np.sin(2 * np.pi * freq * rotated + phase)

        wave_r = float(rng.integers(200, 240))
        wave_g = float(rng.integers(150, 190))
        wave_b = float(rng.integers(170, 210))

        blend = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8) * 0.3
        tile[:, :, 0] = tile[:, :, 0] * (1 - blend) + wave_r * blend
        tile[:, :, 1] = tile[:, :, 1] * (1 - blend) + wave_g * blend
        tile[:, :, 2] = tile[:, :, 2] * (1 - blend) + wave_b * blend

    return np.clip(tile, 0, 255).astype(np.uint8)


def generate_necrosis_tile(rng):
    """
    Necrosis: pale, ghost-cell appearance.
    Near-white with faint ring-shaped cell outlines using PIL draw.
    HSV: S=0.0-0.15, V=0.85-0.95.
    """
    bg_r = int(rng.integers(225, 245))
    bg_g = int(rng.integers(225, 242))
    bg_b = int(rng.integers(220, 240))
    img = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    num_cells = int(rng.integers(30, 80))
    for _ in range(num_cells):
        cx = int(rng.integers(0, TILE_SIZE))
        cy = int(rng.integers(0, TILE_SIZE))
        r = int(rng.integers(15, 40))
        outline_color = (
            int(rng.integers(200, 225)),
            int(rng.integers(195, 220)),
            int(rng.integers(195, 218)),
        )
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=outline_color,
            width=2,
        )

    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, 3, arr.shape).astype(np.float32)
    arr += noise
    return np.clip(arr, 0, 255).astype(np.uint8)


GENERATORS = {
    "tumor": generate_tumor_tile,
    "immune": generate_immune_tile,
    "stroma": generate_stroma_tile,
    "necrosis": generate_necrosis_tile,
}


def stratified_split(labels, train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """Stratified split into train/val/test maintaining class proportions."""
    rng = np.random.default_rng(seed)
    splits = {}

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        for idx in indices[:n_train]:
            splits[idx] = "train"
        for idx in indices[n_train:n_train + n_val]:
            splits[idx] = "val"
        for idx in indices[n_train + n_val:]:
            splits[idx] = "test"

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic histopathology dataset")
    parser.add_argument("--output_dir", type=str, default="data/", help="Output directory")
    parser.add_argument("--num_tiles", type=int, default=2000, help="Total number of tiles")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    output_dir = args.output_dir
    tiles_dir = os.path.join(output_dir, "tiles")
    sample_dir = os.path.join(output_dir, "sample")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Compute tile counts per class
    class_counts = {}
    remaining = args.num_tiles
    classes = list(CLASS_CONFIG.keys())
    for cls in classes[:-1]:
        count = int(args.num_tiles * CLASS_CONFIG[cls])
        class_counts[cls] = count
        remaining -= count
    class_counts[classes[-1]] = remaining

    print("=" * 60)
    print("Synthetic Histopathology Dataset Generator")
    print("=" * 60)
    print(f"Total tiles: {args.num_tiles}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}x{NUM_CHANNELS}")
    print(f"Output: {output_dir}")
    print(f"Seed: {args.seed}")
    print()
    print("Target class distribution:")
    for cls, count in class_counts.items():
        pct = count / args.num_tiles * 100
        print(f"  {cls:>10s}: {count:4d} ({pct:5.1f}%)")
    print()

    # Generate tiles
    filepaths = []
    labels = []
    tile_idx = 0

    sample_counts = {cls: 0 for cls in classes}
    samples_per_class = 5  # 20 total sample tiles (5 per class)

    for cls in classes:
        count = class_counts[cls]
        gen_fn = GENERATORS[cls]
        print(f"Generating {count} {cls} tiles...")

        for i in range(count):
            tile = gen_fn(rng)
            filename = f"{cls}_{i:04d}.png"
            filepath = os.path.join(tiles_dir, filename)
            Image.fromarray(tile).save(filepath)

            filepaths.append(os.path.join("tiles", filename))
            labels.append(cls)

            # Save sample tiles
            if sample_counts[cls] < samples_per_class:
                sample_path = os.path.join(sample_dir, filename)
                Image.fromarray(tile).save(sample_path)
                sample_counts[cls] += 1

            tile_idx += 1
            if tile_idx % 100 == 0:
                print(f"  Generated {tile_idx}/{args.num_tiles} tiles")

    print(f"\nAll {args.num_tiles} tiles generated.")

    # Stratified split
    splits = stratified_split(labels)

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        for i in range(len(filepaths)):
            writer.writerow([filepaths[i], labels[i], splits[i]])

    print(f"\nManifest saved to {manifest_path}")

    # Print split summary
    split_counter = Counter(splits.values())
    print("\nSplit distribution:")
    for split in ["train", "val", "test"]:
        count = split_counter[split]
        print(f"  {split:>5s}: {count:4d} ({count / args.num_tiles * 100:.1f}%)")

    # Print class x split distribution
    print("\nClass x Split distribution:")
    print(f"  {'Class':>10s}  {'train':>6s}  {'val':>5s}  {'test':>5s}")
    for cls in classes:
        cls_indices = [i for i, lbl in enumerate(labels) if lbl == cls]
        cls_splits = Counter(splits[i] for i in cls_indices)
        print(f"  {cls:>10s}  {cls_splits.get('train', 0):>6d}  "
              f"{cls_splits.get('val', 0):>5d}  {cls_splits.get('test', 0):>5d}")

    print(f"\nSample tiles saved to {sample_dir}/ ({sum(sample_counts.values())} tiles)")
    print("Done!")


if __name__ == "__main__":
    main()
