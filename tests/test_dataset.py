"""
Unit tests for HistoDataset and augmentations.

Tests:
- Dataset loads correctly and returns correct tensor shapes
- Class weights are valid and sum to reasonable values
- Augmentations preserve tensor dtype and don't introduce NaN
"""

import os
import sys
import csv

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import HistoDataset, CLASS_TO_IDX
from src.data.augmentations import get_train_transforms, get_val_transforms


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal dataset for testing."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    # Create small synthetic tiles (64x64 for speed)
    manifest_rows = []
    classes = ["tumor", "immune", "stroma", "necrosis"]
    counts = [3, 4, 11, 2]  # Imbalanced

    for cls, count in zip(classes, counts):
        for i in range(count):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            filename = f"{cls}_{i:04d}.png"
            filepath = tiles_dir / filename
            Image.fromarray(img).save(filepath)

            split = "train" if i < count - 1 else "val"
            manifest_rows.append((f"tiles/{filename}", cls, split))

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        writer.writerows(manifest_rows)

    return str(manifest_path), str(tmp_path)


class TestHistoDataset:
    def test_loads_correctly(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        dataset = HistoDataset(manifest_path, split="train", data_root=data_root)
        assert len(dataset) > 0

    def test_returns_correct_shapes(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        dataset = HistoDataset(manifest_path, split="train", data_root=data_root)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # C, H, W format
        assert isinstance(label, int)
        assert 0 <= label < len(CLASS_TO_IDX)

    def test_returns_metadata(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        dataset = HistoDataset(
            manifest_path, split="train", return_metadata=True, data_root=data_root
        )
        image, label, metadata = dataset[0]
        assert "filepath" in metadata
        assert "label_str" in metadata
        assert metadata["label_str"] in CLASS_TO_IDX

    def test_class_weights_valid(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        dataset = HistoDataset(manifest_path, split="train", data_root=data_root)
        weights = dataset.get_class_weights()
        assert len(weights) == len(dataset)
        assert (weights > 0).all()
        assert not torch.isnan(weights).any()

    def test_class_weight_tensor(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        dataset = HistoDataset(manifest_path, split="train", data_root=data_root)
        weight_tensor = dataset.get_class_weight_tensor()
        assert weight_tensor.shape[0] == len(CLASS_TO_IDX)
        assert (weight_tensor >= 0).all()

    def test_split_filtering(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        train_ds = HistoDataset(manifest_path, split="train", data_root=data_root)
        val_ds = HistoDataset(manifest_path, split="val", data_root=data_root)
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(train_ds) != len(val_ds)


class TestAugmentations:
    def test_train_transforms_output_shape(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        transform = get_train_transforms(image_size=224)
        dataset = HistoDataset(
            manifest_path, split="train", transform=transform, data_root=data_root
        )
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32

    def test_val_transforms_output_shape(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        transform = get_val_transforms(image_size=224)
        dataset = HistoDataset(
            manifest_path, split="train", transform=transform, data_root=data_root
        )
        image, label = dataset[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32

    def test_no_nan_after_augmentation(self, sample_dataset):
        manifest_path, data_root = sample_dataset
        transform = get_train_transforms(image_size=224)
        dataset = HistoDataset(
            manifest_path, split="train", transform=transform, data_root=data_root
        )
        for i in range(min(5, len(dataset))):
            image, _ = dataset[i]
            assert not torch.isnan(image).any(), f"NaN found in augmented image {i}"
            assert not torch.isinf(image).any(), f"Inf found in augmented image {i}"
