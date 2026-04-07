"""
PyTorch Dataset for histopathology PNG tiles.

Loads tiles from a manifest CSV, applies transforms, and returns
(image_tensor, label_int) pairs for training. Supports class weight
computation for handling the imbalanced class distribution via
WeightedRandomSampler.
"""

import os
from collections import Counter
from typing import Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


CLASS_TO_IDX = {
    "tumor": 0,
    "immune": 1,
    "stroma": 2,
    "necrosis": 3,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class HistoDataset(Dataset):
    """
    PyTorch Dataset for histopathology PNG tiles.

    Args:
        manifest_path: path to manifest.csv
        split: one of 'train', 'val', 'test'
        transform: optional torchvision transform
        return_metadata: if True, also return filepath and label string
        data_root: root directory containing the tiles (parent of manifest)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        transform: Optional[T.Compose] = None,
        return_metadata: bool = False,
        data_root: Optional[str] = None,
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata

        if data_root is None:
            self.data_root = os.path.dirname(manifest_path)
        else:
            self.data_root = data_root

        # Load and filter manifest
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")

        self.filepaths = self.df["filepath"].tolist()
        self.labels = self.df["label"].tolist()
        self.label_indices = [CLASS_TO_IDX[label] for label in self.labels]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, dict]]:
        filepath = os.path.join(self.data_root, self.filepaths[idx])
        label_str = self.labels[idx]
        label_int = self.label_indices[idx]

        # Load PNG with PIL (lossless)
        image = Image.open(filepath).convert("RGB")

        # Convert to tensor and normalize to [0, 1]
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)  # Converts to [0, 1] float32

        if self.return_metadata:
            metadata = {
                "filepath": filepath,
                "label_str": label_str,
            }
            return image, label_int, metadata

        return image, label_int

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency weights for WeightedRandomSampler.

        Returns a weight for EACH SAMPLE (not each class), where samples
        from minority classes get higher weights. This ensures the sampler
        draws samples from all classes roughly equally.

        Returns:
            torch.Tensor of shape (num_samples,) with per-sample weights
        """
        class_counts = Counter(self.label_indices)
        num_samples = len(self.label_indices)
        num_classes = len(CLASS_TO_IDX)

        # Weight per class = total_samples / (num_classes * class_count)
        class_weights = {
            cls: num_samples / (num_classes * count)
            for cls, count in class_counts.items()
        }

        # Assign per-sample weight based on its class
        sample_weights = torch.tensor(
            [class_weights[label] for label in self.label_indices],
            dtype=torch.float64,
        )

        return sample_weights

    def get_class_weight_tensor(self) -> torch.Tensor:
        """
        Compute class weights as a tensor for loss functions (e.g., FocalLoss alpha).

        Returns:
            torch.Tensor of shape (num_classes,) with weight per class
        """
        class_counts = Counter(self.label_indices)
        num_samples = len(self.label_indices)
        num_classes = len(CLASS_TO_IDX)

        weights = torch.zeros(num_classes, dtype=torch.float32)
        for cls_idx, count in class_counts.items():
            weights[cls_idx] = num_samples / (num_classes * count)

        return weights

    @property
    def class_names(self):
        return list(CLASS_TO_IDX.keys())

    @property
    def num_classes(self):
        return len(CLASS_TO_IDX)
