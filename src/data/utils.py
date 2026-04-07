"""
Utility functions for patch extraction and tissue masking.

These utilities support preprocessing of whole-slide images (WSIs)
into tiles suitable for classification. In a production pipeline,
these would be used upstream of the Dataset class.
"""

import numpy as np
from PIL import Image


def extract_patches(
    image: np.ndarray,
    patch_size: int = 1024,
    stride: int = 1024,
) -> list:
    """
    Extract non-overlapping patches from a large image.

    Args:
        image: numpy array of shape (H, W, C)
        patch_size: size of each square patch
        stride: step size between patches (default = patch_size for non-overlapping)

    Returns:
        List of (patch, row, col) tuples
    """
    h, w = image.shape[:2]
    patches = []

    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patch = image[row:row + patch_size, col:col + patch_size]
            patches.append((patch, row, col))

    return patches


def tissue_mask(
    image: np.ndarray,
    saturation_threshold: float = 0.1,
    brightness_range: tuple = (0.1, 0.95),
) -> np.ndarray:
    """
    Create a binary tissue mask to filter out background (white) regions.

    Tissue regions in H&E slides have higher saturation than background.
    Background is typically white (high value, low saturation).

    Args:
        image: RGB numpy array of shape (H, W, 3), values in [0, 255]
        saturation_threshold: minimum saturation to be considered tissue
        brightness_range: (min, max) value range for tissue

    Returns:
        Binary mask of shape (H, W), True where tissue is present
    """
    # Convert to HSV
    pil_img = Image.fromarray(image.astype(np.uint8))
    hsv = np.array(pil_img.convert("HSV")).astype(np.float32) / 255.0

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (saturation > saturation_threshold)
        & (value > brightness_range[0])
        & (value < brightness_range[1])
    )

    return mask


def compute_tissue_ratio(patch: np.ndarray, threshold: float = 0.1) -> float:
    """
    Compute the fraction of a patch that contains tissue.

    Used to filter out patches that are mostly background.

    Args:
        patch: RGB numpy array
        threshold: saturation threshold for tissue detection

    Returns:
        Float in [0, 1] representing tissue fraction
    """
    mask = tissue_mask(patch, saturation_threshold=threshold)
    return float(mask.sum()) / mask.size
