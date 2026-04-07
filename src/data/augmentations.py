"""
Domain-specific augmentations for histopathology tile classification.

Histopathology images have unique properties:
- No canonical orientation: tissue can be rotated arbitrarily on the slide
- Stain variation: H&E staining intensity varies between labs and protocols
- Scale consistency: tiles are extracted at fixed magnification

The augmentation parameters are intentionally conservative:
- Color jitter is mild because histopathology stain intensity variation
  is less extreme than natural image variation, and aggressive jitter
  could wash out class-discriminative color features (e.g., purple nuclei
  vs. pink stroma)
- Rotation is restricted to 90-degree increments (not arbitrary angles)
  because tissue has no canonical orientation but we want to avoid
  interpolation artifacts from fractional rotations
"""

from torchvision import transforms as T


# ImageNet normalization — used because we fine-tune from ImageNet-pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Training augmentations for histopathology tiles.

    Pipeline:
    1. Resize to image_size (from 1024x1024 source tiles)
    2. Random horizontal flip (tissue has no left/right orientation)
    3. Random 90-degree rotation (tissue has no canonical orientation)
    4. Conservative color jitter (mild stain variation simulation)
    5. Convert to tensor [0, 1]
    6. Normalize to ImageNet statistics

    Args:
        image_size: target spatial resolution (default 224 for ResNet/ViT)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=(180, 180))], p=0.25),
        T.RandomApply([T.RandomRotation(degrees=(270, 270))], p=0.25),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """
    Validation/test transforms for histopathology tiles.

    No augmentation — only resize and normalize for deterministic evaluation.

    Args:
        image_size: target spatial resolution (default 224 for ResNet/ViT)

    Returns:
        torchvision.transforms.Compose pipeline
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
