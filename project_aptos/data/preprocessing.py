"""
Preprocessing utilities for retinal fundus images.

Implements the preprocessing pipeline matching the base paper:
- Resize to 224x224
- ImageNet normalization
"""

from typing import Tuple

from torchvision import transforms


def get_preprocessing_transform(
    image_size: int = 224,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get the basic preprocessing transform (no augmentation).
    Used for validation and test sets.

    Args:
        image_size: Target image size (paper uses 224).
        normalize_mean: Per-channel mean for normalization.
        normalize_std: Per-channel std for normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


def get_inverse_normalize(
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Normalize:
    """
    Get inverse normalization transform for visualization.
    """
    inv_mean = [-m / s for m, s in zip(normalize_mean, normalize_std)]
    inv_std = [1.0 / s for s in normalize_std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)
