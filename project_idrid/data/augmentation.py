"""
Data augmentation pipelines for retinal fundus image classification.

Provides multiple augmentation levels:
- Baseline: minimal (paper reproduction)
- Moderate: standard medical imaging augmentations
- Strong: aggressive augmentation for small datasets
"""

from typing import Tuple

from torchvision import transforms


def get_baseline_augmentation(
    image_size: int = 224,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Baseline augmentation (paper reproduction).
    Minimal augmentation: resize + horizontal flip + normalize.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])


def get_augmentation_by_name(
    name: str,
    image_size: int = 224,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Get augmentation pipeline by name.
    
    Args:
        name: Must be 'baseline'.
    """
    if name != "baseline":
        raise ValueError(f"Unknown augmentation: {name}. Only 'baseline' is supported.")
    
    return get_baseline_augmentation(image_size, normalize_mean, normalize_std)
