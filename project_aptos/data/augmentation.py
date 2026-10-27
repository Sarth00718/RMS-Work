"""
Data augmentation pipelines for retinal fundus image classification.

Provides multiple augmentation levels:
- Baseline: minimal (paper reproduction)
- Enhanced: stronger augmentation for better generalization
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


def get_enhanced_augmentation(
    image_size: int = 224,
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Enhanced augmentation for better generalization on retinal images.
    Includes rotation, color jitter, vertical flip, and random crop.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
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
        name: 'baseline' or 'enhanced'.
    """
    if name == "baseline":
        return get_baseline_augmentation(image_size, normalize_mean, normalize_std)
    elif name == "enhanced":
        return get_enhanced_augmentation(image_size, normalize_mean, normalize_std)
    else:
        raise ValueError(f"Unknown augmentation: {name}. Use 'baseline' or 'enhanced'.")
