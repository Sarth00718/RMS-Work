"""
Model factory for creating models by name.
"""

import torch.nn as nn

from .base_model import DRResNet50


def create_model(
    architecture: str = "resnet50",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create a model by architecture name.

    Args:
        architecture: Model architecture name.
        num_classes: Number of output classes.
        pretrained: Use pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone layers.

    Returns:
        PyTorch model.
    """
    if architecture == "resnet50":
        return DRResNet50(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
