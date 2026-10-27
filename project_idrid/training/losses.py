"""
Loss functions for DR classification.

Includes:
- Standard CrossEntropy
"""

import torch
import torch.nn as nn
from typing import Optional


def create_loss(
    name: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    device: str = "cpu",
) -> nn.Module:
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss: {name}")
