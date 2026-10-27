"""
Base model implementation following the paper's ResNet50 architecture.

Paper uses ResNet50 pretrained on ImageNet with a modified classifier head
for DR classification (5-class or binary).
"""

import torch
import torch.nn as nn
from torchvision import models


class DRResNet50(nn.Module):
    """
    ResNet50-based Diabetic Retinopathy classifier.
    
    Reproduces the paper's architecture:
    - ResNet50 backbone (ImageNet pretrained)
    - Modified final FC layer for num_classes output
    - Optional dropout before classifier
    
    Args:
        num_classes: Number of output classes (5 for grading, 2 for binary).
        pretrained: Whether to use ImageNet pretrained weights.
        dropout: Dropout rate before the classifier head.
        freeze_backbone: If True, freeze all backbone layers.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet50 backbone
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features  # 2048
        
        # Replace the classifier head
        if dropout > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone_layers()
    
    def freeze_backbone_layers(self):
        """Freeze all layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    def unfreeze_backbone_layers(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_last_n_layers(self, n: int = 1):
        """
        Unfreeze the last N residual blocks (layer4, layer3, etc.).
        Useful for progressive fine-tuning.
        """
        # First freeze everything
        self.freeze_backbone_layers()
        
        # Then selectively unfreeze
        layers_to_unfreeze = ["fc"]
        if n >= 1:
            layers_to_unfreeze.append("layer4")
        if n >= 2:
            layers_to_unfreeze.append("layer3")
        if n >= 3:
            layers_to_unfreeze.append("layer2")
        if n >= 4:
            layers_to_unfreeze.append("layer1")
        
        for name, param in self.backbone.named_parameters():
            for layer_name in layers_to_unfreeze:
                if layer_name in name:
                    param.requires_grad = True
                    break
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
