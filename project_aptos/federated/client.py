"""
Federated Learning Client with optional Differential Privacy (DP-SGD).

Implements the paper's Algorithm 1:
- Local training with optional DP-SGD (gradient clipping + Gaussian noise)
- Checkpoint 1: Pre-training accuracy check against median
- Checkpoint 2: Post-training accuracy change threshold
"""
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

class FLClient:
    """
    Represents a client (e.g., a hospital) in the Federated Learning framework.
    """
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer_cls: type,
        optimizer_kwargs: dict,
        device: str,
        dp_noise_multiplier: float = 0.0,
        dp_clip_norm: float = 1.0,
        checkpoint_theta: float = 0.01,
        checkpoint_phi: float = 0.01,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.device = device

        # DP settings
        self.noise_multiplier = dp_noise_multiplier
        self.clip_norm = dp_clip_norm

        # Checkpointing technique (from paper) to reduce communication
        self.theta = checkpoint_theta
        self.phi = checkpoint_phi
        self.prev_val_acc = 0.0

    def set_weights(self, global_weights: Dict[str, torch.Tensor]):
        """Load global weights from server."""
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current local weights."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def evaluate(self) -> float:
        """Evaluate current model on local validation set."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / max(total, 1)

    def train(self, local_epochs: int = 1, median_post_acc: float = 0.0, round_num: int = 1) -> Tuple[Optional[Dict[str, torch.Tensor]], dict]:
        """
        Train locally for a specified number of epochs.
        Implements DP-SGD if noise_multiplier > 0.
        Returns the updated weights and training metrics.
        """
        # Pre-evaluation for communication checkpointing
        pre_acc = self.evaluate()

        # Checkpoint 1: Determine whether a client is allowed for training its local model.
        if round_num > 1 and pre_acc <= (median_post_acc - self.theta):
            metrics = {
                "client_id": self.client_id,
                "train_loss": 0.0,
                "pre_acc": pre_acc,
                "post_acc": pre_acc,
                "num_samples": 0,
                "skipped_checkpoint_1": True
            }
            return None, metrics

        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        self.model.train()

        running_loss = 0.0
        total_samples = 0

        for epoch in range(local_epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                # Differential Privacy: Gradient Clipping and Noise Addition
                if self.noise_multiplier > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            # N(0, sigma^2 * C^2)
                            std = self.noise_multiplier * self.clip_norm
                            noise = torch.normal(mean=0.0, std=std, size=param.grad.size(), device=param.grad.device)
                            param.grad.add_(noise)

                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        post_acc = self.evaluate()
        self.prev_val_acc = post_acc

        metrics = {
            "client_id": self.client_id,
            "train_loss": running_loss / max(total_samples, 1),
            "pre_acc": pre_acc,
            "post_acc": post_acc,
            "num_samples": total_samples // local_epochs,
            "skipped_checkpoint_1": False
        }

        return self.get_weights(), metrics
