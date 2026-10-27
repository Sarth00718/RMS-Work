"""
Training callbacks: early stopping, checkpointing.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a metric and stops training when it stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        monitor: Metric to monitor ('val_loss' or 'val_acc').
        mode: 'min' for loss, 'max' for accuracy.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "auto",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        monitor: Metric to monitor for best model.
        mode: 'min' for loss, 'max' for accuracy.
        save_best: Whether to save the best model.
        save_last: Whether to save the last model.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_loss",
        mode: str = "auto",
        save_best: bool = True,
        save_last: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_best = save_best
        self.save_last = save_last
        self.best_score = None

        if mode == "auto":
            self.mode = "min" if "loss" in monitor else "max"
        else:
            self.mode = mode

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
        metrics: Optional[dict] = None,
    ) -> bool:
        """
        Save checkpoint if score improved.

        Returns:
            True if best model was updated.
        """
        is_best = False

        if self.best_score is None:
            self.best_score = score
            is_best = True
        else:
            if self.mode == "min":
                is_best = score < self.best_score
            else:
                is_best = score > self.best_score

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score,
            "metrics": metrics or {},
        }

        if self.save_last:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.save_dir / "last_model.pth")

        if is_best and self.save_best:
            self.best_score = score
            self.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.save_dir / "best_model.pth")
            if metrics:
                with open(self.save_dir / "best_metrics.json", "w") as f:
                    json.dump({"epoch": epoch, "score": score, **metrics}, f, indent=2)

        return is_best
