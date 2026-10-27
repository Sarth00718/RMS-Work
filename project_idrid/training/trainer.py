"""
Training engine for DR classification.

Implements the complete training loop with:
- Epoch-level training and validation
- Metric tracking and logging
- Early stopping and checkpointing
- Training history recording
"""

import time
import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint


class Trainer:
    """
    Complete training engine for DR classification.
    
    Args:
        model: PyTorch model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Optional LR scheduler.
        device: Device to train on.
        save_dir: Directory to save results.
        experiment_name: Name for this experiment.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        save_dir: str = "results",
        experiment_name: str = "baseline",
        early_stopping_patience: int = 15,
        gradient_clip: float = 0.0,
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.gradient_clip = gradient_clip
        self.seed = seed
        
        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            monitor="val_loss",
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=str(self.save_dir / "checkpoints"),
            monitor="val_loss",
        )
        
        # History
        self.history: List[Dict] = []
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch. Returns (loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.1f}%",
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate model. Returns (loss, accuracy, all_preds, all_targets)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets), np.array(all_probs)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
    ) -> List[Dict]:
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            epochs: Number of epochs to train.
        
        Returns:
            Training history (list of epoch metrics).
        """
        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets, val_probs = self.validate(val_loader)
            
            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Record history
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": current_lr,
                "time": round(epoch_time, 1),
            }
            self.history.append(epoch_metrics)
            
            # Print progress
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Checkpointing
            is_best = self.checkpoint(
                self.model, self.optimizer, epoch, val_loss,
                metrics=epoch_metrics,
            )
            if is_best:
                best_val_acc = val_acc
                print(f"  * New best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n[STOP] Early stopping at epoch {epoch} (patience: {self.early_stopping.patience})")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save history
        self._save_history()
        
        return self.history
    
    def _save_history(self):
        """Save training history to CSV and JSON."""
        # Save as CSV
        csv_path = self.save_dir / f"{self.experiment_name}_history.csv"
        if self.history:
            keys = self.history[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
        
        # Save as JSON
        json_path = self.save_dir / f"{self.experiment_name}_history.json"
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        ckpt_path = self.save_dir / "checkpoints" / "best_model.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print("Warning: No best model checkpoint found.")
