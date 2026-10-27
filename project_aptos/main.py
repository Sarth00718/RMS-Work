"""
Main pipeline for Diabetic Retinopathy Classification on APTOS 2019 Dataset.

Reproduces the base paper methodology on the APTOS 2019 dataset:
1. Centralized local training (ResNet50 baseline)
2. Federated Averaging (FedAvg)
3. Differentially Private Federated Learning (DP-FedAvg) with varying noise.

Paper: "Collaborative Differentially Private Federated Learning
        Framework for the Prediction of Diabetic Retinopathy"
Authors: Ismail Hossain, Sai Puppala, Sajedul Talukder
Conference: IEEE ICAIC 2023

Usage:
    python main.py --config configs/base.yaml --task binary
    python main.py --config configs/base.yaml --task multiclass
    python main.py --config configs/base.yaml --mode full_pipeline
"""

import os
import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import random
import argparse
import hashlib
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import yaml

# Project modules
from data.dataset import APTOSDataset, APTOSSubset
from data.preprocessing import get_preprocessing_transform
from data.augmentation import get_augmentation_by_name
from data.split import create_stratified_split, check_data_leakage, check_image_duplicates
from models.factory import create_model
from training.trainer import Trainer
from training.losses import create_loss
from evaluation.metrics import compute_all_metrics, save_metrics, print_metrics_summary

from federated.client import FLClient
from federated.server import FLServer

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# ============================================================
# Run Centralized Experiment
# ============================================================
def run_experiment(
    config: dict,
    experiment_name: str,
    task: str = "binary",
    architecture: str = "resnet50",
    batch_size: int = 16,
    epochs: int = 50,
    results_dir: str = "results",
    seed: int = 42,
    device: torch.device = None,
    use_balanced_sampling: bool = True,
    label_smoothing: float = 0.1,
    augmentation_name: str = "enhanced",
) -> Dict:
    set_seed(seed)
    if device is None:
        device = get_device()

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    root = Path(config["dataset"]["root_dir"])
    train_img_dir = root / config["dataset"]["train_images"]
    val_img_dir = root / config["dataset"]["val_images"]
    test_img_dir = root / config["dataset"]["test_images"]
    train_csv = root / config["dataset"]["train_labels"]
    val_csv = root / config["dataset"]["val_labels"]
    test_csv = root / config["dataset"]["test_labels"]

    num_classes = 2 if task == "binary" else 5
    class_names = ["No DR", "DR"] if task == "binary" else config["dataset"]["class_names"]

    print(f"\n{'='*60}")
    print(f"  CENTRALIZED EXPERIMENT: {experiment_name}")
    print(f"  Task: {task} ({num_classes} classes)")
    print(f"  Dataset: APTOS 2019")
    print(f"  Architecture: {architecture}")
    print(f"  Balanced Sampling: {use_balanced_sampling}")
    print(f"  Label Smoothing: {label_smoothing}")
    print(f"  Augmentation: {augmentation_name}")
    print(f"{'='*60}")

    img_size = config["preprocessing"]["image_size"]
    norm_mean = tuple(config["preprocessing"]["normalize_mean"])
    norm_std = tuple(config["preprocessing"]["normalize_std"])

    train_transform = get_augmentation_by_name(augmentation_name, img_size, norm_mean, norm_std)
    val_transform = get_preprocessing_transform(img_size, norm_mean, norm_std)

    # APTOS has pre-split train/val/test
    train_dataset = APTOSDataset(
        str(train_img_dir), str(train_csv), transform=train_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )
    val_dataset = APTOSDataset(
        str(val_img_dir), str(val_csv), transform=val_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )
    test_dataset = APTOSDataset(
        str(test_img_dir), str(test_csv), transform=val_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )

    # Print dataset info
    print(f"\n  Dataset Statistics:")
    print(f"  Train: {len(train_dataset)} images")
    train_dist = train_dataset.get_class_distribution()
    for cls_name, count in train_dist.items():
        print(f"    {cls_name}: {count} ({100*count/len(train_dataset):.1f}%)")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")

    # Create data loaders
    if use_balanced_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model, loss, optimizer
    model = create_model(architecture, num_classes=num_classes, pretrained=True).to(device)
    criterion = create_loss("cross_entropy", label_smoothing=label_smoothing, device=str(device))

    lr = config["training"]["learning_rate"]
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create scheduler
    scheduler = None
    sched_config = config["training"].get("scheduler", {})
    if sched_config.get("name") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get("T_max", epochs),
            eta_min=sched_config.get("eta_min", 1e-5),
        )

    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler,
        device=str(device), save_dir=str(results_path), experiment_name=experiment_name,
        early_stopping_patience=config["training"]["early_stopping"]["patience"],
        seed=seed,
    )
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    trainer.load_best_model()

    test_loss, test_acc, test_preds, test_targets, test_probs = trainer.validate(test_loader)
    test_metrics = compute_all_metrics(test_targets, test_preds, test_probs, class_names, num_classes)
    print_metrics_summary(test_metrics, title=f"{experiment_name} - TEST Results")
    save_metrics(test_metrics, str(results_path), experiment_name)

    # Save plots
    _save_training_plots(history, experiment_name, str(results_path))

    return {
        "experiment": experiment_name,
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "auc": test_metrics.get("auc"),
    }


# ============================================================
# Run Federated Experiment
# ============================================================
def run_federated_experiment(
    config: dict,
    experiment_name: str,
    task: str = "binary",
    num_clients: int = 3,
    rounds: int = 20,
    local_epochs: int = 1,
    dp_noise_multiplier: float = 0.0,
    dp_clip_norm: float = 1.0,
    batch_size: int = 16,
    results_dir: str = "results",
    seed: int = 42,
    device: torch.device = None,
    label_smoothing: float = 0.1,
    augmentation_name: str = "enhanced",
    use_checkpointing: bool = True,
) -> Dict:
    set_seed(seed)
    if device is None:
        device = get_device()

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FEDERATED EXPERIMENT: {experiment_name}")
    print(f"  Task: {task}, Clients: {num_clients}, Rounds: {rounds}, Local Epochs: {local_epochs}")
    print(f"  DP Noise: {dp_noise_multiplier}, DP Clip: {dp_clip_norm}")
    print(f"  Checkpointing: {'ENABLED' if use_checkpointing else 'DISABLED'}")
    print(f"{'='*60}")

    root = Path(config["dataset"]["root_dir"])
    train_img_dir = root / config["dataset"]["train_images"]
    val_img_dir = root / config["dataset"]["val_images"]
    test_img_dir = root / config["dataset"]["test_images"]
    train_csv = root / config["dataset"]["train_labels"]
    val_csv = root / config["dataset"]["val_labels"]
    test_csv = root / config["dataset"]["test_labels"]

    num_classes = 2 if task == "binary" else 5
    class_names = ["No DR", "DR"] if task == "binary" else config["dataset"]["class_names"]

    img_size = config["preprocessing"]["image_size"]
    norm_mean = tuple(config["preprocessing"]["normalize_mean"])
    norm_std = tuple(config["preprocessing"]["normalize_std"])
    train_transform = get_augmentation_by_name(augmentation_name, img_size, norm_mean, norm_std)
    val_transform = get_preprocessing_transform(img_size, norm_mean, norm_std)

    # Load datasets
    full_train_dataset = APTOSDataset(
        str(train_img_dir), str(train_csv), transform=train_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )
    val_dataset_raw = APTOSDataset(
        str(train_img_dir), str(train_csv), transform=val_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )
    # Use the provided val set for global evaluation during FL
    val_dataset = APTOSDataset(
        str(val_img_dir), str(val_csv), transform=val_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )
    test_dataset = APTOSDataset(
        str(test_img_dir), str(test_csv), transform=val_transform, task=task,
        id_column=config["dataset"]["id_column"],
        label_column=config["dataset"]["label_column"],
        image_extension=config["dataset"]["image_extension"],
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split training set among K clients (IID for simplicity)
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)
    split_size = len(indices) // num_clients

    clients = []
    # Base model config for server
    server_model = create_model("resnet50", num_classes=num_classes, pretrained=True).to(device)
    server = FLServer(server_model)

    federated_config = config.get("federated", {})
    # Checkpointing thresholds: use very large values when disabled so gates never trigger
    if use_checkpointing:
        checkpoint_theta = federated_config.get("checkpoint_theta", 0.01)
        checkpoint_phi = federated_config.get("checkpoint_phi", 0.01)
    else:
        checkpoint_theta = -999.0   # Checkpoint 1 never blocks (pre_acc always > median - theta)
        checkpoint_phi = -999.0     # Checkpoint 2 never blocks (acc_diff always > phi)

    for i in range(num_clients):
        if i < num_clients - 1:
            client_indices = indices[i*split_size : (i+1)*split_size]
        else:
            client_indices = indices[i*split_size:]

        # Split client indices into 80/20 train/val
        c_train_idx = client_indices[:int(0.8 * len(client_indices))]
        c_val_idx = client_indices[int(0.8 * len(client_indices)):]

        c_train_ds = Subset(full_train_dataset, c_train_idx)
        c_val_ds = Subset(val_dataset_raw, c_val_idx)

        c_train_loader = DataLoader(c_train_ds, batch_size=batch_size, shuffle=True)
        c_val_loader = DataLoader(c_val_ds, batch_size=batch_size, shuffle=False)

        client_model = create_model("resnet50", num_classes=num_classes, pretrained=True).to(device)
        criterion = create_loss("cross_entropy", label_smoothing=label_smoothing, device=str(device))

        client = FLClient(
            client_id=i, model=client_model, train_loader=c_train_loader, val_loader=c_val_loader,
            criterion=criterion, optimizer_cls=optim.SGD,
            optimizer_kwargs={
                "lr": config["training"]["learning_rate"],
                "momentum": config["training"]["momentum"],
                "weight_decay": config["training"]["weight_decay"],
            },
            device=str(device),
            dp_noise_multiplier=dp_noise_multiplier,
            dp_clip_norm=dp_clip_norm,
            checkpoint_theta=checkpoint_theta,
            checkpoint_phi=checkpoint_phi,
        )
        clients.append(client)

    print(f"Created {num_clients} clients with avg {split_size} samples each.")

    history = []
    best_test_acc = 0.0

    # Federated Training Loop
    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---")
        global_weights = server.get_global_weights()
        median_post_acc = server.get_median_post_acc()
        client_updates = []

        for client in clients:
            client.set_weights(global_weights)
            # Local training
            new_weights, metrics = client.train(local_epochs=local_epochs, median_post_acc=median_post_acc, round_num=round_num)

            if metrics.get("skipped_checkpoint_1", False):
                print(f"  Client {client.client_id} blocked by Checkpoint 1 (pre_acc {metrics['pre_acc']:.4f} <= median_post_acc {median_post_acc:.4f} - theta)")
                continue

            # Checkpoint 2: only send if |pre_acc - post_acc| > phi
            # When use_checkpointing=False, phi=-999 so this always passes
            acc_diff = abs(metrics["post_acc"] - metrics["pre_acc"])
            if acc_diff > client.phi or round_num == 1:
                client_updates.append((metrics["num_samples"], new_weights, metrics["post_acc"]))
                print(f"  Client {client.client_id} sent update. Pre: {metrics['pre_acc']:.4f} -> Post: {metrics['post_acc']:.4f} (diff: {acc_diff:.4f})")
            else:
                print(f"  Client {client.client_id} skipped update. Acc diff: {acc_diff:.4f} <= {client.phi}")

        if client_updates:
            server.aggregate_weights(client_updates)

        # Evaluate global model on Test Set
        server_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = server_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        best_test_acc = max(best_test_acc, test_acc)
        print(f"  Global Test Accuracy: {test_acc:.4f} (Best: {best_test_acc:.4f})")
        history.append({"round": round_num, "test_acc": test_acc})

    # Final comprehensive evaluation
    test_loss, test_acc_full, test_preds, test_targets, test_probs = evaluate_model(server_model, test_loader, device)
    test_metrics = compute_all_metrics(test_targets, test_preds, test_probs, class_names, num_classes)

    print_metrics_summary(test_metrics, title=f"{experiment_name} - Final Global Model")
    save_metrics(test_metrics, str(results_path), experiment_name)

    # Save federated training history
    _save_federated_history(history, experiment_name, str(results_path))

    return {
        "experiment": experiment_name,
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "auc": test_metrics.get("auc"),
        "best_test_acc": best_test_acc,
    }


def evaluate_model(model, dataloader, device):
    """Evaluate a model on a dataloader."""
    model.eval()
    criterion = create_loss("cross_entropy", device=str(device))
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return running_loss/total, correct/total, np.array(all_preds), np.array(all_targets), np.array(all_probs)


# ============================================================
# Plotting Utilities
# ============================================================
def _save_training_plots(history: List[Dict], experiment_name: str, save_dir: str):
    """Save training curves (loss and accuracy)."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{experiment_name} - Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{experiment_name} - Accuracy', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / f"{experiment_name}_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


def _save_federated_history(history: List[Dict], experiment_name: str, save_dir: str):
    """Save federated training history."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = save_path / f"{experiment_name}_fl_history.csv"
    if history:
        keys = history[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)

    # Plot federated accuracy over rounds
    if history:
        rounds = [h["round"] for h in history]
        test_accs = [h["test_acc"] for h in history]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, test_accs, 'g-o', label='Global Test Acc', linewidth=2, markersize=4)
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{experiment_name} - Federated Learning Progress', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / f"{experiment_name}_fl_progress.png", dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Summary Table Generator
# ============================================================
def _generate_summary_table(all_results: List[Dict], save_dir: str):
    """Generate a comprehensive summary table of all experiments."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Print to console
    print("\n" + "=" * 80)
    print("  SUMMARY OF ALL EXPERIMENTS (APTOS 2019 Dataset)")
    print("=" * 80)
    print(f"{'Experiment':<45} | {'Accuracy':<10} | {'F1 Score':<10} | {'AUC':<10}")
    print("-" * 80)
    for r in all_results:
        auc_str = f"{r['auc']:.4f}" if r.get('auc') is not None else "N/A"
        print(f"{r['experiment']:<45} | {r['accuracy']:<10.4f} | {r['f1_macro']:<10.4f} | {auc_str:<10}")

    print("\n" + "=" * 80)
    print("  BEST OVERALL ACCURACIES")
    print("=" * 80)

    binary_results = [r for r in all_results if "binary" in r['experiment'].lower()]
    if binary_results:
        best_bin = max(binary_results, key=lambda x: x['accuracy'])
        print(f"  Best Binary Task Accuracy:     {best_bin['accuracy']*100:.2f}%  [{best_bin['experiment']}]")

    multi_results = [r for r in all_results if "multiclass" in r['experiment'].lower()]
    if multi_results:
        best_multi = max(multi_results, key=lambda x: x['accuracy'])
        print(f"  Best Multiclass Task Accuracy: {best_multi['accuracy']*100:.2f}%  [{best_multi['experiment']}]")
    print("=" * 80 + "\n")

    # Save summary to JSON
    with open(save_path / "experiment_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save summary to CSV
    with open(save_path / "experiment_summary.csv", "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)


# ============================================================
# Full Pipeline
# ============================================================
def run_full_pipeline(config_path: str = "configs/base.yaml"):
    """
    Run the complete paper reproduction pipeline:
    1. Centralized baseline (binary + multiclass)
    2. FedAvg without noise (binary + multiclass)
    3. DP-FedAvg with different noise levels (binary + multiclass)
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    results_dir = Path(config["logging"]["save_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    label_smoothing = config["loss"].get("label_smoothing", 0.0)
    use_balanced = config.get("sampling", {}).get("use_balanced_sampling", False)

    federated_config = config.get("federated", {})
    num_clients = federated_config.get("num_clients", 3)
    fl_rounds = federated_config.get("rounds", 20)
    local_epochs = federated_config.get("local_epochs", 1)
    dp_clip_norm = federated_config.get("dp_clip_norm", 1.0)
    noise_multipliers = federated_config.get("noise_multipliers", [0.56, 0.9, 1.2, 3.5, 8.0])

    tasks = ["binary", "multiclass"]

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  STARTING TASK: {task.upper()}")
        print(f"{'#'*60}")

        # 1. Centralized Local Training (Baseline)
        print(f"\n[Phase 1 - {task}] Centralized Local Training (Baseline)")
        res = run_experiment(
            config, f"Centralized_Baseline_{task}",
            task=task, epochs=epochs, batch_size=batch_size,
            results_dir=str(results_dir), device=device,
            use_balanced_sampling=use_balanced,
            label_smoothing=label_smoothing,
        )
        all_results.append(res)

        # 2. Federated Averaging (FedAvg) - No Noise
        print(f"\n[Phase 2 - {task}] Federated Averaging (FedAvg) - No Noise")
        res = run_federated_experiment(
            config, f"FedAvg_NoNoise_{task}",
            task=task, num_clients=num_clients, rounds=fl_rounds,
            local_epochs=local_epochs, dp_noise_multiplier=0.0,
            dp_clip_norm=dp_clip_norm, batch_size=batch_size,
            results_dir=str(results_dir), device=device,
            label_smoothing=label_smoothing,
        )
        all_results.append(res)

        # 3. Differentially Private FedAvg (DP-FedAvg) - Different Noise Multipliers
        print(f"\n[Phase 3 - {task}] Differentially Private FedAvg (DP-FedAvg)")
        for noise in noise_multipliers:
            exp_name = f"DP_FedAvg_Noise_{noise}_{task}".replace(".", "p")
            res = run_federated_experiment(
                config, exp_name,
                task=task, num_clients=num_clients, rounds=fl_rounds,
                local_epochs=local_epochs, dp_noise_multiplier=noise,
                dp_clip_norm=dp_clip_norm, batch_size=batch_size,
                results_dir=str(results_dir), device=device,
                label_smoothing=label_smoothing,
            )
            all_results.append(res)

    # Generate summary
    _generate_summary_table(all_results, str(results_dir))


def run_dp_no_checkpoint(config_path: str = "configs/base.yaml"):
    """
    Run DP-FedAvg (binary) WITHOUT the paper's checkpointing gates.
    Both Checkpoint 1 (pre-acc filter) and Checkpoint 2 (acc-diff filter)
    are disabled so every client trains and sends updates every round.
    This is the fair comparison baseline to see how much accuracy is lost
    by the checkpointing mechanism vs the noise alone.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    results_dir = Path(config["logging"]["save_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    label_smoothing = config["loss"].get("label_smoothing", 0.0)
    batch_size = config["training"]["batch_size"]

    federated_config = config.get("federated", {})
    num_clients = federated_config.get("num_clients", 3)
    fl_rounds = federated_config.get("rounds", 10)
    local_epochs = federated_config.get("local_epochs", 1)
    dp_clip_norm = federated_config.get("dp_clip_norm", 1.0)
    noise_multipliers = federated_config.get("noise_multipliers", [0.56, 0.9, 1.2, 3.5, 8.0])

    task = "binary"
    all_results = []

    print(f"\n{'#'*60}")
    print(f"  DP-FedAvg WITHOUT CHECKPOINTING  (task=binary)")
    print(f"  All clients train and send updates every round.")
    print(f"  No Checkpoint 1 (pre-acc gate) or Checkpoint 2 (acc-diff gate).")
    print(f"{'#'*60}")

    for noise in noise_multipliers:
        exp_name = f"DP_FedAvg_NoCkpt_Noise_{noise}_{task}".replace(".", "p")
        res = run_federated_experiment(
            config, exp_name,
            task=task,
            num_clients=num_clients,
            rounds=fl_rounds,
            local_epochs=local_epochs,
            dp_noise_multiplier=noise,
            dp_clip_norm=dp_clip_norm,
            batch_size=batch_size,
            results_dir=str(results_dir),
            seed=config["seed"],
            device=device,
            label_smoothing=label_smoothing,
            use_checkpointing=False,   # <-- key change
        )
        all_results.append(res)
        print(f"\n[NO-CKPT] noise={noise}  acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}  auc={res.get('auc')}")

    _generate_summary_table(all_results, str(results_dir))



    """Run only the centralized baseline for a specific task."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    results_dir = Path(config["logging"]["save_dir"])
    label_smoothing = config["loss"].get("label_smoothing", 0.0)
    use_balanced = config.get("sampling", {}).get("use_balanced_sampling", False)

    res = run_experiment(
        config, f"Centralized_Baseline_{task}",
        task=task,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        results_dir=str(results_dir),
        device=device,
        use_balanced_sampling=use_balanced,
        label_smoothing=label_smoothing,
    )
    print(f"\nResult: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Classification Pipeline - APTOS 2019 (Paper Reproduction)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--mode", type=str, default="full_pipeline",
                        choices=["full_pipeline", "binary", "multiclass", "dp_no_checkpoint"],
                        help=(
                            "Run mode: "
                            "full_pipeline          - all experiments with checkpointing, "
                            "binary                 - centralized baseline binary only, "
                            "multiclass             - centralized baseline multiclass only, "
                            "dp_no_checkpoint       - DP-FedAvg binary, all noise levels, NO checkpointing"
                        ))

    args = parser.parse_args()

    if args.mode == "full_pipeline":
        run_full_pipeline(args.config)
    elif args.mode == "dp_no_checkpoint":
        run_dp_no_checkpoint(args.config)
    elif args.mode in ["binary", "multiclass"]:
        run_single_experiment(args.config, task=args.mode)
