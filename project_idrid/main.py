"""
Main pipeline for Diabetic Retinopathy Classification.

Reproduces the base paper methodology on the IDRiD dataset:
1. Centralized local training
2. Federated Averaging (FedAvg)
3. Differentially Private Federated Learning (DP-FedAvg) with varying noise.

Paper: "Collaborative Differentially Private Federated Learning 
        Framework for the Prediction of Diabetic Retinopathy"

Usage:
    python main.py --config configs/base.yaml --task binary
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
from data.dataset import IDRiDDataset, IDRiDSubset
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
    epochs: int = 20,
    results_dir: str = "results",
    seed: int = 42,
    device: torch.device = None,
) -> Dict:
    set_seed(seed)
    if device is None:
        device = get_device()
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    root = Path(config["dataset"]["root_dir"])
    train_img_dir = root / config["dataset"]["train_images"]
    test_img_dir = root / config["dataset"]["test_images"]
    train_csv = root / config["dataset"]["train_labels"]
    test_csv = root / config["dataset"]["test_labels"]
    
    num_classes = 2 if task == "binary" else 5
    class_names = ["No DR", "DR"] if task == "binary" else config["dataset"]["class_names"]
    
    print(f"\n{'='*60}")
    print(f"  CENTRALIZED EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    img_size = config["preprocessing"]["image_size"]
    norm_mean = tuple(config["preprocessing"]["normalize_mean"])
    norm_std = tuple(config["preprocessing"]["normalize_std"])
    
    train_transform = get_augmentation_by_name("baseline", img_size, norm_mean, norm_std)
    val_transform = get_preprocessing_transform(img_size, norm_mean, norm_std)
    
    full_train_dataset = IDRiDDataset(str(train_img_dir), str(train_csv), task=task)
    test_dataset = IDRiDDataset(str(test_img_dir), str(test_csv), transform=val_transform, task=task)
    
    train_indices, val_indices = create_stratified_split(full_train_dataset.labels, val_ratio=0.2, random_seed=seed)
    train_subset = IDRiDSubset(full_train_dataset, train_indices, transform=train_transform)
    val_subset = IDRiDSubset(full_train_dataset, val_indices, transform=val_transform)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = create_model(architecture, num_classes=num_classes, pretrained=True).to(device)
    criterion = create_loss("cross_entropy", device=str(device))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    trainer = Trainer(
        model=model, criterion=criterion, optimizer=optimizer,
        device=str(device), save_dir=str(results_path), experiment_name=experiment_name,
        early_stopping_patience=10, seed=seed,
    )
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    trainer.load_best_model()
    
    test_loss, test_acc, test_preds, test_targets, test_probs = trainer.validate(test_loader)
    test_metrics = compute_all_metrics(test_targets, test_preds, test_probs, class_names, num_classes)
    print_metrics_summary(test_metrics, title=f"{experiment_name} - TEST Results")
    save_metrics(test_metrics, str(results_path), experiment_name)
    
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
    test_img_dir = root / config["dataset"]["test_images"]
    train_csv = root / config["dataset"]["train_labels"]
    test_csv = root / config["dataset"]["test_labels"]
    
    num_classes = 2 if task == "binary" else 5
    class_names = ["No DR", "DR"] if task == "binary" else config["dataset"]["class_names"]
    
    img_size = config["preprocessing"]["image_size"]
    norm_mean = tuple(config["preprocessing"]["normalize_mean"])
    norm_std = tuple(config["preprocessing"]["normalize_std"])
    train_transform = get_augmentation_by_name("baseline", img_size, norm_mean, norm_std)
    val_transform = get_preprocessing_transform(img_size, norm_mean, norm_std)
    
    full_train_dataset = IDRiDDataset(str(train_img_dir), str(train_csv), transform=train_transform, task=task)
    val_dataset_raw = IDRiDDataset(str(train_img_dir), str(train_csv), transform=val_transform, task=task)
    test_dataset = IDRiDDataset(str(test_img_dir), str(test_csv), transform=val_transform, task=task)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Split training set among K clients (IID for simplicity)
    indices = list(range(len(full_train_dataset)))
    random.shuffle(indices)
    split_size = len(indices) // num_clients
    
    clients = []
    # Base model config for server
    server_model = create_model("resnet50", num_classes=num_classes, pretrained=True).to(device)
    server = FLServer(server_model)

    # Checkpointing thresholds: use very large negative values when disabled
    # so both gates (Checkpoint 1 and Checkpoint 2) never trigger
    if use_checkpointing:
        checkpoint_theta = 0.01
        checkpoint_phi = 0.01
    else:
        checkpoint_theta = -999.0   # Checkpoint 1 never blocks (pre_acc always > median - theta)
        checkpoint_phi = -999.0     # Checkpoint 2 never blocks (acc_diff always > phi)

    for i in range(num_clients):
        client_indices = indices[i*split_size : (i+1)*split_size if i < num_clients-1 else len(indices)]
        # Split client indices into 80/20 train/val
        c_train_idx = client_indices[:int(0.8 * len(client_indices))]
        c_val_idx = client_indices[int(0.8 * len(client_indices)):]
        
        c_train_ds = Subset(full_train_dataset, c_train_idx)
        c_val_ds = Subset(val_dataset_raw, c_val_idx)
        
        c_train_loader = DataLoader(c_train_ds, batch_size=batch_size, shuffle=True)
        c_val_loader = DataLoader(c_val_ds, batch_size=batch_size, shuffle=False)
        
        client_model = create_model("resnet50", num_classes=num_classes, pretrained=True).to(device)
        criterion = create_loss("cross_entropy", device=str(device))
        
        client = FLClient(
            client_id=i, model=client_model, train_loader=c_train_loader, val_loader=c_val_loader,
            criterion=criterion, optimizer_cls=optim.SGD, 
            optimizer_kwargs={"lr": 0.001, "momentum": 0.9, "weight_decay": 1e-4},
            device=str(device), dp_noise_multiplier=dp_noise_multiplier, dp_clip_norm=dp_clip_norm,
            checkpoint_theta=checkpoint_theta, checkpoint_phi=checkpoint_phi,
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
                print(f"  Client {client.client_id} sent update. Acc diff: {acc_diff:.4f}")
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
        print(f"  Global Test Accuracy: {test_acc:.4f}")
        history.append({"round": round_num, "test_acc": test_acc})
        
    # Final comprehensive evaluation
    test_loss, test_acc_full, test_preds, test_targets, test_probs = evaluate_model(server_model, test_loader, device)
    test_metrics = compute_all_metrics(test_targets, test_preds, test_probs, class_names, num_classes)
    
    print_metrics_summary(test_metrics, title=f"{experiment_name} - Final Global Model")
    save_metrics(test_metrics, str(results_path), experiment_name)
    
    return {
        "experiment": experiment_name,
        "accuracy": test_metrics["accuracy"],
        "f1_macro": test_metrics["f1_macro"],
        "auc": test_metrics.get("auc"),
    }

def evaluate_model(model, dataloader, device):
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

def run_dp_no_checkpoint(config_path: str = "configs/base.yaml"):
    """
    Run DP-FedAvg (binary) WITHOUT the paper's checkpointing gates on the IDRiD dataset.
    Both Checkpoint 1 (pre-acc filter) and Checkpoint 2 (acc-diff filter) are disabled
    so every client trains and sends updates every round.
    This shows the true accuracy cost of DP noise alone, without checkpointing suppressing updates.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device = get_device()
    results_dir = Path(config["logging"]["save_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    noise_multipliers = [0.56, 0.9, 1.2, 3.5, 8.0]
    task = "binary"
    all_results = []

    print(f"\n{'#'*60}")
    print(f"  DP-FedAvg WITHOUT CHECKPOINTING  (IDRiD, task=binary)")
    print(f"  All clients train and send updates every round.")
    print(f"  No Checkpoint 1 (pre-acc gate) or Checkpoint 2 (acc-diff gate).")
    print(f"{'#'*60}")

    for noise in noise_multipliers:
        exp_name = f"DP_FedAvg_NoCkpt_Noise_{noise}_{task}".replace(".", "p")
        res = run_federated_experiment(
            config, exp_name,
            task=task,
            num_clients=3,
            rounds=20,
            local_epochs=1,
            dp_noise_multiplier=noise,
            dp_clip_norm=1.0,
            batch_size=config["training"]["batch_size"],
            results_dir=str(results_dir),
            seed=config["seed"],
            device=device,
            use_checkpointing=False,   # <-- key change: both checkpoints disabled
        )
        all_results.append(res)
        print(f"\n[NO-CKPT] noise={noise}  acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}  auc={res.get('auc')}")

    # Summary table
    print("\n" + "=" * 65)
    print("  SUMMARY: DP-FedAvg WITHOUT Checkpointing (IDRiD binary)")
    print("=" * 65)
    print(f"{'Experiment':<45} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['experiment']:<45} | {r['accuracy']:<10.4f} | {r['f1_macro']:<10.4f}")
    print("=" * 65 + "\n")

    # Save summary JSON
    summary_path = results_dir / "dp_no_checkpoint_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


def run_full_pipeline(config_path: str = "configs/base.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    set_seed(config["seed"])
    device = get_device()
    results_dir = Path(config["logging"]["save_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    tasks = ["binary", "multiclass"]
    
    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"  STARTING TASK: {task.upper()}")
        print(f"{'#'*60}")
        
        # 1. Centralized Local Training (Baseline)
        print(f"\n[Phase 1 - {task}] Centralized Local Training (Baseline)")
        res = run_experiment(config, f"Centralized_Baseline_{task}", task=task, epochs=20, results_dir=str(results_dir), device=device)
        all_results.append(res)
        
        # 2. Federated Averaging (FedAvg) - No Noise
        print(f"\n[Phase 2 - {task}] Federated Averaging (FedAvg) - No Noise")
        res = run_federated_experiment(config, f"FedAvg_NoNoise_{task}", task=task, dp_noise_multiplier=0.0, rounds=20, results_dir=str(results_dir), device=device)
        all_results.append(res)
        
        # 3. Differentially Private FedAvg (DP-FedAvg) - Different Noise Multipliers
        print(f"\n[Phase 3 - {task}] Differentially Private FedAvg (DP-FedAvg)")
        noise_levels = [0.56, 0.9, 1.2, 3.5, 8.0]
        for noise in noise_levels:
            exp_name = f"DP_FedAvg_Noise_{noise}_{task}".replace(".", "p")
            res = run_federated_experiment(config, exp_name, task=task, dp_noise_multiplier=noise, rounds=20, results_dir=str(results_dir), device=device)
            all_results.append(res)
            
    print("\n" + "=" * 60)
    print("  SUMMARY OF ALL EXPERIMENTS")
    print("=" * 60)
    print(f"{'Experiment':<40} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-" * 65)
    for r in all_results:
        print(f"{r['experiment']:<40} | {r['accuracy']:<10.4f} | {r['f1_macro']:<10.4f}")
        
    print("\n" + "=" * 60)
    print("  BEST OVERALL ACCURACIES")
    print("=" * 60)
    
    binary_results = [r for r in all_results if "binary" in r['experiment']]
    if binary_results:
        best_bin = max(binary_results, key=lambda x: x['accuracy'])
        print(f"  Best Binary Task Accuracy:     {best_bin['accuracy']*100:.2f}%  [{best_bin['experiment']}]")
        
    multi_results = [r for r in all_results if "multiclass" in r['experiment']]
    if multi_results:
        best_multi = max(multi_results, key=lambda x: x['accuracy'])
        print(f"  Best Multiclass Task Accuracy: {best_multi['accuracy']*100:.2f}%  [{best_multi['experiment']}]")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Classification Pipeline (Paper Reproduction)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--mode", type=str, default="full_pipeline",
                        choices=["full_pipeline", "dp_no_checkpoint"],
                        help=(
                            "Run mode: "
                            "full_pipeline      - all experiments with checkpointing, "
                            "dp_no_checkpoint   - DP-FedAvg binary, all noise levels, NO checkpointing"
                        ))

    args = parser.parse_args()

    if args.mode == "full_pipeline":
        run_full_pipeline(args.config)
    elif args.mode == "dp_no_checkpoint":
        run_dp_no_checkpoint(args.config)
