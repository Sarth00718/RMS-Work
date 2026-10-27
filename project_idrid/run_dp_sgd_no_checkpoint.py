"""
DP-FedAvg WITHOUT Any Checkpointing — IDRiD Dataset
=====================================================
This script runs Differentially Private Federated Averaging (DP-FedAvg)
for binary DR classification on IDRiD with ZERO checkpoint logic.

Every client trains every round. Every client sends its update every round.
No Checkpoint 1 (pre-acc gate). No Checkpoint 2 (acc-diff gate).
No reference to FLClient or FLServer — all FL logic is self-contained here.

DP-SGD mechanism:
  1. Clip each gradient to norm <= C (dp_clip_norm)
  2. Add Gaussian noise N(0, (noise_multiplier * C)^2) to each gradient
  3. Aggregate with FedAvg (weighted by sample count)

Noise levels tested: 0.56, 0.9, 1.2, 3.5, 8.0  (from paper Table II)

Usage:
    python run_dp_sgd_no_checkpoint.py
    python run_dp_sgd_no_checkpoint.py --rounds 20 --clients 3
"""

import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from torch.utils.data import DataLoader, Subset

# Project modules (same as main.py)
from data.dataset import IDRiDDataset
from data.preprocessing import get_preprocessing_transform
from data.augmentation import get_augmentation_by_name
from models.factory import create_model
from training.losses import create_loss
from evaluation.metrics import compute_all_metrics, save_metrics, print_metrics_summary

# ────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────
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
        d = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("Using CPU")
    return d


# ────────────────────────────────────────────────────────────
# Pure DP-SGD local training (no checkpoint logic at all)
# ────────────────────────────────────────────────────────────
def dp_local_train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    local_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    noise_multiplier: float,
    clip_norm: float,
) -> tuple:
    """
    Train model locally with DP-SGD.
    Returns (updated_state_dict, num_samples_seen, avg_loss).
    No checkpoint gates — always trains, always returns weights.
    """
    model.train()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    total_loss = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # ── DP Step 1: Clip per-sample gradient norm ──────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # ── DP Step 2: Add calibrated Gaussian noise ──────────
            if noise_multiplier > 0:
                sigma = noise_multiplier * clip_norm
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        noise = torch.normal(
                            mean=0.0,
                            std=sigma,
                            size=param.grad.shape,
                            device=param.grad.device,
                        )
                        param.grad.add_(noise)

            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    weights = {k: v.cpu() for k, v in model.state_dict().items()}
    return weights, total_samples // local_epochs, avg_loss


# ────────────────────────────────────────────────────────────
# FedAvg aggregation (standalone, no FLServer class)
# ────────────────────────────────────────────────────────────
def fedavg_aggregate(
    global_weights: dict,
    client_updates: list,  # list of (num_samples, state_dict)
) -> dict:
    """Weighted average of client weights by sample count."""
    total_samples = sum(n for n, _ in client_updates)
    new_weights = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_weights.items()
    }
    for num_samples, client_w in client_updates:
        frac = num_samples / total_samples
        for k in new_weights:
            new_weights[k] += client_w[k].to(torch.float32) * frac

    # Cast back to original dtype
    for k in global_weights:
        new_weights[k] = new_weights[k].to(global_weights[k].dtype)

    return new_weights


# ────────────────────────────────────────────────────────────
# Quick accuracy on a DataLoader
# ────────────────────────────────────────────────────────────
def eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, predicted = model(inputs).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / max(total, 1)


# ────────────────────────────────────────────────────────────
# Full evaluation (loss, acc, preds, probs)
# ────────────────────────────────────────────────────────────
def full_evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = total = 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in loader:
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
    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
        np.array(all_preds),
        np.array(all_targets),
        np.array(all_probs),
    )


# ────────────────────────────────────────────────────────────
# Save FL round history plot
# ────────────────────────────────────────────────────────────
def save_fl_plot(history: list, exp_name: str, save_dir: Path):
    rounds = [h["round"] for h in history]
    accs   = [h["test_acc"] for h in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rounds, accs, "g-o", linewidth=2, markersize=4, label="Global Test Acc")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"{exp_name} — DP-FedAvg (no checkpoint)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{exp_name}_fl_progress.png", dpi=150, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────────────────────────────
# Single DP-FedAvg run (no checkpointing, no FLClient/FLServer)
# ────────────────────────────────────────────────────────────
def run_dp_fedavg(
    config: dict,
    exp_name: str,
    noise_multiplier: float,
    clip_norm: float,
    num_clients: int,
    rounds: int,
    local_epochs: int,
    task: str,
    batch_size: int,
    results_dir: Path,
    device: torch.device,
    seed: int,
) -> dict:
    set_seed(seed)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  DP-FedAvg (NO CHECKPOINT): {exp_name}")
    print(f"  Task: {task} | Clients: {num_clients} | Rounds: {rounds}")
    print(f"  Noise σ={noise_multiplier} | Clip C={clip_norm}")
    print(f"  ALL clients train and send updates EVERY round.")
    print(f"{'='*62}")

    # ── Dataset setup ────────────────────────────────────────
    root = Path(config["dataset"]["root_dir"])
    train_img_dir = root / config["dataset"]["train_images"]
    test_img_dir  = root / config["dataset"]["test_images"]
    train_csv     = root / config["dataset"]["train_labels"]
    test_csv      = root / config["dataset"]["test_labels"]

    num_classes = 2 if task == "binary" else 5
    class_names = ["No DR", "DR"] if task == "binary" else config["dataset"]["class_names"]

    img_size   = config["preprocessing"]["image_size"]
    norm_mean  = tuple(config["preprocessing"]["normalize_mean"])
    norm_std   = tuple(config["preprocessing"]["normalize_std"])

    train_transform = get_augmentation_by_name("baseline", img_size, norm_mean, norm_std)
    val_transform   = get_preprocessing_transform(img_size, norm_mean, norm_std)

    train_ds_aug = IDRiDDataset(str(train_img_dir), str(train_csv), transform=train_transform, task=task)
    train_ds_val = IDRiDDataset(str(train_img_dir), str(train_csv), transform=val_transform,   task=task)
    test_ds      = IDRiDDataset(str(test_img_dir),  str(test_csv),  transform=val_transform,   task=task)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train: {len(train_ds_aug)} | Test: {len(test_ds)}")

    # ── Split data IID across clients ────────────────────────
    indices = list(range(len(train_ds_aug)))
    random.shuffle(indices)
    split_size = len(indices) // num_clients

    client_train_loaders = []
    for i in range(num_clients):
        start = i * split_size
        end   = (i + 1) * split_size if i < num_clients - 1 else len(indices)
        c_idx = indices[start:end]
        c_ds  = Subset(train_ds_aug, c_idx)
        c_loader = DataLoader(c_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        client_train_loaders.append(c_loader)
        print(f"  Client {i}: {len(c_idx)} samples")

    # ── Global model ─────────────────────────────────────────
    global_model = create_model("resnet50", num_classes=num_classes, pretrained=True).to(device)
    global_weights = {k: v.cpu() for k, v in global_model.state_dict().items()}

    criterion = create_loss("cross_entropy", device=str(device))

    lr           = config["training"]["learning_rate"]
    momentum     = config["training"]["momentum"]
    weight_decay = config["training"]["weight_decay"]

    history       = []
    best_test_acc = 0.0

    # ── Federated rounds ─────────────────────────────────────
    for rnd in range(1, rounds + 1):
        print(f"\n  --- Round {rnd}/{rounds} ---")
        client_updates = []

        for i, c_loader in enumerate(client_train_loaders):
            # Give every client a fresh copy of global weights
            client_model = create_model("resnet50", num_classes=num_classes, pretrained=False).to(device)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            # DP local training — NO checkpoint check whatsoever
            new_weights, n_samples, avg_loss = dp_local_train(
                model=client_model,
                train_loader=c_loader,
                criterion=criterion,
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                noise_multiplier=noise_multiplier,
                clip_norm=clip_norm,
            )

            client_updates.append((n_samples, new_weights))
            print(f"    Client {i}: loss={avg_loss:.4f} | samples={n_samples}")

        # Aggregate — every client always contributes
        global_weights = fedavg_aggregate(global_weights, client_updates)
        global_model.load_state_dict(global_weights)

        # Round accuracy
        test_acc = eval_accuracy(global_model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
        print(f"  Global Test Acc: {test_acc:.4f}  (Best: {best_test_acc:.4f})")
        history.append({"round": rnd, "test_acc": test_acc})

    # ── Final comprehensive evaluation ───────────────────────
    _, _, preds, targets, probs = full_evaluate(global_model, test_loader, device, criterion)
    metrics = compute_all_metrics(targets, preds, probs, class_names, num_classes)
    print_metrics_summary(metrics, title=f"{exp_name} — Final")
    save_metrics(metrics, str(results_dir), exp_name)

    # Save history CSV
    hist_csv = results_dir / f"{exp_name}_fl_history.csv"
    with open(hist_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "test_acc"])
        writer.writeheader()
        writer.writerows(history)

    save_fl_plot(history, exp_name, results_dir)

    return {
        "experiment": exp_name,
        "noise_multiplier": noise_multiplier,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "auc": metrics.get("auc"),
        "best_round_acc": best_test_acc,
    }


# ────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DP-FedAvg WITHOUT Checkpointing — IDRiD"
    )
    parser.add_argument("--config",   type=str, default="configs/base.yaml")
    parser.add_argument("--rounds",   type=int, default=20)
    parser.add_argument("--clients",  type=int, default=3)
    parser.add_argument("--epochs",   type=int, default=1,   help="Local epochs per round")
    parser.add_argument("--clip",     type=float, default=1.0, help="DP gradient clip norm C")
    parser.add_argument("--task",     type=str,  default="binary", choices=["binary", "multiclass"])
    parser.add_argument(
        "--noise",
        type=float, nargs="+",
        default=[0.56, 0.9, 1.2, 3.5, 8.0],
        help="List of noise multipliers to evaluate (e.g. --noise 0.56 0.9 1.2)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    device      = get_device()
    results_dir = Path(config["logging"]["save_dir"]) / "dp_no_checkpoint"
    results_dir.mkdir(parents=True, exist_ok=True)

    batch_size = config["training"]["batch_size"]
    all_results = []

    print(f"\n{'#'*62}")
    print(f"  DP-FedAvg WITHOUT CHECKPOINTING — IDRiD ({args.task})")
    print(f"  Noise levels: {args.noise}")
    print(f"  Rounds: {args.rounds} | Clients: {args.clients} | LocalEpochs: {args.epochs}")
    print(f"{'#'*62}")

    for noise in args.noise:
        exp_name = f"DP_NoCkpt_σ{noise}_{args.task}".replace(".", "p")
        result = run_dp_fedavg(
            config        = config,
            exp_name      = exp_name,
            noise_multiplier = noise,
            clip_norm     = args.clip,
            num_clients   = args.clients,
            rounds        = args.rounds,
            local_epochs  = args.epochs,
            task          = args.task,
            batch_size    = batch_size,
            results_dir   = results_dir,
            device        = device,
            seed          = config["seed"],
        )
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY — DP-FedAvg No Checkpoint ({args.task.upper()}, IDRiD)")
    print(f"{'='*70}")
    print(f"  {'Noise σ':<10} {'Accuracy':<12} {'F1 Macro':<12} {'AUC':<10} {'Best Rnd Acc'}")
    print(f"  {'-'*60}")
    for r in all_results:
        auc = f"{r['auc']:.4f}" if r.get("auc") is not None else "N/A"
        print(
            f"  {r['noise_multiplier']:<10} {r['accuracy']:<12.4f} "
            f"{r['f1_macro']:<12.4f} {auc:<10} {r['best_round_acc']:.4f}"
        )
    print(f"{'='*70}\n")

    # Save summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {results_dir}")
    print(f"Summary JSON:     {summary_path}")


if __name__ == "__main__":
    main()
