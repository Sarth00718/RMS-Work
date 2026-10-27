"""
project_aptos/generate_figures.py
=================================
Reproducible IEEE-format publication figure generator for APTOS 2019 Blindness Detection.
Covers:
  1. class_distribution.pdf / .png
  2. sample_images.pdf / .png
  3. training_validation_curves.pdf / .png
  4. confusion_matrix.pdf / .png
  5. model_comparison.pdf / .png
  6. fl_convergence.pdf / .png
  7. dp_noise_comparison.pdf / .png
  8. communication_efficiency.pdf / .png

Outputs are saved in:
  - project_aptos/figures/
  - figures/aptos/ (synced)
"""

import os
import sys
import json
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "sans-serif"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "navy": "#1B365D",
    "blue": "#20639B",
    "teal": "#008080",
    "green": "#2E7D32",
    "amber": "#ED553B",
    "coral": "#D9381E",
    "purple": "#6A1B9A",
    "gold": "#F6D55C",
    "crimson": "#C62828",
}

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Resolve paths whether called from project_aptos/ or root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATASET_DIR = os.path.join(ROOT_DIR, "aptos_dataset")
if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(SCRIPT_DIR, "aptos_dataset")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOCAL_FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
ROOT_FIG_DIR = os.path.join(ROOT_DIR, "figures", "aptos")

for d in [LOCAL_FIG_DIR, ROOT_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

manifest = []

def save_fig(fig, base_filename):
    for target_dir in [LOCAL_FIG_DIR, ROOT_FIG_DIR]:
        pdf_path = os.path.join(target_dir, base_filename + ".pdf")
        png_path = os.path.join(target_dir, base_filename + ".png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {base_filename}.pdf and .png")

# 1. Dataset class distribution
def plot_class_distribution():
    df_train = pd.read_csv(os.path.join(DATASET_DIR, "train_1.csv"))
    df_val = pd.read_csv(os.path.join(DATASET_DIR, "valid.csv"))
    df_test = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

    train_c = [df_train[df_train['diagnosis'] == i].shape[0] for i in range(5)]
    val_c = [df_val[df_val['diagnosis'] == i].shape[0] for i in range(5)]
    test_c = [df_test[df_test['diagnosis'] == i].shape[0] for i in range(5)]
    tot_c = [t + v + te for t, v, te in zip(train_c, val_c, test_c)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={'width_ratios': [2.2, 1.0]})
    x = np.arange(5)
    w = 0.25

    r1 = ax1.bar(x - w, train_c, w, label=f"Train (N={sum(train_c):,})", color=COLORS["navy"])
    r2 = ax1.bar(x, val_c, w, label=f"Val (N={sum(val_c):,})", color=COLORS["blue"])
    r3 = ax1.bar(x + w, test_c, w, label=f"Test (N={sum(test_c):,})", color=COLORS["teal"])

    ax1.set_ylabel("Number of Samples", fontweight="bold")
    ax1.set_title("(a) Multi-Class Distribution by Split", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(Grade {i})" for i, c in enumerate(CLASS_NAMES)], fontsize=8)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.set_ylim(0, max(train_c) * 1.22)

    for i, (r, tot) in enumerate(zip(r1, tot_c)):
        ax1.text(r.get_x() + w*1.5, max(train_c[i], val_c[i], test_c[i]) + 30,
                 f"Tot: {tot}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#222222")

    # Binary Split
    bin_train = [train_c[0], sum(train_c[1:])]
    bin_val = [val_c[0], sum(val_c[1:])]
    bin_test = [test_c[0], sum(test_c[1:])]
    bx = np.arange(2)
    bw = 0.25

    ax2.bar(bx - bw, bin_train, bw, label="Train", color=COLORS["navy"])
    ax2.bar(bx, bin_val, bw, label="Val", color=COLORS["blue"])
    ax2.bar(bx + bw, bin_test, bw, label="Test", color=COLORS["teal"])
    ax2.set_title("(b) Binary DR Split", fontweight="bold", pad=8)
    ax2.set_xticks(bx)
    ax2.set_xticklabels(["No DR\n(Grade 0)", "DR\n(Grades 1-4)"], fontsize=8)
    ax2.set_ylabel("Sample Count", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.suptitle("APTOS 2019 Blindness Detection: Dataset Partition & Label Distribution", fontsize=10.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "class_distribution")

# 2. Representative sample images
def plot_sample_images():
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.1))
    sample_ids = [
        (0, "1b3647865779", "Grade 0: No DR"),
        (1, "1b329a127307", "Grade 1: Mild"),
        (2, "1ae8c165fd53", "Grade 2: Moderate"),
        (3, "1b495ac025b7", "Grade 3: Severe"),
        (4, "1b32e1d775ea", "Grade 4: Proliferative"),
    ]
    for ax, (grade, img_id, title) in zip(axes, sample_ids):
        p1 = os.path.join(DATASET_DIR, "train_images", "train_images", f"{img_id}.png")
        if not os.path.exists(p1):
            p1 = os.path.join(DATASET_DIR, "train_images", f"{img_id}.png")
        im = Image.open(p1)
        ax.imshow(im)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        ax.set_xlabel(f"ID: {img_id[:8]}..\n{im.size[0]}x{im.size[1]}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("APTOS 2019 Representative Clinical Fundus Retinal Images", fontsize=10.0, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "sample_images")

# 3. Training and validation curves
def plot_training_curves():
    df_h_bin = pd.read_csv(os.path.join(RESULTS_DIR, "Centralized_Baseline_binary_history.csv"))
    df_h_mul = pd.read_csv(os.path.join(RESULTS_DIR, "Centralized_Baseline_multiclass_history.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), sharex=True)
    epochs_bin = df_h_bin["epoch"] + 1
    epochs_mul = df_h_mul["epoch"] + 1

    axes[0, 0].plot(epochs_bin, df_h_bin["train_loss"], "o-", color=COLORS["navy"], label="Train Loss")
    axes[0, 0].plot(epochs_bin, df_h_bin["val_loss"], "s--", color=COLORS["coral"], label="Val Loss")
    axes[0, 0].set_ylabel("Loss", fontweight="bold")
    axes[0, 0].set_title("(a) Centralized Binary: Loss vs Epochs", fontweight="bold")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].plot(epochs_bin, df_h_bin["train_acc"]*100, "o-", color=COLORS["navy"], label="Train Acc")
    axes[0, 1].plot(epochs_bin, df_h_bin["val_acc"]*100, "s--", color=COLORS["green"], label="Val Acc")
    axes[0, 1].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[0, 1].set_title("(b) Centralized Binary: Accuracy vs Epochs", fontweight="bold")
    axes[0, 1].legend(loc="lower right", fontsize=8)

    axes[1, 0].plot(epochs_mul, df_h_mul["train_loss"], "o-", color=COLORS["navy"], label="Train Loss")
    axes[1, 0].plot(epochs_mul, df_h_mul["val_loss"], "s--", color=COLORS["coral"], label="Val Loss")
    axes[1, 0].set_xlabel("Epoch", fontweight="bold")
    axes[1, 0].set_ylabel("Loss (Smoothing 0.1)", fontweight="bold")
    axes[1, 0].set_title("(c) Centralized Multiclass: Loss vs Epochs", fontweight="bold")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].plot(epochs_mul, df_h_mul["train_acc"]*100, "o-", color=COLORS["navy"], label="Train Acc")
    axes[1, 1].plot(epochs_mul, df_h_mul["val_acc"]*100, "s--", color=COLORS["green"], label="Val Acc")
    axes[1, 1].set_xlabel("Epoch", fontweight="bold")
    axes[1, 1].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[1, 1].set_title("(d) Centralized Multiclass: Accuracy vs Epochs", fontweight="bold")
    axes[1, 1].legend(loc="lower right", fontsize=8)

    fig.suptitle("APTOS 2019 Centralized ResNet50 Convergence Curves (Cosine Annealing)", fontsize=10.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "training_validation_curves")

# 4. Confusion Matrices (N=366)
def plot_confusion_matrices():
    with open(os.path.join(RESULTS_DIR, "Centralized_Baseline_binary_metrics.json")) as f:
        cm_cb = json.load(f)["confusion_matrix"]
    with open(os.path.join(RESULTS_DIR, "Centralized_Baseline_multiclass_metrics.json")) as f:
        cm_cm = json.load(f)["confusion_matrix"]
    with open(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_binary_metrics.json")) as f:
        cm_fb = json.load(f)["confusion_matrix"]
    with open(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_multiclass_metrics.json")) as f:
        cm_fm = json.load(f)["confusion_matrix"]

    cms = [cm_cb, cm_cm, cm_fb, cm_fm]
    titles = [
        "(a) Centralized Binary (Acc: 98.36%, AUC: 0.996)",
        "(b) Centralized Multiclass 5-Class (Acc: 84.70%, AUC: 0.965)",
        "(c) FedAvg (No Noise) Binary (Acc: 97.27%, AUC: 0.994)",
        "(d) FedAvg (No Noise) Multiclass (Acc: 77.60%, AUC: 0.948)"
    ]
    clists = [["No DR", "DR"], CLASS_NAMES, ["No DR", "DR"], CLASS_NAMES]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    axes = axes.flatten()

    for idx, (cm_data, title, cnames) in enumerate(zip(cms, titles, clists)):
        ax = axes[idx]
        cm = np.array(cm_data)
        n_classes = len(cnames)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=cm_norm, where=row_sums != 0)
        
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=8)
        
        tick_marks = np.arange(n_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(cnames, rotation=25 if n_classes > 2 else 0, ha="right" if n_classes > 2 else "center", fontsize=8)
        ax.set_yticklabels(cnames, fontsize=8)
        ax.set_xlabel("Predicted Label", fontsize=8.5)
        ax.set_ylabel("True Label", fontsize=8.5)
        ax.grid(False)

        for i in range(n_classes):
            for j in range(n_classes):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                text = f"{count}\n({pct:.1f}%)" if n_classes <= 2 else (f"{count}" if count > 0 else "0")
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5 if n_classes <= 2 else 7.5, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label("Recall (Row Normalized)", fontsize=7.5)

    fig.suptitle("APTOS 2019 Held-Out Test Set Confusion Matrices (N=366)", fontsize=10.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "confusion_matrix")

# 5. Model comparison
def plot_model_comparison():
    models_to_compare = [
        ("Centralized (Binary)", "Centralized_Baseline_binary"),
        ("FedAvg (Binary, No Noise)", "FedAvg_NoNoise_binary"),
        ("DP-FedAvg (Binary, σ=0.56)", "DP_FedAvg_Noise_0p56_binary"),
        ("DP-FedAvg (Binary, σ=1.2)", "DP_FedAvg_Noise_1p2_binary"),
        ("DP-FedAvg (Binary, σ=8.0)", "DP_FedAvg_Noise_8p0_binary"),
        ("Centralized (Multiclass)", "Centralized_Baseline_multiclass"),
        ("FedAvg (Multi, No Noise)", "FedAvg_NoNoise_multiclass"),
        ("DP-FedAvg (Multi, σ=1.2)", "DP_FedAvg_Noise_1p2_multiclass"),
    ]
    comp_rows = []
    for label, exp_id in models_to_compare:
        with open(os.path.join(RESULTS_DIR, f"{exp_id}_metrics.json")) as f:
            d = json.load(f)
            comp_rows.append({
                "Model": label,
                "Accuracy": d["accuracy"] * 100,
                "Macro F1": d["f1_macro"] * 100,
                "Macro AUC": d["auc"] * 100,
                "Macro Precision": d["precision_macro"] * 100,
                "Macro Recall": d["recall_macro"] * 100
            })
    df_comp = pd.DataFrame(comp_rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    m_indices = np.arange(len(df_comp))
    b_w = 0.22

    ax.barh(m_indices - b_w*1.5, df_comp["Accuracy"], b_w, label="Accuracy (%)", color=COLORS["navy"])
    ax.barh(m_indices - b_w*0.5, df_comp["Macro F1"], b_w, label="Macro F1 (%)", color=COLORS["blue"])
    ax.barh(m_indices + b_w*0.5, df_comp["Macro AUC"], b_w, label="Macro AUC (%)", color=COLORS["teal"])
    ax.barh(m_indices + b_w*1.5, df_comp["Macro Recall"], b_w, label="Macro Recall (%)", color=COLORS["gold"])

    ax.set_yticks(m_indices)
    ax.set_yticklabels(df_comp["Model"], fontsize=8.5)
    ax.set_xlabel("Metric Score (%)", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    fig.suptitle("APTOS 2019: Quantitative Model Benchmark on Held-Out Test Set (N=366)", fontsize=10.5, fontweight="bold", y=0.98)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=True, framealpha=0.95, fontsize=8, edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "model_comparison")

# 6. FL convergence
def plot_fl_convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=False)
    rounds = np.arange(1, 11)

    df_nn_bin = pd.read_csv(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_binary_fl_history.csv"))
    ax1.plot(rounds, df_nn_bin["test_acc"]*100, "o-", color=COLORS["navy"], lw=2.0, label="FedAvg (No Noise)")

    dp_noises = [("0p56", "σ=0.56", COLORS["green"], "^--"),
                 ("0p9", "σ=0.90", COLORS["blue"], "v--"),
                 ("1p2", "σ=1.20", COLORS["amber"], "s--"),
                 ("3p5", "σ=3.50", COLORS["purple"], "d--"),
                 ("8p0", "σ=8.00", COLORS["crimson"], "x--")]
    for code, lbl, clr, ls in dp_noises:
        p = os.path.join(RESULTS_DIR, f"DP_FedAvg_Noise_{code}_binary_fl_history.csv")
        df_dp = pd.read_csv(p)
        ax1.plot(rounds, df_dp["test_acc"]*100, ls, color=clr, lw=1.3, label=lbl)

    ax1.set_title("(a) Binary FL: Global Test Acc vs Rounds", fontweight="bold", pad=8)
    ax1.set_xlabel("Communication Round", fontweight="bold")
    ax1.set_ylabel("Held-Out Test Accuracy (%)", fontweight="bold")
    ax1.set_xticks(rounds)
    ax1.set_ylim(35, 102)
    ax1.legend(fontsize=7.5, loc="lower left", ncol=2)

    df_nn_mul = pd.read_csv(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_multiclass_fl_history.csv"))
    ax2.plot(rounds, df_nn_mul["test_acc"]*100, "o-", color=COLORS["navy"], lw=2.0, label="FedAvg (No Noise)")

    for code, lbl, clr, ls in dp_noises:
        p = os.path.join(RESULTS_DIR, f"DP_FedAvg_Noise_{code}_multiclass_fl_history.csv")
        df_dp = pd.read_csv(p)
        ax2.plot(rounds, df_dp["test_acc"]*100, ls, color=clr, lw=1.3, label=lbl)

    ax2.set_title("(b) Multiclass FL: Global Test Acc vs Rounds", fontweight="bold", pad=8)
    ax2.set_xlabel("Communication Round", fontweight="bold")
    ax2.set_ylabel("Held-Out Test Accuracy (%)", fontweight="bold")
    ax2.set_xticks(rounds)
    ax2.set_ylim(0, 85)
    ax2.legend(fontsize=7.5, loc="lower left", ncol=2)

    fig.suptitle("APTOS 2019 Federated Learning Convergence across 10 Communication Rounds (K=3)", fontsize=10.0, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fl_convergence")

# 7. DP noise comparison
def plot_dp_noise_comparison():
    noises_val = [0.0, 0.56, 0.9, 1.2, 3.5, 8.0]
    dp_keys = ["FedAvg_NoNoise", "DP_FedAvg_Noise_0p56", "DP_FedAvg_Noise_0p9",
               "DP_FedAvg_Noise_1p2", "DP_FedAvg_Noise_3p5", "DP_FedAvg_Noise_8p0"]

    bin_accs, bin_f1s, bin_aucs = [], [], []
    mul_accs, mul_f1s, mul_aucs = [], [], []

    for k in dp_keys:
        with open(os.path.join(RESULTS_DIR, f"{k}_binary_metrics.json")) as f:
            d = json.load(f)
            bin_accs.append(d["accuracy"] * 100)
            bin_f1s.append(d["f1_macro"] * 100)
            bin_aucs.append(d["auc"] * 100)
        with open(os.path.join(RESULTS_DIR, f"{k}_multiclass_metrics.json")) as f:
            d = json.load(f)
            mul_accs.append(d["accuracy"] * 100)
            mul_f1s.append(d["f1_macro"] * 100)
            mul_aucs.append(d["auc"] * 100)

    ablation_p = os.path.join(RESULTS_DIR, "dp_no_checkpoint", "DP_NoCkpt_σ0p56_binary_metrics.json")
    ablation_acc = None
    if os.path.exists(ablation_p):
        with open(ablation_p, encoding="utf-8") as f:
            ablation_acc = json.load(f)["accuracy"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    ax1.plot(noises_val, bin_accs, "o-", color=COLORS["navy"], lw=1.8, label="Test Accuracy")
    ax1.plot(noises_val, bin_f1s, "s--", color=COLORS["coral"], lw=1.6, label="Macro F1-Score")
    ax1.plot(noises_val, bin_aucs, "^-.", color=COLORS["teal"], lw=1.5, label="Macro AUC")
    if ablation_acc:
        ax1.scatter([0.56], [ablation_acc], color=COLORS["purple"], marker="X", s=90, zorder=5, label="Ablation: No Ckpt (σ=0.56)")
    ax1.set_title("(a) Binary DR Task", fontweight="bold", pad=8)
    ax1.set_xlabel("DP Noise Multiplier (σ)", fontweight="bold")
    ax1.set_ylabel("Metric Score (%)", fontweight="bold")
    ax1.set_ylim(20, 105)
    ax1.legend(loc="lower left", fontsize=7.5)

    ax2.plot(noises_val, mul_accs, "o-", color=COLORS["navy"], lw=1.8, label="Test Accuracy")
    ax2.plot(noises_val, mul_f1s, "s--", color=COLORS["coral"], lw=1.6, label="Macro F1-Score")
    ax2.plot(noises_val, mul_aucs, "^-.", color=COLORS["teal"], lw=1.5, label="Macro AUC")
    ax2.set_title("(b) Multiclass (5-Grade) DR Task", fontweight="bold", pad=8)
    ax2.set_xlabel("DP Noise Multiplier (σ)", fontweight="bold")
    ax2.set_ylabel("Metric Score (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower left", fontsize=7.5)

    fig.suptitle("APTOS 2019: Impact of DP Noise Multiplier (σ) on Model Utility", fontsize=10.0, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "dp_noise_comparison")

# 8. Communication Efficiency
def plot_communication_efficiency():
    rounds = np.arange(1, 11)
    df_nn_bin = pd.read_csv(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_binary_fl_history.csv"))
    df_nn_mul = pd.read_csv(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_multiclass_fl_history.csv"))

    bytes_per_round_mb = 25.56 * 4 * 3  # ~306.7 MB / round
    cum_mb = rounds * bytes_per_round_mb

    fig, ax1 = plt.subplots(figsize=(7.5, 3.4))
    ax2 = ax1.twinx()

    l1 = ax1.plot(rounds, df_nn_bin["test_acc"]*100, "o-", color=COLORS["navy"], lw=2, label="FedAvg Binary Acc (%)")
    l2 = ax1.plot(rounds, df_nn_mul["test_acc"]*100, "s-", color=COLORS["blue"], lw=2, label="FedAvg Multi Acc (%)")
    l3 = ax2.plot(rounds, cum_mb / 1024, "d--", color=COLORS["amber"], lw=1.8, label="Transmitted Data (GB)")

    ax1.set_xlabel("Communication Round (K=3 Clients, E=1 Local Epoch)", fontweight="bold")
    ax1.set_ylabel("Global Test Accuracy (%)", color=COLORS["navy"], fontweight="bold")
    ax2.set_ylabel("Cumulative Transmitted Data (GB)", color=COLORS["amber"], fontweight="bold")
    ax1.set_xticks(rounds)
    ax1.set_ylim(40, 102)
    ax2.set_ylim(0, 3.5)
    ax2.grid(False)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=8, framealpha=0.9)
    plt.title("APTOS 2019: Communication Efficiency and Network Overhead (ResNet50 Backbone)", fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, "communication_efficiency")

if __name__ == "__main__":
    t0 = time.time()
    print("Generating all APTOS 2019 research paper figures...")
    plot_class_distribution()
    plot_sample_images()
    plot_training_curves()
    plot_confusion_matrices()
    plot_model_comparison()
    plot_fl_convergence()
    plot_dp_noise_comparison()
    plot_communication_efficiency()
    print(f"APTOS figures generated successfully in {time.time()-t0:.2f} seconds.")
