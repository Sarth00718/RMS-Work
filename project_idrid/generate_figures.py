"""
project_idrid/generate_figures.py
=================================
Reproducible IEEE-format publication figure generator for IDRiD Disease Grading Dataset.
Covers:
  1. class_distribution.pdf / .png
  2. sample_images.pdf / .png
  3. training_validation_curves.pdf / .png
  4. confusion_matrix.pdf / .png
  5. model_comparison.pdf / .png
  6. dp_noise_comparison.pdf / .png
  7. communication_efficiency.pdf / .png

Outputs are saved in:
  - project_idrid/figures/
  - figures/idrid/ (synced)
"""

import os
import sys
import json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

GT_DIR = os.path.join(ROOT_DIR, "B. Disease Grading", "2. Groundtruths")
if not os.path.exists(GT_DIR):
    GT_DIR = os.path.join(SCRIPT_DIR, "B. Disease Grading", "2. Groundtruths")

IMG_DIR = os.path.join(ROOT_DIR, "B. Disease Grading", "1. Original Images", "a. Training Set")
if not os.path.exists(IMG_DIR):
    IMG_DIR = os.path.join(SCRIPT_DIR, "B. Disease Grading", "1. Original Images", "a. Training Set")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
LOCAL_FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
ROOT_FIG_DIR = os.path.join(ROOT_DIR, "figures", "idrid")

for d in [LOCAL_FIG_DIR, ROOT_FIG_DIR]:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, base_filename):
    for target_dir in [LOCAL_FIG_DIR, ROOT_FIG_DIR]:
        pdf_path = os.path.join(target_dir, base_filename + ".pdf")
        png_path = os.path.join(target_dir, base_filename + ".png")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {base_filename}.pdf and .png")

# 1. Class Distribution
def plot_class_distribution():
    gt_tr = os.path.join(GT_DIR, "a. IDRiD_Disease Grading_Training Labels.csv")
    gt_te = os.path.join(GT_DIR, "b. IDRiD_Disease Grading_Testing Labels.csv")
    df_train = pd.read_csv(gt_tr)
    df_test = pd.read_csv(gt_te)

    tr_counts = [df_train[df_train['Retinopathy grade'] == i].shape[0] for i in range(5)]
    te_counts = [df_test[df_test['Retinopathy grade'] == i].shape[0] for i in range(5)]
    tot_counts = [tr + te for tr, te in zip(tr_counts, te_counts)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={'width_ratios': [2.2, 1.0]})
    x = np.arange(5)
    w = 0.32

    r1 = ax1.bar(x - w/2, tr_counts, w, label=f"Train (N={sum(tr_counts)})", color=COLORS["navy"])
    r2 = ax1.bar(x + w/2, te_counts, w, label=f"Test (N={sum(te_counts)})", color=COLORS["coral"])

    ax1.set_ylabel("Number of Samples", fontweight="bold")
    ax1.set_title("(a) Multi-Class Distribution by Split", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(Grade {i})" for i, c in enumerate(CLASS_NAMES)], fontsize=8)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.set_ylim(0, max(tr_counts) * 1.22)

    for i, (r, tot) in enumerate(zip(r1, tot_counts)):
        ax1.text(r.get_x() + w, max(tr_counts[i], te_counts[i]) + 5,
                 f"Tot: {tot}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#222222")

    # Binary Split
    bin_tr = [tr_counts[0], sum(tr_counts[1:])]
    bin_te = [te_counts[0], sum(te_counts[1:])]
    bx = np.arange(2)
    bw = 0.32

    ax2.bar(bx - bw/2, bin_tr, bw, label="Train", color=COLORS["navy"])
    ax2.bar(bx + bw/2, bin_te, bw, label="Test", color=COLORS["coral"])
    ax2.set_title("(b) Binary DR Split", fontweight="bold", pad=8)
    ax2.set_xticks(bx)
    ax2.set_xticklabels(["No DR\n(Grade 0)", "DR\n(Grades 1-4)"], fontsize=8)
    ax2.set_ylabel("Sample Count", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8)

    fig.suptitle("IDRiD Disease Grading: Dataset Partition & Label Distribution", fontsize=10.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "class_distribution")

# 2. Representative sample images
def plot_sample_images():
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.1))
    sample_ids = [
        (0, "IDRiD_118", "Grade 0: No DR"),
        (1, "IDRiD_021", "Grade 1: Mild"),
        (2, "IDRiD_003", "Grade 2: Moderate"),
        (3, "IDRiD_001", "Grade 3: Severe"),
        (4, "IDRiD_005", "Grade 4: Proliferative"),
    ]
    for ax, (grade, img_name, title) in zip(axes, sample_ids):
        p1 = os.path.join(IMG_DIR, f"{img_name}.jpg")
        im = Image.open(p1)
        ax.imshow(im)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        ax.set_xlabel(f"{img_name}\n{im.size[0]}x{im.size[1]}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    fig.suptitle("IDRiD Representative Clinical Fundus Retinal Images (Original 4288x2848)", fontsize=10.0, fontweight="bold", y=1.02)
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
    axes[0, 0].set_title("(a) IDRiD Centralized Binary: Loss", fontweight="bold")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].plot(epochs_bin, df_h_bin["train_acc"]*100, "o-", color=COLORS["navy"], label="Train Acc")
    axes[0, 1].plot(epochs_bin, df_h_bin["val_acc"]*100, "s--", color=COLORS["green"], label="Val Acc")
    axes[0, 1].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[0, 1].set_title("(b) IDRiD Centralized Binary: Accuracy", fontweight="bold")
    axes[0, 1].legend(loc="lower right", fontsize=8)

    axes[1, 0].plot(epochs_mul, df_h_mul["train_loss"], "o-", color=COLORS["navy"], label="Train Loss")
    axes[1, 0].plot(epochs_mul, df_h_mul["val_loss"], "s--", color=COLORS["coral"], label="Val Loss")
    axes[1, 0].set_xlabel("Epoch", fontweight="bold")
    axes[1, 0].set_ylabel("Loss", fontweight="bold")
    axes[1, 0].set_title("(c) IDRiD Centralized Multiclass: Loss", fontweight="bold")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].plot(epochs_mul, df_h_mul["train_acc"]*100, "o-", color=COLORS["navy"], label="Train Acc")
    axes[1, 1].plot(epochs_mul, df_h_mul["val_acc"]*100, "s--", color=COLORS["green"], label="Val Acc")
    axes[1, 1].set_xlabel("Epoch", fontweight="bold")
    axes[1, 1].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[1, 1].set_title("(d) IDRiD Centralized Multiclass: Accuracy", fontweight="bold")
    axes[1, 1].legend(loc="lower right", fontsize=8)

    fig.suptitle("IDRiD Centralized ResNet50 Convergence Curves (19 Epochs)", fontsize=10.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "training_validation_curves")

# 4. Confusion Matrices (N=103)
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
        "(a) Centralized Binary (Acc: 78.64%, AUC: 0.862)",
        "(b) Centralized Multiclass 5-Class (Acc: 54.37%, AUC: 0.813)",
        "(c) FedAvg (No Noise) Binary (Acc: 76.70%, AUC: 0.841)",
        "(d) FedAvg (No Noise) Multiclass (Acc: 43.69%, AUC: 0.726)"
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

    fig.suptitle("IDRiD Held-Out Test Set Confusion Matrices (N=103)", fontsize=10.5, fontweight="bold", y=0.99)
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
    ax.barh(m_indices - b_w*0.5, df_comp["Macro F1"], b_w, label="Macro F1 (%)", color=COLORS["coral"])
    ax.barh(m_indices + b_w*0.5, df_comp["Macro AUC"], b_w, label="Macro AUC (%)", color=COLORS["teal"])
    ax.barh(m_indices + b_w*1.5, df_comp["Macro Recall"], b_w, label="Macro Recall (%)", color=COLORS["gold"])

    ax.set_yticks(m_indices)
    ax.set_yticklabels(df_comp["Model"], fontsize=8.5)
    ax.set_xlabel("Metric Score (%)", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.invert_yaxis()

    fig.suptitle("IDRiD: Quantitative Model Benchmark on Held-Out Test Set (N=103)", fontsize=10.5, fontweight="bold", y=0.98)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=True, framealpha=0.95, fontsize=8, edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "model_comparison")

# 6. DP Noise comparison
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
    ax1.set_ylim(20, 100)
    ax1.legend(loc="lower left", fontsize=7.5)

    ax2.plot(noises_val, mul_accs, "o-", color=COLORS["navy"], lw=1.8, label="Test Accuracy")
    ax2.plot(noises_val, mul_f1s, "s--", color=COLORS["coral"], lw=1.6, label="Macro F1-Score")
    ax2.plot(noises_val, mul_aucs, "^-.", color=COLORS["teal"], lw=1.5, label="Macro AUC")
    ax2.set_title("(b) Multiclass (5-Grade) DR Task", fontweight="bold", pad=8)
    ax2.set_xlabel("DP Noise Multiplier (σ)", fontweight="bold")
    ax2.set_ylabel("Metric Score (%)", fontweight="bold")
    ax2.set_ylim(0, 90)
    ax2.legend(loc="lower left", fontsize=7.5)

    fig.suptitle("IDRiD: Impact of DP Noise Multiplier (σ) on Model Utility", fontsize=10.0, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "dp_noise_comparison")

# 7. Communication Efficiency
def plot_communication_efficiency():
    rounds = np.arange(1, 11)
    bytes_per_round_mb = 25.56 * 4 * 3  # ~306.7 MB / round
    cum_mb = rounds * bytes_per_round_mb

    fig, ax1 = plt.subplots(figsize=(7.5, 3.4))
    ax2 = ax1.twinx()

    with open(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_binary_metrics.json")) as f:
        bin_final_acc = json.load(f)["accuracy"] * 100
    with open(os.path.join(RESULTS_DIR, "FedAvg_NoNoise_multiclass_metrics.json")) as f:
        mul_final_acc = json.load(f)["accuracy"] * 100

    l1 = ax1.axhline(bin_final_acc, color=COLORS["navy"], ls="-", lw=1.8, label=f"FedAvg Binary Test Acc ({bin_final_acc:.1f}%)")
    l2 = ax1.axhline(mul_final_acc, color=COLORS["blue"], ls="--", lw=1.8, label=f"FedAvg Multi Test Acc ({mul_final_acc:.1f}%)")
    l3 = ax2.plot(rounds, cum_mb / 1024, "d-.", color=COLORS["amber"], lw=1.8, label="Transmitted Data (GB)")

    ax1.set_xlabel("Communication Rounds (K=3 Clients, ResNet50 Backbone)", fontweight="bold")
    ax1.set_ylabel("Global Test Accuracy (%)", color=COLORS["navy"], fontweight="bold")
    ax2.set_ylabel("Cumulative Transmitted Data (GB)", color=COLORS["amber"], fontweight="bold")
    ax1.set_xticks(rounds)
    ax1.set_ylim(30, 95)
    ax2.set_ylim(0, 3.5)
    ax2.grid(False)

    lines = [l1, l2] + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=8, framealpha=0.9)
    plt.title("IDRiD: Communication Cost vs Model Convergence (ResNet50)", fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, "communication_efficiency")

if __name__ == "__main__":
    t0 = time.time()
    print("Generating all IDRiD research paper figures...")
    plot_class_distribution()
    plot_sample_images()
    plot_training_curves()
    plot_confusion_matrices()
    plot_model_comparison()
    plot_dp_noise_comparison()
    plot_communication_efficiency()
    print(f"IDRiD figures generated successfully in {time.time()-t0:.2f} seconds.")
