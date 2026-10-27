"""
project_messidor/generate_figures.py
====================================
Reproducible IEEE-format publication figure generator for UCI Messidor Feature Extraction Dataset.
Covers:
  1. class_distribution.pdf / .png
  2. feature_correlation_and_distribution.pdf / .png
  3. training_validation_curves.pdf / .png
  4. model_comparison.pdf / .png
  5. confusion_matrix.pdf / .png
  6. privacy_utility_tradeoff.pdf / .png
  7. dp_noise_comparison.pdf / .png
  8. fl_convergence.pdf / .png
  9. communication_efficiency.pdf / .png

Outputs are saved in:
  - project_messidor/figures/
  - figures/messidor/ (synced)
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

CLASS_NAMES = ["No DR Risk", "DR Risk"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SCRIPT_DIR)
import mlp_core

CSV_PATH = os.path.join(SCRIPT_DIR, "messidor_features.csv")
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
LOCAL_FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
ROOT_FIG_DIR = os.path.join(ROOT_DIR, "figures", "messidor")

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

# Load and clean data
df_messidor = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)

# 1. Class Distribution
def plot_class_distribution():
    c0 = (df_messidor["Class"] == 0).sum()
    c1 = (df_messidor["Class"] == 1).sum()
    tot = len(df_messidor)

    n_tr_0, n_tr_1 = int(round(c0 * 0.7)), int(round(c1 * 0.7))
    n_va_0, n_va_1 = int(round(c0 * 0.15)), int(round(c1 * 0.15))
    n_te_0, n_te_1 = c0 - n_tr_0 - n_va_0, c1 - n_tr_1 - n_va_1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={'width_ratios': [1.5, 1.0]})
    x = np.arange(2)
    w = 0.25

    ax1.bar(x - w, [n_tr_0, n_tr_1], w, label=f"Train (N={n_tr_0+n_tr_1})", color=COLORS["navy"])
    ax1.bar(x, [n_va_0, n_va_1], w, label=f"Val (N={n_va_0+n_va_1})", color=COLORS["blue"])
    ax1.bar(x + w, [n_te_0, n_te_1], w, label=f"Test (N={n_te_0+n_te_1})", color=COLORS["teal"])

    ax1.set_title("(a) Train / Val / Test Partition", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Class 0: No DR Risk\n(Tot: {c0}, {c0/tot*100:.1f}%)",
                         f"Class 1: DR Risk\n(Tot: {c1}, {c1/tot*100:.1f}%)"], fontsize=8.5)
    ax1.set_ylabel("Number of Patients / Samples", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(0, max(n_tr_0, n_tr_1) * 1.2)

    ax2.pie([c0, c1], labels=["No Risk (0)", "Risk (1)"], autopct="%1.1f%%",
            startangle=90, colors=[COLORS["navy"], COLORS["amber"]],
            wedgeprops=dict(width=0.45, edgecolor="w", linewidth=1.5))
    ax2.set_title("(b) Overall Class Balance", fontweight="bold", pad=8)

    fig.suptitle("Messidor Clinical Feature Distribution & Class Balance (N=1,146)", fontsize=10.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "class_distribution")

# 2. Feature correlation and distribution
def plot_feature_correlation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={'width_ratios': [1.2, 1.0]})
    corr = df_messidor.corr()["Class"].drop("Class").sort_values(key=abs, ascending=False)
    top_names = corr.index[:8].tolist()
    top_vals = corr.values[:8]

    y_pos = np.arange(len(top_names))
    ax1.barh(y_pos, top_vals, color=[COLORS["navy"] if v > 0 else COLORS["coral"] for v in top_vals])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([n.replace('_', ' ') for n in top_names], fontsize=8)
    ax1.set_xlabel("Pearson Correlation with DR Risk", fontweight="bold")
    ax1.set_title("(a) Top Clinical Predictors", fontweight="bold", pad=8)
    ax1.axvline(0, color="gray", lw=0.8)
    ax1.invert_yaxis()

    sns.boxplot(x="Class", y="ma1", data=df_messidor, ax=ax2, hue="Class", palette=[COLORS["navy"], COLORS["amber"]],
                legend=False, showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["No DR Risk (0)", "DR Risk (1)"], fontsize=8.5)
    ax2.set_ylabel("Microaneurysm Count (ma1)", fontweight="bold")
    ax2.set_title("(b) Feature Separation (ma1)", fontweight="bold", pad=8)

    fig.suptitle("Messidor Clinical Features: Statistical Significance & Target Association", fontsize=10.0, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "feature_correlation_and_distribution")

# 3. Training and validation curves
def plot_training_curves():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.metrics import accuracy_score

    feat_m = df_messidor.drop(columns=["Class"]).copy()
    ma_cols = [c for c in feat_m.columns if c.startswith('ma')]
    ex_cols = [c for c in feat_m.columns if c.startswith('exudate')]

    feat_m['ma_mean'] = feat_m[ma_cols].mean(axis=1)
    feat_m['ma_max'] = feat_m[ma_cols].max(axis=1)
    feat_m['ma_slope'] = feat_m['ma6'] - feat_m['ma1']
    feat_m['ex_mean'] = feat_m[ex_cols].mean(axis=1)
    feat_m['ex_max'] = feat_m[ex_cols].max(axis=1)
    feat_m['ex_slope'] = feat_m['exudate8'] - feat_m['exudate1']
    feat_m['ma_ex_ratio'] = feat_m['ma_mean'] / (feat_m['ex_mean'] + 1e-6)
    feat_m['ma1_x_exudate1'] = feat_m['ma1'] * feat_m['exudate1']
    feat_m['distance_to_diameter'] = feat_m['macula_opticdisc_distance'] / (feat_m['opticdisc_diameter'] + 1e-6)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_data = poly.fit_transform(feat_m[['ma_mean', 'ex_mean', 'macula_opticdisc_distance', 'opticdisc_diameter']])
    poly_cols = [f'poly_{i}' for i in range(poly_data.shape[1])]
    poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=feat_m.index)
    feat_m = pd.concat([feat_m, poly_df], axis=1)

    X_m = feat_m.values.astype(np.float64)
    y_m = df_messidor["Class"].values.astype(np.float64)

    SEED = 42
    X_train_m, X_temp_m, y_train_m, y_temp_m = train_test_split(X_m, y_m, test_size=0.30, stratify=y_m, random_state=SEED)
    X_val_m, X_test_m, y_val_m, y_test_m = train_test_split(X_temp_m, y_temp_m, test_size=0.5, stratify=y_temp_m, random_state=SEED)

    scaler_m = StandardScaler().fit(X_train_m)
    X_tr_s, X_va_s, X_te_s = scaler_m.transform(X_train_m), scaler_m.transform(X_val_m), scaler_m.transform(X_test_m)

    mlp_m = mlp_core.NumpyMLP(n_in=X_tr_s.shape[1], h1=48, h2=24, seed=SEED)
    patience_m, bad_m, best_vl_m = 20, 0, np.inf
    m_hist = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for ep in range(120):
        mlp_m.train_standard(X_tr_s, y_train_m, epochs=1, batch_size=16, lr=0.005, l2=0.0005, seed=SEED+ep)
        tl = mlp_m.loss(X_tr_s, y_train_m)
        vl = mlp_m.loss(X_va_s, y_val_m)
        ta = accuracy_score(y_train_m, mlp_m.predict(X_tr_s))
        va = accuracy_score(y_val_m, mlp_m.predict(X_va_s))
        m_hist['epoch'].append(ep + 1)
        m_hist['train_loss'].append(tl)
        m_hist['val_loss'].append(vl)
        m_hist['train_acc'].append(ta)
        m_hist['val_acc'].append(va)
        if vl < best_vl_m - 1e-5:
            best_vl_m = vl
            bad_m = 0
        else:
            bad_m += 1
            if bad_m >= patience_m:
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))
    ax1.plot(m_hist['epoch'], m_hist['train_loss'], "o-", color=COLORS["navy"], label="Train Loss")
    ax1.plot(m_hist['epoch'], m_hist['val_loss'], "s--", color=COLORS["coral"], label="Val Loss")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontweight="bold")
    ax1.set_title("(a) Centralized MLP Loss", fontweight="bold", pad=8)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(m_hist['epoch'], np.array(m_hist['train_acc'])*100, "o-", color=COLORS["navy"], label="Train Acc")
    ax2.plot(m_hist['epoch'], np.array(m_hist['val_acc'])*100, "s--", color=COLORS["green"], label="Val Acc")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.set_title("(b) Centralized MLP Accuracy", fontweight="bold", pad=8)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(50, 95)

    fig.suptitle("Messidor: Centralized MLP Convergence with Early Stopping", fontsize=10.0, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "training_validation_curves")

# 4. Model comparison (10 ML Models CV)
def plot_model_comparison():
    df_cv = pd.read_csv(os.path.join(OUTPUTS_DIR, "cv_model_comparison_full.csv")).sort_values("roc_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    m_idx = np.arange(len(df_cv))
    b_w = 0.22

    ax.barh(m_idx - b_w*1.5, df_cv["accuracy"]*100, b_w, xerr=df_cv["std_accuracy"]*100,
            label="Accuracy (%) ± Std", color=COLORS["navy"], capsize=2)
    ax.barh(m_idx - b_w*0.5, df_cv["f1"]*100, b_w, label="F1-Score (%)", color=COLORS["blue"])
    ax.barh(m_idx + b_w*0.5, df_cv["roc_auc"]*100, b_w, label="ROC-AUC (%)", color=COLORS["teal"])
    ax.barh(m_idx + b_w*1.5, df_cv["precision"]*100, b_w, label="Precision (%)", color=COLORS["gold"])

    ax.set_yticks(m_idx)
    ax.set_yticklabels(df_cv["Unnamed: 0"], fontsize=8.5)
    ax.set_xlabel("Cross-Validation Score (5-Fold Stratified, %)", fontweight="bold")
    ax.set_xlim(50, 95)

    fig.suptitle("Messidor: 5-Fold Cross-Validation Benchmark across 10 Model Families", fontsize=10.5, fontweight="bold", y=0.98)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=True, framealpha=0.95, fontsize=8, edgecolor="#cccccc")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "model_comparison")

# 5. Confusion Matrices (N=172)
def plot_confusion_matrices():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import confusion_matrix

    feat = df_messidor.drop(columns=["Class"]).copy()
    ma_cols = [c for c in feat.columns if c.startswith('ma')]
    ex_cols = [c for c in feat.columns if c.startswith('exudate')]

    feat['ma_mean'] = feat[ma_cols].mean(axis=1)
    feat['ma_max'] = feat[ma_cols].max(axis=1)
    feat['ma_slope'] = feat['ma6'] - feat['ma1']
    feat['ex_mean'] = feat[ex_cols].mean(axis=1)
    feat['ex_max'] = feat[ex_cols].max(axis=1)
    feat['ex_slope'] = feat['exudate8'] - feat['exudate1']
    feat['ma_ex_ratio'] = feat['ma_mean'] / (feat['ex_mean'] + 1e-6)
    feat['ma1_x_exudate1'] = feat['ma1'] * feat['exudate1']
    feat['distance_to_diameter'] = feat['macula_opticdisc_distance'] / (feat['opticdisc_diameter'] + 1e-6)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_data = poly.fit_transform(feat[['ma_mean', 'ex_mean', 'macula_opticdisc_distance', 'opticdisc_diameter']])
    poly_cols = [f'poly_{i}' for i in range(poly_data.shape[1])]
    poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=feat.index)
    feat = pd.concat([feat, poly_df], axis=1)

    X = feat.values.astype(np.float64)
    y = df_messidor["Class"].values.astype(np.float64)

    SEED = 42
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

    with open(os.path.join(OUTPUTS_DIR, "preprocessor.pkl"), "rb") as f:
        scaler = pickle.load(f)["scaler"]
    X_test_s = scaler.transform(X_test)

    # 1. Centralized MLP
    with open(os.path.join(OUTPUTS_DIR, "centralized_model.pkl"), "rb") as f:
        cm_obj = pickle.load(f)
        mlp = cm_obj["model"]
        th = cm_obj["threshold"]
        p_c = mlp.predict_proba(X_test_s)
        cm_mlp = confusion_matrix(y_test, (p_c >= th).astype(int))

    # 2. FedAvg 3c IID
    with open(os.path.join(OUTPUTS_DIR, "fedavg_model_3c_iid.pkl"), "rb") as f:
        m3_iid = pickle.load(f)
        p_f3 = m3_iid.predict_proba(X_test_s)
        cm_fed3 = confusion_matrix(y_test, (p_f3 >= 0.5).astype(int))

    # 3. FedAvg 5c IID
    with open(os.path.join(OUTPUTS_DIR, "fedavg_model_5c_iid.pkl"), "rb") as f:
        m5_iid = pickle.load(f)
        p_f5 = m5_iid.predict_proba(X_test_s)
        cm_fed5 = confusion_matrix(y_test, (p_f5 >= 0.5).astype(int))

    # 4. FedAvg+DP 3c very strong privacy (epsilon=4.00)
    with open(os.path.join(OUTPUTS_DIR, "fedavg_dp_model_3c_very_strong_privacy.pkl"), "rb") as f:
        mdp3 = pickle.load(f)
        p_dp = mdp3.predict_proba(X_test_s)
        cm_dp = confusion_matrix(y_test, (p_dp >= 0.5).astype(int))

    cms = [cm_mlp, cm_fed3, cm_fed5, cm_dp]
    titles = [
        "(a) Centralized MLP (Acc: 74.42%, AUC: 0.822)",
        "(b) FedAvg IID (3 Clients, Acc: 70.35%, AUC: 0.810)",
        "(c) FedAvg IID (5 Clients, Acc: 72.09%, AUC: 0.791)",
        "(d) FedAvg+DP (ε=4.00, 3 Clients, Acc: 59.88%, AUC: 0.677)"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    axes = axes.flatten()

    for idx, (cm_data, title) in enumerate(zip(cms, titles)):
        ax = axes[idx]
        cm = np.array(cm_data)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=float)
        np.divide(cm, row_sums, out=cm_norm, where=row_sums != 0)
        
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=8)
        
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted Label", fontsize=8.5)
        ax.set_ylabel("True Label", fontsize=8.5)
        ax.grid(False)

        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=8.5, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.set_label("Recall (Row Normalized)", fontsize=7.5)

    fig.suptitle("Messidor Held-Out Test Set Confusion Matrices (N=172)", fontsize=10.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "confusion_matrix")

# 6. Privacy-Utility Tradeoff Curve (Analytical Epsilon)
def plot_privacy_utility_tradeoff():
    df_dp_res = pd.read_csv(os.path.join(OUTPUTS_DIR, "fedavg_dp_results.csv"))
    df_dp_3 = df_dp_res[df_dp_res["n_clients"] == 3].sort_values("epsilon")
    df_dp_5 = df_dp_res[df_dp_res["n_clients"] == 5].sort_values("epsilon")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), sharex=True)

    ax1.plot(df_dp_3["epsilon"], df_dp_3["accuracy"]*100, "o-", color=COLORS["navy"], lw=1.8, label="FedAvg+DP (3 Clients)")
    ax1.plot(df_dp_5["epsilon"], df_dp_5["accuracy"]*100, "s--", color=COLORS["coral"], lw=1.8, label="FedAvg+DP (5 Clients)")
    ax1.axhline(70.35, ls=":", color=COLORS["navy"], alpha=0.7, label="FedAvg 3c (No DP, 70.4%)")
    ax1.axhline(72.09, ls=":", color=COLORS["coral"], alpha=0.7, label="FedAvg 5c (No DP, 72.1%)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy Budget ε (log scale, δ=1e-4)", fontweight="bold")
    ax1.set_ylabel("Held-Out Test Accuracy (%)", fontweight="bold")
    ax1.set_title("(a) Accuracy vs Privacy Budget (ε)", fontweight="bold", pad=8)
    ax1.set_ylim(50, 78)
    ax1.legend(loc="lower right", fontsize=7.5)

    ax2.plot(df_dp_3["epsilon"], df_dp_3["roc_auc"]*100, "o-", color=COLORS["navy"], lw=1.8, label="FedAvg+DP (3 Clients)")
    ax2.plot(df_dp_5["epsilon"], df_dp_5["roc_auc"]*100, "s--", color=COLORS["coral"], lw=1.8, label="FedAvg+DP (5 Clients)")
    ax2.axhline(81.02, ls=":", color=COLORS["navy"], alpha=0.7, label="FedAvg 3c (No DP, 81.0%)")
    ax2.axhline(79.07, ls=":", color=COLORS["coral"], alpha=0.7, label="FedAvg 5c (No DP, 79.1%)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Privacy Budget ε (log scale, δ=1e-4)", fontweight="bold")
    ax2.set_ylabel("Held-Out Test ROC-AUC (%)", fontweight="bold")
    ax2.set_title("(b) ROC-AUC vs Privacy Budget (ε)", fontweight="bold", pad=8)
    ax2.set_ylim(60, 85)
    ax2.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(r"Messidor: Privacy-Utility Frontier under Rényi Differential Privacy ($\delta = 10^{-4}$)", fontsize=10.0, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "privacy_utility_tradeoff")

# 7. DP noise multiplier comparison
def plot_dp_noise_comparison():
    df_dp_res = pd.read_csv(os.path.join(OUTPUTS_DIR, "fedavg_dp_results.csv"))
    sigma_map = {"low_privacy": 0.4, "medium_privacy": 0.8, "strong_privacy": 1.5, "very_strong_privacy": 4.0}
    df_dp_res["sigma"] = df_dp_res["config"].map(sigma_map)
    d3_s = df_dp_res[df_dp_res["n_clients"] == 3].sort_values("sigma")
    d5_s = df_dp_res[df_dp_res["n_clients"] == 5].sort_values("sigma")

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(d3_s["sigma"], d3_s["accuracy"]*100, "o-", color=COLORS["navy"], lw=1.8, label="3 Clients: Test Acc")
    ax.plot(d3_s["sigma"], d3_s["f1"]*100, "s--", color=COLORS["blue"], lw=1.6, label="3 Clients: Test F1")
    ax.plot(d5_s["sigma"], d5_s["accuracy"]*100, "^-.", color=COLORS["coral"], lw=1.8, label="5 Clients: Test Acc")
    ax.plot(d5_s["sigma"], d5_s["f1"]*100, "d:", color=COLORS["amber"], lw=1.6, label="5 Clients: Test F1")

    ax.set_xlabel("Per-Sample Noise Multiplier (σ)", fontweight="bold")
    ax.set_ylabel("Metric Score (%)", fontweight="bold")
    ax.set_title("Messidor: Test Accuracy & F1 vs DP Noise Multiplier (σ)", fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(50, 75)

    plt.tight_layout()
    save_fig(fig, "dp_noise_comparison")

# 8. FL convergence
def plot_fl_convergence():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from mlp_core import set_seed
    set_seed(42)

    feat_m = df_messidor.drop(columns=["Class"]).copy()
    ma_cols = [c for c in feat_m.columns if c.startswith('ma')]
    ex_cols = [c for c in feat_m.columns if c.startswith('exudate')]

    feat_m['ma_mean'] = feat_m[ma_cols].mean(axis=1)
    feat_m['ma_max'] = feat_m[ma_cols].max(axis=1)
    feat_m['ma_slope'] = feat_m['ma6'] - feat_m['ma1']
    feat_m['ex_mean'] = feat_m[ex_cols].mean(axis=1)
    feat_m['ex_max'] = feat_m[ex_cols].max(axis=1)
    feat_m['ex_slope'] = feat_m['exudate8'] - feat_m['exudate1']
    feat_m['ma_ex_ratio'] = feat_m['ma_mean'] / (feat_m['ex_mean'] + 1e-6)
    feat_m['ma1_x_exudate1'] = feat_m['ma1'] * feat_m['exudate1']
    feat_m['distance_to_diameter'] = feat_m['macula_opticdisc_distance'] / (feat_m['opticdisc_diameter'] + 1e-6)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_data = poly.fit_transform(feat_m[['ma_mean', 'ex_mean', 'macula_opticdisc_distance', 'opticdisc_diameter']])
    poly_cols = [f'poly_{i}' for i in range(poly_data.shape[1])]
    poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=feat_m.index)
    feat_m = pd.concat([feat_m, poly_df], axis=1)

    X_m = feat_m.values.astype(np.float64)
    y_m = df_messidor["Class"].values.astype(np.float64)

    X_train_m, X_temp_m, y_train_m, y_temp_m = train_test_split(X_m, y_m, test_size=0.30, stratify=y_m, random_state=42)
    X_val_m, X_test_m, y_val_m, y_test_m = train_test_split(X_temp_m, y_temp_m, test_size=0.5, stratify=y_temp_m, random_state=42)
    scaler = StandardScaler().fit(X_train_m)
    X_tr_s = scaler.transform(X_train_m)
    X_te_s = scaler.transform(X_test_m)

    def part_iid(X, y, n_clients):
        rng = np.random.RandomState(42); idx = rng.permutation(len(y))
        return [idx[i::n_clients] for i in range(n_clients)]

    def part_noniid(X, y, n_clients, alpha=0.3):
        rng = np.random.RandomState(42); client_idx = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            c_idx = np.where(y==c)[0]; rng.shuffle(c_idx)
            props = rng.dirichlet([alpha]*n_clients)
            splits = (np.cumsum(props)*len(c_idx)).astype(int)[:-1]
            for i,s in enumerate(np.split(c_idx, splits)): client_idx[i].extend(s.tolist())
        return [np.array(sorted(ci)) for ci in client_idx]

    c3_i = part_iid(X_tr_s, y_train_m, 3)
    c3_n = part_noniid(X_tr_s, y_train_m, 3)
    c5_i = part_iid(X_tr_s, y_train_m, 5)
    c5_n = part_noniid(X_tr_s, y_train_m, 5)

    def run_quick_fl(c_indices, rounds=60):
        gm = mlp_core.NumpyMLP(n_in=X_tr_s.shape[1], h1=48, h2=24, seed=42)
        m_g = [np.zeros_like(p) for p in gm.get_params()]
        v_g = [np.zeros_like(p) for p in gm.get_params()]
        hist = []
        for r in range(rounds):
            c_params, c_sizes = [], []
            for idx in c_indices:
                lm = gm.clone()
                lm.train_standard(X_tr_s[idx], y_train_m[idx], epochs=3, batch_size=min(16, len(idx)), lr=0.005, l2=0.0005, seed=42+r*100)
                c_params.append(lm.get_params()); c_sizes.append(len(idx))
            tot = sum(c_sizes)
            agg = [sum(cp[i]*(sz/tot) for cp,sz in zip(c_params, c_sizes)) for i in range(len(c_params[0]))]
            old = gm.get_params()
            grads = [o - a for o, a in zip(old, agg)]
            new_p = []
            for i, g in enumerate(grads):
                m_g[i] = 0.9 * m_g[i] + 0.1 * g
                v_g[i] = 0.999 * v_g[i] + 0.001 * (g**2)
                mh = m_g[i] / (1 - 0.9**(r+1))
                vh = v_g[i] / (1 - 0.999**(r+1))
                new_p.append(old[i] - 0.01 * mh / (np.sqrt(vh) + 1e-8))
            gm.set_params(new_p)
            preds = (gm.predict_proba(X_te_s) >= 0.5).astype(int)
            hist.append((preds == y_test_m).mean())
        return np.array(hist)

    h_3_i = run_quick_fl(c3_i)
    h_3_n = run_quick_fl(c3_n)
    h_5_i = run_quick_fl(c5_i)
    h_5_n = run_quick_fl(c5_n)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    r_axis = np.arange(1, 61)
    ax.plot(r_axis, h_3_i*100, "-", color=COLORS["navy"], lw=1.8, label="3 Clients (IID)")
    ax.plot(r_axis, h_3_n*100, "--", color=COLORS["blue"], lw=1.5, label="3 Clients (Non-IID, α=0.3)")
    ax.plot(r_axis, h_5_i*100, "-", color=COLORS["coral"], lw=1.8, label="5 Clients (IID)")
    ax.plot(r_axis, h_5_n*100, "--", color=COLORS["amber"], lw=1.5, label="5 Clients (Non-IID, α=0.3)")

    ax.set_xlabel("Communication Round (60 Rounds Total)", fontweight="bold")
    ax.set_ylabel("Global Test Accuracy (%)", fontweight="bold")
    ax.set_title("Messidor: FedAvg Convergence Trajectory (IID vs Non-IID Dirichlet Skew)", fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(55, 78)

    plt.tight_layout()
    save_fig(fig, "fl_convergence")

# 9. Communication efficiency
def plot_communication_efficiency():
    df_dp_res = pd.read_csv(os.path.join(OUTPUTS_DIR, "fedavg_dp_results.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x_c = np.arange(4)
    w_c = 0.35

    t3 = df_dp_res[df_dp_res["n_clients"] == 3].sort_values("epsilon")["training_time_sec"].values
    t5 = df_dp_res[df_dp_res["n_clients"] == 5].sort_values("epsilon")["training_time_sec"].values
    cfg_labels = ["Low Privacy\n(σ=0.4)", "Medium\n(σ=0.8)", "Strong\n(σ=1.5)", "Very Strong\n(σ=4.0)"]

    ax.bar(x_c - w_c/2, t3, w_c, label="3 Clients", color=COLORS["navy"])
    ax.bar(x_c + w_c/2, t5, w_c, label="5 Clients", color=COLORS["coral"])
    ax.set_xticks(x_c)
    ax.set_xticklabels(cfg_labels, fontsize=8.5)
    ax.set_ylabel("Runtime per 60 Rounds (Seconds)", fontweight="bold")
    ax.set_title("Messidor: Computational & Communication Runtime Across DP Regimes", fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, max(max(t3), max(t5)) * 1.25)

    for i in range(4):
        ax.text(x_c[i] - w_c/2, t3[i] + 0.12, f"{t3[i]:.2f}s", ha="center", fontsize=7.5)
        ax.text(x_c[i] + w_c/2, t5[i] + 0.12, f"{t5[i]:.2f}s", ha="center", fontsize=7.5)

    plt.tight_layout()
    save_fig(fig, "communication_efficiency")

if __name__ == "__main__":
    t0 = time.time()
    print("Generating all Messidor research paper figures...")
    plot_class_distribution()
    plot_feature_correlation()
    plot_training_curves()
    plot_model_comparison()
    plot_confusion_matrices()
    plot_privacy_utility_tradeoff()
    plot_dp_noise_comparison()
    plot_fl_convergence()
    plot_communication_efficiency()
    print(f"Messidor figures generated successfully in {time.time()-t0:.2f} seconds.")
