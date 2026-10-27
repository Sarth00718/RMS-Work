# Privacy-preserving federated learning for diabetic retinopathy detection

This repository contains implementation pipelines for privacy-preserving federated learning applied to diabetic retinopathy (DR) screening and classification across three clinical cohorts:

1. **APTOS 2019 Blindness Detection** (Retinal fundus image classification: binary triage and 5-stage ICDR grading)
2. **IDRiD Indian Diabetic Retinopathy Image Dataset** (Retinal fundus image classification: binary triage and 5-stage ICDR grading)
3. **Messidor Tabular Feature Benchmark** (Primary care clinical biomarkers: binary risk screening with analytical Rényi DP accounting)

The framework evaluates centralized baselines, standard Federated Averaging (FedAvg), dual-checkpoint communication regulation, and Differentially Private Stochastic Gradient Descent (DP-SGD).

---

## Repository layout

```
├── project_aptos/          # APTOS pipeline (ResNet50, FedAvg, DP-SGD, checkpoint regulation)
├── project_idrid/          # IDRiD pipeline (ResNet50, FedAvg, DP-SGD, client/server logic)
├── project_messidor/       # Messidor tabular pipeline (NumPy/scikit-learn MLP, Rényi DP accountant)
├── DATASET_SOURCES.md      # Direct download URLs, Kaggle commands, and data specifications
├── .gitignore              # Ignores large raw image directories and checkpoint weights (>100 MB)
└── README.md               # Main repository documentation
```

---

## Datasets

Raw medical images and archives are not stored in this repository to keep it lightweight. Detailed dataset source links, split distributions, and CLI commands are documented in [DATASET_SOURCES.md](DATASET_SOURCES.md).

To download the datasets via the Kaggle CLI:

```bash
# APTOS 2019 Blindness Detection
kaggle competitions download -c aptos2019-blindness-detection

# IDRiD dataset mirror
kaggle datasets download -d mariaherrerot/idrid-dataset

# Messidor tabular features
kaggle datasets download -d abdallahwagih/messidor-features
```

Place extracted datasets into their corresponding directories:
- APTOS: `aptos_dataset/`
- IDRiD: `B. Disease Grading/`
- Messidor: `project_messidor/messidor_features.csv` (already included)

---

## Environment setup

Requirements vary slightly across image and tabular pipelines.

### Python environment

Recommended: Python 3.10 or higher with CUDA-enabled PyTorch.

```bash
# Install core dependencies for image pipelines (APTOS / IDRiD)
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn pyyaml tqdm opencv-python albumentations
```

---

## Running the pipelines

### 1. APTOS 2019 pipeline

```bash
cd project_aptos

# Run binary triage classification
python main.py --config configs/base.yaml --mode binary

# Run 5-stage multiclass grading
python main.py --config configs/base.yaml --mode multiclass

# Run complete pipeline (centralized + FedAvg + DP-SGD sweeps)
python main.py --config configs/base.yaml --mode full_pipeline
```

### 2. IDRiD pipeline

```bash
cd project_idrid

# Run end-to-end evaluation
python main.py --config configs/base.yaml --mode full_pipeline
```

### 3. Messidor tabular pipeline

```bash
cd project_messidor

# Run the notebook or inspection scripts
jupyter notebook messidor_fl_dp_pipeline.ipynb
```

---

## Methodology summary

- **Federated Averaging (FedAvg):** Clients train on isolated local partitions and transmit weight updates to a central coordinator.
- **Differential Privacy (DP-SGD):** Per-sample gradient clipping with threshold $C$ and calibrated Gaussian noise injection $\sigma \in \{0.56, 0.90, 1.20, 3.50, 8.00\}$.
- **Dual Checkpoint Regulation:** Evaluates pre-training performance to skip lagging clients and post-training accuracy shift to drop redundant updates.
- **Rényi DP Accounting:** Computes analytical $(\epsilon, \delta)$ privacy bounds under subsampled Gaussian mechanisms at $\delta = 10^{-4}$.
