# Diabetic Retinopathy Classification — Differentially Private Federated Learning Framework

## Project Overview

This project reproduces the exact methodology from the paper **"Collaborative Differentially Private Federated Learning Framework for the Prediction of Diabetic Retinopathy"** (ICAIC 2023) on the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**.

The framework simulates multiple collaborative nodes (hospitals) that jointly train a deep learning model (ResNet50) without sharing raw patient data, employing **Federated Averaging (FedAvg)**. To prevent adversarial model inversion and membership inference attacks, it applies **Differentially Private Stochastic Gradient Descent (DP-SGD)** with varying noise multipliers.

## Methodology

As proposed in the base paper, this implementation executes the following exact pipeline:
1. **Centralized Local Training**: The baseline standard approach where all data is pooled centrally.
2. **Federated Learning (FedAvg)**: A collaborative approach without noise. The dataset is distributed among $K$ clients. Clients compute local updates which the server averages.
3. **Differentially Private Federated Learning (DP-FedAvg)**: The server averages local weights from clients that have been injected with Gaussian noise $\mathcal{N}(0, \sigma^2 \cdot C^2 \cdot I)$ and clipped to a norm $C$. Tested over multiple noise levels: `0.56, 0.9, 1.2, 3.5, 8.0` matching the paper's experiments.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- scikit-learn, numpy, pandas, matplotlib, tqdm, pyyaml

## Running the Pipeline

You can run the full, end-to-end pipeline using the command below. It will automatically run the centralized baseline followed by standard FedAvg and all Differentially Private FedAvg scenarios.

```bash
python main.py --config configs/base.yaml --mode full_pipeline
```

The script distributes the 413 training images dynamically into 3 virtual clients, simulating 3 hospital silos. It runs the entire suite of experiments across **both Binary and 5-Class Multiclass** classification tasks automatically.

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml               # Base configuration
├── data/                       # Datasets, splits, and preprocessing
├── federated/
│   ├── client.py               # FL Client with DP-SGD implementation
│   ├── server.py               # FL Server handling FedAvg
│   └── __init__.py
├── models/                     # ResNet50 definitions
├── training/                   # Standard centralized training modules
├── evaluation/                 # Comprehensive metrics, AUC, CMs
├── results/                    # Output directory for models/metrics
└── main.py                     # Primary pipeline entry point
```

## Dataset: IDRiD Disease Grading

- **Training**: 413 Images
- **Testing**: 103 Images
- **Task**: Binary Classification (No DR vs. DR) AND 5-Class Disease Grading (No DR, Mild, Moderate, Severe, Proliferative DR)

## Differential Privacy Details

- **Noise Multipliers ($\sigma$)**: 0.56, 0.9, 1.2, 3.5, 8.0
- **Clipping Norm ($C$)**: 1.0 (default)
- **Communication Optimization**: Clients only send weights back to the server if the local validation accuracy varies by more than a threshold ($\phi = 0.01$) compared to the pre-training evaluation, identically matching the paper's optimization technique to save 49% of communication costs.
