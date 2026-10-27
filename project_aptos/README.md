# Collaborative Differentially Private Federated Learning for Diabetic Retinopathy Prediction
## APTOS 2019 Blindness Detection Dataset

### Paper Reference
**Title**: "Collaborative Differentially Private Federated Learning Framework for the Prediction of Diabetic Retinopathy"  
**Authors**: Ismail Hossain, Sai Puppala, Sajedul Talukder  
**Conference**: IEEE 2nd International Conference on AI in Cybersecurity (ICAIC), 2023

---

### Project Overview

This project reproduces the methodology described in the above paper using the **APTOS 2019 Blindness Detection** dataset (Kaggle). The paper proposes a federated learning framework with differential privacy for diabetic retinopathy prediction.

### Pipeline

The pipeline runs three phases for both binary (DR vs No-DR) and multiclass (5-class grading) tasks:

1. **Phase 1: Centralized Baseline** — Standard ResNet50 training with ImageNet pretraining
2. **Phase 2: Federated Averaging (FedAvg)** — K=3 client federated training without noise
3. **Phase 3: DP-FedAvg** — Differentially private federated training with noise multipliers: 0.56, 0.9, 1.2, 3.5, 8.0

### Dataset

| Split | Images | Format |
|-------|--------|--------|
| Train | 2,930  | PNG    |
| Val   | 366    | PNG    |
| Test  | 366    | PNG    |

**Classes**: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR

### Key Features

- **ResNet50** backbone with ImageNet pretraining
- **Balanced sampling** with `WeightedRandomSampler` for imbalanced data
- **Enhanced data augmentation**: rotation, color jitter, random crop, flips
- **Label smoothing** (0.1) for regularization
- **Cosine annealing** learning rate scheduler
- **Communication checkpointing** (paper's Algorithm 1): dual checkpoint mechanism to reduce communication overhead
- **DP-SGD**: gradient clipping + Gaussian noise injection

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run binary classification only (fastest)
python main.py --config configs/base.yaml --mode binary

# Run multiclass classification only
python main.py --config configs/base.yaml --mode multiclass

# Run FULL pipeline (all experiments — takes many hours)
python main.py --config configs/base.yaml --mode full_pipeline
```

### Project Structure

```
project_aptos/
├── configs/
│   └── base.yaml              # Configuration file
├── data/
│   ├── __init__.py
│   ├── dataset.py             # APTOSDataset & APTOSSubset classes
│   ├── preprocessing.py       # Image preprocessing transforms
│   ├── augmentation.py        # Data augmentation pipelines
│   └── split.py               # Stratified splitting & leakage checks
├── models/
│   ├── __init__.py
│   ├── base_model.py          # DRResNet50 architecture
│   └── factory.py             # Model factory
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Training engine
│   ├── losses.py              # Loss functions
│   └── callbacks.py           # Early stopping & checkpointing
├── federated/
│   ├── __init__.py
│   ├── client.py              # FL Client with DP-SGD
│   └── server.py              # FL Server with FedAvg
├── evaluation/
│   ├── __init__.py
│   └── metrics.py             # Comprehensive evaluation metrics
├── results/                   # Output directory (auto-created)
├── main.py                    # Main pipeline entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Expected Results (from Paper)

| Experiment | Binary Acc | Multiclass Acc |
|-----------|-----------|---------------|
| Centralized (no noise) | ~83% | ~75% |
| FedAvg (no noise) | ~80% | ~72% |
| DP-FedAvg (noise=0.56) | ~79% | ~70% |
| DP-FedAvg (noise=8.0) | ~75% | ~65% |

### Configuration

All hyperparameters are in `configs/base.yaml`. Key settings:
- Image size: 224×224
- Batch size: 16
- Learning rate: 0.001 with cosine annealing
- Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
- Epochs: 50 (centralized), 20 rounds (federated)
- Early stopping patience: 15

### Outputs

Results are saved to `results/` including:
- `*_metrics.json` — Full classification metrics
- `*_classification_report.txt` — Detailed per-class report
- `*_training_curves.png` — Training/validation loss & accuracy plots
- `*_fl_history.csv` — Federated round-by-round accuracy
- `*_fl_progress.png` — Federated training progress plots
- `experiment_summary.json` — Summary of all experiments
