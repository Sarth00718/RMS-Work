"""
Evaluation metrics computation for DR classification.

Computes all required metrics:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1 (macro, weighted, per-class)
- ROC-AUC (one-vs-rest)
- Classification report
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    num_classes: int = 5,
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC computation).
        class_names: List of class names.
        num_classes: Number of classes.
    
    Returns:
        Dictionary of all metrics.
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # Precision, Recall, F1
    avg_modes = ["macro", "weighted"]
    for avg in avg_modes:
        metrics[f"precision_{avg}"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"f1_{avg}"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
        )
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    metrics["per_class"] = {}
    for i, name in enumerate(class_names):
        if i < len(per_class_precision):
            metrics["per_class"][name] = {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
                "support": int(np.sum(y_true == i)),
            }
    
    # ROC-AUC
    if y_prob is not None:
        try:
            if num_classes == 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                )
        except ValueError as e:
            metrics["auc"] = None
            metrics["auc_error"] = str(e)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Classification report string
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names[:len(np.unique(y_true))],
        zero_division=0,
    )
    
    return metrics


def save_metrics(
    metrics: Dict,
    save_path: str,
    experiment_name: str = "baseline",
):
    """Save metrics to JSON and text files."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full metrics as JSON
    json_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(save_dir / f"{experiment_name}_metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=2, default=str)
    
    # Save classification report as text
    with open(save_dir / f"{experiment_name}_classification_report.txt", "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Classification Report: {experiment_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(metrics["classification_report"])
        f.write(f"\n\nAccuracy:          {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1:          {metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted F1:       {metrics['f1_weighted']:.4f}\n")
        if metrics.get("auc") is not None:
            f.write(f"AUC:               {metrics['auc']:.4f}\n")
    
    print(f"\nMetrics saved to {save_dir}")


def print_metrics_summary(metrics: Dict, title: str = "Results"):
    """Print a formatted metrics summary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    if metrics.get("auc") is not None:
        print(f"  AUC:               {metrics['auc']:.4f}")
    print(f"{'='*60}")
    
    # Per-class
    if "per_class" in metrics:
        print(f"\n  Per-class metrics:")
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*60}")
        for cls_name, cls_metrics in metrics["per_class"].items():
            print(
                f"  {cls_name:<20} "
                f"{cls_metrics['precision']:>10.4f} "
                f"{cls_metrics['recall']:>10.4f} "
                f"{cls_metrics['f1']:>10.4f} "
                f"{cls_metrics['support']:>10d}"
            )
