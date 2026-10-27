"""
Data splitting utilities.

Implements stratified train/validation splitting with leakage checks.
"""

from typing import Tuple, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_stratified_split(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a stratified train/validation split.
    
    Args:
        labels: Array of integer class labels.
        val_ratio: Fraction of data for validation.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (train_indices, val_indices).
    """
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=random_seed,
    )
    
    indices = np.arange(len(labels))
    
    for train_idx, val_idx in splitter.split(indices, labels):
        return train_idx, val_idx


def check_data_leakage(
    train_names: List[str],
    val_names: List[str],
    test_names: List[str],
) -> dict:
    """
    Check for data leakage between splits.
    
    Checks:
    - Exact duplicates between splits
    - Overlapping image IDs
    
    Args:
        train_names: List of training image names.
        val_names: List of validation image names.
        test_names: List of test image names.
    
    Returns:
        Dictionary with leakage report.
    """
    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)
    
    report = {
        "train_val_overlap": list(train_set & val_set),
        "train_test_overlap": list(train_set & test_set),
        "val_test_overlap": list(val_set & test_set),
        "train_duplicates": len(train_names) - len(train_set),
        "val_duplicates": len(val_names) - len(val_set),
        "test_duplicates": len(test_names) - len(test_set),
        "leakage_detected": False,
    }
    
    if (report["train_val_overlap"] or 
        report["train_test_overlap"] or 
        report["val_test_overlap"]):
        report["leakage_detected"] = True
    
    return report


def check_image_duplicates(
    image_dir: str,
    image_names: List[str],
) -> dict:
    """
    Check for exact duplicate images using file hashes.
    
    Args:
        image_dir: Directory containing images.
        image_names: List of image filenames (without extension).
    
    Returns:
        Dictionary mapping hash to list of duplicate filenames.
    """
    import hashlib
    from pathlib import Path
    
    hashes = {}
    duplicates = {}
    
    for name in image_names:
        img_path = Path(image_dir) / f"{name}.jpg"
        if img_path.exists():
            with open(img_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in hashes:
                if file_hash not in duplicates:
                    duplicates[file_hash] = [hashes[file_hash]]
                duplicates[file_hash].append(name)
            else:
                hashes[file_hash] = name
    
    return duplicates
