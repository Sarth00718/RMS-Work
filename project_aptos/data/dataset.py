"""
APTOS 2019 Diabetic Retinopathy Dataset Module.

Implements PyTorch Dataset for the APTOS 2019 Blindness Detection dataset,
supporting both 5-class DR grading and binary (DR vs No-DR) classification.

Dataset: https://www.kaggle.com/c/aptos2019-blindness-detection
- 3662 retinal fundus images (train: 2930, val: 366, test: 366)
- Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class APTOSDataset(Dataset):
    """
    APTOS 2019 Diabetic Retinopathy Dataset.

    Supports:
    - 5-class DR severity grading (0-4)
    - Binary classification (DR vs No-DR)

    Args:
        image_dir: Path to directory containing images.
        labels_csv: Path to CSV file with labels.
        transform: Optional torchvision transforms to apply.
        task: 'multiclass' for 5-class, 'binary' for DR vs No-DR.
        id_column: Column name for image IDs (default: 'id_code').
        label_column: Column name for labels (default: 'diagnosis').
        image_extension: Extension for image files (default: '.png').
    """

    CLASS_NAMES_5 = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    CLASS_NAMES_2 = ["No DR", "DR"]

    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        transform: Optional[Callable] = None,
        task: str = "multiclass",
        id_column: str = "id_code",
        label_column: str = "diagnosis",
        image_extension: str = ".png",
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.task = task
        self.id_column = id_column
        self.label_column = label_column
        self.image_extension = image_extension

        # Load labels
        self.df = pd.read_csv(labels_csv)
        # Clean column names (remove trailing whitespace)
        self.df.columns = self.df.columns.str.strip()

        # Drop any fully empty columns
        self.df = self.df.dropna(axis=1, how='all')

        # Verify all images exist
        self.valid_indices = []
        missing_count = 0
        for idx, row in self.df.iterrows():
            img_name = row[self.id_column]
            img_path = self.image_dir / f"{img_name}{self.image_extension}"
            if img_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count} images not found in {self.image_dir}")

        self.df = self.df.loc[self.valid_indices].reset_index(drop=True)

        # Get labels
        self.labels = self.df[self.label_column].values.astype(int)

        # Convert to binary if needed
        if self.task == "binary":
            self.labels = (self.labels > 0).astype(int)
            self.class_names = self.CLASS_NAMES_2
            self.num_classes = 2
        else:
            self.class_names = self.CLASS_NAMES_5
            self.num_classes = 5

        self.image_names = self.df[self.id_column].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.image_names[idx]
        img_path = self.image_dir / f"{img_name}{self.image_extension}"

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for cls, count in zip(unique, counts):
            name = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
            dist[name] = int(count)
        return dist

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        # Normalize so weights sum to num_classes
        weights = weights / weights.sum() * len(unique)

        full_weights = np.ones(self.num_classes)
        for cls, w in zip(unique, weights):
            full_weights[cls] = w

        return torch.FloatTensor(full_weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels]
        return sample_weights


class APTOSSubset(Dataset):
    """
    A subset of APTOSDataset that allows different transforms.
    Used for creating train/val splits with different augmentations.
    """

    def __init__(
        self,
        dataset: APTOSDataset,
        indices: List[int],
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

        self.labels = dataset.labels[indices]
        self.num_classes = dataset.num_classes
        self.class_names = dataset.class_names

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        img_name = self.dataset.image_names[real_idx]
        img_path = self.dataset.image_dir / f"{img_name}{self.dataset.image_extension}"

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        elif self.dataset.transform:
            image = self.dataset.transform(image)

        label = int(self.dataset.labels[real_idx])
        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in this subset."""
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {}
        for cls, count in zip(unique, counts):
            name = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
            dist[name] = int(count)
        return dist

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        weights = total / (len(unique) * counts)
        weights = weights / weights.sum() * len(unique)
        full_weights = np.ones(self.num_classes)
        for cls, w in zip(unique, weights):
            full_weights[cls] = w
        return torch.FloatTensor(full_weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels]
        return sample_weights
