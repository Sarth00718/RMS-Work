"""
IDRiD Diabetic Retinopathy Dataset Module.

Implements PyTorch Dataset for the IDRiD Disease Grading dataset,
supporting both 5-class DR grading and binary (DR vs No-DR) classification.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class IDRiDDataset(Dataset):
    """
    IDRiD Diabetic Retinopathy Disease Grading Dataset.
    
    Supports:
    - 5-class DR severity grading (0-4)
    - Binary classification (DR vs No-DR)
    - Optional macular edema risk prediction
    
    Args:
        image_dir: Path to directory containing images.
        labels_csv: Path to CSV file with labels.
        transform: Optional torchvision transforms to apply.
        task: 'multiclass' for 5-class, 'binary' for DR vs No-DR.
        target_column: Which column to use as label ('Retinopathy grade' or 'Risk of macular edema').
    """
    
    CLASS_NAMES_5 = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    CLASS_NAMES_2 = ["No DR", "DR"]
    
    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        transform: Optional[Callable] = None,
        task: str = "multiclass",
        target_column: str = "Retinopathy grade",
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.task = task
        self.target_column = target_column
        
        # Load labels
        self.df = pd.read_csv(labels_csv)
        # Clean column names (remove trailing whitespace)
        self.df.columns = self.df.columns.str.strip()
        
        # Drop any fully empty columns
        self.df = self.df.dropna(axis=1, how='all')
        
        # Verify all images exist
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_name = row['Image name']
            img_path = self.image_dir / f"{img_name}.jpg"
            if img_path.exists():
                self.valid_indices.append(idx)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        self.df = self.df.loc[self.valid_indices].reset_index(drop=True)
        
        # Get labels
        self.labels = self.df[self.target_column].values.astype(int)
        
        # Convert to binary if needed
        if self.task == "binary":
            self.labels = (self.labels > 0).astype(int)
            self.class_names = self.CLASS_NAMES_2
            self.num_classes = 2
        else:
            self.class_names = self.CLASS_NAMES_5
            self.num_classes = 5
        
        self.image_names = self.df['Image name'].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.image_names[idx]
        img_path = self.image_dir / f"{img_name}.jpg"
        
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


class IDRiDSubset(Dataset):
    """
    A subset of IDRiDDataset that allows different transforms.
    Used for creating train/val splits with different augmentations.
    """
    
    def __init__(
        self,
        dataset: IDRiDDataset,
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
        img_path = self.dataset.image_dir / f"{img_name}.jpg"
        
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
