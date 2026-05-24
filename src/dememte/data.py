"""Flowers102 data loaders.

Default protocol = `historical_trainval_resplit`: concatenate official train + val
and re-split with StratifiedShuffleSplit (seed=42, 20% val). The official test set
is left intact. This matches the protocol used to produce the E5 checkpoints stored
under `experiments/atracctor/out/artifacts/dememte_e5_critical/`, so reloaded
checkpoints reproduce their saved metrics.
"""

from __future__ import annotations

import random
from typing import Tuple, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, ConcatDataset, Subset


def seed_everything(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _make_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def _extract_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "_labels"):
        return np.array(dataset._labels)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def build_loaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 2,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    protocol: str = "historical_trainval_resplit",
    download: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    train_tf, eval_tf = _make_transforms()
    tr = torchvision.datasets.Flowers102(root=data_dir, split="train", download=download, transform=train_tf)
    va_train_tf = torchvision.datasets.Flowers102(root=data_dir, split="val", download=download, transform=train_tf)
    va_eval = torchvision.datasets.Flowers102(root=data_dir, split="val", download=download, transform=eval_tf)
    te = torchvision.datasets.Flowers102(root=data_dir, split="test", download=download, transform=eval_tf)

    if protocol == "historical_trainval_resplit":
        cv_ds = ConcatDataset([tr, va_train_tf])
        cv_y = np.concatenate([_extract_labels(tr), _extract_labels(va_train_tf)], axis=0)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=split_seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(cv_y)), cv_y))
        tr_ds = Subset(cv_ds, train_idx.tolist())
        va_ds = Subset(cv_ds, val_idx.tolist())
        meta = {
            "protocol": protocol,
            "split_seed": split_seed,
            "train_size": len(tr_ds),
            "val_size": len(va_ds),
            "test_size": len(te),
        }
    elif protocol == "official":
        tr_ds, va_ds = tr, va_eval
        meta = {
            "protocol": protocol,
            "split_seed": None,
            "train_size": len(tr_ds),
            "val_size": len(va_ds),
            "test_size": len(te),
        }
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    generator = torch.Generator()
    generator.manual_seed(split_seed)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, generator=generator)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr_loader, va_loader, te_loader, meta
