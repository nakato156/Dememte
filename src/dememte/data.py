"""Dataset loaders for ImageNet-C and legacy Flowers102.

The project default is now ImageNet-C. Flowers102 remains available through the
legacy ``build_flowers_loaders`` / ``build_loaders`` API so old experiment
artifacts can still be reproduced.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset


IMAGENET_CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
IMAGENET_C_DEFAULT_CORRUPTIONS = ("gaussian_noise", "motion_blur", "pixelate", "jpeg_compression")
IMAGENET_C_DEFAULT_SEVERITIES = (3, 5)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def make_imagenet_eval_transform() -> transforms.Compose:
    """Transform matching torchvision ImageNet-pretrained ResNet weights."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _default_class_index_path() -> Path:
    return Path.home() / ".cache" / "dememte" / "imagenet_class_index.json"


def load_imagenet_class_index(
    path: str | Path | None = None,
    *,
    download: bool = True,
) -> Dict[str, int]:
    """Load ImageNet ``wnid -> class_index`` mapping.

    The public JSON maps numeric class ids to ``[wnid, label]``. ImageNet-C is
    stored by wnid directories, so evaluation needs this inverse mapping.
    """
    mapping_path = Path(path).expanduser() if path is not None else _default_class_index_path()
    if not mapping_path.exists():
        if not download:
            raise FileNotFoundError(f"ImageNet class-index file not found: {mapping_path}")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(IMAGENET_CLASS_INDEX_URL, mapping_path)
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if all(isinstance(v, int) for v in raw.values()):
        return {str(k): int(v) for k, v in raw.items()}
    return {str(value[0]): int(key) for key, value in raw.items()}


def _image_files(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


class ImageNetCDataset(Dataset):
    """ImageNet-C condition dataset for ``root/corruption/severity/wnid/*.JPEG``."""

    def __init__(
        self,
        root: str | Path,
        corruption: str,
        severity: int | str,
        transform=None,
        *,
        class_index_path: str | Path | None = None,
        download_class_index: bool = True,
        max_samples_per_class: int | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.root = Path(root).expanduser()
        self.corruption = str(corruption)
        self.severity = str(severity)
        self.condition_root = self.root / self.corruption / self.severity
        self.transform = transform or make_imagenet_eval_transform()
        self.wnid_to_idx = load_imagenet_class_index(class_index_path, download=download_class_index)
        if not self.condition_root.exists():
            raise FileNotFoundError(f"ImageNet-C condition not found: {self.condition_root}")

        rng = random.Random(seed)
        samples: list[tuple[Path, int]] = []
        for class_dir in sorted(p for p in self.condition_root.iterdir() if p.is_dir()):
            wnid = class_dir.name
            if wnid not in self.wnid_to_idx:
                continue
            files = _image_files(class_dir)
            if max_samples_per_class is not None and len(files) > max_samples_per_class:
                files = sorted(rng.sample(files, int(max_samples_per_class)))
            samples.extend((path, self.wnid_to_idx[wnid]) for path in files)
        if max_samples is not None and len(samples) > max_samples:
            samples = sorted(rng.sample(samples, int(max_samples)), key=lambda item: str(item[0]))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ImageNetFolderDataset(Dataset):
    """Clean ImageNet folder dataset using canonical ImageNet class indices."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        *,
        class_index_path: str | Path | None = None,
        download_class_index: bool = True,
        max_samples_per_class: int | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.root = Path(root).expanduser()
        self.split = split
        self.split_root = self.root / split
        self.transform = transform or make_imagenet_eval_transform()
        self.wnid_to_idx = load_imagenet_class_index(class_index_path, download=download_class_index)
        if not self.split_root.exists():
            raise FileNotFoundError(f"ImageNet split not found: {self.split_root}")

        rng = random.Random(seed)
        samples: list[tuple[Path, int]] = []
        for class_dir in sorted(p for p in self.split_root.iterdir() if p.is_dir()):
            wnid = class_dir.name
            if wnid not in self.wnid_to_idx:
                continue
            files = _image_files(class_dir)
            if max_samples_per_class is not None and len(files) > max_samples_per_class:
                files = sorted(rng.sample(files, int(max_samples_per_class)))
            samples.extend((path, self.wnid_to_idx[wnid]) for path in files)
        if max_samples is not None and len(samples) > max_samples:
            samples = sorted(rng.sample(samples, int(max_samples)), key=lambda item: str(item[0]))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def make_imagenet_train_transform() -> transforms.Compose:
    """Standard ImageNet train transform for frozen-backbone DeMemte fitting."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_imagenet_loaders(
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    *,
    class_index_path: str | Path | None = None,
    max_train_samples_per_class: int | None = None,
    max_val_samples_per_class: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, dict]:
    train_ds = ImageNetFolderDataset(
        root,
        "train",
        transform=make_imagenet_train_transform(),
        class_index_path=class_index_path,
        max_samples_per_class=max_train_samples_per_class,
        max_samples=max_train_samples,
        seed=seed,
    )
    val_ds = ImageNetFolderDataset(
        root,
        "val",
        transform=make_imagenet_eval_transform(),
        class_index_path=class_index_path,
        max_samples_per_class=max_val_samples_per_class,
        max_samples=max_val_samples,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    meta = {
        "dataset": "imagenet",
        "root": str(Path(root).expanduser()),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "num_classes": 1000,
    }
    return train_loader, val_loader, meta


def build_imagenet_c_loader(
    root: str | Path,
    corruption: str,
    severity: int | str,
    batch_size: int = 64,
    num_workers: int = 4,
    *,
    class_index_path: str | Path | None = None,
    max_samples_per_class: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> tuple[DataLoader, dict]:
    dataset = ImageNetCDataset(
        root,
        corruption,
        severity,
        transform=make_imagenet_eval_transform(),
        class_index_path=class_index_path,
        max_samples_per_class=max_samples_per_class,
        max_samples=max_samples,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    meta = {
        "dataset": "imagenet_c",
        "root": str(Path(root).expanduser()),
        "corruption": corruption,
        "severity": int(severity),
        "size": len(dataset),
        "num_classes": 1000,
    }
    return loader, meta


def build_imagenet_c_loaders(
    root: str | Path,
    corruptions: Sequence[str] = IMAGENET_C_DEFAULT_CORRUPTIONS,
    severities: Sequence[int] = IMAGENET_C_DEFAULT_SEVERITIES,
    batch_size: int = 64,
    num_workers: int = 4,
    *,
    class_index_path: str | Path | None = None,
    max_samples_per_class: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[dict[tuple[str, int], DataLoader], dict]:
    loaders: dict[tuple[str, int], DataLoader] = {}
    records = []
    for corruption in corruptions:
        for severity in severities:
            loader, meta = build_imagenet_c_loader(
                root,
                corruption,
                severity,
                batch_size=batch_size,
                num_workers=num_workers,
                class_index_path=class_index_path,
                max_samples_per_class=max_samples_per_class,
                max_samples=max_samples,
                seed=seed,
                pin_memory=pin_memory,
            )
            loaders[(corruption, int(severity))] = loader
            records.append(meta)
    return loaders, {"dataset": "imagenet_c", "conditions": records}


def _extract_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "_labels"):
        return np.array(dataset._labels)
    if hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def build_flowers_loaders(
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


def build_loaders(*args, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Legacy alias for Flowers102 loaders.

    New ImageNet-C experiments should call ``build_imagenet_c_loader(s)``.
    """
    return build_flowers_loaders(*args, **kwargs)
