"""CPU tests for the CIFAR-C (.npy) loader — severity slicing and label alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dememte.data import CIFAR_C_CORRUPTIONS, CIFARCDataset


def _write_cifar_c(root: Path, dirname: str = "CIFAR-10-C", corruption: str = "gaussian_noise"):
    """Write a synthetic CIFAR-C condition: 5 severity blocks of 10000, label == severity per block."""
    c_dir = root / dirname
    c_dir.mkdir(parents=True, exist_ok=True)
    # encode the severity into pixel value AND label so we can assert the slice picked the right block
    images = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
    labels = np.zeros((50000,), dtype=np.int64)
    for s in range(1, 6):
        lo, hi = (s - 1) * 10000, s * 10000
        images[lo:hi] = s          # block s is filled with value s
        labels[lo:hi] = s % 10     # label tied to the block
    np.save(c_dir / f"{corruption}.npy", images)
    np.save(c_dir / "labels.npy", labels)


def test_severity_slice_selects_correct_block(tmp_path):
    _write_cifar_c(tmp_path)
    for s in range(1, 6):
        ds = CIFARCDataset(tmp_path, "cifar10", "gaussian_noise", s)
        assert len(ds) == 10000
        img, label = ds[0]
        # transform resizes to 224 and normalizes; label must match the block we asked for
        assert img.shape[-2:] == (224, 224)
        assert label == s % 10


def test_labels_align_with_images(tmp_path):
    _write_cifar_c(tmp_path)
    ds = CIFARCDataset(tmp_path, "cifar10", "gaussian_noise", 3)
    # raw block-3 labels should all equal 3 % 10 in our synthetic fixture
    assert set(int(x) for x in ds.labels) == {3 % 10}
    assert len(ds.labels) == len(ds.images)


def test_max_samples_is_deterministic(tmp_path):
    _write_cifar_c(tmp_path)
    a = CIFARCDataset(tmp_path, "cifar10", "gaussian_noise", 2, max_samples=128, seed=7)
    b = CIFARCDataset(tmp_path, "cifar10", "gaussian_noise", 2, max_samples=128, seed=7)
    assert len(a) == len(b) == 128
    assert np.array_equal(a.labels, b.labels)


def test_invalid_inputs_raise(tmp_path):
    _write_cifar_c(tmp_path)
    with pytest.raises(ValueError):
        CIFARCDataset(tmp_path, "cifar10", "not_a_corruption", 1)
    with pytest.raises(ValueError):
        CIFARCDataset(tmp_path, "mnist", "gaussian_noise", 1)
    with pytest.raises(ValueError):
        CIFARCDataset(tmp_path, "cifar10", "gaussian_noise", 6)
    with pytest.raises(FileNotFoundError):
        CIFARCDataset(tmp_path / "missing", "cifar10", "gaussian_noise", 1)


def test_canonical_corruptions_count():
    assert len(CIFAR_C_CORRUPTIONS) == 15
    assert len(set(CIFAR_C_CORRUPTIONS)) == 15
