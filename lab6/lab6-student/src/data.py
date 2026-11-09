"""Fashion-MNIST data pipeline for DCGAN."""

from __future__ import annotations

import gzip
import os
from pathlib import Path
import shutil

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_RAW_FILE_SIZES = {
    "train-images-idx3-ubyte": 16 + 60_000 * 28 * 28,
    "train-labels-idx1-ubyte": 8 + 60_000,
    "t10k-images-idx3-ubyte": 16 + 10_000 * 28 * 28,
    "t10k-labels-idx1-ubyte": 8 + 10_000,
}


def _repair_incomplete_raw_files(root: str) -> bool:
    """Re-extract valid archives if a previous download left partial IDX files."""

    raw_dir = Path(root) / "FashionMNIST" / "raw"
    repaired = False
    for filename, expected_size in _RAW_FILE_SIZES.items():
        target = raw_dir / filename
        archive = raw_dir / f"{filename}.gz"
        if not archive.is_file() or (target.is_file() and target.stat().st_size == expected_size):
            continue
        temporary = raw_dir / f"{filename}.tmp"
        with gzip.open(archive, "rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if temporary.stat().st_size != expected_size:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Fashion-MNIST archive produced an invalid {filename}")
        os.replace(temporary, target)
        repaired = True
    return repaired


def fashion_mnist_loader(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
) -> DataLoader:
    """Return shuffled images normalized from [0, 1] to [-1, 1]."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    if download:
        _repair_incomplete_raw_files(root)
    try:
        dataset = datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
    except RuntimeError:
        # torchvision regards an existing IDX path as downloaded even if an
        # interrupted extraction left it incomplete. Recover from the verified
        # .gz archive once, then load without another network request.
        if not download or not _repair_incomplete_raw_files(root):
            raise
        dataset = datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
