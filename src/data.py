"""Dataset loading for the two public TB chest X-ray sets.

Expected layout (see README, "Getting the data"):

    data/montgomery/*.png     e.g. MCUCXR_0001_0.png
    data/shenzhen/*.png       e.g. CHNCXR_0001_1.png

Both sets encode the label in the file name: the digit after the last
underscore is 0 for normal and 1 for TB. Each file is a different patient,
so a random split of files is already a split by patient. If you add a set
where one patient has several images, split on a patient id instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

DATASETS = ("montgomery", "shenzhen")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def label_from_name(path: Path) -> int:
    """'CHNCXR_0001_1.png' -> 1. Raises on anything unexpected."""
    suffix = path.stem.rsplit("_", 1)[-1]
    if suffix not in ("0", "1"):
        raise ValueError(f"Cannot read a label from file name: {path.name}")
    return int(suffix)


def list_samples(data_root: Path, name: str) -> list[tuple[Path, int]]:
    folder = Path(data_root) / name
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png files in {folder}. See the README for how to download the data.")
    return [(f, label_from_name(f)) for f in files]


def split_samples(samples, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15):
    """Stratified train/val/test split. One image per patient, so this is patient-level."""
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    p_train, p_rest, y_train, y_rest = train_test_split(
        paths, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    p_val, p_test, y_val, y_test = train_test_split(
        p_rest, y_rest, test_size=test_frac / (val_frac + test_frac), stratify=y_rest, random_state=seed
    )
    return (
        list(zip(p_train, y_train)),
        list(zip(p_val, y_val)),
        list(zip(p_test, y_test)),
    )


def make_transforms(train: bool, size: int = 224):
    ops = [transforms.Resize((size, size))]
    if train:
        # Mild augmentation only: no flips (heart side matters), no heavy colour jitter.
        ops += [transforms.RandomRotation(7), transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


_RESIZED: dict[tuple[Path, int], Image.Image] = {}


def _load_resized(path: Path, size: int) -> Image.Image:
    """Decode once and keep the resized image: the source PNGs are ~3000x3000 and decoding
    them every epoch left the GPU idle."""
    key = (Path(path), size)
    if key not in _RESIZED:
        image = Image.open(path).convert("RGB")  # X-rays are grayscale; the backbone wants 3 channels
        _RESIZED[key] = transforms.Resize((size, size))(image)
    return _RESIZED[key]


class CxrDataset(Dataset):
    def __init__(self, samples, train: bool = False, size: int = 224):
        self.samples = list(samples)
        # Same pipeline as make_transforms; its leading Resize is already applied by the cache.
        self.tf = transforms.Compose(make_transforms(train, size).transforms[1:])
        self.images = [_load_resized(p, size) for p, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.samples[i][1]

    def labels(self) -> np.ndarray:
        return np.array([y for _, y in self.samples])
