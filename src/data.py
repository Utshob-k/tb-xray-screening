"""Load the two TB chest X-ray datasets.

Layout (see the README):

    data/montgomery/*.png     e.g. MCUCXR_0001_0.png
    data/shenzhen/*.png       e.g. CHNCXR_0001_1.png

The final filename digit is the label: 0 is normal and 1 is TB. Each file is a
different patient, so splitting files also splits patients. Datasets with more
than one image per patient would need a patient-level split instead.
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
    """Read the final label digit: CHNCXR_0001_1.png becomes 1."""
    suffix = path.stem.rsplit("_", 1)[-1]
    if suffix not in ("0", "1"):
        raise ValueError(f"Cannot read a label from file name: {path.name}")
    return int(suffix)


def list_samples(data_root: Path, name: str) -> list[tuple[Path, int]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {', '.join(DATASETS)}")
    folder = Path(data_root) / name
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No .png files in {folder}. See the README for how to download the data.")
    return [(f, label_from_name(f)) for f in files]


def split_samples(samples, seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15):
    """Make a stratified train/validation/test split by patient."""
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be positive and sum to less than 1")
    samples = list(samples)
    if not samples:
        raise ValueError("samples must be a non-empty sequence of (path, binary_label) pairs")
    paths, labels = zip(*samples)
    if any(label not in (0, 1) for label in labels):
        raise ValueError("samples must use binary labels 0 and 1")
    counts = np.bincount(labels, minlength=2)
    if np.any(counts < 3):
        raise ValueError("each class needs at least three samples for a stratified train/validation/test split")
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
        # Keep augmentation mild and preserve left/right anatomy.
        ops += [transforms.RandomRotation(7), transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


_RESIZED: dict[tuple[Path, int], Image.Image] = {}


def _load_resized(path: Path, size: int) -> Image.Image:
    """Reuse a resized image instead of decoding the large PNG every epoch."""
    key = (Path(path), size)
    if key not in _RESIZED:
        # A hidden cache beside the data keeps later runs from decoding again.
        disk = Path(path).parent / f".cache{size}" / f"{Path(path).stem}.png"
        if disk.exists():
            with Image.open(disk) as image:
                _RESIZED[key] = image.convert("RGB")
        else:
            with Image.open(path) as image:
                # The pretrained backbone expects three channels.
                _RESIZED[key] = transforms.Resize((size, size))(image.convert("RGB"))
            disk.parent.mkdir(exist_ok=True)
            _RESIZED[key].save(disk)
    return _RESIZED[key]


class CxrDataset(Dataset):
    def __init__(self, samples, train: bool = False, size: int = 224):
        self.samples = list(samples)
        # Resize already happened when the image entered the cache.
        self.tf = transforms.Compose(make_transforms(train, size).transforms[1:])
        self.images = [_load_resized(p, size) for p, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.samples[i][1]
