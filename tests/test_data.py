from pathlib import Path

import pytest

from src.data import split_samples


def test_split_samples_is_stratified_and_repeatable():
    samples = [(Path(f"image_{i}_{label}.png"), label) for label in (0, 1) for i in range(20)]

    first = split_samples(samples, seed=7)
    second = split_samples(samples, seed=7)

    assert first == second
    assert tuple(map(len, first)) == (28, 6, 6)
    for split in first:
        labels = [label for _, label in split]
        assert labels.count(0) == labels.count(1)


@pytest.mark.parametrize("val_frac,test_frac", [(0, 0.15), (0.2, 0), (0.6, 0.4)])
def test_split_samples_rejects_invalid_fractions(val_frac, test_frac):
    samples = [(Path(f"image_{i}_{label}.png"), label) for label in (0, 1) for i in range(4)]
    with pytest.raises(ValueError, match="must be positive"):
        split_samples(samples, val_frac=val_frac, test_frac=test_frac)
