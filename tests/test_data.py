from pathlib import Path

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
