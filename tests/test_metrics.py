from pathlib import Path

import numpy as np
import pytest

from src.metrics import sens_spec, threshold_at_specificity
from src.data import label_from_name


def test_label_from_name():
    assert label_from_name(Path("CHNCXR_0001_1.png")) == 1
    assert label_from_name(Path("MCUCXR_0123_0.png")) == 0
    with pytest.raises(ValueError):
        label_from_name(Path("something.png"))


def test_sens_spec_perfect_and_threshold():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert sens_spec(y, p, 0.5) == (1.0, 1.0)
    t = threshold_at_specificity(y, p, 0.9)
    sens, spec = sens_spec(y, p, t)
    assert spec >= 0.9
