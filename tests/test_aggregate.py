import pytest

from src.aggregate import aggregate_reports, mean_and_sd


def test_mean_and_sd_for_one_run_has_zero_spread():
    assert mean_and_sd([0.82]) == (0.82, 0.0)


def test_aggregate_reports_rejects_mismatched_sample_counts():
    first = {"external": {"n": 10, "n_tb": 4, "auroc": 0.8, "sensitivity": 0.7, "specificity": 0.9, "threshold": 0.5}}
    second = {"external": {"n": 11, "n_tb": 4, "auroc": 0.7, "sensitivity": 0.6, "specificity": 0.8, "threshold": 0.4}}
    with pytest.raises(ValueError, match="sample counts"):
        aggregate_reports([first, second])
