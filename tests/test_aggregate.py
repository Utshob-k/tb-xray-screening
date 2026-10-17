from src.aggregate import mean_and_sd


def test_mean_and_sd_for_one_run_has_zero_spread():
    assert mean_and_sd([0.82]) == (0.82, 0.0)
