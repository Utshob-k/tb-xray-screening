from types import SimpleNamespace

import pytest

from src import run_experiments


def test_runner_isolates_seed_outputs_and_aggregates(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(run_experiments.subprocess, "run", lambda command, check: commands.append(command))
    args = SimpleNamespace(
        train_on=["shenzhen"], seeds=[0, 2], data_root="data", results_dir=tmp_path,
        epochs=1, batch_size=2, lr=1e-3, workers=0,
    )

    run_experiments.run_experiments(args)

    assert len(commands) == 5
    assert commands[0][2] == "src.train"
    assert str(tmp_path / "seeds" / "shenzhen_seed0") in commands[0]
    assert commands[-1][2] == "src.aggregate"
    assert str(tmp_path / "shenzhen" / "seeds_summary.json") in commands[-1]


def test_runner_refuses_to_overwrite_a_run(monkeypatch, tmp_path):
    existing = tmp_path / "seeds" / "shenzhen_seed0"
    existing.mkdir(parents=True)
    monkeypatch.setattr(run_experiments.subprocess, "run", lambda *args, **kwargs: pytest.fail("should not run"))
    args = SimpleNamespace(
        train_on=["shenzhen"], seeds=[0], data_root="data", results_dir=tmp_path,
        epochs=1, batch_size=2, lr=1e-3, workers=0,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_experiments.run_experiments(args)
