"""Run training, evaluation, and aggregation across datasets and seeds.

    python -m src.run_experiments --seeds 0 1 2 3 4

Each seed gets its own output folder, so a batch run does not overwrite a
single-run result.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .data import DATASETS


def command(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(arg) for arg in args)]


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


def run_experiments(args: argparse.Namespace) -> None:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be unique, non-negative integers")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        raise ValueError("--epochs, --batch-size, and --lr must be positive; --workers cannot be negative")

    results_root = Path(args.results_dir)
    for train_on in args.train_on:
        runs = []
        for seed in args.seeds:
            run_dir = results_root / "seeds" / f"{train_on}_seed{seed}"
            if run_dir.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing run directory: {run_dir}. "
                    "Choose a new --results-dir for a fresh experiment."
                )
            runs.append(run_dir)
            train = command(
                "src.train", "--train-on", train_on, "--data-root", args.data_root,
                "--epochs", args.epochs, "--batch-size", args.batch_size, "--lr", args.lr,
                "--seed", seed, "--workers", args.workers, "--run-dir", run_dir,
            )
            evaluate = command(
                "src.evaluate", "--train-on", train_on, "--data-root", args.data_root,
                "--workers", args.workers, "--run-dir", run_dir,
            )
            run(train)
            run(evaluate)

        aggregate = command(
            "src.aggregate", "--train-on", train_on, "--runs", *runs,
            "--out", results_root / train_on / "seeds_summary.json",
        )
        run(aggregate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeatable TB X-ray experiments across seeds.")
    parser.add_argument("--train-on", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-dir", default="results/reproduction")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    run_experiments(parser.parse_args())


if __name__ == "__main__":
    main()
