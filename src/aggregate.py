"""Summarize results across random seeds.

    python -m src.aggregate --train-on shenzhen --runs results/shenzhen results/seeds/shenzhen_seed1 ...

Read report.json from each run folder and write mean, standard deviation, and
range per metric to results/<train_on>/seeds_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("auroc", "sensitivity", "specificity", "threshold")


def mean_and_sd(values) -> tuple[float, float]:
    """Return the mean and sample SD; one run has no observed spread."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("cannot aggregate an empty set of values")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def aggregate_reports(reports: list[dict]) -> dict:
    """Summarize reports only when they describe the same evaluation setup."""
    if not reports:
        raise ValueError("at least one report is required")
    settings = tuple(reports[0])
    if not settings:
        raise ValueError("reports must contain at least one evaluation setting")
    if any(tuple(report) != settings for report in reports[1:]):
        raise ValueError("all reports must contain the same evaluation settings")

    summary = {}
    for setting in settings:
        rows = [report[setting] for report in reports]
        reference = (rows[0]["n"], rows[0]["n_tb"])
        if any((row["n"], row["n_tb"]) != reference for row in rows[1:]):
            raise ValueError(f"reports disagree on sample counts for {setting}")
        summary[setting] = {"n_runs": len(rows), "n": reference[0], "n_tb": reference[1]}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            mean, sd = mean_and_sd(values)
            summary[setting][metric] = {"values": values.tolist(), "mean": mean, "sd": sd}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders, one per seed")
    ap.add_argument("--out", default=None, help="summary JSON path (default: results/<train_on>/seeds_summary.json)")
    args = ap.parse_args()

    reports = [json.loads((Path(r) / "report.json").read_text()) for r in args.runs]
    summary = aggregate_reports(reports)
    for setting, setting_summary in summary.items():
        print(f"\n{setting}  (n={setting_summary['n']}, TB={setting_summary['n_tb']}, runs={setting_summary['n_runs']})")
        for m in METRICS:
            details = setting_summary[m]
            v = np.asarray(details["values"])
            mean, sd = details["mean"], details["sd"]
            print(f"  {m:12s} mean {mean:.3f}  sd {sd:.3f}  min {v.min():.3f}  max {v.max():.3f}  "
                  f"values {[round(x, 3) for x in v]}")
    out = Path(args.out) if args.out else Path("results") / args.train_on / "seeds_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
