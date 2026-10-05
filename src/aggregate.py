"""Mean and spread of the results across seeds.

    python -m src.aggregate --train-on shenzhen --runs results/shenzhen results/seeds/shenzhen_seed1 ...

Reads report.json from each run folder and prints the mean, standard
deviation and range per metric. Writes results/<train_on>/seeds_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("auroc", "sensitivity", "specificity", "threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders, one per seed")
    args = ap.parse_args()

    reports = [json.loads((Path(r) / "report.json").read_text()) for r in args.runs]
    summary = {}
    for setting in reports[0]:
        rows = [rep[setting] for rep in reports]
        summary[setting] = {"n_runs": len(rows), "n": rows[0]["n"], "n_tb": rows[0]["n_tb"]}
        print(f"\n{setting}  (n={rows[0]['n']}, TB={rows[0]['n_tb']}, runs={len(rows)})")
        for m in METRICS:
            v = np.array([r[m] for r in rows])
            summary[setting][m] = {"values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
            print(f"  {m:12s} mean {v.mean():.3f}  sd {v.std(ddof=1):.3f}  min {v.min():.3f}  max {v.max():.3f}  "
                  f"values {[round(x, 3) for x in v]}")
    out = Path("results") / args.train_on / "seeds_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
