"""Evaluate a trained model in-domain and on the OTHER dataset (external validation).

    python -m src.evaluate --train-on shenzhen

The in-domain number uses the held-out test split saved at training time.
The external number runs on EVERY image of the other dataset. The gap
between the two is the headline result of this project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import DATASETS, CxrDataset, label_from_name, list_samples
from .metrics import summarize
from .model import build_model, predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", required=True, choices=list(DATASETS))
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path("results") / args.train_on
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(device)
    threshold = json.loads((run / "threshold.json").read_text())["threshold"]  # fixed from validation

    split = json.loads((run / "split.json").read_text())
    in_domain = [(Path(p), label_from_name(Path(p))) for p in split["test"]]
    other = next(d for d in DATASETS if d != args.train_on)
    external = list_samples(Path(args.data_root), other)

    report = {}
    for name, samples in ((f"in-domain ({args.train_on} test split)", in_domain), (f"external ({other}, all images)", external)):
        dl = DataLoader(CxrDataset(samples), batch_size=32, num_workers=2)
        probs, ys = predict_proba(model, dl, device)
        report[name] = summarize(ys, probs, threshold)
        r = report[name]
        print(f"{name}: AUROC {r['auroc']:.3f} {tuple(round(v, 3) for v in r['auroc_ci95'])}  "
              f"sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  (n={r['n']}, TB={r['n_tb']})")

    (run / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
