"""Train on one dataset, keep the best epoch by validation AUROC.

    python -m src.train --train-on shenzhen --epochs 15

Writes results/<train_on>/model.pt and the validation-chosen threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import CxrDataset, list_samples, split_samples
from .metrics import threshold_at_specificity
from .model import build_model, predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", required=True, choices=["montgomery", "shenzhen"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None, help="output folder (default: results/<train_on>)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.run_dir) if args.run_dir else Path("results") / args.train_on
    out.mkdir(parents=True, exist_ok=True)

    train_s, val_s, test_s = split_samples(list_samples(Path(args.data_root), args.train_on), seed=args.seed)
    (out / "split.json").write_text(json.dumps({
        "train": [str(p) for p, _ in train_s],
        "val": [str(p) for p, _ in val_s],
        "test": [str(p) for p, _ in test_s],
    }, indent=1))

    train_dl = DataLoader(CxrDataset(train_s, train=True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(CxrDataset(val_s), batch_size=args.batch_size, num_workers=2)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # weight TB by neg/pos so both classes count about equally
    labels = CxrDataset(train_s).labels()
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.float().to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        probs, ys = predict_proba(model, val_dl, device)
        auc = roc_auc_score(ys, probs)
        print(f"epoch {epoch:2d}  train loss {total / len(train_dl.dataset):.4f}  val AUROC {auc:.4f}")
        if auc > best_auc:
            best_auc, best = auc, (epoch, {k: v.cpu().clone() for k, v in model.state_dict().items()}, probs, ys)

    epoch, state, probs, ys = best
    threshold = threshold_at_specificity(ys, probs, 0.90)
    torch.save(state, out / "model.pt")
    (out / "threshold.json").write_text(json.dumps({"best_epoch": epoch, "val_auroc": best_auc, "threshold": threshold}))
    print(f"saved best epoch {epoch} (val AUROC {best_auc:.4f}, threshold {threshold:.3f}) to {out}")


if __name__ == "__main__":
    main()
