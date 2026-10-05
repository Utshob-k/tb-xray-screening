"""Grad-CAM heatmaps: where is the model looking?

    python -m src.gradcam --train-on shenzhen --images data/montgomery/MCUCXR_0001_0.png ...

Worth checking by eye: a model that lights up on the image border, a
text marker, or a scanner artefact instead of the lungs has learned a
shortcut, and that is exactly what a cross-dataset test tends to expose.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import make_transforms
from .model import build_model


def grad_cam(model, x: torch.Tensor) -> np.ndarray:
    """Heatmap for the TB logit from the last ResNet block, scaled to 0-1 at input size."""
    feats, grads = {}, {}
    layer = model.layer4
    h1 = layer.register_forward_hook(lambda m, i, o: feats.update(a=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(a=go[0]))
    model.zero_grad()
    model(x).squeeze().backward()
    h1.remove(), h2.remove()
    weights = grads["a"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp(min=1e-8)).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="results/gradcam")
    args = ap.parse_args()

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(Path("results") / args.train_on / "model.pt", map_location="cpu"))
    model.eval()
    tf = make_transforms(train=False)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for p in args.images:
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0)
        prob = torch.sigmoid(model(x)).item()
        cam = grad_cam(model, x)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img.resize((224, 224)), cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"{Path(p).name}\nP(TB) = {prob:.2f}", fontsize=9)
        ax.axis("off")
        fig.savefig(Path(args.out) / f"{Path(p).stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {Path(args.out) / (Path(p).stem + '.png')}")


if __name__ == "__main__":
    main()
