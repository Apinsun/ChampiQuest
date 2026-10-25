"""Explicabilité du modèle : Grad-CAM.

Responsabilités :
- calculer la carte Grad-CAM pour une image (dernière couche Conv2d du backbone) ;
- superposer la heatmap à l'image ;
- générer un montage d'exemples (paires cible↔sosie) montrant *où* le modèle regarde.

Usage :
    python -m champiquest.vision.explain --checkpoint data/models/vision_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from . import datasets
from .models import load_checkpoint, softmax

REPO = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO / "reports"


def _find_last_conv(model: nn.Module) -> nn.Module:
    """Dernière couche Conv2d du modèle (bonne cible pour Grad-CAM sur un CNN)."""
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("aucune couche Conv2d trouvée dans le modèle")


def grad_cam(model: nn.Module, image_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """Heatmap Grad-CAM (H, W) normalisée [0, 1] pour la classe `target_class`."""
    features: dict = {}
    gradients: dict = {}

    def fwd(_m, _i, out):
        features["v"] = out

    def bwd(_m, _gin, gout):
        gradients["v"] = gout[0]

    layer = _find_last_conv(model)
    fh = layer.register_forward_hook(fwd)
    bh = layer.register_full_backward_hook(bwd)

    model.zero_grad()
    logits = model(image_tensor.unsqueeze(0))
    logits[0, target_class].backward()

    fh.remove()
    bh.remove()

    feats = features["v"]  # (1, C, H, W)
    grads = gradients["v"]  # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * feats).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = torch.relu(cam).squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay(image_pil: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Superpose la heatmap Grad-CAM sur l'image (retourne un array RGB uint8)."""
    size = image_pil.size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR))
    heatmap = matplotlib.colormaps["jet"](cam_resized / 255.0)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    base = np.array(image_pil.convert("RGB")).astype(np.float32)
    blended = base * (1 - alpha) + heatmap * alpha
    return blended.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="data/models/vision_best.pt")
    parser.add_argument("--n-per-class", type=int, default=1)
    args = parser.parse_args()

    model, ckpt = load_checkpoint(Path(args.checkpoint))
    config = ckpt["config"]
    classes: list[str] = ckpt["classes"]
    model = model.cuda().eval()

    input_size = config["model"]["input_size"]
    transform = datasets.build_transforms(input_size=input_size, is_train=False)
    _, _, _, (_, _, test_df) = datasets.build_dataloaders(config=config, seed=config["train"]["seed"])

    # classes d'intérêt : cibles + sosies (paires de confusion)
    interesting = [
        "Boletus edulis", "Boletus reticulatus",
        "Cantharellus cibarius", "Hygrophoropsis aurantiaca",
        "Gyromitra esculenta", "Morchella spp.",
        "Macrolepiota procera", "Lepiota castanea",
        "Imleria badia", "Boletus subtomentosus",
    ]

    examples = []
    for cls in interesting:
        sub = test_df[test_df["class_label"] == cls]
        for _, row in sub.head(args.n_per_class).iterrows():
            examples.append((cls, row["file"]))
    print(f"{len(examples)} exemples à expliquer")

    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 3.2 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]

    for i, (true_cls, fpath) in enumerate(examples):
        img_pil = Image.open(fpath).convert("RGB").resize((input_size, input_size))
        tensor = transform(img_pil).cuda()
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))
        probs = softmax(logits.detach().cpu().numpy())[0]
        pred = int(probs.argmax())
        pred_name = classes[pred] if pred < len(classes) else ("no_mushroom" if pred == len(classes) else "other_fungus")

        cam = grad_cam(model, tensor, pred)
        over = overlay(img_pil, cam)

        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"réel : {true_cls}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(over)
        axes[i, 1].set_title(f"prédit : {pred_name} (p={probs[pred]:.2f})", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    out = REPORTS_DIR / "partie2_gradcam.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"montage Grad-CAM : {out}")


if __name__ == "__main__":
    main()
