"""Inférence end-to-end : photo(s) → décision (orienté contrat d'API, §10 du rapport).

Usage CLI :
    python -m champiquest.vision.predict image1.jpg [image2.jpg ...]

La classe `Predictor` est réutilisable par l'API FastAPI (partie 3).
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import datasets
from .decision import CostDecision
from .evaluate import load_checkpoint, softmax

REPO = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO / "data" / "models"


class Predictor:
    """Charge le modèle + calibration, expose `identify(images_bytes, views=None)`."""

    def __init__(self, checkpoint_path: Path | None = None, device: str = "cuda") -> None:
        checkpoint_path = checkpoint_path or MODELS_DIR / "vision_best.pt"
        self.model, self.ckpt = load_checkpoint(checkpoint_path)
        self.config = self.ckpt["config"]
        self.classes: list[str] = self.ckpt["classes"]
        self.danger: dict[str, str] = self.ckpt["danger"]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.input_size = self.config["model"]["input_size"]
        self.transform = datasets.build_transforms(input_size=self.input_size, is_train=False)

        tfile = MODELS_DIR / "temperature.txt"
        self.temperature = float(tfile.read_text().strip()) if tfile.exists() else 1.0

        self.common_names = {c["name"]: c["common_name"] for c in self.config["classes"]}
        self.no_mushroom_idx = len(self.classes)
        self.other_fungus_idx = len(self.classes) + 1
        self.dec = CostDecision(
            self.classes,
            self.danger,
            no_mushroom_idx=self.no_mushroom_idx,
            other_fungus_idx=self.other_fungus_idx,
            cost_edible_as_toxic=self.config["decision"]["cost_edible_as_toxic"],
            cost_toxic_as_edible=self.config["decision"]["cost_toxic_as_edible"],
            abstain_threshold=self.config["decision"]["abstain_threshold"],
        )

    def _probs(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack(
            [self.transform(im.convert("RGB").resize((self.input_size, self.input_size))) for im in images]
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(tensors)
        logits = logits.mean(dim=0, keepdim=True)  # fusion multi-photos : moyenne des logits
        return softmax(logits.cpu().numpy(), self.temperature)[0]

    def identify(self, images: list[bytes], views: list[str] | None = None) -> dict:
        """N photos (bytes) → décision au format API (cf. §10)."""
        pils = [Image.open(BytesIO(b)) for b in images]
        probs = self._probs(pils)
        d = self.dec.decide(probs)

        top = []
        for i in np.argsort(probs)[::-1][:3]:
            name = (
                self.classes[i]
                if i < len(self.classes)
                else ("no_mushroom" if i == self.no_mushroom_idx else "other_fungus")
            )
            top.append(
                {
                    "species": name,
                    "common_name": self.common_names.get(name, name),
                    "probability": float(probs[i]),
                    "edible": i < len(self.classes) and self.danger.get(name) == "comestible",
                }
            )

        status = {
            "no_mushroom": "no_mushroom",
            "other_fungus": "other_fungus",
            "abstain": "low_confidence",
        }.get(d["decision"], "ok")

        return {
            "status": status,
            "decision": d["decision"],
            "top_predictions": top,
            "confidence": d.get("confidence", 0.0),
            "calibrated": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="chemins d'images (1..N photos du même spécimen)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None, device=args.device
    )
    images = [Path(p).read_bytes() for p in args.images]
    result = predictor.identify(images)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
