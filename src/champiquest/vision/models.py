"""Backbone pré-entraîné + tête de classification + utilitaires d'inférence.

Module volontairement LÉGER (timm + torch + numpy) : importé par le runtime
(predict.py) SANS tirer mlflow/seaborn (qui vivent dans evaluate.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn


def build_model(
    backbone: str = "convnext_base",
    num_classes: int = 17,
    pretrained: bool = True,
) -> nn.Module:
    """Construit un backbone `timm` pré-entraîné avec une tête à `num_classes` sorties.

    `num_classes` = nombre d'espèces + 1 (la classe « pas de champi »).
    `timm.create_model` remplace automatiquement la tête de classification
    (1000 classes ImageNet) par un linéaire de la bonne dimension.
    """
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def load_checkpoint(path: Path | str):
    """Charge un checkpoint (state_dict + config) et reconstruit le modèle."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Softmax avec température T (stable numériquement)."""
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
