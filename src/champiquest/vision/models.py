"""Backbone pré-entraîné + tête de classification (N+1 classes)."""
from __future__ import annotations

import timm
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
