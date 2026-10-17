"""Explicabilité du modèle : Grad-CAM.

Responsabilités :
- calculer la carte Grad-CAM pour une image donnée ;
- générer des figures superposant la heatmap à l'image (en particulier sur les
  paires cible↔sosie, pour montrer où le modèle regarde : chapeau, lamelles, pied).
"""
from __future__ import annotations

import numpy as np


def grad_cam(model, image, target_layer: str) -> np.ndarray:
    """Renvoie la heatmap Grad-CAM (2D, même taille que la feature map)."""
    raise NotImplementedError
