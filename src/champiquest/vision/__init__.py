"""Reconnaissance d'espèce par image (partie 2 de ChampiQuest).

Sous-package dédié à la vision, séparé des modules de la partie 1
(prédiction météo → pousse). Chaîne type :

    datasets → models → train → evaluate + decision → explain → predict
"""

import torch

# cuDNN plante au chargement ("Invalid handle. Cannot load symbol cudnnGetVersion")
# quel que soit torch/cuDNN (testé cu130 ET cu128) → probablement le driver 595.99.02
# (CUDA 13.2, bêta) incompatible avec cuDNN 9.x.
# On désactive cuDNN → PyTorch utilise ses kernels CUDA natifs (stables, ~10-20 % plus lents).
# TODO : réactiver cuDNN après une mise à jour du driver NVIDIA.
torch.backends.cudnn.enabled = False
