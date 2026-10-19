"""Couche de décision « sécurité » : calibration + coût asymétrique + abstention.

Responsabilités :
- **temperature scaling** : calibrer les probabilités sur un split de validation dédié ;
- **matrice de coût** : coût(prédire comestible | vrai toxique/mortel) ≫ coût(prédire
  toxique | vrai comestible) ; idem pour « autre champignon » prédit comestible ;
- **décision bayésienne** : prédire la classe qui minimise le coût attendu
  `Σ_y P(y|x)·coût(c, y)` ;
- **abstention** : si la confiance (max P) est trop faible, ne pas répondre.

Catégories de sortie : no_mushroom, other_fungus, edible, dangerous, neutral, abstain.
Les deux classes de rejet (no_mushroom, other_fungus) n'émettent jamais d'avis de
comestibilité.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

EDIBLE = "comestible"
DANGEROUS = {"toxique", "mortel"}
NEUTRAL = "non_toxique"
NONE = "no_mushroom"
OTHER = "other_fungus"


def calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Estime le paramètre T du temperature scaling (NLL minimale sur le split val).

    `logits` : (N, C) ; `labels` : (N,). T > 1 adoucit, T < 1 accentue.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def nll(T: float) -> float:
        if T <= 0:
            return np.inf
        scaled = logits / T
        logp = scaled - np.log(np.sum(np.exp(scaled), axis=1, keepdims=True))
        return -float(np.mean(logp[np.arange(len(labels)), labels]))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def _category(idx: int, no_mushroom_idx: int, other_fungus_idx: int, danger_of: dict, classes: list) -> str:
    if idx == no_mushroom_idx:
        return NONE
    if idx == other_fungus_idx:
        return OTHER
    d = danger_of[classes[idx]]
    if d == EDIBLE:
        return EDIBLE
    if d in DANGEROUS:
        return "dangerous"
    return NEUTRAL


def build_cost_matrix(
    classes: list[str],
    danger_of: dict[str, str],
    no_mushroom_idx: int,
    other_fungus_idx: int,
    cost_edible_as_toxic: float = 10.0,
    cost_toxic_as_edible: float = 1.0,
) -> np.ndarray:
    """Matrice de coût (C, C) : `C[c, y]` = coût de prédire c alors que la vérité est y.

    Coûts clés :
    - comestible prédit / toxique réel = 10 (l'erreur fatale) ;
    - comestible prédit / autre champignon réel = 8 (un intrus peut être toxique) ;
    - neutre prédit / toxique réel = 8 ;
    - comestible prédit / neutre réel = 3 ;
    - toxique prédit / comestible réel = 1 (fausse alerte) ;
    - prédire un rejet (no_mushroom/other_fungus) sur une espèce = 1 (faux rejet) ;
    - correct = 0.
    """
    n = len(classes) + 2
    C = np.ones((n, n))
    np.fill_diagonal(C, 0.0)
    for c in range(n):
        for y in range(n):
            if c == y:
                continue
            cc = _category(c, no_mushroom_idx, other_fungus_idx, danger_of, classes)
            cy = _category(y, no_mushroom_idx, other_fungus_idx, danger_of, classes)
            if cc == EDIBLE and cy == "dangerous":
                C[c, y] = cost_edible_as_toxic
            elif cc == EDIBLE and cy == OTHER:
                C[c, y] = 8.0
            elif cc == EDIBLE and cy == NEUTRAL:
                C[c, y] = 3.0
            elif cc == NEUTRAL and cy == "dangerous":
                C[c, y] = 8.0
            elif cc == "dangerous" and cy == EDIBLE:
                C[c, y] = cost_toxic_as_edible
            elif cc == "dangerous" and cy == NEUTRAL:
                C[c, y] = 2.0
            elif cc == EDIBLE and cy == NONE:
                C[c, y] = 2.0
            elif cc in (NONE, OTHER) and cy in (EDIBLE, "dangerous", NEUTRAL):
                C[c, y] = 1.0  # faux rejet
            elif cc in (NONE, OTHER) and cy in (NONE, OTHER):
                C[c, y] = 1.0
            else:
                C[c, y] = 1.0
    return C


class CostDecision:
    """Décision à coût asymétrique sur des probabilités calibrées."""

    def __init__(
        self,
        classes: list[str],
        danger_of: dict[str, str],
        no_mushroom_idx: int,
        other_fungus_idx: int,
        cost_edible_as_toxic: float = 10.0,
        cost_toxic_as_edible: float = 1.0,
        abstain_threshold: float = 0.6,
    ) -> None:
        self.classes = classes
        self.danger_of = danger_of
        self.no_mushroom_idx = no_mushroom_idx
        self.other_fungus_idx = other_fungus_idx
        self.cost = build_cost_matrix(
            classes, danger_of, no_mushroom_idx, other_fungus_idx, cost_edible_as_toxic, cost_toxic_as_edible
        )
        self.abstain_threshold = abstain_threshold

    def decide(self, probs: np.ndarray) -> dict:
        """Prend une décision à partir d'un vecteur de probabilités calibrées."""
        probs = np.asarray(probs, dtype=np.float64)
        max_conf = float(probs.max())
        if max_conf < self.abstain_threshold:
            return {
                "decision": "abstain",
                "predicted_idx": int(np.argmax(probs)),
                "confidence": max_conf,
            }
        expected_cost = probs @ self.cost.T  # (C,) = Σ_y P(y)·C(c, y)
        c = int(np.argmin(expected_cost))
        cat = _category(c, self.no_mushroom_idx, self.other_fungus_idx, self.danger_of, self.classes)
        decision = {
            NONE: "no_mushroom",
            OTHER: "other_fungus",
            EDIBLE: "edible",
            "dangerous": "dangerous",
            NEUTRAL: "neutral",
        }[cat]
        name = "no_mushroom" if c == self.no_mushroom_idx else ("other_fungus" if c == self.other_fungus_idx else self.classes[c])
        return {
            "decision": decision,
            "predicted_idx": c,
            "predicted_class": name,
            "confidence": float(probs[c]),
            "expected_cost": float(expected_cost[c]),
        }
