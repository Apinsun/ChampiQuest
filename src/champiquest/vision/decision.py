"""Couche de décision « sécurité » : calibration + coût asymétrique + abstention.

Responsabilités :
- **temperature scaling** : calibrer les probabilités sur un split de validation dédié ;
- **matrice de coût** : coût(prédire comestible | vrai toxique/mortel) ≫ coût(prédire
  toxique | vrai comestible) ;
- **décision bayésienne** : prédire la classe qui minimise le coût attendu
  `Σ_y P(y|x)·coût(c, y)` ;
- **abstention** : si la confiance (max P) est trop faible, ne pas répondre.

Flux final (partagé avec `predict.py`) :
1. si la classe « pas de champi » domine → rejet ;
2. sinon → décision espèce : comestible / dangereux / neutre, ou abstention.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

EDIBLE = "comestible"
DANGEROUS = {"toxique", "mortel"}
NEUTRAL = "non_toxique"
NONE = "no_mushroom"


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


def _category(idx: int, no_mushroom_idx: int, danger_of: dict, classes: list) -> str:
    if idx == no_mushroom_idx:
        return NONE
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
    cost_edible_as_toxic: float = 10.0,
    cost_toxic_as_edible: float = 1.0,
) -> np.ndarray:
    """Matrice de coût (C, C) : `C[c, y]` = coût de prédire c alors que la vérité est y.

    Coûts clés :
    - comestible prédit / toxique réel = 10 (l'erreur fatale) ;
    - neutre prédit / toxique réel = 8 (dire « inoffensif » alors que mortel) ;
    - comestible prédit / neutre réel = 3 ;
    - toxique prédit / comestible réel = 1 (fausse alerte, sans danger) ;
    - erreurs au sein d'une même catégorie = 1 ;
    - correct = 0.
    """
    n = len(classes) + 1
    C = np.ones((n, n))
    np.fill_diagonal(C, 0.0)
    for c in range(n):
        for y in range(n):
            if c == y:
                continue
            cc = _category(c, no_mushroom_idx, danger_of, classes)
            cy = _category(y, no_mushroom_idx, danger_of, classes)
            if cc == EDIBLE and cy == "dangerous":
                C[c, y] = cost_edible_as_toxic
            elif cc == NEUTRAL and cy == "dangerous":
                C[c, y] = 8.0
            elif cc == EDIBLE and cy == NEUTRAL:
                C[c, y] = 3.0
            elif cc == "dangerous" and cy == EDIBLE:
                C[c, y] = cost_toxic_as_edible
            elif cc == "dangerous" and cy == NEUTRAL:
                C[c, y] = 2.0
            elif cc == EDIBLE and cy == NONE:
                C[c, y] = 2.0
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
        cost_edible_as_toxic: float = 10.0,
        cost_toxic_as_edible: float = 1.0,
        abstain_threshold: float = 0.6,
    ) -> None:
        self.classes = classes
        self.danger_of = danger_of
        self.no_mushroom_idx = no_mushroom_idx
        self.cost = build_cost_matrix(
            classes, danger_of, no_mushroom_idx, cost_edible_as_toxic, cost_toxic_as_edible
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
        cat = _category(c, self.no_mushroom_idx, self.danger_of, self.classes)
        decision = "no_mushroom" if cat == NONE else ("edible" if cat == EDIBLE else ("dangerous" if cat == "dangerous" else "neutral"))
        name = "no_mushroom" if c == self.no_mushroom_idx else self.classes[c]
        return {
            "decision": decision,
            "predicted_idx": c,
            "predicted_class": name,
            "confidence": float(probs[c]),
            "expected_cost": float(expected_cost[c]),
        }
