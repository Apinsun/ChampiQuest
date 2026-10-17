"""Inférence end-to-end : photo(s) → décision (orienté contrat d'API).

Responsabilités :
- charger une ou plusieurs photos (éventuellement avec une vue optionnelle) ;
- agréger les logits de N photos du même spécimen (moyenne) ;
- appliquer calibration + décision à coût (`decision.CostDecision`) ;
- renvoyer un dictionnaire au format du contrat d'API (§10 du rapport) :
  status, decision, top_predictions, confidence, calibrated.
"""
from __future__ import annotations


def identify(images: list[bytes], views: list[str] | None = None) -> dict:
    """Point d'entrée : N photos (bytes) → décision au format API."""
    raise NotImplementedError
