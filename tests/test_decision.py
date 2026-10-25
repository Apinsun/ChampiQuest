"""Tests de la couche de décision : calibration + matrice de coût + abstention.

Pourquoi tester ça : c'est la couche de SÉCURITÉ. Un bug ici = quelqu'un mange
un champignon toxique. C'est le test à plus forte valeur du projet.
"""
import numpy as np

from champiquest.vision.decision import CostDecision, calibrate_temperature

CLASSES = ["Boletus edulis", "Rubroboletus satanas", "Tylopilus felleus"]
DANGER = {
    "Boletus edulis": "comestible",
    "Rubroboletus satanas": "toxique",
    "Tylopilus felleus": "non_toxique",
}


def _dec() -> CostDecision:
    # indices : 0=comestible, 1=toxique, 2=non_toxique, 3=no_mushroom, 4=other_fungus
    return CostDecision(CLASSES, DANGER, no_mushroom_idx=3, other_fungus_idx=4)


def test_toxique_confiant_n_est_jamais_edible():
    p = np.array([0.05, 0.90, 0.02, 0.01, 0.02])  # toxique très probable
    d = _dec().decide(p)
    assert d["decision"] != "edible"


def test_comestible_confiant_est_edible():
    p = np.array([0.95, 0.01, 0.01, 0.01, 0.02])
    assert _dec().decide(p)["decision"] == "edible"


def test_confiance_faible_provoque_abstention():
    p = np.array([0.40, 0.30, 0.10, 0.10, 0.10])  # max conf 0.40 < 0.6
    assert _dec().decide(p)["decision"] == "abstain"


def test_temperature_positive_et_argmax_invariant():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 5, (300, 5))  # logits très dispersés → sur-confiance
    labels = logits.argmax(1)
    T = calibrate_temperature(logits, labels)
    assert T > 0
    # diviser par T est monotone → la prédiction (argmax) ne doit jamais changer
    assert np.array_equal((logits / T).argmax(1), logits.argmax(1))
