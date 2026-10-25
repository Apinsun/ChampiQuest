"""Constantes partagées des modèles (features, configs LightGBM, chemins).

Module volontairement LÉGER (aucun import lourd) : importé par le runtime
(growth.py, mapgen.py) SANS tirer mlflow/lightgbm.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "processed" / "dataset.csv"

CAT_FEATURE = "clc_code"

# Jeu complet : toutes les fenêtres (forte redondance entre fenêtres)
FULL_NUM_FEATURES = [
    "precip_J7", "precip_J14", "precip_J30", "precip_J60", "precip_J90",
    "temp_mean_J14", "temp_min_J14", "temp_max_J14",
    "choc_thermique", "tendance_temp", "amplitude_thermique",
    "bilan_hydrique_J30", "bilan_hydrique_J60", "bilan_hydrique_J90",
    "temp_sol_J14", "rechauffement_sol", "humidite_sol_J30",
    "altitude", "pente", "northness", "eastness",
    "phh2o_0-5cm", "sand_0-5cm", "clay_0-5cm", "soc_0-5cm",
]

# Jeu élagué (data-driven) : corrélations + SHAP → 12 features non redondantes.
# 1 représentant par cluster (temp_sol plutôt que temp_mean, bilan_J30 plutôt que
# J60/J90, tendance/choc plutôt que rechauffement_sol) ; vire les quasi-mortes
# (clc_code, relief, texture du sol hors sable).
PRUNED_NUM_FEATURES = [
    "precip_J7", "precip_J30", "precip_J90",
    "temp_min_J14", "temp_max_J14",
    "choc_thermique", "tendance_temp", "amplitude_thermique",
    "bilan_hydrique_J30",
    "temp_sol_J14", "humidite_sol_J30",
    "sand_0-5cm",
]

FEATURE_SETS = {
    "full": FULL_NUM_FEATURES + [CAT_FEATURE],
    "pruned": PRUNED_NUM_FEATURES,  # sans clc_code (importance ~0)
}

LGB_CONFIGS = {
    "A_simple": {"n_estimators": 50, "num_leaves": 7, "min_child_samples": 50, "max_depth": 4, "random_state": 0},
    "B_moyen": {"n_estimators": 100, "num_leaves": 15, "min_child_samples": 30, "max_depth": 6, "random_state": 0},
    "C_complexe": {"n_estimators": 300, "num_leaves": 31, "min_child_samples": 20, "max_depth": -1, "random_state": 0},
}
