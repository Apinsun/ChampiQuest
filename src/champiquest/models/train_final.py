"""Entraînement des modèles finaux (production) : 1 LightGBM par espèce, set pruned.

- Entraîne sur **tout** le dataset (pas de split temporel) → modèle « de production »
  destiné à l'API / la carte de prédiction du jour.
- Config `A_simple` (validée en LOOCV dans evaluate.py).
- Sauvegarde `joblib` (data/models/growth_<espèce>.joblib) + log MLflow
  (`mlflow.lightgbm.log_model`).

Usage :
    poetry run python -m champiquest.models.train_final                    # les 8 espèces
    poetry run python -m champiquest.models.train_final --species boletus_edulis
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from champiquest.models.constants import LGB_CONFIGS, PRUNED_NUM_FEATURES
from champiquest.models.evaluate import load_species
from champiquest.models.mlflow_config import setup

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "data" / "models"

# Les 8 espèces modélisées (groupes écologiques de la partie 1).
SPECIES = [
    "boletus_edulis",
    "macrolepiota_procera",
    "cantharellus_cibarius",
    "morchella",
    "imleria_badia",
    "boletus_reticulatus",
    "craterellus_cornucopioides",
    "hydnum_repandum",
]

FEATURES = PRUNED_NUM_FEATURES  # 13 features (pruned + day_of_year), validé en LOOCV


def train_one(species: str):
    """Entraîne + calibre (Platt, CV=5) le modèle final d'une espèce. Renvoie (model, df).

    `CalibratedClassifierCV(cv=5)` entraîne le LightGBM sur 5 folds, ajuste une
    sigmoïde (Platt) sur les prédictions hors-fold, puis moyenne les modèles.
    Le modèle sauvegardé expose un `predict_proba` calibré → le chemin de
    prédiction (growth.py / mapgen.py) n'a pas besoin de changer.
    """
    df = load_species(species, FEATURES)
    base = lgb.LGBMClassifier(**LGB_CONFIGS["A_simple"], verbose=-1)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    calibrated.fit(df[FEATURES], df["target"])
    return calibrated, df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default=None, help="espèce unique (sinon les 8)")
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    args = parser.parse_args()

    setup()  # expérience MLflow « champiquest »
    species_list = [args.species] if args.species else SPECIES
    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in species_list:
        model, df = train_one(sp)
        auc_train = roc_auc_score(df["target"], model.predict_proba(df[FEATURES])[:, 1])

        # sauvegarde locale (pour l'API / la carte)
        path = out_dir / f"growth_{sp}.joblib"
        joblib.dump({"model": model, "features": FEATURES, "species": sp}, path)

        # versionnage MLflow
        with mlflow.start_run(run_name=sp):
            mlflow.log_params({"species": sp, "n_rows": len(df), **LGB_CONFIGS["A_simple"]})
            mlflow.log_metrics({"auc_train": auc_train})
            mlflow.log_artifact(str(path))

        print(f"{sp:22s} : {len(df):6d} lignes | AUC train {auc_train:.3f} | {path.name}")


if __name__ == "__main__":
    main()
