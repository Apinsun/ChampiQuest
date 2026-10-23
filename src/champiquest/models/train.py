"""Entraînement d'un modèle LightGBM (baseline) avec suivi MLflow.

Usage :
    poetry run python -m champiquest.models.train --species boletus_edulis

Pour l'instant : split temporel simple (train < année de test, test ≥ année de test).
La CV par blocs annuels viendra ensuite.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score

from champiquest.models.mlflow_config import setup

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "processed" / "dataset.csv"

# Features du modèle (clc_code = catégorielle, les autres numériques)
FEATURES = [
    "precip_J7", "precip_J14", "precip_J30", "precip_J60", "precip_J90",
    "temp_mean_J14", "temp_min_J14", "temp_max_J14",
    "choc_thermique", "tendance_temp", "amplitude_thermique",
    "bilan_hydrique_J30", "bilan_hydrique_J60", "bilan_hydrique_J90",
    "altitude", "pente", "northness", "eastness",
    "phh2o_0-5cm", "sand_0-5cm", "clay_0-5cm", "soc_0-5cm",
    "clc_code",
]

HYPERPARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 0,
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="boletus_edulis", help="clé du taxon à modéliser")
    parser.add_argument("--test-year", type=int, default=2024, help="année de début du jeu de test")
    args = parser.parse_args()

    setup()  # expérience MLflow « champiquest »

    df = load_dataset(DATASET)
    df = df[df["taxon_key"] == args.species].copy()
    df = df.dropna(subset=FEATURES)
    # clc_code est un code nominal (111=urbain, 311=feuillus…) → catégoriel
    df["clc_code"] = df["clc_code"].astype(int).astype("category")

    train = df[df["year"] < args.test_year]
    test = df[df["year"] >= args.test_year]
    print(
        f"{args.species}: train {len(train)} (années {train['year'].min()}–{train['year'].max()}), "
        f"test {len(test)} (années {test['year'].min()}–{test['year'].max()})"
    )
    if len(test) == 0:
        raise SystemExit("Jeu de test vide : ajuste --test-year.")

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    model = lgb.LGBMClassifier(**HYPERPARAMS, verbose=-1)
    model.fit(X_train, y_train, categorical_feature=["clc_code"])

    auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"ROC-AUC : train {auc_train:.3f} | test {auc_test:.3f}")

    # --- Suivi MLflow ---
    with mlflow.start_run(run_name=f"baseline_{args.species}"):
        mlflow.log_params(
            {
                "species": args.species,
                "test_year": args.test_year,
                "n_train": len(train),
                "n_test": len(test),
                "features": FEATURES,
                **HYPERPARAMS,
            }
        )
        mlflow.log_metrics({"roc_auc_train": auc_train, "roc_auc_test": auc_test})
        mlflow.lightgbm.log_model(model, name="model")

    # --- Importance des features (lecture rapide) ---
    importance = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )
    print("\nImportance des features (top) :")
    print(importance.head(10).round(3).to_string())


if __name__ == "__main__":
    main()
