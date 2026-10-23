"""Évaluation du modèle multi-espèces (espèce = feature catégorielle).

Compare à la stratégie « un modèle par espèce » : est-ce qu'entraîner sur toutes
les espèces (et laisser le modèle apprendre les différences via la feature
``taxon_key``) aide à mieux prédire chaque espèce ?

Usage :
    poetry run python -m champiquest.models.evaluate_multi
"""

from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from champiquest.models.evaluate import FULL_NUM_FEATURES, LGB_CONFIGS
from champiquest.models.mlflow_config import setup

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "processed" / "dataset.csv"

# Deux features catégorielles : occupation du sol + espèce
CAT_FEATURES = ["clc_code", "taxon_key"]


def load_all() -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["clc_code"] = df["clc_code"].astype(int).astype("category")
    df["taxon_key"] = df["taxon_key"].astype("category")
    df = df.dropna(subset=FULL_NUM_FEATURES + CAT_FEATURES)
    return df


def make_logistic():
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FULL_NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ]
    )
    return Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=0)),
    ])


def make_lightgbm(cfg: dict):
    return lgb.LGBMClassifier(**cfg, verbose=-1)


def loocv_per_species(df: pd.DataFrame, build_model, is_lgb: bool) -> dict:
    """LOOCV par année, mais l'AUC est calculée espèce par espèce sur le test."""
    features = FULL_NUM_FEATURES + CAT_FEATURES
    results: dict[str, list[float]] = {s: [] for s in df["taxon_key"].unique()}
    for year in sorted(df["year"].unique()):
        train = df[df["year"] != year]
        test = df[df["year"] == year]
        model = build_model()
        if is_lgb:
            model.fit(train[features], train["target"], categorical_feature=CAT_FEATURES)
        else:
            model.fit(train[features], train["target"])
        proba = model.predict_proba(test[features])[:, 1]
        test = test.copy()
        test["proba"] = proba
        for species, grp in test.groupby("taxon_key", observed=True):
            if grp["target"].nunique() < 2:
                continue
            results[species].append(roc_auc_score(grp["target"], grp["proba"]))
    return results


def main() -> None:
    setup()
    df = load_all()
    print(f"{len(df)} lignes, {df['taxon_key'].nunique()} espèces")

    models = [("logistic", make_logistic, False)]
    for cfg_name, cfg in LGB_CONFIGS.items():
        models.append((f"lightgbm_{cfg_name}", (lambda c=cfg: make_lightgbm(c)), True))

    for name, build, is_lgb in models:
        results = loocv_per_species(df, build, is_lgb)
        print(f"\n{name} (modèle global) :")
        summary = {}
        for species, aucs in sorted(results.items()):
            m = float(np.mean(aucs))
            summary[species] = m
            print(f"  {species:24s} → AUC {m:.3f}  ({len(aucs)} folds)")
        with mlflow.start_run(run_name=f"multi_{name}"):
            mlflow.log_params({"model": name, "mode": "multi"})
            mlflow.log_metrics({f"auc_{s}": v for s, v in summary.items()})
            mlflow.log_dict({"per_species": summary}, "per_species.json")


if __name__ == "__main__":
    main()
