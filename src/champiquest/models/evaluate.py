"""Comparaison de baselines par CV Leave-One-Year-Out (blocs annuels), avec MLflow.

Compare, sur une même espèce :
- une régression logistique (pipeline StandardScaler + OneHotEncoder) ;
- plusieurs configs LightGBM (du simple au sur-ajusté).

Deux jeux de features : ``full`` (toutes les fenêtres météo) et ``pruned``
(fenêtres non redondantes, pour limiter la collinéarité).

Usage :
    poetry run python -m champiquest.models.evaluate --species boletus_edulis --feature-set pruned
    poetry run python -m champiquest.models.evaluate --species boletus_edulis --tune  # random search (expérience)
"""

import argparse
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

from champiquest.models.mlflow_config import setup

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


def load_species(species: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df = df[df["taxon_key"] == species].copy()
    df["year"] = df["date"].dt.year
    df["clc_code"] = df["clc_code"].fillna(0).astype(int).astype("category")
    df = df.dropna(subset=features)
    return df


def make_logistic(features: list[str]):
    """Pipeline logistique ; gère la présence éventuelle de la feature catégorielle."""
    num_feats = [f for f in features if f != CAT_FEATURE]
    cat_feats = [f for f in features if f == CAT_FEATURE]
    transformers = [("num", StandardScaler(), num_feats)]
    if cat_feats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats))
    preprocess = ColumnTransformer(transformers=transformers)
    return Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=0)),
    ])


def make_lightgbm(cfg: dict):
    return lgb.LGBMClassifier(**cfg, verbose=-1)


def loocv(df: pd.DataFrame, build_model, is_lgb: bool, features: list[str]) -> list[dict]:
    """Leave-One-Year-Out : chaque année est testée une fois.

    Les proxies (espèces co-occurrentes ajoutées en présences) servent uniquement à
    l'entraînement : on les exclut du test pour mesurer la prédiction de la cible réelle.
    """
    folds = []
    for year in sorted(df["year"].unique()):
        train = df[df["year"] != year]
        test = df[(df["year"] == year) & ~((df["species"] == "proxy") & (df["target"] == 1))]
        if test["target"].nunique() < 2:
            continue
        model = build_model()
        cat = [CAT_FEATURE] if CAT_FEATURE in features else []
        if is_lgb:
            model.fit(train[features], train["target"], categorical_feature=cat)
        else:
            model.fit(train[features], train["target"])
        proba = model.predict_proba(test[features])[:, 1]
        auc = roc_auc_score(test["target"], proba)
        folds.append({"year": int(year), "n_test": len(test), "auc": float(auc)})
    return folds


# ---------------------------------------------------------------------------
# Random search LightGBM (EXPÉRIENCE)
# ---------------------------------------------------------------------------
# ⚠️ Sélection sur les MÊMES folds LOOCV que l'évaluation → fuite légère :
# l'AUC reporté est optimiste. Résultat indicatif pour guider la régularisation,
# pas un score final. À mentionner comme tel à la soutenance.

TUNE_SPACE = {
    "n_estimators": [30, 50, 80, 120, 200],
    "learning_rate": [0.02, 0.05, 0.08, 0.1, 0.15],
    "num_leaves": [5, 7, 10, 15, 20],
    "max_depth": [3, 4, 5, 6, -1],
    "min_child_samples": [10, 20, 30, 50, 80, 120],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}


def _sample_cfg(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": int(rng.choice(TUNE_SPACE["n_estimators"])),
        "learning_rate": float(rng.choice(TUNE_SPACE["learning_rate"])),
        "num_leaves": int(rng.choice(TUNE_SPACE["num_leaves"])),
        "max_depth": int(rng.choice(TUNE_SPACE["max_depth"])),
        "min_child_samples": int(rng.choice(TUNE_SPACE["min_child_samples"])),
        "subsample": float(rng.choice(TUNE_SPACE["subsample"])),
        "colsample_bytree": float(rng.choice(TUNE_SPACE["colsample_bytree"])),
        "reg_lambda": float(rng.choice(TUNE_SPACE["reg_lambda"])),
        "reg_alpha": float(rng.choice(TUNE_SPACE["reg_alpha"])),
        "random_state": 0,
    }


def _cfg_str(cfg: dict) -> str:
    return (
        f"leaves={cfg['num_leaves']} depth={cfg['max_depth']} "
        f"mcs={cfg['min_child_samples']} lr={cfg['learning_rate']} "
        f"n={cfg['n_estimators']} sub={cfg['subsample']} "
        f"col={cfg['colsample_bytree']} l2={cfg['reg_lambda']} l1={cfg['reg_alpha']}"
    )


def random_search(species: str, features: list[str], n_iters: int, seed: int) -> None:
    """Random search LightGBM sur les folds LOOCV (expérience, cf. avertissement)."""
    df = load_species(species, features)
    rng = np.random.default_rng(seed)

    print(f"Random search : {species} ({len(df)} lignes, {n_iters} itérations, {len(features)} features)")
    results = []
    for it in range(n_iters):
        cfg = _sample_cfg(rng)
        build = (lambda c=cfg: make_lightgbm(c))
        folds = loocv(df, build, True, features)
        auc = float(np.mean([f["auc"] for f in folds]))
        results.append((auc, cfg))
        print(f"[{it + 1}/{n_iters}] AUC {auc:.3f} | {_cfg_str(cfg)}", flush=True)

    results.sort(key=lambda r: -r[0])
    print("\nTop 5 :")
    for auc, cfg in results[:5]:
        print(f"  AUC {auc:.3f} | {_cfg_str(cfg)}")

    best_auc, best_cfg = results[0]
    with mlflow.start_run(run_name=f"tune_{species}_lightgbm"):
        mlflow.log_params({"species": species, "n_iters": n_iters, "seed": seed})
        mlflow.log_metric("best_mean_auc", best_auc)
        mlflow.log_dict({"best_cfg": best_cfg, "top5": results[:5]}, "tune_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="boletus_edulis")
    parser.add_argument("--feature-set", choices=list(FEATURE_SETS), default="full")
    parser.add_argument("--tune", action="store_true", help="Random search LightGBM (expérience, fuite légère)")
    parser.add_argument("--tune-iters", type=int, default=40, help="nombre d'itérations du random search")
    parser.add_argument("--tune-seed", type=int, default=0, help="graine du tirage des configs")
    args = parser.parse_args()

    setup()
    features = FEATURE_SETS[args.feature_set]
    if args.tune:
        random_search(args.species, features, args.tune_iters, args.tune_seed)
        return
    df = load_species(args.species, features)
    print(f"{args.species}: {len(df)} lignes, {df['year'].nunique()} années, {len(features)} features")

    models = [("logistic", (lambda: make_logistic(features)), False)]
    for cfg_name, cfg in LGB_CONFIGS.items():
        models.append((f"lightgbm_{cfg_name}", (lambda c=cfg: make_lightgbm(c)), True))

    for name, build, is_lgb in models:
        folds = loocv(df, build, is_lgb, features)
        aucs = [f["auc"] for f in folds]
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        print(f"{name:22s} → {len(folds)} folds, AUC {mean_auc:.3f} ± {std_auc:.3f}")

        with mlflow.start_run(run_name=f"cv_{args.species}_{args.feature_set}_{name}"):
            mlflow.log_params({"species": args.species, "model": name, "feature_set": args.feature_set})
            mlflow.log_metrics({"mean_auc": mean_auc, "std_auc": std_auc})
            mlflow.log_dict({"folds": folds}, "folds.json")


if __name__ == "__main__":
    main()
