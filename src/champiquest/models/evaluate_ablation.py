"""Ablation : modèle unique (espèce = feature catégorielle) vs un modèle par espèce.

Répond à la question de soutenance « pourquoi N modèles plutôt qu'un seul ? ».
Compare, sur le MÊME protocole LOOCV (leave-one-year-out) et les MÊMES données,
les 4 variantes :

  1. baseline      : un LightGBM par espèce (features `pruned`, 12 features)
  2. baseline+doy  : idem + feature `day_of_year` (le modèle peut apprendre la saison)
  3. unique        : un seul LightGBM, `taxon_key` en feature catégorielle
  4. unique+doy    : idem + feature `day_of_year`

Métrique : AUC par espèce (slicée). Pourquoi pas l'AUC globale ?
→ dans un modèle unique, la feature `taxon_key` porte le taux de base de chaque
espèce ; l'AUC globale (toutes espèces mélangées) serait gonflée par ce
« raccourci de prévalence ». L'AUC par espèce y est insensible (dans le sous-
ensemble d'une espèce, `taxon_key` est constant, donc inutile au classement).

Usage :
    poetry run python -m champiquest.models.evaluate_ablation
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from champiquest.models.constants import DATASET, LGB_CONFIGS

# Les 8 espèces modélisées (mêmes clés que growth.py / configs/species.yaml).
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

CONFIG = LGB_CONFIGS["A_simple"]

# Set de base GELÉ pour cette expérience historique : les 12 features `pruned`
# SANS day_of_year (qui était alors une feature « extra »). Depuis, day_of_year a
# été intégré à PRUNED_NUM_FEATURES ; ce script reste indépendant pour être
# reproductible tel quel.
BASE_FEATURES = [
    "precip_J7", "precip_J30", "precip_J90",
    "temp_min_J14", "temp_max_J14",
    "choc_thermique", "tendance_temp", "amplitude_thermique",
    "bilan_hydrique_J30",
    "temp_sol_J14", "humidite_sol_J30",
    "sand_0-5cm",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATASET, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df = df[df["taxon_key"].isin(SPECIES)].copy()
    df["taxon_key"] = df["taxon_key"].astype("category")
    df = df.dropna(subset=BASE_FEATURES)
    return df


def test_mask(df: pd.DataFrame) -> pd.Series:
    """Exclut du test les présences « proxy » (co-fructification), comme evaluate.py.

    Une présence proxy = « une autre espèce a fructifié ici → habitat favorable »,
    utile à l'entraînement, mais la tester serait « prédire une autre espèce ».
    """
    return ~((df["species"] == "proxy") & (df["target"] == 1))


def auc_mean(aucs: list[float]) -> float:
    return float(np.mean(aucs)) if aucs else float("nan")


# --- Variantes 1 & 2 : un modèle par espèce ---------------------------------


def per_species_loocv(df: pd.DataFrame, extra: list[str]) -> dict[str, list[float]]:
    features = BASE_FEATURES + extra
    results: dict[str, list[float]] = {sp: [] for sp in SPECIES}
    for sp in SPECIES:
        sub = df[df["taxon_key"] == sp]
        for year in sorted(sub["year"].unique()):
            train = sub[sub["year"] != year]
            test = sub[(sub["year"] == year) & test_mask(sub)]
            if test["target"].nunique() < 2:
                continue
            model = lgb.LGBMClassifier(**CONFIG, verbose=-1)
            model.fit(train[features], train["target"])
            proba = model.predict_proba(test[features])[:, 1]
            results[sp].append(roc_auc_score(test["target"], proba))
    return results


# --- Variantes 3 & 4 : un seul modèle ---------------------------------------


def single_loocv(df: pd.DataFrame, extra: list[str]) -> tuple[dict[str, list[float]], float]:
    features = BASE_FEATURES + extra
    cat = ["taxon_key"] if "taxon_key" in features else []
    results: dict[str, list[float]] = {sp: [] for sp in SPECIES}
    all_proba, all_target = [], []  # pour l'AUC globale (diagnostic raccourci)
    for year in sorted(df["year"].unique()):
        train = df[df["year"] != year]
        test = df[(df["year"] == year) & test_mask(df)]
        model = lgb.LGBMClassifier(**CONFIG, verbose=-1)
        model.fit(train[features], train["target"], categorical_feature=cat)
        proba = model.predict_proba(test[features])[:, 1]
        all_proba.extend(proba)
        all_target.extend(test["target"].values)
        for sp, grp in pd.DataFrame({"taxon_key": test["taxon_key"], "target": test["target"], "p": proba}).groupby(
            "taxon_key", observed=True
        ):
            if grp["target"].nunique() < 2:
                continue
            results[sp].append(roc_auc_score(grp["target"], grp["p"]))
    global_auc = roc_auc_score(all_target, all_proba) if len(set(all_target)) > 1 else float("nan")
    return results, float(global_auc)


# --- Affichage --------------------------------------------------------------


def print_table(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    for sp in SPECIES:
        row = df.loc[sp]
        print(f"  {sp:26s} AUC {row['auc']:.3f}  ({row['n_folds']} folds)")
    print(f"  {'moyenne':26s} AUC {df['auc'].mean():.3f}")


def main() -> None:
    df = load()
    print(f"dataset : {len(df)} lignes, {df['year'].nunique()} années, {len(SPECIES)} espèces")
    print(f"features pruned (base, gelé) : {len(BASE_FEATURES)}")

    # 1. baseline (un modèle par espèce)
    r1 = per_species_loocv(df, [])
    # 2. baseline + day_of_year
    r2 = per_species_loocv(df, ["day_of_year"])
    # 3. unique (taxon_key)
    r3, g3 = single_loocv(df, ["taxon_key"])
    # 4. unique + day_of_year
    r4, g4 = single_loocv(df, ["taxon_key", "day_of_year"])

    def summarize(r: dict[str, list[float]]) -> pd.DataFrame:
        return pd.DataFrame(
            {"auc": [auc_mean(r[sp]) for sp in SPECIES], "n_folds": [len(r[sp]) for sp in SPECIES]},
            index=SPECIES,
        )

    s1, s2, s3, s4 = summarize(r1), summarize(r2), summarize(r3), summarize(r4)

    print_table(s1, "1. baseline : un modèle par espèce (pruned)")
    print_table(s2, "2. baseline + day_of_year")
    print_table(s3, "3. modèle unique (pruned + taxon_key)")
    print_table(s4, "4. modèle unique (pruned + taxon_key + day_of_year)")

    print(f"\n  AUC globale du modèle unique (diagnostic) :")
    print(f"    unique       : {g3:.3f}  ← gonflée par la feature espèce (taux de base)")
    print(f"    unique + doy : {g4:.3f}")

    # Comparaison côte à côte
    cmp = pd.DataFrame(
        {
            "baseline": s1["auc"],
            "base+doy": s2["auc"],
            "unique": s3["auc"],
            "unique+doy": s4["auc"],
        },
        index=SPECIES,
    )
    cmp.loc["MOYENNE"] = cmp.mean()
    cmp["Δ unique - baseline"] = cmp["unique"] - cmp["baseline"]
    cmp["Δ unique+doy - baseline"] = cmp["unique+doy"] - cmp["baseline"]
    print("\n=== Comparaison (AUC par espèce) ===")
    print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
