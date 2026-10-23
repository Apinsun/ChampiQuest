"""Génère les graphes d'importance SHAP pour un modèle LightGBM.

Usage :
    poetry run python -m champiquest.models.explain --species boletus_edulis

Sorties : ``reports/shap_importance_<espèce>.png`` et ``reports/shap_beeswarm_<espèce>.png``.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import shap

from champiquest.models.evaluate import CAT_FEATURE, FULL_NUM_FEATURES, LGB_CONFIGS, load_species

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS = REPO_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="boletus_edulis")
    parser.add_argument("--config", default="A_simple", choices=list(LGB_CONFIGS))
    args = parser.parse_args()

    df = load_species(args.species, FULL_NUM_FEATURES)
    cfg = LGB_CONFIGS[args.config]
    model = lgb.LGBMClassifier(**cfg, verbose=-1)
    X = df[FULL_NUM_FEATURES + [CAT_FEATURE]]
    model.fit(X, df["target"], categorical_feature=[CAT_FEATURE])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 1) Importance moyenne (bar plot)
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    bar_path = REPORTS / f"shap_importance_{args.species}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar plot    → {bar_path}")

    # 2) Beeswarm (distribution par observation, top 10)
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.tight_layout()
    bee_path = REPORTS / f"shap_beeswarm_{args.species}.png"
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Beeswarm    → {bee_path}")


if __name__ == "__main__":
    main()
