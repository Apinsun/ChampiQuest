# ChampiQuest 🍄

Prévision de **pousse de champignons** (météo → fructification) et **reconnaissance
visuelle d'espèce**, exposées via une API, une carte interactive et une page de démo.

Projet de fin de formation (AI Engineer) — tout en français, chaque choix documenté
et validé par des baselines/ablations (voir `AGENTS.md` et `reports/`).

---

## Les 3 parties

| Partie | Objectif | Approche | Résultat clé |
|---|---|---|---|
| **1. Pousse** | Prédire quand/où un champignon fructifie | LightGBM **un modèle par espèce** (8 espèces) | **AUC 0.811** (LOOCV) |
| **2. Vision** | Reconnaître l'espèce depuis une photo | **ConvNeXt-Base** fine-tuné (20 classes) + calibration + décision à coût | **0 erreur dangereuse** |
| **3. API & carte** | Rendre les modèles utilisables | **FastAPI** + Leaflet + Docker/CI-CD | `/identify`, `/growth/predict`, `/map`, `/demo` |

---

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  GBIF        │   │  Open-Meteo  │   │  SoilGrids   │
│ (présences)  │   │ ERA5+forecast│   │  (sol)       │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │  pseudo-absences  │                  │
       ▼                   ▼                  ▼
   dataset.csv ──────► features (13) ──► LightGBM × 8 ──► carte / API
```

- **Partie 1** : pseudo-absences « effort prouvé » + features météo/sol (13) + un
  LightGBM par espèce. Évaluation **LOOCV par année**.
- **Partie 2** : DF20 + `other_fungus` + `non-champi` → 20 classes, ConvNeXt-Base.
  Calibration (temperature scaling) puis décision à **coût asymétrique** (une erreur
  « comestible sur toxique » vaut 10×).
- **Partie 3** : FastAPI, store météo Parquet, carte `mapgen`, frontend Leaflet,
  Docker + GitHub Actions (CI tests / CD GHCR).

---

## Installation (développement)

Prérequis : **Python 3.12** et [Poetry](https://python-poetry.org/).

```bash
poetry install              # deps runtime + train + test
poetry run pytest -q        # 14 tests
```

Groupes de dépendances (`pyproject.toml`) :
- `main` : runtime (torch CPU/torchvision, timm, lightgbm, scikit-learn, fastapi…)
- `train` : entraînement & analyses (jupyter, mlflow, shap, seaborn, folium…)
- `test` : pytest, httpx

> ⚠️ `data/` et `models/` sont **gitignorés** (données volumineuses). Les modèles et
> le store météo se placent dans `data/models/` et `data/weather/` localement.

---

## Lancer l'API en local

```bash
poetry run uvicorn champiquest.api:app --host 127.0.0.1 --port 8000
```

| Endpoint | Rôle |
|---|---|
| `GET /health` | statut |
| `POST /identify` | 1..N photos (base64) → espèce + décision + top-3 |
| `POST /growth/predict` | `{lat, lon, date}` → proba de pousse par espèce |
| `GET /map?date=…&species=…` | heatmap PNG (probabilités 0→1) |
| `GET /species` | liste des 8 espèces |
| `GET /` et `GET /demo` | frontend carte + page démo vision |

Swagger : `http://127.0.0.1:8000/docs`.

### Générer la carte

```bash
poetry run python -m champiquest.mapgen               # refresh météo + génération
poetry run python -m champiquest.mapgen --skip-refresh  # génération seule
poetry run python -m champiquest.mapgen --limit 10       # test rapide
```

La carte couvre ~1634 points (grille ~5 km, Vosges / Forêt-Noire), 4 dates (J..J+3),
échelle fixe 0→1, interpolation + lissage.

---

## Déploiement Docker

```bash
./deploy.sh                     # pull GHCR + up (prod)
./scripts/smoke_test.sh         # vérifie les 5 endpoints
./stop.sh
```

- Image : `python:3.12-slim`, deps runtime uniquement, **modèles/données en volume**.
- **CI** (`ci.yml`) : tests sur toutes branches/PR.
- **CD** (`cd.yml`) : build + push GHCR sur `main` et tags `v*`, avec cache de layers
  (pull incrémental) et filtre de chemins (pas de rebuild si seuls docs/scripts changent).

---

## Structure du projet

```
configs/            species.yaml, vision.yaml, season_windows.yaml
src/champiquest/
  data/             GBIF, Open-Meteo, make_dataset (pseudo-absences), environnement
  features/         weather (fenêtres), season (fenêtres de fructification)
  models/           train, evaluate (LOOCV), train_final, explain (SHAP), constants
  vision/           datasets, train, predict, decision (coût/calibration), explain
  growth.py         prédiction de pousse (runtime)
  mapgen.py         génération de la carte
  weather_store.py  store Parquet ERA5 + forecast
  api.py            FastAPI
  static/           index.html (carte), identify.html (démo)
tests/              14 tests (pytest + httpx)
reports/            rapports de design & résultats (markdown) + figures
presentation/       slides de soutenance (index.html, imprimable en PDF)
```

---

## Résultats (résumé)

**Partie 1 — pousse (AUC LOOCV, 8 espèces, 13 features)** : moyenne **0.811**

| Espèce | AUC |
|---|---|
| Trompette de la mort | 0.885 |
| Lépiote élevée | 0.882 |
| Cèpe d'été | 0.862 |
| Bolet bai | 0.821 |
| Pied de mouton | 0.817 |
| Morille | 0.753 |
| Cèpe de Bordeaux | 0.749 |
| Girolle | 0.722 |

**Partie 2 — vision (20 classes)** : test accuracy **0.879**, macro-F1 **0.841**,
**0 erreur dangereuse** après calibration + décision à coût (étranger→comestible : 1.4 %, bénin).

---

## Documentation

- `reports/partie-1-prediction-pousse.md` — design partie 1 (données, features, modèles).
- `reports/partie-2-reconnaissance-image.md` — vision, calibration, coût.
- `reports/partie-3-api-carte.md` — architecture stockage/calcul/serveur.
- `reports/feature-engineering.md` — élagage, proxies, ablation modèle unique vs N.
- `reports/ajout-especes-calibration.md` — process d'ajout d'espèce + calibration en profondeur.
- `AGENTS.md` — règles de collaboration humain ↔ agent.
