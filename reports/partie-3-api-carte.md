# Partie 3 — API, carte de prédiction & industrialisation

> Journal de la session du 23/10/2026 : ce qui a été construit, pourquoi, les
> problèmes rencontrés, et ce qui reste.

## 1. Architecture cible

Deux temps : du **calcul offline** (cron/manuel) et du **service online** (API).

```
STOCKAGE (data/, gitignoré)
├── weather/history.parquet   ← météo brute accumulée (ERA5 + forecast)
├── maps/<date>/probs.npz     ← probas de la carte (précalculées)
└── models/
    ├── growth_*.joblib       ← 6 LightGBM (partie 1)
    └── vision_best.pt        ← 1 ConvNeXt (partie 2)

CALCUL (offline)
  Parquet → mapgen.py → features (12) → 6 modèles → npz

SERVEUR (online, FastAPI)
  GET  /             → frontend Leaflet
  POST /identify     → modèle vision (torch)
  POST /growth/predict → modèles growth (joblib) + météo live
  GET  /map          → lit les npz → rendu PNG (échelle fixe 0→1)
```

## 2. Choix architecturaux (et pourquoi)

| Choix | Pourquoi |
|---|---|
| **Store météo Parquet** (`weather_store.py`) | Accumuler la météo passée (pas de re-fetch quotidien), source de vérité brute dont dérivent les features. Parquet = colonnaire, compressé, standard data engineering |
| **Hybride ERA5 + forecast** | ERA5 = réanalyse *précise* du passé (~5 j de délai) ; forecast = récent + futur. On stocke l'ERA5 (stable) et on re-fetch le forecast (transitoire) à chaque run |
| **Champ `source` (era5/forecast)** | Flag demandé pour savoir quoi écraser : le forecast est supprimé + re-fetché à chaque run (sans cron, l'écart entre 2 runs le rend obsolète), l'ERA5 est conservé |
| **npz intermédiaire** | Les probas précalculées (tableaux NumPy compressés), servies en PNG. Découple le calcul lourd (offline) du service léger (online) |
| **Modèles finaux entraînés sur tout le dataset** | `train_final.py` : 6 LightGBM (set pruned), sauvegardés joblib **+** loggés MLflow. Les vrais chiffres restent LOOCV ~0.76 (pas l'AUC train ~0.9, optimiste) |
| **Season gate** | Mettre à 0 la proba hors de la fenêtre de fructification (la morille en octobre ressortait à 0.47 sans ça) |
| **Échelle couleur fixe 0→1** | La couleur encode la probabilité absolue (comparable entre espèces/dates). D'abord dynamique (p2→p98), corrigé : 0.25 ne doit jamais paraître « rouge foncé » |
| **Interpolation + alpha** | Lissage bicubique ×10 (fini les gros carrés) + alpha progressif (les zones faibles laissent voir le terrain) |

## 3. Problèmes rencontrés & solutions

1. **cuDNN plante** (`Invalid handle... cudnnGetVersion`) → lié au **driver CUDA 13.2 bêta**, pas à torch (testé cu128 et cu130). Workaround : `torch.backends.cudnn.enabled = False` (kernels CUDA natifs, ~10-20 % plus lents).
2. **`poetry add` a rétrogradé torch** (2.11.0+cu128 → +cu130) → sans gravité (CUDA OK, cuDNN désactivé de toute façon).
3. **`.gitignore` trop large** : `data/` et `models/` ignoraient aussi `src/champiquest/data/` et `models/` (le code partie 1 n'était jamais commité). Corrigé avec `/data/` et `/models/` (racine seulement) + récupération depuis l'ancien PC.
4. **Limites API Open-Meteo** : une requête 26 ans ≈ 475 appels (formule jours/14 × variables/10), plafond 10 000/jour → le re-fetch sol a dû passer en « sol uniquement » (2 variables ≈ 136 appels/requête) + fusion dans les CSV existants.
5. **`max()` dominé par une seule espèce** : en octobre, les lépiotes (0.59-0.88 partout) écrasent les 5 autres → la vue « tout confondu » = carte des lépiotes. Accepté pour l'instant (info « où trouver un champi ») ; options futures : enrichissement relatif `p_i/médiane_i`, ou couche « richesse ».
6. **Bugs de chemins** `parents[N]` (growth.py / api.py / STATIC_DIR) → corrigés au cas par cas.
7. **Mapgen non-resumable** → rendu resumable (écriture par lots de 50 points + reprise per-point de l'ERA5).

## 4. Ce qui reste

- [ ] **Backfill ERA5 complet** (~7000 appels, en cours de lancement, reprendra où il s'est arrêté).
- [ ] **Multi-dates J..J+3** : le nouveau mapgen génère 4 npz ; à valider une fois le store rempli.
- [ ] **Docker** : conteneuriser l'API (+ volume pour le Parquet/npz) + **CI/CD** GitHub Actions.
- [ ] **Enrichissement « tout confondu »** (option, cf. problème 5).
- [ ] README + rapport de conduite de projet (template AIE) pour la soutenance.

## 5. Commandes clés

```bash
poetry run uvicorn champiquest.api:app --host 0.0.0.0 --port 8000   # API + frontend
poetry run python -m champiquest.mapgen --refresh-only              # backfill ERA5 (resumable)
poetry run python -m champiquest.mapgen                             # refresh + generate (4 dates)
poetry run python -m champiquest.models.train_final                 # (re)entraîner les 6 modèles
```
