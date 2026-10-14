# ChampiQuest — Partie 1 : Prédiction de pousse (météo → fructification)

> Document de design & décisions, rédigé après discussion préalable.
> Statut : **draft** — à faire évoluer après l'EDA.
> Aucun code écrit à ce stade : ce document fige les choix avant l'implémentation.

---

## 1. Objectif

Prédire, pour une espèce de champignon donnée et un lieu donné dans la zone
Vosges + Forêt-Noire, si les conditions météorologiques récentes sont favorables
à la fructification.

C'est la **première partie** d'un projet plus large qui comprendra ensuite :

1. **Partie 1 (ce document)** : prédiction météo → pousse (modèle tabulaire).
2. **Partie 2** : reconnaissance d'espèce par image (ViT/CNN en transfer learning).
3. **Partie 3** : carte interactive (heatmap des zones favorables).

### Périmètre du MVP

- Zone géographique : **Vosges + Forêt-Noire** (bbox ~ lat 47.4–49.0, lon 6.8–8.3,
  incluant la partie allemande et la bordure suisse).
- **5 taxons** : cèpe de Bordeaux, bolet bai, girolle, coulmelle, morilles (genre).
- Modèle tabulaire : **LightGBM**.

---

## 2. Décisions verrouillées (récapitulatif)

| Sujet | Décision | Justification / remarque |
|---|---|---|
| Langage / modèle | Python + **LightGBM** | Robuste, rapide, gère bien le petit volume de données |
| Granularité du modèle | **Un modèle par espèce** (config-driven) | Fenêtres de saison et pseudo-absences déjà par espèce ; question utilisateur par espèce |
| Fallback données rares | **Regroupement par genre** (bolets ensemble, chanterelles ensemble) | Si une espèce a < ~100–150 observations |
| Dépendances | **Poetry 2.x** | Habitude du projet, lockfile partagé CI/Docker |
| Tracking expériences | **MLflow local** (par machine) | `mlflow.db`/`mlartifacts/` **gitignorés** ; git = source de vérité (code + config + seed → reproductible) |
| Cache météo | `requests-cache` | Éviter de re-requêter Open-Meteo à chaque itération |
| API | **Une seule FastAPI**, routers par domaine | `growth` maintenant, `species` plus tard ; interface = client séparé |
| Équilibrage classes | **50/50 par construction** (1 absence générée pour 1 présence) | Pas de poids (`scale_pos_weight`) : les absences sont synthétiques, pas besoin de subir un déséquilibre |
| Features spatiales | **Pas de lat/lon brutes** dans le modèle | Remplacées par altitude, pente, orientation, type de forêt (CLC), type de sol (SoilGrids), humidité du sol |
| Règle hors-saison | **Règle métier** (filtre en amont, sans appeler le modèle) | Évite l'inférence out-of-distribution |

---

## 3. Architecture

### Principe

**Un monorepo, une API FastAPI avec des routers par domaine.** Pas deux API
distinctes (météo / reconnaissance) : un seul déploiement, règles métier
partagées, config partagée. L'interface (Streamlit) est un simple client HTTP.

> Important : la "prédiction météo" n'est **pas** une API à construire.
> Open-Meteo est une source de données externe appelée en interne par le router
> `growth` (récupérer l'historique → features → `model.predict()`).

### Flux d'un appel `POST /predict/growth`

```
lat/lon + espèce + date
   │
   ├─ Règle métier : date hors fenêtre de saison ? ──► {prediction: 0, raison: "hors_saison"}
   │
   ├─ Récupération historique Open-Meteo (J-60 → J-1, cache)
   ├─ Calcul des features (météo + géostatiques)
   ├─ model.predict()  (modèle LightGBM chargé via MLflow)
   │
   └─ {probability, ...} JSON
```

### Structure du dépôt (cible)

```
ChampiQuest/
├── pyproject.toml              # Poetry, deps + entry points
├── README.md
├── .gitignore                  # data/, mlruns/, .venv/, cache, models/
├── configs/
│   ├── species.yaml            # espèces, taxonKey GBIF, fenêtres de saison
│   └── model.yaml              # features, hyperparams, version modèle
├── data/                       # gitignoré
│   ├── raw/                    # sorties GBIF, cache Open-Meteo
│   └── processed/              # datasets train/val par année
├── notebooks/                  # EDA / exploration uniquement
├── reports/                    # documents de design (ce fichier)
├── src/champiquest/
│   ├── config.py               # charge YAML + env
│   ├── data/
│   │   ├── gbif.py             # ingestion occurrences
│   │   ├── openmeteo.py        # historique météo + cache
│   │   ├── pseudo_absences.py  # génération Target=0
│   │   └── make_dataset.py     # orchestration → dataset final
│   ├── features/
│   │   ├── weather.py          # agrégats courts/longs, choc thermique, bilan hydrique
│   │   └── season.py           # percentiles 5/95 + buffer 14j
│   ├── models/
│   │   ├── train.py            # LightGBM + log MLflow
│   │   ├── evaluate.py         # métriques + CV par blocs annuels
│   │   └── predict.py          # inférence (appelée par l'API, indépendante de train)
│   ├── api/
│   │   ├── main.py             # app FastAPI
│   │   ├── schemas.py          # Pydantic = contrat entrée/sortie
│   │   ├── business_rules.py   # filtre hors-saison
│   │   └── routers/
│   │       ├── growth.py       # partie 1
│   │       └── species.py      # partie 2 (plus tard)
│   └── recognition/            # partie 2 (plus tard)
├── frontend/                   # Streamlit (partie 3) — client de l'API
├── tests/
├── .github/workflows/
│   └── ci.yml                  # lint + tests
├── Dockerfile                  # image de l'API
└── mlflow/                     # tracking local (gitignoré)
```

### Trois frontières à respecter

1. **`train` ≠ `predict`** : l'API charge un modèle versionné via MLflow (artifact +
   params + métriques), pas un `.pkl` posé à la main. Le code d'inférence n'importe
   pas le code d'entraînement.
2. **Couches `data` / `features` / `models` / `api`** : l'API ne connaît que
   `features` + `predict`, jamais `gbif.py` directement.
3. **Config hors du code** : fenêtres de saison et liste d'espèces dans
   `configs/species.yaml`, lues au démarrage (modifiable sans redéployer de code).

---

## 4. Pipeline de données

### 4.0 Zone d'étude & qualité géographique

- La zone couvre **Vosges + Forêt-Noire** (bbox 47.4–49.0 lat, 6.8–8.3 lon), incluant
  la partie allemande et la bordure suisse — milieux très similaires, conservés
  volontairement (~3123 occurrences, 74 % côté ouest du Rhin).
- **Découverte EDA** : la colonne `countryCode` GBIF est **peu fiable** près des
  frontières (60 % des lignes étiquetées « CH » alors que géographiquement en
  France). On ne l'utilise donc **pas** pour délimiter la zone : en cas de besoin,
  filtrage par **géométrie** (polygone de frontière + `geopandas`), pas par code pays.
- Conséquence : les sources environnementales sont **paneuropéennes/globales**
  (CLC, SoilGrids, EU-DEM) pour couvrir uniformément les deux côtés de la frontière.

### 4.1 Sources

- **Occurrences** : GBIF (via `pygbif`), filtrées par taxon, bbox Vosges + Forêt-Noire,
  coordonnées obligatoires. iNaturalist en complément (souvent déjà agrégé dans GBIF).
- **Météo historique** : Open-Meteo (API archive), avec `requests-cache`.
- **Environnement** : Corine Land Cover (type de forêt), SoilGrids (type de sol),
  Copernicus EU-DEM (altitude/pente/orientation).

### 4.2 Croisement

Pour chaque observation : récupérer l'historique météo des **60 jours** précédant
l'observation (pas seulement J-14), pour capturer la "mémoire hydrique".

### 4.3 Fenêtre de saison

Définie **par espèce** à partir des observations, robuste aux outliers :

- `day_of_year` → quantile 5 % (début) et 95 % (fin).
- **Buffer de ±14 jours** (réchauffement climatique, décalages).

Ces deux bornes vont dans `configs/species.yaml`. Hors de cette fenêtre, l'API
répond par la règle métier, sans appeler le modèle.

### 4.4 Pseudo-absences (le point le plus subtil)

Le dataset de présences seules est inutilisable (Target toujours = 1). Il faut
générer des **pseudo-absences** (Target = 0).

**Principe de pertinence** : une absence n'est informative que si elle provient
d'un endroit où une présence *aurait pu* être enregistrée (**même habitat + même
effort de prospection**). Sinon, le modèle apprend "pas de randonneurs = pas de
champi" (fuite déguisée).

**Contraintes d'échantillonnage :**

- **Temporellement** : intra-saison uniquement. Pas de janvier/février en
  "absence" (le modèle réinventerait le calendrier).
- **Spatialement** : zones forestées/naturelles uniquement. Pas de points
  aléatoires dans la bbox (ville/lac/champ = absence triviale).

**Techniques retenues, par ordre de priorité :**

1. **Jitter autour des présences** : décalage spatial/temporel d'une présence
   (même site, autre semaine/année). Prouve habitat + visite.
2. **Absences cross-années sur sites à présence** : site observé en 2021 et
   2023, pas en 2022 → semaines en saison de 2022 candidates.
3. **Co-occurrence d'autres espèces** (amélioration semaine 2 si besoin) :
   présence d'autres champignons à un lieu+date sans l'espèce cible = absence
   quasi-réelle (preuve d'effort).

**Équilibrage** : générer **1 absence pour 1 présence** (50/50 par construction),
pas de poids. `scale_pos_weight` uniquement si on décide de sur-échantillonner
les absences pour refléter la rareté (option de raffinement).

### 4.4bis — Implémentation réelle & littérature (validation a posteriori)

Ce qui est implémenté dans `make_dataset.py` :

- **1 absence par présence** (50/50), même point de grille 0.25° (jitter spatial).
- **Date aléatoire intra-saison** du groupe (pas de janvier pour une espèce d'été).
- **Buffer ±14 j** de toute observation réelle même espèce + même point
  (un carpophore persiste ~1-2 semaines → évite une « absence » contaminée).
- **≥30 j de météo** avant (complétude des features).

Ce que dit la littérature (et où on se situe) :

1. **Barbet-Massin et al. 2012** (*Methods Ecol Evol*) — la référence sur
   « combien/où/comment » échantillonner les pseudo-absences. Le ratio 1:1 est
   correct pour les arbres boostés (GLM/GBM/RF) ; l'échantillonnage **aléatoire
   sur toute la zone** est biaisé par l'accessibilité (le modèle apprend
   « route = présence »).
2. **Phillips et al. 2009** — échantillonner le bruit de fond avec **le même
   biais que les présences** (target-group background). Notre règle « même cellule
   qu'une présence » en est une version spatiale stricte : cohérente, car on
   modélise **quand** une espèce fructifie (à un endroit connu), pas **où** elle vit.
3. **VanDerWal et al. 2009** — les pseudo-absences ne doivent pas s'éloigner des
   conditions connues (restriction environnementale) : « même cellule + intra-saison »
   l'applique.

**Point améliorable identifié** : l'**année** des absences est tirée
**uniformément** sur 2000-2026, alors que les présences suivent l'effort de
prospection (plus dense après ~2015) → biais temporel d'échantillonnage.
Piste : tirer l'année de l'absence **proportionnellement au nombre de présences
de l'année** (proxy d'effort), ou « même année que la présence jumelle ».

**Limite épistémologique assumée** : une pseudo-absence n'est pas une vraie
absence (personne n'a peut-être regardé ce jour-là). L'AUC mesuré est donc une
**borne inférieure** du pouvoir discriminant réel (les absences contaminées
diluent le signal).

**Résultats des tests** (LOOCV, LightGBM A_simple) :

- **Année pondérée par l'effort** : +0.01-0.02 sur 4/6 espèces (morille +0.02,
  cèpes +0.01, bolet bai +0.01, cèpe d'été +0.01) → **adopté**.
- **Buffer 7/14/30 j** : pas de vainqueur net ; 30 j plombe la morille (0.63) →
  **14 j conservé**.
- **Ratio 1:1 vs 1:2** : 1:2 n'aide que les lépiotes (+0.03), neutre ailleurs →
  **1:1 conservé**.

### 4.5 Limite assumée

Données présence-seul ⇒ impossible de garantir une "vraie" absence. Le modèle
prédit *"probabilité de fructification, habitat adapté, en saison, selon la météo"*,
avec des négatifs bruités. À assumer explicitement à la soutenance.

---

## 5. Feature engineering

### 5.1 Deux échelles temporelles

- **Fenêtre courte (J-14)** :
  - cumul précipitations ;
  - température min / max / moyenne ;
  - **choc thermique** : différence température moyenne (J-21→J-10) vs (J-10→J-0) ;
  - **bilan hydrique** : précipitations − évapotranspiration (ET0).
- **Fenêtre longue (J-60, ou "depuis le 1er juin")** — répond à "un été sec
  influence l'automne" :
  - cumul précipitations ;
  - **humidité du sol** (meilleur indicateur de sécheresse) ;
  - bilan hydrique estival.

### 5.2 Features géostatiques (remplacement des coordonnées)

Pas de lat/lon brutes dans le modèle (elles encodent le biais d'échantillonnage
et empêchent la généralisation). À la place :

- **Altitude** (Open-Meteo ou EU-DEM) ;
- **Pente + orientation** (à dériver d'un DEM, ex. Copernicus EU-DEM 25 m) ;
- **Type de forêt** (Corine Land Cover) — feuillus / résineux / mixte ;
- **Type de sol** (SoilGrids) ;
- **Humidité du sol** (Open-Meteo).

> Les coordonnées restent nécessaires en **entrée d'API** (pour aller chercher la
> météo et dériver ces features), elles ne sont simplement pas **injectées dans le
> modèle**.

> Note géographique : dans la zone, le gradient nord-sud est faible ; le vrai
> gradient est **est-ouest** (plaine du Rhin ~150 m vs Vosges/Forêt-Noire
> ~1400 m), capturé par l'altitude/pente/orientation.

---

## 6. Modélisation & évaluation

### 6.1 Jeu d'entraînement

- **Uniquement la fenêtre de saison** (présences + pseudo-absences intra-saison).
- **Jamais l'année entière** : l'hors-saison est géré par la règle métier, pas
  appris (fuite calendrier).

### 6.2 Split — temporel par blocs annuels, jamais aléatoire

Un `train_test_split` aléatoire jour-par-jour fait fuiter la météo d'une même
semaine entre train et test (le 12 oct en train, le 14 oct en test).

- **Sélection de modèle / hyperparamètres** : **CV par blocs annuels**
  (Leave-One-Year-Out ou KFold groupé par année) → distribution de métriques
  (moyenne ± écart-type), chaque année devient tour à tour test.
- **Évaluation finale honnête** : **holdout = année la plus récente**, jamais
  touchée pendant le dev (promesse "ça marche la saison prochaine").
- Vocabulaire : tester sur des années passées = backtesting ; tester sur la
  dernière année = forecasting. Les deux sont utiles.

### 6.3 Limites connues à mentionner

- **Fuite spatiale partielle** : mêmes forêts en train et test (années
  différentes). Acceptée, à signaler comme limite.
- **Peu d'années de données** (3-4) → la CV par blocs est indispensable pour ne
  pas gaspiller.

---

## 7. MLOps

- **MLflow local** par machine ; `mlflow.db`/`mlartifacts/` **gitignorés**.
- **git = source de vérité** : code + config + seed ⇒ un run est reproductible
  sur n'importe quelle machine (laptop ↔ desktop).
- **Vue unifiée des runs (optionnel, semaine 3)** : `mlflow server` sur le
  desktop + Tailscale, `MLFLOW_TRACKING_URI` pointé dessus.
- **CI/CD** : GitHub Actions (lint + tests à chaque push), build d'image Docker
  de l'API.

---

## 8. EDA à mener (avant de figer la structure)

L'EDA tranche plusieurs décisions encore ouvertes :

1. **Répartition spatiale des présences** (carte folium/kepler) → ampleur du
   clustering = biais d'échantillonnage.
2. **Nb d'observations par espèce et par année** → tranche "modèle par espèce vs
   regroupement par genre".
3. **Histogramme jour-de-l'année par espèce** → valide les fenêtres de saison,
   repère les outliers (cèpe séché en décembre).
4. **Couverture des features** (sol, altitude, humidité partout ? trous ?).
5. **Doublons / erreurs de coordonnées** (points dans un lac, coordonnées arrondies).
6. **Test de comparabilité absence/présence** (validation de la pertinence des
   pseudo-absences) :
   - *altitude / type de forêt / type de sol / pente* — les distributions doivent
     se ressembler (même habitat) ;
   - ce qui doit différer : **uniquement la météo récente** ;
   - *(En réserve)* : *distance aux routes/sentiers* (BD TOPO, proxy d'effort de
     prospection) — écarté pour l'instant, à réévaluer si besoin.
7. **EDA sur Corine Land Cover, SoilGrids et DEM** → décider si on intègre type de
   forêt, type de sol + pente/orientation dès le départ.

---

## 9. Questions encore ouvertes

- [ ] Masque de forêt pour les absences : Corine Land Cover ou jitter seul pour commencer ?
- [ ] Intégrer pente/orientation (DEM) dès le MVP, ou altitude + humidité du sol
      (Open-Meteo) suffisent ?
- [ ] Nombre exact d'espèces et choix (cèpe, girolle, + ?) — dépend de l'EDA.
- [ ] *(En réserve)* Proxy "distance aux routes/sentiers" (BD TOPO) pour le test de
      comparabilité — écarté pour l'instant.

---

## 10. Prochaines étapes

1. Initialiser le repo (pyproject Poetry, `.gitignore`, arborescence `src/`).
2. **EDA en notebook** (GBIF + Open-Meteo) : répondre aux points de la section 8.
3. Trancher les questions ouvertes (section 9) sur la base de l'EDA.
4. Implémenter `data/` (gbif, openmeteo, pseudo_absences, make_dataset).
5. Implémenter `features/` (weather, season).
6. Implémenter `models/` (train, evaluate) avec MLflow.
7. Implémenter `api/routers/growth.py` + `business_rules.py`.

---

## 11. Auto-critiques & défense (soutenance)

### 11.1 Placement des absences (procédure exacte + défense)

1 absence par présence, même cellule 0.25° que la présence (jitter spatial) :

1. **Date intra-saison** : jour tiré dans la fenêtre de fructification du groupe
   (percentiles 5/95 du jour-de-l'année observé, ±14 j).
2. **Année pondérée par l'effort** : tirée proportionnellement au nombre de
   présences de l'année (target-group background temporel).
3. **Buffer ±14 j** de toute observation réelle même espèce + même cellule (un
   carpophore persiste ~1-2 semaines → évite de compter le même événement 2 fois).
4. **≥30 j de météo** avant (complétude des features).

**Défense** : c'est le *target-group background* (Phillips 2009) — on échantillonne
le négatif avec le même biais que le positif. Cohérent car on modélise **quand** une
espèce fructifie (à un endroit connu), pas **où** elle vit. La littérature valide
(Barbet-Massin 2012 : 1:1 pour les arbres boostés ; VanDerWal 2009 : restriction
environnementale). Limite assumée : pseudo-absence ≠ vraie absence (personne n'a
peut-être regardé ce jour-là) → l'AUC est une **borne inférieure**.

### 11.2 Pourquoi restreindre à la saison (pas l'année entière)

Question attendue : « pourquoi ne pas entraîner sur l'année entière, le modèle
apprendrait les bornes min/max de fructification ? »

Réponse : c'est exactement ce qu'on veut **éviter**. Sur l'année entière, le modèle
apprend un signal **trivial** (« janvier = pas de champi, octobre = champi », via les
features de température) : l'AUC gonflerait (~0.95) sans rien apprendre sur la relation
météo→fructification. La vraie question utile est : « **étant en saison**, la météo
récente prédit-elle la pousse ? ». Le hors-saison est géré par la règle métier
(calendrier), pas appris. Le coût calcul/API n'est qu'un bénéfice secondaire (on a de
toute façon les séries complètes).

### 11.3 Que veulent dire nos chiffres (AUC → taux d'erreur)

AUC ≠ précision. AUC 0.78 = « sur une paire (jour de pousse, jour sans), le modèle
classe le bon en premier 78 % des fois ». C'est un **rang**, pas un taux de réussite.

En classification concrète (seuil 0.5, prédictions LOOCV agrégées) :

| Espèce | AUC | Précision | Rappel | Erreur |
|---|---|---|---|---|
| bolet bai | 0.78 | 71 % | 77 % | ~29 % |
| lépiotes | 0.76 | 69 % | 75 % | ~31 % |
| cèpes | 0.74 | 68 % | 71 % | ~32 % |
| girolle | 0.72 | 66 % | 65 % | ~34 % |
| cèpe d'été | 0.70 | 65 % | 64 % | ~35 % |
| morille | 0.67 | 62 % | 65 % | ~38 % |

→ On se trompe **~30-35 % du temps** (vs 50 % au hasard). Modeste mais réel : la
météo explique une partie, pas tout (patchiness du mycélium, microclimat, bruit des
présences-seules). À assumer. Note : l'AUC agrégé (prédictions concaténées) diffère
légèrement de l'AUC moyen par année (pondération par observation vs par année).

### 11.4 Holdout 2025 (année jamais vue à l'entraînement)

Entraîné sur ≤2024, testé sur 2025 (config finale : absences à effort prouvé + top-N
par espèce, 12 features) :

| Espèce | 2025 (prés) | AUC | Précision |
|---|---|---|---|
| cèpes | 293 (97) | **0.84** | 75 % |
| cèpe d'été | 197 (21) | 0.74 | 62 % |
| lépiotes | 388 (49) | 0.72 | 57 % |
| girolle | 158 (43) | 0.71 | 62 % |
| bolet bai | 281 (59) | 0.66 | 56 % |
| morille | 30 (6) | 0.54 | 40 % |

→ **Pas d'effondrement** : cèpes et cèpe d'été généralisent bien (0.84/0.74). La morille
n'a que 6 présences 2025 (peu fiable). Note : proxies exclus du test (entraînement seul).

### 11.5 Autres sources de données fongiques (explorées, non retenues)

Question attendue : « y a-t-il d'autres datasets que GBIF ? ». Exploration faite :

- **FongiBase** (FongiFrance, 1,77 M) : présent dans GBIF (3 jeux), mais ~10 obs de nos
  espèces dans notre bbox (3 cèpes, 4 girolles, 2 morilles, 1 bolet bai) → négligeable.
  Le reste n'est accessible que via leur portail (pas d'API en vrac).
- **Pilze Deutschlands** (DGfM, 4,5 M) : **absent de GBIF** (occurrences), mais l'accès
  est un site PHP à cartes **images** (pas de téléchargement structuré) → non exploitable.
- **INPN / OpenObs** : la fonge vosgienne y serait sans doute, mais **OpenObs est hors
  service** depuis la cyber-attaque MNHN (été 2025, page WAF « BunkerWeb »). Dans GBIF,
  l'INPN fonge est limité : ONF « flore et fonge » = 0 occ dans notre bbox, le reste =
  faune/bibliographie ; l'Atlas Mycologique des Hauts-de-France est hors zone (nord).
- **Andrew et al. 2018** (méta-base Dryad, 6 M) : presence-only, 8 pays.

→ **Bilan** : notre GBIF est à **75 % issu de SwissFungi** (base suisse) — les Vosges (FR)
et la Forêt-Noire (DE) sont sous-représentées dans GBIF. Les bases nationales (FongiBase,
Pilzkartierung) contiendraient des présences non redondantes, mais ne sont **pas accessibles
en vrac avec coordonnées + dates**. L'approche pseudo-absences reste nécessaire ; le biais
suisse est une limite assumée.
