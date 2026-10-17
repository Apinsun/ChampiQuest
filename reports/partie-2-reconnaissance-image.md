# Partie 2 — Reconnaissance d'espèce par image

> Document de cadrage : « vers où on va ». Évolutif — à mettre à jour quand le
> design avance, comme `partie-1-prediction-pousse.md`.

## 1. Objectif

Reconnaître l'espèce d'un champignon à partir d'une photo, dans la continuité du
projet ChampiQuest :

- **Partie 1** (bouclée) : *quand/où* ça pousse (météo → fructification).
- **Partie 2** (ce document) : *de quelle espèce il s'agit* (image → espèce).

L'angle retenu est la **sécurité de cueillette** : reconnaître nos comestibles **et
surtout ne pas les confondre avec leurs sosies toxiques**.

## 2. Décisions actées

| Question | Décision |
|---|---|
| Structure | **Monorepo** — sous-package `champiquest.vision`, sans toucher à la partie 1 |
| Périmètre | **6 cibles + sosies** (toxiques **et** non-toxiques) |
| Approche | **Fine-tuning** d'un backbone pré-entraîné (pas d'embeddings+LightGBM) |
| Backbone | **ConvNeXt-Base (384px)** puis **ViT-B/16 ImageNet-21k** en comparaison |
| Librairie | **timm** (+ torch, torchvision) |
| Données | **DF20** (nos classes) + **COCO** (négatifs « pas de champi ») |
| Négatifs | classe **« pas de champi »** → modèle à **N+1 classes** |
| Calibration | **temperature scaling** → probabilités fiables |
| Sécurité | **matrice de coût** (comestible prédit = toxique réel → fortement pénalisé) |
| Tracking | **MLflow** (réutilise l'installation de la partie 1) |
| Produit | pas d'appli, mais **contrat d'API** réfléchi (soutenance) |

## 3. Données

### 3.1 Danish Fungi 2020 (DF20)

- Référence académique : Picek et al., *Danish Fungi 2020* (WACV 2022, ~90 citations).
- ~**296 000 images**, ~**1600 espèces**, photos **in-situ**, labels validés par des
  mycologues, **espèces européennes**.
- Fournit un **`observation_id`** par photo : plusieurs photos d'une même observation
  partagent le même label d'espèce → base du split anti-fuite (§9).

### 3.2 Négatifs « pas de champi » — COCO

- DF20 ne contient que des champignons → il faut des images « sans champignon ».
- **COCO** (scènes réelles) retenu pour la vitesse et la diversité.
- ⚠️ piège à éviter : des négatifs trop « studio » (ImageNet) créeraient un raccourci
  « fond neutre = pas un champignon ». COCO (scènes in-situ) limite ce biais.
- Si des raccourcis apparaissent : compléter par des **non-fungi iNaturalist/GBIF**
  (plantes, insectes — pipeline GBIF déjà maîtrisé en partie 1).
- Volume cible : ~2000-5000 images.

### 3.3 Autres datasets (écartés)

`FungiTastic` (trop gros), `MO106` (petit), `Mushrooms-215` (trop petit),
`FungiCLEF/FGVC8` (compétition, hors scope).

## 4. Classes : liste RÉELLE (validée après EDA DF20)

16 classes, **6076 images** au total (hors classe « pas de champi »).

| Classe | Rôle | Images |
|---|---|---|
| *Boletus edulis* (cèpe de Bordeaux) | comestible | 901 |
| *Boletus reticulatus* (cèpe d'été) | comestible | 343 |
| *Cantharellus cibarius* (girolle) | comestible | 673 |
| *Macrolepiota procera* (lépiote élevée) | comestible | 472 |
| *Imleria badia* (bolet bai) | comestible | 925 |
| *Morchella* spp. (morille) | comestible | 173 |
| *Rubroboletus satanas* (bolet Satan) | toxique | 155 |
| *Lepiota cristata* (lépiote crêtée) | toxique | 486 |
| *Gyromitra esculenta* (fausse morille) | **mortel** | 189 |
| *Lepiota castanea* (lépiote châtain) | **mortel** | 198 |
| *Tylopilus felleus* (bolet amer) | non-toxique | 358 |
| *Hygrophoropsis aurantiaca* (fausse girolle) | non-toxique | 620 |
| *Boletus subtomentosus* (bolet subtomenteux) | non-toxique | 198 |
| *Boletus pinophilus* (cèpe des pins) | non-toxique | 70 |
| *Chlorophyllum rachodes* (lépiote déguenillée) | non-toxique | 281 |
| *Verpa bohemica* (verpe de Bohême) | non-toxique | 34 |

**Ajustements vs la liste initiale (appris de l'EDA)** :
- *Omphalotus olearius* : absent de DF20 → la girolle est couverte par
  *Hygrophoropsis aurantiaca* (620).
- *Chlorophyllum molybdites* : absent (espèce nord-américaine) → remplacé par
  *Chlorophyllum rachodes* (281) + les petites lépiotes mortelles.
- *Xerocomus subtomentosus* : présent sous l'ancien nom *Boletus subtomentosus* (198).
- *Lepiota brunneoincarnata* (44) : remplacée par *Lepiota castanea* (198, mortelle),
  mieux représentée.
- Morille : *esculenta* (142) + *conica* (31) fusionnées en « *Morchella* spp. » (173).
- Classes faibles : *Verpa bohemica* (34) et *Boletus pinophilus* (70) — seuil à
  réévaluer selon les premiers résultats.

## 5. Matériel

- **RTX 3090 (24 Go)** — CUDA disponible. Le code vit dans **ce dépôt** (monorepo),
  l'env Poetry est installé sur cette machine.
- **torch 2.11.0+cu128** (stable, installé depuis l'index PyTorch cu128).
- ⚠️ **cuDNN planté** (`Invalid handle. Cannot load symbol cudnnGetVersion`) quel que
  soit torch/cuDNN (testé cu130 ET cu128) → probablement le **driver 595.99.02
  (CUDA 13.2, bêta)** incompatible avec cuDNN 9.x. Workaround :
  `torch.backends.cudnn.enabled = False` (kernels CUDA natifs, ~10-20 % plus lents).
  À réexaminer après une mise à jour du driver NVIDIA.

## 6. Structure du code

```
src/champiquest/vision/
  __init__.py
  datasets.py  # chargement DF20 + COCO, split par observation, augmentations
  models.py    # backbone pré-entraîné + tête de classification (N+1)
  train.py     # boucle de fine-tuning (AMP, scheduler, early stopping)
  evaluate.py  # accuracy, macro-F1, matrice de confusion, par-classe
  decision.py  # calibration (temperature scaling) + matrice de coût + abstention
  explain.py   # Grad-CAM
  predict.py   # inference end-to-end (photo(s) → décision)
configs/vision.yaml   # classes, hyperparamètres, coûts
data/raw/df20/        # images (gitignoré)
data/raw/coco/        # négatifs (gitignoré)
reports/partie-2-reconnaissance-image.md
```

## 7. Pipeline

1. Télécharger DF20 (sous-ensemble de nos classes) + COCO.
2. **EDA** : nombre d'images/classe, distribution, échantillons visuels.
3. **Split par observation** (pas par image) → anti-fuite.
4. Backbone pré-entraîné + tête **N+1 classes** (espèces + « pas de champi »).
5. **Fine-tuning** complet, LR bas (~1e-4), augmentations (random resized crop,
   flip, color jitter), AdamW + cosinus + AMP, early stopping sur macro-F1.
6. **Calibration** (temperature scaling) sur un split de validation dédié.
7. **Évaluation** : métriques + matrice de confusion + décision à coût.

## 8. Sécurité : calibration + coût asymétrique + abstention

C'est le cœur « produit » du modèle. Trois briques qui se combinent :

1. **Calibration** : temperature scaling → `P(espèce|x)` est une vraie probabilité.
2. **Matrice de coût** (à la décision, pas dans la loss) :
   - coût(prédire *comestible* | vrai *toxique*) = **élevé** (ex. 10)
   - coût(prédire *toxique* | vrai *comestible*) = **faible** (ex. 1)
   - coût(correct) = 0
3. **Décision bayésienne** : on prédit la classe qui minimise le coût attendu
   `Σ_y P(y|x)·coût(c,y)`.

**Flux de décision final** :
1. si la classe « pas de champi » domine → **rejet** (« je ne vois pas de champignon ») ;
2. sinon → décision espèce avec la matrice de coût : **comestible** / **toxique** /
   **abstention** (si confiance trop faible).

Le déséquilibre de classes, lui, se gère pendant l'entraînement (poids de classe /
focal loss) — orthogonal au coût asymétrique.

## 9. Fusion multi-vues

- **MVP** : le modèle est **mono-photo** ; au moment de l'inférence on **agrège** les
  prédictions des N photos (moyenne des logits). Aucun entraînement multi-vue.
- **Option (stretch)** : entraîner un **set classifier** faiblement supervisé via
  l'`observation_id` de DF20 (N photos → N embeddings → agrégation → 1 prédiction,
  supervisée par le label d'observation). Pas besoin de labels de vue.
- **Précision** : l'`observation_id` garantit *même espèce* (pas forcément *même
  individu* : une observation peut couvrir un groupe). C'est suffisant — même
  bénéfique — pour l'agrégation.
- Le **guidage par vue** (demander chapeau/lamelles/pied) relève de l'API, pas de
  l'entraînement ; l'auto-reconnaissance de vue est un stretch (CLIP zero-shot).

## 10. Contrat d'API (pour la soutenance, implémenté en partie 3)

```
POST /api/identify
  body: { "images": [ { "data": "<base64>", "view": "cap|gills|stem|whole|auto" } ] }
  → 200 {
    "status": "ok" | "no_mushroom" | "low_confidence",
    "decision": "edible" | "toxic" | "abstain",   # issue de la matrice de coût
    "top_predictions": [
      { "species": "Boletus edulis", "common_name": "Cèpe de Bordeaux",
        "probability": 0.87, "edible": true }
    ],
    "confidence": 0.87,
    "calibrated": true
  }
```

- `images[]` accepte 1..N photos du même spécimen → le serveur agrège (moyenne logits).
- `view` est **optionnel** (guidage utilisateur) ; la valeur `auto` = pas de vue fournie.
- `status` : `no_mushroom` si la classe négative domine, `low_confidence` si
  abstention, `ok` sinon. `decision` n'est `edible` que si la confiance ET le coût le
  justifient.

## 11. Métriques

- accuracy top-1, **macro-F1** (jeu déséquilibré), précision/rappel **par classe**.
- **Matrice de confusion** : lire directement les confusions cible↔sosie.
- **Coverage vs accuracy** pour l'abstention (on refuse, on mesure le gain).
- **Taux d'erreur dangereuse** (comestible prédit = toxique réel) → la métrique clé
  de sécurité, à minimiser en priorité.

## 12. Explicabilité

**Grad-CAM** sur des exemples de test, en particulier les paires cible↔sosie, pour
montrer *où* le modèle regarde (chapeau, lamelles, pied…). Continuité avec le SHAP
de la partie 1.

## 13. Dépendances à ajouter

À valider avant ajout au `pyproject.toml` :

- `torch` (build CUDA)
- `torchvision`
- `timm` (backbones pré-entraînés : ConvNeXt, ViT-21k, Swin, DINOv2…)

(Pillow arrive avec torchvision.)

## 14. Feuille de route

1. [x] Rapport de cadrage + squelette `champiquest.vision/`.
2. [x] Installer l'env Poetry + ajouter torch/torchvision/timm.
3. [x] Télécharger DF20 (nos classes) + COCO (négatifs).
4. [x] EDA : nb images/classe, disponibilité des classes, échantillons.
5. [x] Finaliser la liste des classes (seuil d'images) + `configs/vision.yaml`.
6. [x] `datasets.py` : chargement + split par observation + augmentations.
7. [x] `models.py` + `train.py` : fine-tuning ConvNeXt-Base, MLflow.
8. [x] `evaluate.py` + `decision.py` : calibration, coût, abstention.
9. [ ] `explain.py` : Grad-CAM.
10. [ ] `predict.py` + contrat d'API.

## 15. Questions ouvertes

- Seuil minimal d'images/classe (après l'EDA, selon les résultats).
- Backbone final : ConvNeXt-Base vs ViT-B/16-21k (à comparer) ; DINOv2 en option.
- Set classifier multi-vue : stretch ou non ?
- Auto-reconnaissance de vue (CLIP) : stretch ou non ?

## 16. Résultats (premier entraînement complet)

ConvNeXt-Base (384px), 20 epochs, AdamW lr 1e-4, batch 32, AMP, poids de classe.

| Métrique | Valeur |
|---|---|
| test accuracy | **0.896** |
| test macro-F1 | **0.847** |
| température calibrée | 1.426 (sur-confiance → adoucissement) |
| abstentions | 74/1073 (6,9 %) |
| erreur dangereuse (argmax) | 1,4 % |
| erreur dangereuse (coût) | **0,0 %** |

**Confusions principales (bénignes)** : cèpes ↔ bolet bai ↔ bolet amer ↔ bolet
subtomenteux ; girolle ↔ fausse girolle. Les 2 confusions dangereuses
(*Gyromitra esculenta* et *Rubroboletus satanas* → girolle) sont **éliminées par la
matrice de coût** (d'où 0,0 %).

Graphes : `reports/partie2_confusion_matrix.png`, `partie2_per_class_metrics.png`,
`partie2_training_curves.png` (aussi loggés dans MLflow).
