# Partie 2 — Reconnaissance d'espèce par image

> Document de cadrage : « vers où on va ». Évolutif — à mettre à jour quand le
> design avance, comme `partie-1-prediction-pousse.md`.

## 1. Objectif

Reconnaître l'espèce d'un champignon à partir d'une photo, dans la continuité du
projet ChampiQuest :

- **Partie 1** (bouclée) : *quand/où* ça pousse (météo → fructification).
- **Partie 2** (ce document) : *de quelle espèce il s'agit* (image → espèce).

L'angle retenu est celui de la **sécurité de cueillette** : reconnaître nos espèces
comestibles **et surtout ne pas les confondre avec leurs sosies toxiques**.

## 2. Décisions actées

| Question | Décision |
|---|---|
| Structure | **Monorepo** — nouveau sous-package `champiquest.vision`, sans toucher aux modules de la partie 1 |
| Périmètre | **6 espèces cibles + leurs sosies** (~12-16 classes) |
| Approche | **Fine-tuning** d'un CNN pré-entraîné (pas d'embeddings+LightGBM) |
| Matériel | Développement et entraînement sur **un autre PC équipé d'une RTX 3090** (GPU) |
| Données | **Danish Fungi 2020 (DF20)**, réduit à nos classes |
| Tracking | **MLflow** (réutilise l'installation de la partie 1) |

## 3. Données : Danish Fungi 2020 (DF20)

- **Référence académique** : Picek et al., *Danish Fungi 2020 — Not Just Another
  Image Recognition Dataset* (WACV 2022, ~90 citations).
- ~**296 000 images**, ~**1600 espèces**, photos **in-situ** (Atlas fongique danois),
  labels validés par des mycologues.
- Espèces **européennes** → nos 6 espèces y figurent a priori (à vérifier, §12).

**Pourquoi DF20** plutôt que d'autres datasets Kaggle :

- `FungiTastic` (2024) : plus gros et plus riche (masques, climat), mais trop lourd
  pour notre besoin.
- `MO106` : propre mais petit (29 k images / 106 espèces).
- `Mushrooms-215` : trop petit (~4 images/classe).
- `FungiCLEF / FGVC8` : compétition, millions d'images → hors scope.

> Alternative de secours si DF20 manque de classes : les **images GBIF** (on a déjà
> l'accès via la partie 1) pour compléter des espèces rares — mais bruitées, à
> n'utiliser qu'en appoint.

## 4. Périmètre : espèces cibles + sosies

| Cible (comestible) | Sosie(s) — danger |
|---|---|
| Cèpe de Bordeaux (*Boletus edulis*) | Bolet amer (*Tylopilus felleus*) — amer, gâche le plat ; Bolet Satan (*Rubroboletus satanas*) — toxique (gastro) |
| Cèpe d'été (*Boletus reticulatus*) | idem |
| Girolle (*Cantharellus cibarius*) | Fausse girolle (*Hygrophoropsis aurantiaca*) ; Clitocybe de l'olivier (*Omphalotus olearius*) — toxique (gastro) |
| Morille (*Morchella* spp.) | Fausse morille (*Gyromitra esculenta*) — **mortelle** (gyromitrine) |
| Bolet bai (*Imleria badia*) | Bolet subtomenteux (*Xerocomus subtomentosus*) — confusion fréquente (comestible) |
| Lépiote élevée (*Macrolepiota procera*) | Lépiote brun incarnat (*Lepiota brunneoincarnata*) — **mortelle** (amatoxines) ; Faux parasol (*Chlorophyllum molybdites*) — toxique (gastro) |

- ~**12-16 classes** au total (le « + sosies » peut être ajusté selon la
  disponibilité réelle dans DF20).
- Règle de filtrage : ne garder que les classes ayant **assez d'images** (seuil à
  fixer, ex. ≥ 50-100 images/classe) ; compléter/supprimer des sosies en conséquence.

## 5. Matériel & contraintes

- Machine actuelle : **pas de GPU** → le fine-tuning se fait sur le **PC RTX 3090**.
- Le code reste dans **ce dépôt** (monorepo) → portable, l'autre PC clone le repo,
  installe les dépendances et télécharge DF20 localement (dans `data/` gitignoré).

## 6. Structure du code (monorepo)

```
src/champiquest/
  data/          # partie 1 — inchangé
  features/      # partie 1 — inchangé
  models/        # partie 1 — inchangé
  vision/        # partie 2 — NOUVEAU
    __init__.py
    datasets.py  # chargement DF20, split par observation, augmentations
    models.py    # backbone pré-entraîné + tête de classification
    train.py     # boucle de fine-tuning (AMP, scheduler, early stopping)
    evaluate.py  # accuracy, macro-F1, matrice de confusion, par-classe
    explain.py   # Grad-CAM
    predict.py   # inference sur 1 image (+ seuil d'abstention)
data/raw/df20/        # images (gitignoré)
configs/vision.yaml   # liste des classes (cibles + sosies), hyperparamètres
reports/partie-2-reconnaissance-image.md   # ce document
```

## 7. Pipeline de fine-tuning

1. **Télécharger DF20** (sur le PC GPU), filtrer sur nos classes.
2. **Split par observation/spécimen** (pas par image) : plusieurs photos d'un même
   champignon partagent le même identifiant d'observation → elles ne doivent jamais
   être réparties entre train et test (fuite). Même rigueur que la partie 1.
3. **Backbone pré-entraîné ImageNet** + nouvelle tête de classification (nombre de
   classes = nos espèces). Défaut : `ResNet50` ou `EfficientNet-B0` ; alternatives à
   tester si le temps le permet : `ConvNeXt-Tiny`, `ViT-B/16` (souvent meilleurs en
   classification fine-grained).
4. **Fine-tuning complet** (toutes les couches) avec un **learning rate bas**
   (ex. 1e-4), car on part de poids ImageNet.
5. **Augmentations** : random resized crop, flip horizontal, color jitter ;
   éventuellement MixUp/CutMix si le surapprentissage apparaît.
6. **Optimisation** : AdamW, scheduler cosinus, **mixed precision (AMP)** sur 3090,
   early stopping sur la macro-F1 de validation.

## 8. Méthodologie

- **Split** : train / validation / test par **observation** (stratifié par espèce si
  possible). Le test n'est touché qu'à la fin.
- **Métriques** : accuracy top-1, **macro-F1** (le jeu est déséquilibré : cèpes et
  girolles sur-représentés), précision/rappel **par classe**, matrice de confusion
  (lire directement les confusions cible↔sosie).
- **Gestion du déséquilibre** : poids de classe ou focal loss.
- **MLflow** : log hyperparamètres, métriques, matrice de confusion, artefacts
  (modèle, exemples Grad-CAM). Réutilise `mlflow-workflow.md`.

## 9. Explicabilité

**Grad-CAM** sur des exemples de test — en particulier sur les paires cible↔sosie
pour montrer *où* le modèle regarde (chapeau, pied, lamelles…). Continuité avec le
SHAP de la partie 1 : « on explique nos modèles, pas seulement leurs scores ».

## 10. Bonus (si le temps le permet)

- **Abstention « je ne sais pas »** : seuil de confiance sous lequel le modèle
  refuse de répondre. Très pertinent pour une app de cueillette (mieux vaut ne rien
  dire que se tromper). Mesurable via la courbe *coverage vs accuracy*.
- **Lien partie 1 ↔ partie 2** : l'app finale combine « météo = ça pousse ici »
  (partie 1) + « l'image confirme l'espèce » (partie 2).
- ~~Classification hiérarchique genre→espèce~~ : over-engineering pour le MVP, écarté.

## 11. Dépendances à ajouter

À valider avant ajout au `pyproject.toml` (règle 2) :

- `torch` (+ build CUDA sur le PC GPU)
- `torchvision`
- `timm` (backbones pré-entraînés) — ou torchvision seul si on s'en tient à ResNet/EfficientNet
- éventuellement `albumentations` (augmentations) — ou torchvision.transforms

## 12. Feuille de route

1. [ ] Vérifier la disponibilité de **chaque classe** dans DF20 (nombre d'images),
       fixer le seuil minimal et finaliser la liste cibles+sosies.
2. [ ] Créer `configs/vision.yaml` (classes, hyperparamètres).
3. [ ] Créer le squelette `champiquest.vision/` (`__init__.py` + modules vides).
4. [ ] Sur le PC GPU : clone du repo, env Poetry + CUDA, téléchargement DF20.
5. [ ] `datasets.py` : chargement, split par observation, augmentations.
6. [ ] `train.py` + `models.py` : fine-tuning, MLflow.
7. [ ] `evaluate.py` : métriques + matrice de confusion.
8. [ ] `explain.py` : Grad-CAM sur les paires cible↔sosie.
9. [ ] Bonus : abstention « je ne sais pas ».

## 13. Questions ouvertes

- Seuil minimal d'images par classe (50 ? 100 ?).
- Backbone de départ (ResNet50/EfficientNet-B0, ou directement ConvNeXt/ViT ?).
- Faut-il inclure des **sosies non toxiques** (ex. bolet subtomenteux) pour montrer
  les confusions « inoffensives » en plus des toxiques ?
- `timm` ou `torchvision` seul ?
