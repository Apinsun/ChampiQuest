# Évaluation & calibration — méthodes

> Comment on mesure les modèles, et comment on rend leurs probabilités fiables.
> Scripts : `models/evaluate.py` (LOOCV), `models/train_final.py` (modèles finaux
> calibrés), `vision/evaluate.py` + `vision/decision.py` (vision).

## 1. Évaluation — partie 1 (pousse)

### LOOCV par année

On entraîne sur **toutes les années sauf une**, on prédit l'année retirée, répété
pour chaque année. Pourquoi : les jours J et J+1 sont très corrélés — un split
aléatoire les mettrait des deux côtés (fuite temporelle). Le bloc annuel casse
cette corrélation.

Les **proxies** (espèces co-fructifiantes ajoutées en présences) sont **exclus du
test** : ils ne servent qu'à augmenter l'entraînement, sinon ils gonfleraient l'AUC.

### Métriques (par espèce ; F1 et confusion au seuil 0,5)

| Métrique | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| **AUC** (ROC) | classement global positif vs négatif | indépendante du seuil, standard |
| **AP** (PR-AUC) | classement des *positifs* | complément de l'AUC sur données déséquilibrées |
| **Brier** | erreur quadratique des probabilités (0 = parfait, 0,25 = aléatoire) | diagnostic de calibration |
| **F1** | compromis précision/rappel | lisible, seuil fixe |
| **Confusion** | matrice cumulée sur les folds | où sont les erreurs |

Résultat : AUC LOOCV moyenne **0,811** (0,72–0,89 par espèce).

## 2. Évaluation — partie 2 (vision)

Split **par observation** (pas par image) → anti-fuite (aucun identifiant partagé
entre train/val/test). Métriques : accuracy, **macro-F1** (non pondéré : les classes
rares comptent autant que les majoritaires), précision/rappel/F1 par classe, matrice
de confusion.

Métrique de **sécurité** : fraction des toxiques/mortels prédits « comestible ».
Résultat : **0 erreur dangereuse** (sur 145 images de test toxiques/mortelles),
étranger → comestible réduit à 1,4 % (bénin).

## 3. Calibration — partie 1 (Platt, cv=5)

`CalibratedClassifierCV(method="sigmoid", cv=5)` :

- 5 folds ; pour chaque fold, un LightGBM est entraîné sur les 4 autres, puis une
  sigmoïde de Platt `σ(a·score + b)` est ajustée sur les scores **hors-fold** ;
- à la prédiction, on **moyenne les 5 probabilités calibrées**.

Pourquoi `cv=5` plutôt qu'un split 85/15 : aucune donnée perdue (chaque point sert
à l'entraînement *et* à la calibration), ensemble plus robuste.

**Point clé** : Platt est monotone → l'AUC et l'AP ne changent pas, seule la *valeur*
des probabilités est redressée. Brier brut ≈ 0,20 → la correction était justifiée.
Le modèle sauvegardé (`joblib`) expose un `predict_proba` calibré : le chemin de
prédiction (API / carte) n'a rien à changer.

## 4. Calibration — partie 2 (temperature scaling + coût)

- **Temperature scaling** : on divise les logits par une température T apprise sur
  un split de validation → vraies probabilités, l'argmax ne change pas (T ≈ 1,44 :
  le modèle était sur-confiant).
- **Matrice de coût** : choisir la classe qui minimise le **coût attendu**
  (comestible prédit sur toxique ×10, comestible sur « étranger » ×8), pas la plus
  probable.
- **Abstention** : confiance maximale < 0,6 → « je ne sais pas ».

## En résumé

| | Métrique honnête | Calibration |
|---|---|---|
| Partie 1 (pousse) | AUC LOOCV **0,811** | Platt (cv=5) |
| Partie 2 (vision) | macro-F1 **0,841**, 0 erreur dangereuse | temperature scaling + matrice de coût |
