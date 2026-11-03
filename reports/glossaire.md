# Glossaire

Explications pédagogiques des termes techniques utilisés dans le
[rapport de conduite de projet](rapport-conduite-projet.md) et les `reports/`.
Chaque entrée pointe vers le document détaillé correspondant.

---

## AUC (ROC)

L'**aire sous la courbe ROC** mesure la capacité d'un modèle à **classer** les
exemples : séparer les positifs des négatifs.

- **1,0** = classement parfait ; **0,5** = hasard (autant dire pile ou face) ;
  en dessous de 0,5, le modèle fait pire que le hasard (classement inversé).
- On la calcule **par espèce** (et pas globalement) : chaque espèce a sa propre
  prévalence, et un score global serait gonflé par les espèces faciles.

→ Détails et résultats : `partie-1-prediction-pousse.md`, `feature-engineering.md`.

## LOOCV (Leave-One-Year-Out)

Méthode de **validation croisée temporelle** : on retire **une année entière** du jeu
d'entraînement, on entraîne sur toutes les autres, puis on prédit l'année retirée.
On répète pour chaque année, et on moyenne.

**Pourquoi par année et pas par tirage aléatoire ?** Les données météo sont
**corrélées dans le temps** (le jour J ressemble au jour J+1). Un split aléatoire
mélangerait des jours voisins entre train et test → le modèle « verrait » l'avenir
proche, et le score serait **optimiste** (fuite de données). Le découpage par blocs
annuels garantit que le test est réellement « jamais vu ».

→ Détails : `partie-1-prediction-pousse.md` §6.2.

## GBIF (et pourquoi les coordonnées sont imprécises)

Le **Global Biodiversity Information Facility** est une base mondiale qui agrège des
observations naturalistes (ici, des champignons) : espèce, date, localisation.

Pourquoi la localisation n'est pas précise :
- les observations sont souvent **floutées ou arrondies** (protection d'espèces,
  saisie manuelle, GPS approximatif) ;
- en pratique, la précision est de l'ordre de **0,5 à 1 km**.

Conséquence directe pour le projet : impossible de relier une observation à
« l'arbre sous lequel elle a poussé » — c'est le plafond du modèle de pousse.

→ Détails : `partie-1-prediction-pousse.md` §4.0.

## Pseudo-absences

Le jeu de données ne contient que des **présences** : on sait où un champignon **a été
vu**, jamais où il **n'était pas**. Or un modèle a besoin des deux pour apprendre.

On fabrique donc des **négatifs « honnêtes »** : au même endroit et à la même période,
un **autre** champignon a été observé sans le nôtre → c'est un vrai « on a regardé,
pas trouvé » (absence **à effort prouvé**), et pas « personne n'est sorti ».

→ Détails : `partie-1-prediction-pousse.md` §4.4.

## Calibration (temperature scaling)

Un classifieur sort des **scores**, qui ne sont pas forcément de **vraies
probabilités** (un score de 0,9 peut ne correspondre qu'à 70 % de réussite réelle).

La calibration **ajuste** ces scores — ici via un unique paramètre de « température »
appris sur un jeu de validation — pour qu'un score de 0,8 veuille réellement dire
« 80 % de chances ». C'est indispensable avant d'appliquer une décision à coût.

→ Détails : `ajout-especes-calibration.md` §2.

## Matrice de coût

Toutes les erreurs **ne se valent pas**. Annoncer « comestible » un champignon
**toxique** est bien plus grave que l'inverse. On attribue donc un **coût** à chaque
type d'erreur (comestible-sur-toxique = ×10, comestible-sur-étranger = ×8), et on
choisit la décision qui **minimise le coût attendu** — quitte à s'abstenir
(« je ne sais pas ») quand la confiance est trop faible.

→ Détails : `ajout-especes-calibration.md` §2.3–2.5.

## SHAP

Méthode d'**interprétation** qui mesure la contribution de chaque feature à la
prédiction d'un modèle (même « boîte noire »). Elle a permis d'identifier des
features quasi inutiles (relief, occupation du sol) et d'élaguer le modèle sans
perdre en performance.

→ Détails : `feature-engineering.md`.

## Fine-tuning / ConvNeXt

**Fine-tuning** : partir d'un réseau de neurones **pré-entraîné** sur des millions
d'images (ImageNet) et le **ré-entraîner** sur nos données — bien plus efficace que
d'entraîner depuis zéro avec peu d'images.

**ConvNeXt** : architecture moderne de classification d'images, utilisée pour la
reconnaissance d'espèces (20 classes).

→ Détails : `partie-2-reconnaissance-image.md`.
