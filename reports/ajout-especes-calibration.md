# Ajout de nouvelles espèces & calibration — méthode, décisions, résultats

> Document de synthèse : comment on a étendu ChampiQuest à deux nouvelles espèces
> (trompette de la mort, pied de mouton), et en profondeur comment marche la
> couche de décision « sécurité » (calibration + coût).

---

## 1. Ajouter une espèce : le processus complet

Objectif : traiter les nouvelles espèces **comme les 6 existantes** — un modèle
LightGBM par espèce (growth), une classe par espèce (vision), pas un « modèle
champi comestible » générique.

### 1.1 Vérifier la donnée AVANT de coder

| espèce | occurrences GBIF (all_fungi.csv) | images DF20 |
|---|---|---|
| Trompette de la mort (*Craterellus cornucopioides*) | 942 | 251 (classe 343) |
| Pied de mouton (*Hydnum repandum*) | 1233 | 371 (classe 628) |

Les deux étaient **déjà téléchargées** : `all_fungi.csv` contient tous les
champignons de la zone, et DF20 contient les deux espèces. → **0 re-téléchargement**.

### 1.2 Partie 1 — growth (un LightGBM par espèce)

1. **Config** (`species.yaml`) : `hydnum_repandum` ajouté (clé GBIF `2554716`) ;
   `craterellus_cornucopioides` y était déjà.
2. **CSV par espèce** : `make_dataset` lit un CSV par espèce dans `data/raw/gbif/`.
   Le pied de mouton n'en avait pas → créé depuis `all_fungi.csv` (1233 lignes).
3. **Météo** : les points de grille (0.25°) des 2 espèces étaient **déjà tous en
   cache** (72/72) → **0 appel API, 0 quota consommé**.
4. **RNG des absences** : corrigé au passage (cf. §1.4).
5. **Proxies** : candidats sélectionnés par co-occurrence (co_occ ~0.5–0.6), balayage top-N :

| espèce | sans proxy | top-3 | top-5 |
|---|---|---|---|
| trompette | 0.689 | **0.791** | 0.792 |
| pied de mouton | 0.566 | 0.720 | **0.753** |

→ **trompette = top-3, pied de mouton = top-5** (réglage par espèce).

6. **Modèles finaux** : 8 LightGBM (A_simple, set pruned), sauvegardés `joblib` + MLflow.
7. **Carte / API / frontend** : `SPECIES` étendu à 8, carte régénérée, 2 cases à cocher.

**Résultat growth** (LOOCV, 8 espèces, moyenne **0.760**) :

| espèce | LOOCV |
|---|---|
| cèpe d'été | 0.846 |
| lépiotes | 0.805 |
| **trompette de la mort** | **0.791** |
| bolet bai | 0.764 |
| **pied de mouton** | **0.734** |
| cèpes | 0.733 |
| girolle | 0.727 |
| morille | 0.679 |

### 1.3 Partie 2 — vision (2 classes dans le ConvNeXt)

1. **Config** (`vision.yaml`) : 2 classes **comestibles**, sans sosie dangereux
   (trompette → *Craterellus tubaeformis* comestible ; pied de mouton → autres
   *Hydnum* comestibles). Donc **aucune classe de sécurité à ajouter**.
2. **Extraction** : métadonnées (6076 → 6698) + 622 images du tarball DF20.
3. **Retrain** : ConvNeXt-Base 384px, 20 époques, **20 classes** (18 espèces +
   `no_mushroom` + `other_fungus`).
4. **Bug trouvé en évaluant** : 22 images de *Hydnum repandum* étaient **à la fois**
   dans la classe cible et dans `other_fungus` (duplication héritée de l'époque où
   le pied de mouton n'était pas une cible). Conséquence : « étranger → comestible »
   montait artificiellement à 3.4 %. → retirées de `other_fungus` (980 → 958),
   **re-train propre**.

**Résultat vision** (20 classes) :

| métrique | valeur |
|---|---|
| test acc / macro-F1 | 0.879 / 0.841 |
| température calibrée | 1.437 |
| erreur dangereuse (coût) | **0 %** |
| étranger → comestible (coût) | 1.4 % (confusions bénignes) |

### 1.4 Leçon : le RNG des absences

Avant, le seed des absences était la **position alphabétique** de l'espèce
(`enumerate(sorted(taxon_key))`). Ajouter une espèce décalait donc les seeds de
toutes les suivantes → leurs absences (et leurs AUC) changeaient sans raison.

Fix : un **dict en dur** (`RNG_SEEDS`) avec un seed figé par espèce. Les valeurs
0–12 reproduisent les anciennes positions (→ le dataset existant reste identique),
13–15 sont réservées aux nouvelles. **Ne jamais changer une valeur.**

Leçon générale : **toute source d'aléa doit être nommée, figée et indépendante de
l'ordre d'ajout des données.**

---

## 2. La calibration, en profondeur

### 2.1 Pourquoi calibrer ?

Le `softmax` d'un réseau de neurones produit des scores **qui ressemblent à des
probabilités mais n'en sont pas** : un ConvNeXt est typiquement **sur-conflant**
(il annonce 0.99 alors que la vraie fréquence de succès est ~0.85).

Or toute la couche de décision « sécurité » repose sur `P(y|x)`. Si ces nombres
sont faux, la matrice de coût (qui multiplie `P(y|x)` par des coûts) prend des
décisions sur des probabilités mensongères. **Calibrer = rendre `P(y|x)` honnête** :
quand le modèle dit « 0.8 », il doit se tromper ~20 % du temps.

### 2.2 Temperature scaling (notre méthode)

**Idée** : une seule transformation, un seul paramètre `T > 0`, appris sur un split
de validation **dédié** (jamais le test) :

```
softmax_T(z)_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

- `T > 1` **adoucit** (réduit la confiance, corrige la sur-confiance) ;
- `T < 1` **accentue** (renforce les pics) ;
- `T = 1` = softmax classique.

**Propriété clé** : `z_i / T` est une transformation **monotone** → l'`argmax` ne
change jamais. La température **ne change pas l'accuracy/F1**, elle ne corrige que
la **confiance**. C'est ce qui la rend sûre à appliquer après l'entraînement.

**Estimation** : on minimise la **NLL** (negative log-likelihood) sur le split de
validation :

```
T* = argmin_T  − (1/N) Σ log softmax_T(z_n)[y_n]
```

Chez nous : `minimize_scalar` (scipy) borné sur `[0.1, 10]`.

**Nos valeurs** : `T ≈ 1.44` (18 et 20 classes) → le modèle était bien
sur-conflant, et on l'adoucit d'un facteur ~1.4.

### 2.3 La matrice de coût (décision bayésienne)

Une fois calibré, on **ne prend pas l'argmax**. On choisit la classe `c` qui
minimise le **coût attendu** :

```
décision(x) = argmin_c  Σ_y P(y|x) · C(c, y)
```

où `C(c, y)` = coût de prédire `c` alors que la vérité est `y`. Coûts asymétriques :

| prédit ↓ / vrai → | comestible | toxique/mortel | autre champi | no_mushroom |
|---|---|---|---|---|
| **comestible** | 0 | **10** | **8** | 2 |
| toxique/mortel | 1 | 0 | 2 | 1 |
| autre champi / no_mushroom | 1 (faux rejet) | 1 | 0 | 0 |

L'asymétrie est le cœur du design : **dire « comestible » sur un champignon toxique
est 10× plus cher** que l'inverse. Et « comestible » sur un champignon étranger
(potentiellement toxique) coûte 8.

### 2.4 L'abstention

Si la confiance maximale `max P` est **sous un seuil** (0.6), on refuse de répondre
(`low_confidence`). Principe : **mieux vaut ne rien dire que se tromper**.

### 2.5 Résultats avant / après coût

| métrique | argmax (brut) | **après calibration + coût** |
|---|---|---|
| erreur dangereuse (18 classes) | 2.1 % | **0.0 %** |
| étranger → comestible (18 classes) | 5.4 % | **0.0 %** |
| erreur dangereuse (20 classes) | 2.8 % | **0.0 %** |
| étranger → comestible (20 classes) | 6.2 % | **1.4 %** |

Le **1.4 % résiduel** (20 classes) est **bénin** : ce sont des sosies comestibles
(*Hydnum rufescens* → pied de mouton). L'erreur **dangereuse** reste à 0 %.

### 2.6 Pourquoi la dangerosité n'est PAS dans la loss

On aurait pu mettre la matrice de coût **dans la fonction de perte** (coût-sensitive
learning). On a choisi de l'appliquer **en aval** car :

1. **mathématiquement équivalent** si le modèle est bien calibré (l'argmin du coût
   attendu est la décision optimale de Bayes) ;
2. **plus explicable** : le réseau reste un classifieur « pur », et toute la
   logique de sécurité vit dans un module dédié, lisible et testable ;
3. **plus flexible** : changer les coûts (ex. être plus/moins conservateur) ne
   nécessite **aucun re-entraînement**.

**Trade-off assumé** : le modèle est **conservateur** — quand il hésite entre un
comestible et un champignon étranger, il préfère dire « champignon non identifié »
plutôt que « comestible » (ex. une trompette reconnue à 0.73 mais classée
`other_fungus`). C'est le comportement voulu : **la précision de la comestibilité
est sacrifiée au profit de la sécurité**.

---

## 3. Leçons transverses

1. **La donnée prime** : ajouter une espèce a été « facile » car les occurrences et
   les images étaient déjà là. Le goulot, c'est la donnée (quota météo), pas le code.
2. **Vigilance sur les duplications** : la fuite des 22 images ne se voyait qu'en
   scrutant une métrique (« étranger → comestible ») qui montait bizarrement. Toujours
   croiser les sources quand on change les classes.
3. **Nommer et figer l'aléa** : le bug RNG ne se manifestait que quand on ajoutait
   une espèce. Tout seed doit être explicite et stable.
4. **Séparer reconnaissance et décision** : le réseau classe ; la couche coût décide.
   Ça rend chaque étage testable et la sécurité auditable.
