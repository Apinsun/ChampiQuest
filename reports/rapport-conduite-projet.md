# Rapport de conduite de projet — ChampiQuest

> Synthèse **gestion de projet** du projet technique. Les détails techniques sont
> dans les `reports/` dédiés ; les termes marqués d'un * renvoient au
> [glossaire](glossaire.md).

---

## 1. Contexte & objectifs

Projet **personnel technique** mené dans le cadre de la formation **AI Engineer**
(OpenClassrooms), sur **3 semaines** (une partie par semaine). Objectif : démontrer,
sur un cas réel, la capacité à mener un projet d'IA **de bout en bout** — de la
donnée brute au déploiement.

Trois parties :

| Partie | Objectif | Livrable |
|---|---|---|
| 1 | Prédire **quand/où** un champignon va fructifier | 8 modèles LightGBM + carte |
| 2 | **Reconnaître** l'espèce sur photo, en sécurité | ConvNeXt 20 classes + démo |
| 3 | **Exposer** les modèles | API FastAPI + carte + Docker/CI-CD |

---

## 2. Analyse des besoins

**Contexte** : projet personnel technique pour démontrer une capacité à mener un
projet IA complet (donnée → modèle → déploiement).

### Le besoin, à deux niveaux

Le besoin se décline à deux niveaux : celui du **porteur de projet** — démontrer, sur
un cas réel, ma capacité à mener un projet d'IA de bout en bout — et celui de
l'**utilisateur final**, le cueilleur (amateur ou pro), qui veut savoir **où et quand**
chercher un champignon, et **identifier** une espèce sur photo **en sécurité**.

### Contraintes

Plusieurs contraintes ont cadré la solution. Les données GBIF* ne fournissent que des
**présences** (pas d'absences), ce qui impose de fabriquer des négatifs honnêtes ;
l'absence de **GPU en production** impose une inférence CPU ; la **sécurité** exige
zéro erreur dangereuse (un comestible annoncé sur toxique) ; les **quotas des API
météo** imposent un stockage local avec cache ; enfin, le **délai** est de trois semaines.

### Matrice des risques

| Risque | Impact | Parade |
|---|---|---|
| **Technique** — environnement GPU instable (driver CUDA bêta, cuDNN planté) | entraînement ralenti | cuDNN désactivé (workaround) ; déploiement en inférence CPU |
| **Données** — GBIF presence-only, coordonnées floutées (~1 km) | plafonne la précision | pseudo-absences « effort prouvé » ; habitat fin écarté après test |
| **Métier/sécurité** — annoncer comestible un toxique | critique | calibration + matrice de coût + abstention → 0 erreur dangereuse |
| **Juridique/éthique** — santé publique, licences des données | responsabilité | disclaimer pédagogique dans la démo ; données en usage recherche |

### Critères de succès (mesurables)

| Critère | Objectif | Atteint |
|---|---|---|
| Partie 1 — AUC* LOOCV* par espèce (moyenne) | > 0,7 | **0,811** (min 0,722 – max 0,885) |
| Partie 2 — erreur dangereuse (coût) | 0 | **0,0 %** (sur 145 images de test toxiques/mortelles) |
| Partie 3 — API + carte + démo déployées, tests verts | oui | **oui** (14 tests) |

### Périmètre MVP

8 espèces (partie 1), 20 classes (partie 2), zone Vosges / Forêt-Noire. Écartés et
documentés comme perspectives : habitat fin (BD Forêt), SDM, cron d'automatisation.
Règle appliquée partout : **anti-sur-ingénierie** (on alerte plutôt que d'ajouter de
la complexité hors périmètre).

---

## 3. Élaboration de la solution

Chaque choix clé a été validé par une **baseline ou une ablation** — jamais « parce
que c'est mieux ».

| Choix | Alternative écartée | Justification |
|---|---|---|
| **Un modèle par espèce** (LightGBM) | un seul modèle multi-espèces | ablation : le modèle unique fait moins bien (0,703) et la morille s'effondre à 0,465 |
| **LightGBM** (`A_simple`) | régression logistique | baseline : LightGBM supérieur, sans sur-ajustement |
| **Pseudo-absences « effort prouvé »** | absences aléatoires | des négatifs crédibles (« on a regardé, pas trouvé ») |
| **Calibration + matrice de coût** | argmax brut | sécurité : 2,8 % d'erreurs dangereuses → 0 % |
| **Store Parquet + ERA5 + forecast** | appels API en direct | quotas + latence + reprise sur panne |

Détails : architecture (`architecture.md`), features et ablations
(`feature-engineering.md`), partie 1 (`partie-1-prediction-pousse.md`), partie 2
(`partie-2-reconnaissance-image.md`), partie 3 (`partie-3-api-carte.md`).

---

## 4. Réalisation

### Planning & arbitrages

| Semaine | Prévu | Réalisé / imprévu |
|---|---|---|
| 1 | partie 1 (météo → pousse) | + changement de fournisseur sol (indispo au forecast), fuite des proxies corrigée |
| 2 | partie 2 (vision) | + workaround GPU (cuDNN désactivé, entraînement sur RTX 3090) |
| 3 | API, carte, Docker/CI-CD | + arbitrage périmètre : cron et habitat fin reportés hors MVP |

Le délai a été tenu en arbitrant sur le **périmètre**, jamais sur la vérification :
baselines, ablations et tests sont restés non négociables.

### Méthode

Le projet a été mené de façon **itérative** : partir d'une baseline simple, la mesurer
(LOOCV*), puis l'améliorer par petites étapes dont chacune est validée par la mesure,
plutôt que de tenter d'emblée une solution complexe. Chaque résultat reporté a été
**vérifié par exécution** avant d'être affirmé, et les décisions — avec leur « pourquoi » —
ont été consignées au fil de l'eau dans les `reports/`, ce qui garantit la traçabilité
du raisonnement.

### Obstacles rencontrés & solutions

Le projet a rencontré quelques difficultés de fond, au-delà des aléas techniques habituels.

**L'environnement GPU** a d'abord été une source de friction : des problèmes de drivers
(CUDA en version bêta, incompatibilités cuDNN) ont obligé à **désactiver cuDNN** à
l'entraînement — le fine-tuning tournait bien sur une **RTX 3090**, mais ~10-20 % plus
lent — et, plus largement, à concevoir le déploiement en **inférence CPU**. Ce choix,
contraint au départ, s'est révélé cohérent avec une démo légère qui n'exige pas de GPU.

**Le manque de données** a été la difficulté centrale : GBIF ne fournit que des présences,
et certaines espèces n'étaient représentées que par un faible nombre d'observations. On
a pallié ce manque en **augmentant** les données de deux façons : des pseudo-absences
« à effort prouvé » pour fabriquer des négatifs crédibles, et des **proxies** (des
espèces co-fructifiantes ajoutées comme présences de la cible). C'est cette augmentation
qui a fait passer la moyenne LOOCV d'environ 0,71 à 0,76, avant même l'ajout de la
saisonnalité.

**La qualité géographique** a imposé une limite de fond : il s'est avéré difficile de
trouver des données avec une **géolocalisation précise**, les coordonnées GBIF étant
souvent floutées (≈ 0,5–1 km). Cette imprécision a plafonné le modèle de pousse et a
conduit à écarter les features d'habitat fin (occupation du sol, relief), quasi sans
signal — une limite assumée et documentée.

À ces difficultés s'ajoutent deux leçons de rigueur : une **fuite de données** (des
proxies utilisés en test gonflaient artificiellement l'AUC), corrigée en excluant les
proxies du test ; et une **non-reproductibilité** (un générateur aléatoire global rendait
les résultats instables), corrigée par des graines par espèce, figées.

### MLOps

Quatorze tests (pytest) couvrent l'anti-fuite, la décision à coût, le store météo et
l'API — l'anti-fuite par un test qui vérifie qu'aucun identifiant d'observation n'est
partagé entre les splits de la vision, la décision à coût par quatre tests unitaires sur
la couche de sécurité (un toxique très probable n'est jamais annoncé comestible, une
confiance faible déclenche l'abstention, la calibration ne change jamais l'argmax). La
**CI** les exécute sur toutes les branches et les pull requests, tandis que la
**CD** construit et pousse l'image Docker vers GHCR sur `main` et les tags, avec un cache
de layers qui rend les pulls incrémentaux.

**Ressources maîtrisées** : store météo ~4 Mo, 8 modèles LightGBM ~400 Ko, checkpoint
vision 335 Mo, cartes ~40 Ko par date. Le seul poste lourd est l'**image Docker (~9 Go à
tirer)**, qui embarque torch CUDA alors que l'inférence est CPU — un allègement documenté,
non retenu au MVP.

**Maintien en condition opérationnelle** (identifié, non outillé au MVP) : surveiller le
**data drift** (des saisons qui se décalent dégraderaient le modèle de pousse) et re-tester
le store après tout changement de fournisseur météo.

---

## 5. Bilan

### Résultats vs objectifs

| Partie | Résultat |
|---|---|
| 1 — pousse | AUC LOOCV moyenne **0,811** (8 espèces, 13 features) |
| 2 — vision | accuracy **0,879**, macro-F1 **0,841**, **0 erreur dangereuse** |
| 3 — API/carte | endpoints + carte + démo déployés, 14 tests verts |

### Limites (assumées et documentées)

Le modèle de pousse est plafonné par la **précision des coordonnées GBIF (~1 km)** :
impossible de relier une observation à « l'arbre sous lequel elle a poussé ». La vision
couvre vingt classes, et son extension à d'autres espèces reste possible grâce au
pipeline d'ajout documenté (`ajout-especes-calibration.md`). Enfin, la carte est générée
manuellement — son automatisation (cron) est une perspective.

### Ce que je ferais différemment

Si c'était à refaire, je vérifierais dès le départ qu'une feature d'entraînement est
réellement **disponible au moment de la prédiction**. On a en effet utilisé, pour
l'entraînement, des variables de sol que notre premier fournisseur ne livrait qu'avec
retard — donc inutilisables pour un forecast « au jour J » —, ce qui a obligé à changer
de fournisseur de données en cours de route. C'est la leçon la plus importante du
projet : aligner les données d'entraînement sur celles réellement disponibles à
l'inférence.

### Axes d'amélioration du projet

Deux pistes prolongeraient le projet. D'abord, **fiabiliser l'environnement GPU** : le
fine-tuning a bien eu lieu sur une RTX 3090, mais avec cuDNN désactivé (incompatibilité
driver, ~10-20 % de vitesse perdus) — réactiver cuDNN rendrait les futurs entraînements
plus confortables. Ensuite, trouver des données **géolocalisées plus précises** pour
faire jouer davantage l'**environnement** — exposition, habitat — dans les prédictions :
c'est aujourd'hui le principal verrou, borné par la précision des coordonnées GBIF.

---

**Renvois** : [glossaire](glossaire.md) · [architecture](architecture.md) ·
[feature engineering](feature-engineering.md) · [partie 1](partie-1-prediction-pousse.md) ·
[partie 2](partie-2-reconnaissance-image.md) · [partie 3](partie-3-api-carte.md) ·
[ajout d'espèces & calibration](ajout-especes-calibration.md)
