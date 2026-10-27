# ChampiQuest — Feature Engineering

> Bilan des expériences de feature engineering (météo + regroupement d'espèces).
> Évaluation : CV Leave-One-Year-Out (ROC-AUC), jeux de features "full".

## 1. Features météo

### Baseline (v1)
7 features : `precip_J7/J14/J30`, `temp_min/max/moy_J14`, `bilan_hydrique_J30`.
→ cèpe : AUC **0.655** (logistique) / 0.645 (LightGBM A).

### Ajout des features dynamiques et longues (v2)
- **choc thermique** : temp moyenne (J-21→J-11) − (J-10→J-1) ;
- **amplitude thermique** : moyenne de (Tmax − Tmin) sur J-14 ;
- **precip_J60/J90** et **bilan_hydrique_J60/J90** (mémoire longue).

→ cèpe : AUC **0.678** (+0.023). Toutes les configs progressent ; le surapprentissage
de la config LightGBM complexe disparaît (0.601 → 0.676).

### Test d'élagage (pruning)
Retrait des fenêtres jugées redondantes (precip_J7/J14/J60, bilan_J60/J90, temp_min).
→ AUC **0.633–0.650** (en baisse). **Conclusion : on garde le jeu complet** — les
fenêtres multiples ne sont pas redondantes, elles apportent de la résolution temporelle.

### Interprétation (SHAP)
- **`bilan_hydrique_J30` est la feature n°1** (mémoire hydrique / sécheresse), sens
  cohérent : pluie nette → plus de pousse.
- La météo domine ; altitude / sol / CLC sont faibles. **Normal** : nos absences étant
  ancrées aux points de présence (même habitat), les features statiques ne discriminent
  pas présence/absence — elles serviraient à un modèle d'habitat (partie 3), pas au
  modèle de timing.

### Features relief (pente/orientation) — non retenues
Ajout de `pente`, `northness` (cos aspect), `eastness` (sin aspect), calculées par
gradient du DEM (76 m/px).
→ **Pas d'effet** (cèpes 0.704→0.699, lépiotes 0.739→0.737). L'altitude capture déjà
le relief (corrélée à la pente) et le DEM à 76 m lisse la micro-pente.
→ **Conclusion : neutres, on ne les exploite pas.**

### Round « littérature » (pluie/événements/tendance)
D'après la littérature (Kauserud 2008, Andrew 2018, Guo 2025) : pluies antécédentes =
déclencheur n°1, baisse de température = déclencheur d'automne, humidité du sol = moteur
principal. On a testé : `pluie_max_J14`, `jours_depuis_grosse_pluie`, `gel_J14`,
`tendance_temp` (pente de température sur 30 j), `bilan_hydrique_J120`.
→ **Seule `tendance_temp` apporte** (SHAP top 5, girolle +0.012). Les 4 autres sont
redondantes (pluie_max ~ precip, −0.62) ou quasi constantes (gel ≈ 0) → **écartées**.
→ La **morille** régresse (−0.07) : son déclencheur printanier est le **réchauffement du
sol**, qu'on n'a pas (`soil_temperature` horaire → écarté, remplacé par le bilan hydrique).
→ **Plateau atteint** : le signal principal (précipitations + bilan hydrique + dynamique
thermique) est capturé.

## 2. Regroupement d'espèces (augmentation de données)

### Méthode : 3 critères avant de regrouper
1. **Saison** (jour-de-l'année) — même période de fructification ?
2. **Habitat** (altitude) — même étage ?
3. **Édibilité** — même valeur pour le cueilleur ?

### Pré-test : modèle global multi-espèces
Entraîner un modèle sur les 5 espèces (espèce = feature catégorielle) → la **morille
s'effondre** (0.693 → 0.528), espèce de printemps noyée par les 96 % d'automne.
→ **Modèles spécialisés confirmés.**

### Regroupements validés

| Groupe | Ajout | Corrélation | AUC avant → après |
|---|---|---|---|
| **Cèpes** | + Boletus aereus (+56 obs) | saison + altitude ✅ | 0.678 → **0.704** |
| **Lépiotes** | + Chlorophyllum rhacodes (+92 obs) | saison + altitude ✅ | 0.692 → **0.739** |

### Regroupements rejetés

| Candidat | Raison du rejet |
|---|---|
| Boletus reticulatus (cèpe d'été) | saison d'été ≠ automne → **gardé comme modèle séparé** |
| Xerocomellus/Xerocomus (petits bolets) | altitude −150 m + édibilité médiocre |
| Craterellus tubaeformis/cornucopioides | altitude −150 m + saison d'automne ≠ girolle d'été |

### Leçon
**Genre taxonomique ≠ saison ≠ habitat.** Vérifier le moment ET le lieu (et l'édibilité)
avant de regrouper — sinon on mélange des signaux différents.

## 3. Baselines actuelles (full features, LightGBM `A_simple`)

Progression en 3 étapes : (a) zone initiale → (b) zone élargie (×2 data) →
(c) + absences à année pondérée par l'effort.

| Taxa | Zone initiale | + zone élargie | + absences effort-pondéré |
|---|---|---|---|
| bolet bai | 0.656 | 0.739 | **0.749** |
| lépiotes (procera + rhacodes) | 0.740 | 0.729 | 0.724 |
| girolle | 0.728 | 0.712 | 0.713 |
| cèpes (edulis + aereus) | 0.703 | 0.710 | 0.721 |
| cèpe d'été (reticulatus) | 0.638 | 0.696 | 0.706 |
| morilles (genre) | 0.625 | 0.662 | 0.682 |

**Lecture** : l'élargissement a surtout aidé les espèces *data-limited*
(bolet bai ×3 → +0.08, cèpe d'été +0.06, morille +0.04) ; les grosses espèces
plafonnent (~0.71-0.73, bornées par le signal). L'**année d'absence pondérée par
l'effort** (target-group background temporel) apporte +0.01-0.02 sur 4 espèces
(dont morille +0.02). La morille reste en retrait (température du sol absente).
La régression logistique s'effondre sur la morille (0.42 < 0.5) → relation
printemps **non-linéaire**, LightGBM nécessaire.

**Configs LightGBM** : `A_simple` (7 feuilles, prof 4) ≥ `B_moyen` (15 feuilles)
> `C_complexe` (31 feuilles) → le modèle simple gagne (le complexe surapprend).

**Pseudo-absences validées** : buffer 7/14/30 j et ratio 1:1/1:2 testés →
**14 j + 1:1** reste le meilleur compromis (buffer 30 j plombe la morille 0.63,
ratio 1:2 neutre sauf lépiotes). Seule vraie amélioration : année pondérée par l'effort.

### Round « sol » (NASA POWER : température de surface + humidité du sol)

Le proxy « bilan hydrique » (précip − ET0) a été remplacé/complété par les vraies
variables de sol, en **daily** via NASA POWER (MERRA-2, sans clé) :

- `TS` (Earth Skin Temperature, °C) → `temp_sol_J14`, `rechauffement_sol` (J-7 − J-30) ;
- `GWETTOP` (Surface Soil Wetness, 0-1) → `humidite_sol_J30`.

Résultats (LightGBM A_simple) : **bolet bai +0.020 (0.769)**, **morille +0.013 (0.695,
son meilleur score)**, girolle +0.009, les espèces d'automne neutres.

→ **Valide la littérature** : le déclencheur printanier de la morille est le
**réchauffement du sol** (Guo 2025), et l'**humidité du sol** (Boukary 2024) est le
moteur principal de la fructification — le proxy précip−ET0 la sous-estimait.

### ⚠️ Bascule NASA POWER → Open-Meteo (sol, pour la carte live)

NASA POWER est de la **réanalyse** (MERRA-2) avec ~2-3 mois de délai → indisponible en
temps réel. Pour la **carte de prédiction du jour** (partie 3), on a basculé les
features sol sur **Open-Meteo** (`soil_temperature_0_to_7cm_mean`,
`soil_moisture_0_to_7cm_mean`, ERA5-land), disponibles en **archive ET en forecast**.

- Mêmes définitions de features : `temp_sol_J14`, `rechauffement_sol`, `humidite_sol_J30`.
- **Résultat : AUC moyenne 0.761 vs 0.767 (NASA POWER)** → **pas de perte significative**
  (−0.006, dans le bruit des folds LOOCV, σ ~0.1-0.18). Détail : cèpes 0.717 (+0.011),
  lépiotes 0.796 (−0.033), girolle 0.744 (+0.014), morille 0.694 (−0.023),
  bolet bai 0.770 (+0.004), cèpe d'été 0.846 (−0.007).
- **Gain** : toutes les features sont désormais forecast-capables → carte live possible.

### Random search LightGBM (expérience) — `A_simple` conservé

40 itérations de random search (espace orienté régularisation) sur 2 espèces :
cèpes +0.017 (0.735), bolet bai +0.005 (0.774). Gains **dans le bruit** de la LOOCV
et **optimistes** (sélection sur les mêmes folds → fuite légère). Aucune config ne
transforme le score ; les meilleures restent dans un régime proche de `A_simple`.

→ **`A_simple` conservé** : le modèle simple (7 feuilles, prof 4, 50 arbres) est déjà
proche de l'optimum. Ce résultat valide le refus de sur-ingénierier.

### Analyse SHAP : morille (printemps) vs bolet bai (automne)

`temp_sol_J14` est la **feature n°1 pour les deux espèces** (|SHAP| = 0.44 morille,
0.58 bolet bai) — la température du sol (NASA POWER) était bien le chaînon manquant.

**Direction opposée selon la phénologie :**

- **Morille (printemps)** : fenêtre de sol **étroite** (~6-12 °C). Sol froid (<6 °C)
  SHAP −0.17, sol chaud (>12 °C) SHAP −0.93 → les deux extrêmes défavorisent, le pic
  est au milieu. `rechauffement_sol` corr +0.59 (le réchauffement favorise) → confirme
  le déclencheur « réchauffement du sol » (Guo 2025).
- **Bolet bai (automne)** : sol froid favorise (+0.20), sol chaud défavorise (−0.27)
  → fructifie quand le sol **refroidit**. `bilan_hydrique_J30` corr **+0.90** : la pluie
  récente (30 j) est LE déclencheur (Kauserud 2008).

**Humidité** : chez la morille elle est toujours élevée (fonte des neiges) → précondition,
pas discriminant. Chez le bolet bai, le bilan hydrique 30 j domine.

**Bémol** : les corrélations SHAP↔feature peuvent être confondues (ex. `humidite_sol_J30`
« sec → positif » chez le bolet bai = artefact de saison, n=103). Le rang d'importance est
robuste, les directions extrêmes à prendre avec prudence.

### Élagage de features (data-driven) — set « pruned » 12 features

Corrélations + SHAP agrégé → redondances majeures : `temp_sol_J14` ↔ `temp_mean_J14`
**+0.97**, `tendance_temp` ↔ `rechauffement_sol` +0.85, fenêtres de pluie/bilan imbriquées ;
et quasi-mortes (|SHAP| < 0.03) : `clc_code` (0.003), relief (pente/northness/eastness/
altitude ~0.01), texture du sol (phh2o/clay/soc ~0.02).

Set « pruned » (12 features) : `precip_J7/J30/J90`, `temp_min/max_J14`, `choc_thermique`,
`tendance_temp`, `amplitude_thermique`, `bilan_hydrique_J30`, `temp_sol_J14`,
`humidite_sol_J30`, `sand_0-5cm`.

→ **Meilleur ou égal sur les 6 espèces** (cèpes +0.015, cèpe d'été +0.013, morille
+0.002…) avec **moins de la moitié des features**. LightGBM tolère la colinéarité, mais
les features redondantes/mortes ajoutaient du bruit (surtout sur les petits datasets).
Le set pruned est désormais **recommandé**.

**Règle de choix quand deux features sont corrélées** (|corr| > 0.75) :

1. on garde la plus importante selon |SHAP| ;
2. à SHAP proche, préférence écologique : **sol > air** (`temp_sol` plutôt que
   `temp_mean`), **extrêmes > moyenne** (`temp_min`/`temp_max` plutôt que `temp_mean`),
   **bilan hydrique > pluie brute** (il ajoute l'ET0) ;
3. pour les fenêtres cumulatives imbriquées (J7→J90), on garde un **éventail
   court/moyen/long** (déclencheur + mémoire) plutôt que les top-SHAP ;
4. cas limite ~0.84 (`precip_J30` ↔ `bilan_J30`) : on garde les deux car la différence
   (l'ET0) est écologiquement porteuse et les deux ont un SHAP élevé.

Le critère final reste l'**AUC LOOCV** : on teste et on garde le meilleur.

### Augmentation par proxies (espèces co-occurrentes)

**Idée** : la co-fructification (`cofruct`) fuitait (feature indisponible en prod). On
l'utilise autrement : les espèces qui co-apparaissent fortement avec notre cible sont
ajoutées comme **présences** de la cible à l'entraînement (features restent 100 % météo).

**Méthode de sélection d'un proxy** : espèce F vue dans la même cellule 0.25° à ±7 j
d'une fraction ≥ `co_occ` des observations de la cible, avec un recouvrement de saison
≥ `saison`. Les scores sont sauvegardés dans `data/processed/proxies_scores.csv`
(pour filtrer sans recalculer).

**Résultats (LightGBM A_simple, pruned 12 features)** :

| Espèce | sans proxy | co_occ≥0.4 | co_occ≥0.5 | top-3 |
|---|---|---|---|---|
| lépiotes | 0.648 | 0.803 | 0.642 | **0.794** |
| cèpe d'été | 0.728 | 0.754 | 0.754 | **0.782** |
| bolet bai | 0.704 | 0.700 | 0.695 | **0.777** |
| girolle | 0.771 | 0.721 | 0.734 | **0.774** |
| cèpes | 0.708 | **0.720** | 0.701 | 0.702 |
| morille | 0.705 | 0.653* | **0.720** | 0.684 |

\* la morille à 0.4 était buggée (proxies = ses propres espèces de Morchella, corrigé
en excluant les taxons déjà téléchargés). Le correctif a aussi nettoyé son pool d'effort.

**Conclusion** : le **top-N par espèce** (seuil relatif) est le bon réglage — top-3 améliore
la moyenne de **0.711 → 0.752** (+0.041), avec 4/6 espèces en hausse. La morille reste la
seule perdante (probable bruit RNG : le générateur est global, l'ordre d'augmentation change
ses absences — à corriger par un RNG par espèce).

**Réglage final** : RNG par espèce (reproductibilité) + top-N par espèce
(`PER_SPECIES_TOPK`) : cèpes=1, lépiotes=5, girolle=0, morille=0, bolet bai=3, cèpe d'été=1.

| Espèce | AUC LOOCV final |
|---|---|
| cèpe d'été | **0.853** |
| lépiotes | 0.829 |
| bolet bai | 0.766 |
| girolle | 0.730 |
| morille | 0.717 |
| cèpes | 0.706 |

Moyenne **0.767**. Bémols assumés : (1) le N par espèce est choisi sur les mêmes folds
LOOCV (sélection légèrement optimiste) ; (2) les proxies sont **exclus du test** (ils ne
servent qu'à l'entraînement — sinon ils gonflent l'AUC en tant que positifs « faciles »).

---

## Élagage re-validé avec proxies (post-augmentation)

Le set `pruned` avait été figé **avant** l'ajout des proxies. Vérification faite *a
posteriori* sur le dataset final (proxies inclus) — LOOCV, LightGBM A_simple :

| Espèce | full | pruned | Δ (pruned − full) |
|---|---|---|---|
| cèpes | 0.713 | 0.717 | +0.004 |
| lépiotes | 0.805 | 0.796 | −0.009 |
| girolle | 0.752 | 0.744 | −0.008 |
| morille | 0.711 | 0.694 | −0.017 |
| bolet bai | 0.773 | 0.770 | −0.003 |
| cèpe d'été | 0.842 | 0.846 | +0.004 |
| **moyenne** | **0.766** | **0.761** | **−0.005** |

`full` ressort en tête de **+0.005**, dans le bruit (la morille, dataset le plus petit,
oscille à elle seule de ±0.017). Ablation par groupe de features :

| config | AUC moy | Δ vs pruned |
|---|---|---|
| pruned (12) | 0.761 | — |
| pruned + `clc_code` | 0.763 | +0.002 |
| pruned + relief (alt/pente/orient.) | 0.760 | −0.001 |
| pruned + sol (ph/clay/soc) | 0.767 | +0.006 |
| pruned + fenêtres météo | 0.765 | +0.004 |
| full (26) | 0.766 | +0.005 |

**Conclusion** : les proxies n'ont **pas** ressuscité les features habitat — `clc_code`
(+0.002) et le relief (−0.001) restent quasi-morts. Le petit gain de `full` vient de la
texture du sol (ph/clay/soc, +0.006) et de fenêtres météo redondantes (+0.004), pas de
l'occupation du sol → `pruned` conservé (parcimonie). Seule nuance écologique : le pH/argile/SOC
est le seul groupe « habitat » avec un léger signal — cohérent (le pH compte), mais trop
faible pour 3 features de plus au MVP.

---

---

## Ablation : modèle unique vs N modèles, et feature `day_of_year`

Deux questions de soutenance, testées sur le **même protocole LOOCV** (leave-one-year-out)
et les **mêmes données** (8 espèces, LightGBM `A_simple`, set pruned) :

1. Un **seul modèle** (espèce = feature catégorielle `taxon_key`) ferait-il aussi bien
   qu'**un modèle par espèce** ?
2. La **saison** est-elle mieux gérée par une règle (fenêtre + gate) ou **apprise** par
   le modèle (feature `day_of_year`) ?

| Variante | AUC moy. par espèce | Δ vs baseline |
|---|---|---|
| **N modèles (baseline, 12 features)** | **0.760** | — |
| N modèles + `day_of_year` | **0.811** | **+0.051** |
| Modèle unique (`taxon_key`) | 0.703 | −0.057 |
| Modèle unique + `day_of_year` | 0.789 | +0.029 |

### Résultat 1 : la phénologie intra-saison portait un gros signal

Ajouter `day_of_year` aux modèles par espèce : **0.760 → 0.811** (+0.051). La fenêtre
de fructification (percentiles 5–95 % ± 14 j) borne le **domaine** (hors saison = non
applicable), mais à l'intérieur, la phénologie fine (début vs plein vs fin de saison)
restait inexploitée : le modèle n'avait aucune feature de date.

→ **Décision** : `day_of_year` ajouté au set pruned (**13 features**), modèles finaux
retrainés. La gate « hors saison » reste nécessaire : le modèle n'a jamais vu de dates
hors fenêtre (les absences sont échantillonnées **dans** la fenêtre, cf. morille doy
67–148), il ne peut pas apprendre cette frontière. Pas de fuite : le jour-de-l'année est
connu à l'inférence (on prédit pour une date donnée).

### Résultat 2 : le modèle unique perd — la morille s'effondre

Sans `day_of_year`, le modèle unique tombe à **0.703**, et la **morille à 0.465** (pire
que le hasard : classement inversé). Cause : la morille est la **seule espèce de
printemps** (doy 67–148), noyée dans ~60 000 lignes d'espèces d'automne. Un seul
LightGBM à 50 arbres répartit sa capacité entre 8 espèces → la morille (1,9 % des
données) n'a presque pas de capacité dédiée, et la structure partagée « automne » la
prédit à l'envers. Avec `day_of_year`, elle récupère (0.698) — preuve que c'est la
phénologie divergente, pas un manque de données.

Même avec `day_of_year`, **N modèles (0.811) > modèle unique (0.789)** : mutualiser
~60 000 lignes ne compense pas la dilution de capacité ni le mélange printemps/automne.

### Leçon

- **N modèles** à 8 espèces à phénologies opposées : la mutualisation ne paie pas, et
  l'espèce minoritaire trinque. Un modèle unique se justifierait à 100+ espèces, pas ici.
- **Métrique** : l'AUC **par espèce** (slicée) est la comparaison honnête. L'AUC globale
  du modèle unique (0.743–0.818) est gonflée par `taxon_key` (taux de base par espèce).

Script : `src/champiquest/models/evaluate_ablation.py` (set de base gelé, reproductible).

---

## Perspectives : habitat / essences (non implémenté)

**Limite structurante** : le modèle growth est plafonné par la **précision des coordonnées
GBIF** (≈ 0.5–1 km). Relier une observation à « l'arbre sous lequel elle a poussé » est
illusoire — c'est la même cause que la classe CLC majoritaire « terres arables » (champi de
lisière géolocalisé dans le champ voisin). Plus la couche habitat est fine, plus ce bruit
pénalise.

**Piste essence** : la **BD Forêt® V2 (IGN)** — open data, nomenclature 32 essences
(chêne, hêtre, pin, sapin, épicéa…), résolution 0,5 ha — serait le bon outil pour une
**couche d'habitat potentiel par espèce** (cèpe → chênaies/hêtraies/pessières ; girolle →
hêtraies-sapinières). C'est un projet de type **SDM presence-only** (MaxEnt-like), où la
précision des coordonnées devient moins critique (agrégation à la cellule). Pas pour le MVP :
l'ablation montre que le type de forêt (`clc_code` 311/312/313) n'apporte déjà rien au modèle
growth (+0.002).
