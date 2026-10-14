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
