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

## 3. Baselines actuelles (full features, modèles spécialisés/regroupés)

| Taxa | Meilleur AUC |
|---|---|
| lépiotes (procera + rhacodes) | **0.739** |
| girolle | 0.716 |
| cèpes (edulis + aereus) | 0.704 |
| morilles (genre) | 0.693 |
| bolet bai | 0.668 |
| cèpe d'été (reticulatus) | 0.642 |
