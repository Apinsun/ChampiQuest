# AGENTS.md

> Document de collaboration humain ↔ agent. Évolutif : à mettre à jour
> incrémentalement quand notre façon de travailler change.

## Contexte & rôle

- Projet **ChampiQuest** (ce dépôt).
- Partie 1 en cours : prédiction de pousse météo → fructification.
  → Voir `reports/partie-1-prediction-pousse.md` (design & décisions).
- **Utilisateur = étudiant AI Engineer** : il doit comprendre et s'approprier
  tout le code produit. Aucun "vibecode" aveugle.
- Objectif : soutenance finale.

## Échéancier

- **Semaine 1** : partie 1 — prédiction météo → pousse.
- **Semaine 2** : partie 2 — reconnaissance d'espèce par image.
- **Semaine 3** : polissage, FastAPI, préparation de la présentation.

## Langue

- Tout en **français** : discussions, docs, reports, commentaires de code,
  messages de commit.

## Règles de collaboration

1. **Demander avant de créer ou supprimer un fichier.**
   Exception : les fichiers techniques triviaux (`__init__.py`, `.gitignore`)
   peuvent être créés sans demande, en le signalant après coup.
2. **Demander avant d'ajouter une dépendance** au projet.
3. **Reports avec parcimonie** : pas de génération en chaîne. Un report doit
   retenir (notamment pour la soutenance) les points critiques, les réflexions
   intéressantes, tout ce qui a mené aux designs et aux résultats.
   Qualité > quantité.
4. **Comprendre avant de coder, expliquer en codant** : les choix de design et
   d'architecture sont discutés et argumentés *avant* l'implémentation ; le
   "pourquoi" est expliqué, pas seulement le "quoi". Les simplifications par
   rapport à la production sont signalées.
5. **Vérifier par exécution avant d'affirmer** : ne jamais prétendre qu'un code
   marche sans l'avoir exécuté (scripts, tests) et rapporter ce qui a été testé.
6. **Anti-sur-ingénierie** : rester sur le périmètre MVP de l'échéancier ;
   alerter en cas de scope creep plutôt que d'ajouter de la complexité.
7. **Secrets & coûts** : jamais de `.env` ni de clé/token commité ; **demander
   avant** toute commande longue ou coûteuse (entraînement de plusieurs minutes,
   gros téléchargements GBIF, batch d'appels API).
8. **Tests unitaires** : demander explicitement avant d'en créer, en expliquant
   à quoi le test correspond et ce qu'il vérifie réellement.

## Git / commits

- **L'utilisateur commit lui-même.** L'agent prépare les changements et propose
  un message de commit, sans exécuter `git commit`.

## Ce que l'agent peut faire sans demander

- Lire des fichiers, explorer le dépôt.
- Commandes en lecture seule : `git status`, `git log`, `ls`, `grep`, `find`, etc.
- Créer des fichiers techniques triviaux (`__init__.py`, `.gitignore`), en
  signalant après coup.
- Proposer (designs, plans, messages de commit, contenu de fichiers) sans exécuter.

## Ce qui nécessite une validation explicite de l'utilisateur

- Création / suppression de tout fichier métier : code, config, reports, docs, tests.
- Ajout de toute dépendance.
- Commandes longues ou coûteuses (entraînement, téléchargements, API en masse).
- Tout commit (l'utilisateur commit lui-même).
